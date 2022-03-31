/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "fp16_80.h"
#include "integer.h"

#include <complex>
typedef std::complex<double> Complex;

static Complex rint(const Complex & z) { return Complex(::rint(z.real()), ::rint(z.imag())); }

static void forward_out(Complex * const z, const size_t n, const size_t m_io, const Complex * const w)
{
	for (size_t l = 0; l < m_io; ++l)
	{
		for (size_t m = n / 2, s = 1; m >= m_io; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex r = w[s + j];
				for (size_t i = 0; i < m; i += m_io)
				{
					const size_t k = 2 * m * j + i + l;
					const Complex u = z[k], v = z[k + m] * r;
					z[k] = u + v; z[k + m] = u - v;
				}
			}
		}
	}
}

static void backward_out(Complex * const z, const size_t n, const size_t m_io, const Complex * const w)
{
	const size_t s_io = (n / 2) / m_io;

	for (size_t l = 0; l < m_io; ++l)
	{
		for (size_t m = m_io, s = s_io; m <= n / 2; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex r = std::conj(w[s + j]);
				for (size_t i = 0; i < m; i += m_io)
				{
					const size_t k = 2 * m * j + i + l;
					const Complex u = z[k], v = z[k + m];
					z[k] = u + v; z[k + m] = (u - v) * r;
				}
			}
		}
	}
}

static void pass1(Complex * const z, const size_t n, const size_t m_io_2, const Complex * const w)
{
	const size_t s_io_2 = (n / 2) / m_io_2;

	for (size_t l = 0; l < s_io_2; ++l)	// parallel
	{
		// forward_in
		for (size_t m = m_io_2, s = 1; m >= 1; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex r = w[(s_io_2 + l) * s + j];
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m_io_2 * l + 2 * m * j + i;
					const Complex u = z[k], v = z[k + m] * r;
					z[k] = u + v; z[k + m] = u - v;
				}
			}
		}

		// square
		for (size_t j = 0; j < 2 * m_io_2; ++j)
		{
			const size_t k = 2 * m_io_2 * l + j;
			z[k] *= z[k];
		}

		// backward_in
		for (size_t m = 1, s = m_io_2; m <= m_io_2; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const Complex r = std::conj(w[(s_io_2 + l) * s + j]);
				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m_io_2 * l + 2 * m * j + i;
					const Complex u = z[k], v = z[k + m];
					z[k] = u + v; z[k + m] = (u - v) * r;
				}
			}
		}
	}
}

static double pass2(Complex * const z, const size_t n, const size_t m_io, const Complex * const w, Complex * const f,
					const bool dup, const double b, const double sb, const double isb, const double fsb)
{
	const size_t s_io = (n / 2) / m_io;
	const double b_inv = 1.0 / b, sb_inv = 1.0 / sb, n_inv = 1.0 / n, g = dup ? 2 : 1;

	double err = 0;
	for (size_t lh = 0; lh < m_io / 2 / 4; ++lh)	// parallel
	{
		// backward_out
		for (size_t ll = 0; ll < 8; ++ll)
		{
			const size_t l = 8 * lh + ll;
			for (size_t m = m_io, s = s_io; m <= n / 2; m *= 2, s /= 2)
			{
				for (size_t j = 0; j < s; ++j)
				{
					const Complex r = std::conj(w[s + j]);
					for (size_t i = 0; i < m; i += m_io)
					{
						const size_t k = 2 * m * j + i + l;
						const Complex u = z[k], v = z[k + m];
						z[k] = u + v; z[k + m] = (u - v) * r;
					}
				}
			}
		}

		// carry_out
		for (size_t ll = 0; ll < 4; ++ll)
		{
			for (size_t j = 0; j < (n / 2) / (m_io / 2); ++j)
			{
				const size_t k_4 = (m_io / 2) / 4 * j + lh;
				const size_t k = 4 * k_4 + ll;
				const Complex o = n_inv * (z[2 * k + 0] + sb * z[2 * k + 1]), oi = rint(o), d = fabs(o - oi);
				const Complex f_i = f[k_4] + g * oi;
				err = std::max(err, std::max(d.real(), d.imag()));
				const Complex f_o = rint(f_i * b_inv);
				const Complex r = f_i - f_o * b;
				f[k_4] = f_o;
				const Complex irh = rint(r * sb_inv);
				z[2 * k + 0] = (r - isb * irh) - fsb * irh;
				z[2 * k + 1] = irh;
			}
		}
	}
	return err;
}

static void pass3(Complex * const z, const size_t n, const size_t m_io, const Complex * const w, Complex * const f,
				  const double b, const double sb, const double isb, const double fsb)
{
	const double b_inv = 1.0 / b, sb_inv = 1.0 / sb;

	for (size_t lh = 0; lh < m_io / 2 / 4; ++lh)	// parallel
	{
		// carry_in
		for (size_t j = 0; j < (n / 2) / (m_io / 2); ++j)
		{
			const size_t k_4 = (m_io / 2) / 4 * j + lh;
			const size_t carry_i = (k_4 == 0) ? n / 2 / 4 - 1 : k_4 - 1;
			Complex ff = (k_4 == 0) ? Complex(-f[carry_i].imag(), f[carry_i].real()) : f[carry_i];
			f[carry_i] = Complex(0, 0);
			for (size_t ll = 0; ll < 4; ++ll)
			{
				const size_t k = 4 * k_4 + ll;
				const Complex o = z[2 * k + 0] + sb * z[2 * k + 1], oi = rint(o);
				ff += oi;
				const Complex f_o = rint(ff * b_inv);
				const Complex r = ff - f_o * b;
				ff = f_o;
				const Complex irh = rint(r * sb_inv);
				z[2 * k + 0] = (r - isb * irh) - fsb * irh;
				z[2 * k + 1] = irh;
				if (ff == Complex(0, 0)) break;
			}
			if (ff != Complex(0, 0)) std::cout << "Error!" << std::endl;	// TODO
		}

		// forward_out
		for (size_t ll = 0; ll < 8; ++ll)
		{
			const size_t l = 8 * lh + ll;
			for (size_t m = n / 2, s = 1; m >= m_io; m /= 2, s *= 2)
			{
				for (size_t j = 0; j < s; ++j)
				{
					const Complex r = w[s + j];
					for (size_t i = 0; i < m; i += m_io)
					{
						const size_t k = 2 * m * j + i + l;
						const Complex u = z[k], v = z[k + m] * r;
						z[k] = u + v; z[k + m] = u - v;
					}
				}
			}
		}
	}
}

static void reducePos(Complex * const z, const size_t n, const size_t n_io, const uint32_t b, const double sb)
{
	const double n_io_inv = n_io / double(n);

	int64_t f = 0;
	for (size_t k = 0; k < 2 * n; k += 2)
	{
		const double o1 = (k < n) ? z[k].real() : z[k - n].imag();
		const double o2 = (k < n) ? z[k + 1].real() : z[k + 1 - n].imag();
		const double o = (o1 + sb * o2) * n_io_inv;
		f += llrint(o);
		int32_t r = f % b; if (r < 0) r += b;
		f -= r; f /= b;
		if (k < n) z[k].real(r); else z[k - n].imag(r);
		if (k < n) z[k + 1].real(0); else z[k + 1 - n].imag(0);
	}
	while (f != 0)
	{
		f = -f;
		for (size_t k = 0; k < 2 * n; k += 2)
		{
			const double o = (k < n) ? z[k].real() : z[k - n].imag();
			f += llrint(o);
			int32_t r = f % b; if (r < 0) r += b;
			f -= r; f /= b;
			if (k < n) z[k].real(r); else z[k - n].imag(r);
			if (r == 0) break;
		}
	}
}

class transform
{
private:
	const size_t n_io = 32;

	const size_t _n;
	Complex * const _w;
	Complex * const _z;
	Complex * const _f;
	const uint32_t _b;
	double _sb, _isb, _fsb;

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

public:
	transform(const size_t n, const uint32_t b) : _n(n), _w(new Complex[n]), _z(new Complex[n]), _f(new Complex[n / 2 / 4]), _b(b)
	{
		Complex * const w = _w;
		for (size_t s = 1; s < n; s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const long double C2PI = 6.2831853071795864769252867665590057684L;
				const long double theta = C2PI * (bitRev(j, 4 * s) + 1) / (8 * s);
				w[s + j] = Complex(double(cosl(theta)), double(sinl(theta)));
			}
		}

		Complex * const z = _z;
		z[0] = Complex(2, 0);
		for (size_t k = 1; k < n; ++k) z[k] = Complex(0, 0);

		Complex * const f = _f;
		for (size_t k = 0; k < n / 2 / 4; ++k) f[k] = Complex(0, 0);

		const fp16_80 sqrt_b = fp16_80::sqrt(b);
		_sb = double(sqrtl(b)); _isb = sqrt_b.hi(); _fsb = sqrt_b.lo();

		forward_out(_z, _n, n_io, _w);
	}

	virtual ~transform()
	{
		delete[] _w;
		delete[] _z;
	}

	double squareDup(const bool dup)
	{
		pass1(_z, _n, n_io / 2, _w);
		const double err = pass2(_z, _n, n_io, _w, _f, dup, double(_b), _sb, _isb, _fsb);
		pass3(_z, _n, n_io, _w, _f, double(_b), _sb, _isb, _fsb);
		return err;
	}

	bool isPrime()
	{
		backward_out(_z, _n, n_io, _w);
		reducePos(_z, _n, n_io, _b, _sb);

		Complex * const z = _z;
		bool isPrime = (z[0] == Complex(1, 0));
		if (isPrime)
		{
			for (size_t k = 1, n = _n; k < n; ++k) isPrime &= (z[k] == Complex(0, 0));
		}

		return isPrime;
	}
};

class genefer
{
public:
	bool check(const uint32_t b, const size_t n, double & maxError)
	{
		transform t(n, b);

		const integer exponent(b, n);

		double err = 0;
		for (int i = int(exponent.bitSize()) - 1; i >= 0; --i)
		{
			const double e = t.squareDup(exponent.bit(i));
			err  = std::max(err, e);
		}

		maxError = err;
		return t.isPrime();
	}
};

int main(/*int argc, char * argv[]*/)
{
	try
	{
		genefer g;
		const size_t n = 256;
		for (uint32_t b = 1000000862; b <= 1000000886; b += 2)
		{
			double maxError = 1;
			const bool isPrime = g.check(b, n, maxError);
			std::cout << b;
			if (isPrime) std::cout << "*";
			std::cout << ", " << maxError << std::endl;
		}
	}
	catch (const std::runtime_error & e)
	{
		std::ostringstream ss; ss << std::endl << "error: " << e.what() << ".";
		std::cerr << ss.str() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
