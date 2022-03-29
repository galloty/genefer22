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

static void forward(Complex * const z, const size_t n, const Complex * const w)
{
	for (size_t m = n / 2, s = 1; m > 0; m /= 2, s *= 2)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Complex r = w[s + j];
			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const Complex u = z[k], v = z[k + m] * r;
				z[k] = u + v; z[k + m] = u - v;
			}
		}
	}
}

static void backward(Complex * const z, const size_t n, const Complex * const w)
{
	for (size_t m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Complex r = std::conj(w[s + j]);
			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const Complex u = z[k], v = z[k + m];
				z[k] = u + v; z[k + m] = (u - v) * r;
			}
		}
	}
}

static void square(Complex * const z, const size_t n)
{
	for (size_t k = 0; k < n; ++k) z[k] *= z[k];
}

static double reduceDup(Complex * const z, const size_t n, const bool dup, const double b, const double sb, const double isb, const double fsb)
{
	const double b_inv = 1.0 / b, sb_inv = 1.0 / sb, n_inv = 1.0 / n, g = dup ? 2 : 1;

	double err = 0;
	Complex f = Complex(0, 0);
	for (size_t k = 0; k < n; k += 2)
	{
		const Complex o = n_inv * (z[k] + sb * z[k + 1]), oi = rint(o), d = fabs(o - oi);
		f += g * oi;
		err = std::max(err, std::max(d.real(), d.imag()));
		const Complex f_b = rint(f * b_inv);
		const Complex r = f - f_b * b;
		f = f_b;
		const Complex irh = rint(r * sb_inv);
		z[k] = (r - isb * irh) - fsb * irh;
		z[k + 1] = irh;
	}
	while (f != Complex(0, 0))
	{
		f = Complex(-f.imag(), f.real());
		for (size_t k = 0; k < n; k += 2)
		{
			const Complex o = z[k] + sb * z[k + 1], oi = rint(o);
			f += oi;
			const Complex f_b = rint(f * b_inv);
			const Complex r = f - f_b * b;
			f = f_b;
			const Complex irh = rint(r * sb_inv);
			z[k] = (r - isb * irh) - fsb * irh;
			z[k + 1] = irh;
			if (f == Complex(0, 0)) break;
		}
	}
	return err;
}

static void reducePos(Complex * const z, const size_t n, const uint32_t b, const double sb)
{
	int64_t f = 0;
	for (size_t k = 0; k < 2 * n; k += 2)
	{
		const double o1 = (k < n) ? z[k].real() : z[k - n].imag();
		const double o2 = (k < n) ? z[k + 1].real() : z[k + 1 - n].imag();
		const double o = o1 + sb * o2;
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
	const size_t _n;
	Complex * const _w;
	Complex * const _z;
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
	transform(const size_t n, const uint32_t b) : _n(n), _w(new Complex[n]), _z(new Complex[n]), _b(b)
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

		const fp16_80 sqrt_b = fp16_80::sqrt(b);
		_sb = double(sqrtl(b)); _isb = sqrt_b.hi(); _fsb = sqrt_b.lo();
	}

	virtual ~transform()
	{
		delete[] _w;
		delete[] _z;
	}

	double squareDup(const bool dup)
	{
		forward(_z, _n, _w);
		square(_z, _n);
		backward(_z, _n, _w);
		return reduceDup(_z, _n, dup, double(_b), _sb, _isb, _fsb);
	}

	bool isPrime()
	{
		reducePos(_z, _n, _b, _sb);

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
		for (size_t i = exponent.bitSize() - 1; i != 0; --i)
		{
			const double e = t.squareDup(exponent.bit(i));
			err  = std::max(err, e);
		}
		const double e = t.squareDup(exponent.bit(0));
		maxError = std::max(err, e);

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
			if (isPrime) std::cout << b << "*, " << maxError << std::endl;
			else std::cout << b << ", " << maxError << std::endl;
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
