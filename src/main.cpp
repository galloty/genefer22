/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include <omp.h>

#include "fp16_80.h"
#include "integer.h"

class Complex
{
private:
	double re, im;

public:
	Complex() {}
	constexpr explicit Complex(double real) : re(real), im(0.0) {}
	constexpr Complex(const Complex & rhs) : re(rhs.re), im(rhs.im) {}
	constexpr explicit Complex(double real, double imag) : re(real), im(imag) {}
	Complex & operator=(const Complex & rhs) { re = rhs.re; im = rhs.im; return *this; }

	double real() const { return re; }
	double imag() const { return im; }

	bool isZero() const { return ((re == 0.0) & (im == 0.0)); }

	Complex & operator+=(const Complex & rhs) { re += rhs.re; im += rhs.im; return *this; }
	Complex & operator-=(const Complex & rhs) { re -= rhs.re; im -= rhs.im; return *this; }
	Complex & operator*=(const double & f) { re *= f; im *= f; return *this; }

	Complex operator+(const Complex & rhs) const { return Complex(re + rhs.re, im + rhs.im); }
	Complex operator-(const Complex & rhs) const { return Complex(re - rhs.re, im - rhs.im); }
	Complex addi(const Complex & rhs) const { return Complex(re - rhs.im, im + rhs.re); }
	Complex subi(const Complex & rhs) const { return Complex(re + rhs.im, im - rhs.re); }
	Complex sub_i(const Complex & rhs) const { return Complex(rhs.im - im, re - rhs.re); }

	Complex operator*(const Complex & rhs) const { return Complex(re * rhs.re - im * rhs.im, im * rhs.re + re * rhs.im); }
	Complex operator*(const double & f) const { return Complex(re * f, im * f); }
	// Complex muli() const { return Complex(-im, re); }
	// Complex mulmi() const { return Complex(im, -re); }
	Complex mul1i() const { return Complex(re - im, im + re); }
	Complex mul1mi() const { return Complex(re + im, im - re); }

	Complex sqr() const { return Complex(re * re - im * im, (re + re) * im); }

	Complex mulW(const Complex & rhs) const { return Complex((re - im * rhs.im) * rhs.re, (im + re * rhs.im) * rhs.re); }
	Complex mulWconj(const Complex & rhs) const { return Complex((re + im * rhs.im) * rhs.re, (im - re * rhs.im) * rhs.re); }

	Complex abs() const { return Complex(std::fabs(re), std::fabs(im)); }

	Complex round() const
	{
		// const float MAGIC = 6755399441055744.0f;		// 52-bit fraction => 2^52 + 2^51
		// return Complex((re + MAGIC) - MAGIC, (im + MAGIC) - MAGIC);
		return Complex(std::rint(re), std::rint(im));
	}

	Complex rotate() const
	{
		// f x^n = -f
		return Complex(-im, re);
	}

	static Complex exp2iPi(const size_t a, const size_t b)
	{
#define	C2PI	6.2831853071795864769252867665590057684L
		const long double alpha = C2PI * (long double)a / (long double)b;
		const double cs = (double)cosl(alpha), sn = (double)sinl(alpha);
		return Complex(cs, sn / cs);
	}
};

class ComplexITransform
{
public:
	virtual bool isPrime() = 0;
	virtual double squareDup(const bool dup, const size_t num_threads) = 0;
};

template<size_t N>
class CZIT_CPU_vec_mt : public ComplexITransform
{
private:
	static const size_t l_shift = 4;
	static const size_t n_io = 16 * 4;		// multiple of 4, n_io >= 2 * l_shift, n >= 16 * n_io, n >= num_threads * n_io

	static constexpr double csqrt2_2 = 0.707106781186547524400844362104849039284835937688;
	static constexpr Complex cs2pi_1_16 = Complex(0.92387953251128675612818318939678828682, 0.41421356237309504880168872420969807857);
	static constexpr Complex cs2pi_1_32 = Complex(0.98078528040323044912618223613423903697, 0.19891236737965800691159762264467622860);
	static constexpr Complex cs2pi_5_32 = Complex(0.55557023301960222474283081394853287438, 1.49660576266548901760113513494247691870);

	const fp16_80 sqrt_b;

	const uint32_t _b;
	const size_t _num_threads;
	const double _sb, _isb, _fsb;
	Complex * const _z;
	Complex * const _w122i;
	Complex * const _ws;
	Complex * const _f;

private:
	static size_t index(const size_t k) { return k; }	// { const size_t i = k % n_io, j = k / n_io; return i * (N / n_io) + j; }

	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	void reducePos()
	{
		const uint32_t b = _b;
		const double sb = _sb, n_io_inv = double(n_io) / N;
		Complex * const z = _z;

		int64_t f = 0;
		for (size_t k = 0; k < 2 * N; k += 2)
		{
			const double o1 = (k < N) ? z[index(k + 0)].real() : z[index(k + 0 - N)].imag();
			const double o2 = (k < N) ? z[index(k + 1)].real() : z[index(k + 1 - N)].imag();
			const double o = (o1 + sb * o2) * n_io_inv;
			f += llrint(o);
			int32_t r = f % b; if (r < 0) r += b;
			f -= r; f /= b;
			if (k < N) z[index(k + 0)] = Complex(r, z[index(k + 0)].imag()); else z[index(k + 0 - N)] = Complex(z[index(k + 0 - N)].real(), r);
			if (k < N) z[index(k + 1)] = Complex(0, z[index(k + 1)].imag()); else z[index(k + 1 - N)] = Complex(z[index(k + 1 - N)].real(), 0);
		}
		while (f != 0)
		{
			f = -f;
			for (size_t k = 0; k < 2 * N; k += 2)
			{
				const double o = (k < N) ? z[index(k)].real() : z[index(k - N)].imag();
				f += llrint(o);
				int32_t r = f % b; if (r < 0) r += b;
				f -= r; f /= b;
				if (k < N) z[index(k)] = Complex(r, z[index(k)].imag()); else z[index(k - N)] = Complex(z[index(k - N)].real(), r);
				if (r == 0) break;
			}
		}
	}

	template <size_t step, size_t count>
	static void forward4(const size_t m, Complex * const z, const size_t k, const Complex * const w)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				const Complex u0 = z0, u2 = z2.mulW(w[0]), u1 = z1, u3 = z3.mulW(w[0]);
				const Complex v0 = u0 + u2, v2 = u0 - u2, v1 = Complex(u1 + u3).mulW(w[1]), v3 = Complex(u1 - u3).mulW(w[1]);
				z0 = v0 + v1; z1 = v0 - v1; z2 = v2.addi(v3); z3 = v2.subi(v3);
			}
		}
	}

	template <size_t step, size_t count>
	static void forward4i(const size_t m, Complex * const z, const size_t k, const Complex * const w)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				const Complex u0 = z0, u2 = z2.mulW(w[0]), u1 = z1, u3 = z3.mulW(w[0]);
				const Complex v0 = u0.addi(u2), v2 = u0.subi(u2), v1 = u1.addi(u3).mulW(w[2]), v3 = u1.subi(u3).mulW(w[2]);
				z0 = v0 + v1; z1 = v0 - v1; z2 = v2.addi(v3); z3 = v2.subi(v3);
			}
		}
	}

	template <size_t step, size_t count>
	static void backward4(const size_t m, Complex * const z, const size_t k, const Complex * const w)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				const Complex v0 = z0, v1 = z1, v2 = z2, v3 = z3;
				const Complex u0 = v0 + v1, u1 = Complex(v0 - v1).mulWconj(w[1]), u2 = v2 + v3, u3 = Complex(v2 - v3).mulWconj(w[1]);
				z0 = u0 + u2; z2 = Complex(u0 - u2).mulWconj(w[0]); z1 = u1.subi(u3); z3 = u1.addi(u3).mulWconj(w[0]);
			}
		}
	}

	template <size_t step, size_t count>
	static void backward4i(const size_t m, Complex * const z, const size_t k, const Complex * const w)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				const Complex v0 = z0, v1 = z1, v2 = z2, v3 = z3;
				const Complex u0 = v0 + v1, u1 = Complex(v0 - v1).mulWconj(w[2]), u2 = v2 + v3, u3 = Complex(v2 - v3).mulWconj(w[2]);
				z0 = u0 + u2; z2 = u2.sub_i(u0).mulWconj(w[0]); z1 = u1.subi(u3); z3 = u3.subi(u1).mulWconj(w[0]);
			}
		}
	}

	template <size_t step, size_t count>
	static void forward4_0(const size_t m, Complex * const z, const size_t k)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				const Complex u0 = z0, u2 = z2.mul1i(), u1 = z1.mulW(cs2pi_1_16), u3 = z3.mulWconj(cs2pi_1_16);
				const Complex v0 = u0 + u2 * csqrt2_2, v2 = u0 - u2 * csqrt2_2, v1 = u1.addi(u3), v3 = u3.addi(u1);
				z0 = v0 + v1; z1 = v0 - v1; z2 = v2 + v3; z3 = v2 - v3;
			}
		}
	}

	template <size_t step, size_t count>
	static void backward4_0(const size_t m, Complex * const z, const size_t k)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				const Complex v0 = z0, v1 = z1, v2 = z2, v3 = z3;
				const Complex u0 = v0 + v1, u1 = v0 - v1, u2 = v2 + v3, u3 = v2 - v3;
				z0 = u0 + u2; z2 = Complex(u0 - u2).mul1mi() * csqrt2_2; z1 = u1.subi(u3).mulWconj(cs2pi_1_16); z3 = u3.subi(u1).mulW(cs2pi_1_16);
			}
		}
	}

	template <size_t step, size_t count>
	static void forward8_0(const size_t m, Complex * const z, const size_t k)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				Complex & z4 = z[index(k + 4 * m + j + i)]; Complex & z5 = z[index(k + 5 * m + j + i)];
				Complex & z6 = z[index(k + 6 * m + j + i)]; Complex & z7 = z[index(k + 7 * m + j + i)];
				const Complex u0 = z0, u4 = z4.mul1i(), u2 = z2.mulW(cs2pi_1_16), u6 = z6.mul1i().mulW(cs2pi_1_16);
				const Complex u1 = z1, u5 = z5.mul1i(), u3 = z3.mulW(cs2pi_1_16), u7 = z7.mul1i().mulW(cs2pi_1_16);
				const Complex v0 = u0 + u4 * csqrt2_2, v4 = u0 - u4 * csqrt2_2, v2 = u2 + u6 * csqrt2_2, v6 = u2 - u6 * csqrt2_2;
				const Complex v1 = Complex(u1 + u5 * csqrt2_2).mulW(cs2pi_1_32), v5 = Complex(u1 - u5 * csqrt2_2).mulW(cs2pi_5_32);
				const Complex v3 = Complex(u3 + u7 * csqrt2_2).mulW(cs2pi_1_32), v7 = Complex(u3 - u7 * csqrt2_2).mulW(cs2pi_5_32);
				const Complex s0 = v0 + v2, s2 = v0 - v2, s1 = v1 + v3, s3 = v1 - v3;
				const Complex s4 = v4.addi(v6), s6 = v4.subi(v6), s5 = v5.addi(v7), s7 = v5.subi(v7);
				z0 = s0 + s1; z1 = s0 - s1; z2 = s2.addi(s3); z3 = s2.subi(s3);
				z4 = s4 + s5; z5 = s4 - s5; z6 = s6.addi(s7); z7 = s6.subi(s7);
			}
		}
	}

	template <size_t step, size_t count>
	static void backward8_0(const size_t m, Complex * const z, const size_t k)
	{
		for (size_t j = 0; j < m; j += step)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Complex & z0 = z[index(k + 0 * m + j + i)]; Complex & z1 = z[index(k + 1 * m + j + i)];
				Complex & z2 = z[index(k + 2 * m + j + i)]; Complex & z3 = z[index(k + 3 * m + j + i)];
				Complex & z4 = z[index(k + 4 * m + j + i)]; Complex & z5 = z[index(k + 5 * m + j + i)];
				Complex & z6 = z[index(k + 6 * m + j + i)]; Complex & z7 = z[index(k + 7 * m + j + i)];
				const Complex s0 = z0, s1 = z1, s2 = z2, s3 = z3, s4 = z4, s5 = z5, s6 = z6, s7 = z7;
				const Complex v0 = s0 + s1, v1 = Complex(s0 - s1).mulWconj(cs2pi_1_32), v2 = s2 + s3, v3 = Complex(s2 - s3).mulWconj(cs2pi_1_32);
				const Complex v4 = s4 + s5, v5 = Complex(s4 - s5).mulWconj(cs2pi_5_32), v6 = s6 + s7, v7 = Complex(s6 - s7).mulWconj(cs2pi_5_32);
				const Complex u0 = v0 + v2, u2 = v0 - v2, u4 = v4 + v6, u6 = v4 - v6;
				const Complex u1 = v1.subi(v3), u3 = v1.addi(v3), u5 = v5.subi(v7), u7 = v5.addi(v7);
				z0 = u0 + u4; z4 = Complex(u0 - u4).mul1mi() * csqrt2_2; z2 = u2.subi(u6).mulWconj(cs2pi_1_16); z6 = u6.subi(u2).mulW(cs2pi_1_16);
				z1 = u1 + u5; z5 = Complex(u1 - u5).mul1mi() * csqrt2_2; z3 = u3.subi(u7).mulWconj(cs2pi_1_16); z7 = u7.subi(u3).mulW(cs2pi_1_16);
			}
		}
	}

	static void square4(Complex * const z, const size_t k, const Complex & w)
	{
		Complex & z0 = z[index(k + 0)]; Complex & z1 = z[index(k + 1)]; Complex & z2 = z[index(k + 2)]; Complex & z3 = z[index(k + 3)];
		const Complex u0 = z0, u2 = z2.mulW(w), u1 = z1, u3 = z3.mulW(w);
		const Complex v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const Complex s0 = v0.sqr() + v1.sqr().mulW(w), s1 = (v0 + v0) * v1, s2 = v2.sqr() - v3.sqr().mulW(w), s3 = (v2 + v2) * v3;
		z0 = s0 + s2; z2 = Complex(s0 - s2).mulWconj(w); z1 = s1 + s3; z3 = Complex(s1 - s3).mulWconj(w);
	}

	static void square4i(Complex * const z, const size_t k, const Complex & w)
	{
		Complex & z0 = z[index(k + 0)]; Complex & z1 = z[index(k + 1)]; Complex & z2 = z[index(k + 2)]; Complex & z3 = z[index(k + 3)];
		const Complex u0 = z0, u2 = z2.mulW(w), u1 = z1, u3 = z3.mulW(w);
		const Complex v0 = u0.addi(u2), v2 = u0.subi(u2), v1 = u1.addi(u3), v3 = u3.addi(u1);
		const Complex s0 = v1.sqr().mulW(w).subi(v0.sqr()), s1 = (v0 + v0) * v1, s2 = v2.sqr().addi(v3.sqr().mulW(w)), s3 = (v2 + v2) * v3;
		z0 = s2.addi(s0); z2 = s0.addi(s2).mulWconj(w); z1 = s1.subi(s3); z3 = s3.subi(s1).mulWconj(w);
	}

	static void forward_out(Complex * const z, const size_t lh, const Complex * const w122i)
	{
		size_t s = (N / 4) / n_io / 2; for (; s >= 4 * 2; s /= 4);

		if (s == 4) forward8_0<n_io, 2 * l_shift>(N / 8, z, 2 * l_shift * lh);
		else        forward4_0<n_io, 2 * l_shift>(N / 4, z, 2 * l_shift * lh);

		for (size_t m = (s == 4) ? N / 32 : N / 16; m >= n_io; m /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 2 * l_shift * lh + 8 * m * j;
				const Complex * const w_j = &w122i[s + 3 * j];
				forward4<n_io, 2 * l_shift>(m, z, k + 0 * 4 * m, w_j);
				forward4i<n_io, 2 * l_shift>(m, z, k + 1 * 4 * m, w_j);
			}
		}
	}

	static void backward_out(Complex * const z, const size_t lh, const Complex * const w122i)
	{
		size_t s = (N / 4) / n_io / 2;
		for (size_t m = n_io; s >= 2; m *= 4, s /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 2 * l_shift * lh + 8 * m * j;
				const Complex * const w_j = &w122i[s + 3 * j];
				backward4<n_io, 2 * l_shift>(m, z, k + 0 * 4 * m, w_j);
				backward4i<n_io, 2 * l_shift>(m, z, k + 1 * 4 * m, w_j);
			}
		}

		if (s == 1) backward8_0<n_io, 2 * l_shift>(N / 8, z, 2 * l_shift * lh);
		else        backward4_0<n_io, 2 * l_shift>(N / 4, z, 2 * l_shift * lh);
	}

	void pass1(const size_t thread_id)
	{
		Complex * const z = _z;
		const Complex * const w122i = _w122i;
		const Complex * const ws = _ws;

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			// forward_in
			const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];

			if (l % 2 == 0) forward4<1, 1>(n_io / 4, z, n_io * l, w);
			else            forward4i<1, 1>(n_io / 4, z, n_io * l, w);

			for (size_t m = n_io / 16, s = 2; m >= 4; m /= 4, s *= 4)
			{
				for (size_t j = 0; j < s; ++j)
				{
					const size_t k = n_io * l + 8 * m * j;
					const Complex * const w_j = &w122i[(s_io + 3 * l) * s + 3 * j];
					forward4<1, 1>(m, z, k + 0 * 4 * m, w_j);
					forward4i<1, 1>(m, z, k + 1 * 4 * m, w_j);
				}
			}

			// square
			for (size_t j = 0; j < n_io / 8; ++j)
			{
				const Complex ws_j = ws[l * n_io / 8 + j];
				square4(z, n_io * l + 8 * j + 0, ws_j);
				square4i(z, n_io * l + 8 * j + 4, ws_j);
			}

			// backward_in
			for (size_t m = 4, s = n_io / 4 / m / 2; m <= n_io / 16; m *= 4, s /= 4)
			{
				for (size_t j = 0; j < s; ++j)
				{
					const size_t k = n_io * l + 8 * m * j;
					const Complex * const w_j = &w122i[(s_io + 3 * l) * s + 3 * j];
					backward4<1, 1>(m, z, k + 0 * 4 * m, w_j);
					backward4i<1, 1>(m, z, k + 1 * 4 * m, w_j);
				}
			}

			if (l % 2 == 0) backward4<1, 1>(n_io / 4, z, n_io * l, w);
			else            backward4i<1, 1>(n_io / 4, z, n_io * l, w);
		}
	}

	double pass2(const size_t thread_id, const bool dup)
	{
		const double b = double(_b);
		Complex * const z = _z;
		const Complex * const w122i = _w122i;
		Complex * const f = _f;
		const double sb = _sb, isb = _isb, fsb = _fsb;

		const double b_inv = 1.0 / b, sb_inv = 1.0 / sb, t2_n = 2.0 / N, g = dup ? 2 : 1;

		double err = 0;

		const size_t num_threads = _num_threads, n_io_s = n_io / l_shift / 2;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			backward_out(z, lh, w122i);

			// carry_out
			for (size_t l = 0; l < l_shift; ++l)
			{
				for (size_t j = 0; j < (N / 2) / (n_io / 2); ++j)
				{
					const size_t k_f = n_io_s * j + lh;
					const size_t k = l_shift * k_f + l;
					Complex & z0 = z[index(2 * k + 0)]; Complex & z1 = z[index(2 * k + 1)];
					const Complex o = (z0 + z1 * sb) * t2_n, oi = o.round(), d = Complex(o - oi).abs();
					const Complex f_i = f[k_f] + oi * g;
					err = std::max(err, std::max(d.real(), d.imag()));
					const Complex f_o = Complex(f_i * b_inv).round();
					const Complex r = f_i - f_o * b;
					f[k_f] = f_o;
					const Complex irh = Complex(r * sb_inv).round();
					z0 = (r - irh * isb) - irh * fsb;
					z1 = irh;
				}
			}
		}

		return err;
	}

	void pass3(const size_t thread_id)
	{
		const double b = double(_b);
		Complex * const z = _z;
		const Complex * const w122i = _w122i;
		Complex * const f = _f;
		const double sb = _sb, isb = _isb, fsb = _fsb;

		const double b_inv = 1.0 / b, sb_inv = 1.0 / sb;

		const size_t num_threads = _num_threads, n_io_s = n_io / l_shift / 2;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			// carry_in
			for (size_t j = 0; j < (N / 2) / (n_io / 2); ++j)
			{
				const size_t k_f = n_io_s * j + lh;
				const size_t carry_i = (k_f == 0) ? N / l_shift / 2 - 1 : k_f - 1;
				Complex f_j = (k_f == 0) ? f[carry_i].rotate() : f[carry_i];
				f[carry_i] = Complex(0, 0);
				for (size_t l = 0; l < l_shift; ++l)
				{
					const size_t k = l_shift * k_f + l;
					Complex & z0 = z[index(2 * k + 0)]; Complex & z1 = z[index(2 * k + 1)];
					const Complex o = z0 + z1 * sb, oi = Complex(o).round();
					f_j += oi;
					const Complex f_o = Complex(f_j * b_inv).round();
					const Complex r = f_j - f_o * b;
					f_j = f_o;
					const Complex irh = Complex(r * sb_inv).round();
					z0 = (r - irh * isb) - irh * fsb;
					z1 = irh;
					if (f_j.isZero()) break;
				}
				if (!f_j.isZero()) { std::cout << "Error!" << std::endl; exit(0); }	// TODO
			}

			forward_out(z, lh, w122i);
		}
	}

public:
	CZIT_CPU_vec_mt(const uint32_t b, const size_t num_threads) : sqrt_b(fp16_80::sqrt(b)), _b(b),
		_num_threads(num_threads), _sb(double(sqrtl(b))), _isb(sqrt_b.hi()), _fsb(sqrt_b.lo()),
		_z(new Complex[N]), _w122i(new Complex[N / 8]), _ws(new Complex[N / 8]), _f(new Complex[N / l_shift / 2])
	{
		Complex * const z = _z;
		z[0] = Complex(2, 0);
		for (size_t k = 1; k < N; ++k) z[k] = Complex(0, 0);

		Complex * const w122i = _w122i;
		for (size_t s = N / 16; s >= 4; s /= 4)
		{
			Complex * const w_s = &w122i[2 * s / 4];
			for (size_t j = 0; j < s / 2; ++j)
			{
				const size_t r = bitRev(j, 2 * s) + 1;
				w_s[3 * j + 0] = Complex::exp2iPi(r, 8 * s);
				w_s[3 * j + 1] = Complex::exp2iPi(r, 2 * 8 * s);
				w_s[3 * j + 2] = Complex::exp2iPi(r + 2 * s, 2 * 8 * s);
			}
		}

		Complex * const ws = _ws;
		for (size_t j = 0; j < N / 8; ++j)
		{
			ws[j] = Complex::exp2iPi(bitRev(j, 2 * (N / 4)) + 1, 8 * (N / 4));
		}

		Complex * const f = _f;
		for (size_t k = 0; k < N / l_shift / 2; ++k) f[k] = Complex(0, 0);

		for (size_t lh = 0; lh < n_io / l_shift / 2; ++lh)
		{
			forward_out(z, lh, w122i);
		}
	}

	virtual ~CZIT_CPU_vec_mt()
	{
		delete[] _z;
		delete[] _w122i;
		delete[] _ws;
		delete[] _f;
	}

	double squareDup(const bool dup, const size_t num_threads) override
	{
		double e[num_threads];

#pragma omp parallel
		{
			const size_t thread_id = size_t(omp_get_thread_num());

			pass1(thread_id);
#pragma omp barrier
			e[thread_id] = pass2(thread_id, dup);
#pragma omp barrier
			pass3(thread_id);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		return err;
	}

	bool isPrime() override
	{
		Complex * const z = _z;
		const Complex * const w122i = _w122i;

		for (size_t lh = 0; lh < n_io / l_shift / 2; ++lh)
		{
			backward_out(z, lh, w122i);
		}

		reducePos();

		bool isPrime = Complex(z[0] - Complex(1)).isZero();
		if (isPrime)
		{
			for (size_t k = 1; k < N; ++k) isPrime &= z[k].isZero();
		}

		return isPrime;
	}
};

class genefer
{
private:

public:
	void check(const uint32_t b, const size_t n)
	{
		omp_set_num_threads(3);
		size_t num_threads = 0;
#pragma omp parallel 
		{
#pragma omp single
			num_threads = size_t(omp_get_num_threads());
		}
		std::cout << num_threads << " thread(s)." << std::endl;

		const integer exponent(b, n);

		ComplexITransform * t = nullptr;;
		if (n == 1024)      t = new CZIT_CPU_vec_mt<1024>(b, num_threads);
		else if (n == 2048) t = new CZIT_CPU_vec_mt<2048>(b, num_threads);
		else if (n == 4096) t = new CZIT_CPU_vec_mt<4096>(b, num_threads);
		else if (n == 8192) t = new CZIT_CPU_vec_mt<8192>(b, num_threads);
		if (t == nullptr) throw std::runtime_error("exponent is not supported");

		auto t0 = std::chrono::steady_clock::now();

		double err = 0;
		for (int i = int(exponent.bitSize()) - 1; i >= 0; --i)
		{
			const double e = t->squareDup(exponent.bit(size_t(i)), num_threads);
			err  = std::max(err, e);
		}

		const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		const bool isPrime = t->isPrime();
		std::cout << b << "^" << n << " + 1";
		if (isPrime) std::cout << " is prime";
		std::cout << ", err = " << err << ", " << time << " sec." << std::endl;
	}
};

int main(/*int argc, char * argv[]*/)
{
	std::cerr << "genefer22: search for Generalized Fermat primes" << std::endl;
	std::cerr << " Copyright (c) 2022, Yves Gallot" << std::endl;
	std::cerr << " genefer22 is free source code, under the MIT license." << std::endl << std::endl;

	try
	{
		genefer g;
		g.check(399998298, 1024);
		g.check(399998572, 2048);
		g.check(399987078, 4096);
		g.check(399992284, 8192);
	}
	catch (const std::runtime_error & e)
	{
		std::ostringstream ss; ss << std::endl << "error: " << e.what() << ".";
		std::cerr << ss.str() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
