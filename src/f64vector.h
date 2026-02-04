/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include "simd128d.h"

#define finline	__attribute__((always_inline))

namespace transformCPU_namespace
{

struct Complex
{
	double real, imag;

	explicit Complex() {}
	constexpr explicit Complex(const double re, const double im) : real(re), imag(im) {}

	static Complex exp2iPi(const size_t a, const size_t b)
	{
#define	C2PI	6.2831853071795864769252867665590057684L
		const long double alpha = C2PI * (long double)a / (long double)b;
		const double cs = static_cast<double>(cosl(alpha)), sn = static_cast<double>(sinl(alpha));
		return Complex(cs, sn / cs);
	}
};

// Is not used because of full template specialization
template<size_t N>
class Vd
{
private:
	double __attribute__((aligned(sizeof(double) * N))) r[N];

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) { r[0] = f; for (size_t i = 1; i < N; ++i) r[i] = 0.0; }
	finline Vd(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = rhs.r[i]; }
	finline Vd & operator=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = rhs.r[i]; return *this; }

	finline static Vd broadcast(const double & f) { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = f; return vd; }
	finline static Vd broadcast(const double & f_l, const double & f_h)
	{
		Vd vd;
		for (size_t i = 0; i < N / 2; ++i) vd.r[i + 0 * N / 2] = f_l;
		for (size_t i = 0; i < N / 2; ++i) vd.r[i + 1 * N / 2] = f_h;
		return vd;
	}

	finline double operator[](const size_t i) const { return r[i]; }
	finline void set(const size_t i, const double & f) { r[i] = f; }

	finline bool isZero() const { bool zero = true; for (size_t i = 0; i < N; ++i) zero &= (r[i] == 0.0); return zero; }

	// finline Vd operator-() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = -r[i]; return vd; }

	finline Vd & operator+=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] += rhs.r[i]; return *this; }
	finline Vd & operator-=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] -= rhs.r[i]; return *this; }
	finline Vd & operator*=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] *= rhs.r[i]; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	void shift(const double f) { for (size_t i = N - 1; i > 0; --i) r[i] = r[i - 1]; r[0] = f; }

	finline Vd round() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = std::round(r[i]); return vd; }

	finline Vd abs() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = std::fabs(r[i]); return vd; }
	finline Vd & max(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = std::max(r[i], rhs.r[i]); return *this; }
	finline double max() const { double m = r[0]; for (size_t i = 1; i < N; ++i) m = std::max(m, r[i]); return m; }

	finline void interleave(Vd & rhs) { for (size_t i = 0; i < N / 2; ++i) { std::swap(r[i + N / 2], rhs.r[i]); } }	// N = 8

	finline static void transpose(Vd vd[N])
	{
		for (size_t i = 0; i < N; ++i)
		{
			for (size_t j = 0; j < i; ++j)
			{
				std::swap(vd[i].r[j], vd[j].r[i]);
			}
		}
	}
};

template<>
class Vd<2>
{
private:
	simd128d r;

private:
	constexpr explicit Vd(const simd128d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r(set_pd(0.0, f)) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd(set1_pd(f)); }
	finline static Vd broadcast(const double &, const double &) { return Vd(0.0); }	// unused

	finline double operator[](const size_t i) const { return r[i]; }
	finline void set(const size_t i, const double & f) { r[i] = f; }

	finline bool isZero() const { return is_zero_pd(r); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline void shift(const double f) { r = set_pd(r[0], f); }

	finline Vd round() const { return Vd(round_pd(r)); } 

	finline Vd abs() const { return Vd(abs_pd(r)); }
	finline Vd & max(const Vd & rhs) { r = max_pd(r, rhs.r); return *this; }
	finline double max() const { return std::max(r[0], r[1]); }

	finline void interleave(Vd &) {}	// unused

	finline static void transpose(Vd vd[2])
	{
		const simd128d t = unpackhi_pd(vd[0].r, vd[1].r);
		vd[0].r = unpacklo_pd(vd[0].r, vd[1].r); vd[1].r = t;
	}
};

#if defined(__AVX__)
template<>
class Vd<4>
{
private:
	__m256d r;

private:
	constexpr explicit Vd(const __m256d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r(_mm256_set_pd(0.0, 0.0, 0.0, f)) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd(_mm256_set1_pd(f)); }
	finline static Vd broadcast(const double &, const double &) { return Vd(0.0); }	// unused

	finline double operator[](const size_t i) const { return r[i]; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
	finline void set(const size_t i, const double & f) { r[i] = f; }
#pragma GCC diagnostic pop

	finline bool isZero() const { return (_mm256_movemask_pd(_mm256_cmp_pd(r, _mm256_setzero_pd(), _CMP_NEQ_OQ)) == 0); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline void shift(const double f) { r = _mm256_set_pd(r[2], r[1], r[0], f); }

	finline Vd round() const { return Vd(_mm256_round_pd(r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); } 

	finline Vd abs() const { return Vd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), r)); }
	finline Vd & max(const Vd & rhs) { r = _mm256_max_pd(r, rhs.r); return *this; }
	finline double max() const { const double m01 = std::max(r[0], r[1]), m23 = std::max(r[2], r[3]); return std::max(m01, m23); }

	finline void interleave(Vd &) {}	// unused

	finline static void transpose(Vd vd[4])
	{
		const __m256d r0 = _mm256_shuffle_pd(vd[0].r, vd[1].r, 0b0000), r1 = _mm256_shuffle_pd(vd[0].r, vd[1].r, 0b1111);
		const __m256d r2 = _mm256_shuffle_pd(vd[2].r, vd[3].r, 0b0000), r3 = _mm256_shuffle_pd(vd[2].r, vd[3].r, 0b1111);
		vd[0].r = _mm256_permute2f128_pd(r0, r2, _MM_SHUFFLE(0, 2, 0, 0));
		vd[2].r = _mm256_permute2f128_pd(r0, r2, _MM_SHUFFLE(0, 3, 0, 1));
		vd[1].r = _mm256_permute2f128_pd(r1, r3, _MM_SHUFFLE(0, 2, 0, 0));
		vd[3].r = _mm256_permute2f128_pd(r1, r3, _MM_SHUFFLE(0, 3, 0, 1));
	}
};
#else
template<>
class Vd<4>
{
private:
	simd128d rl, rh;

private:
	constexpr explicit Vd(const simd128d & _rl, const simd128d & _rh) : rl(_rl), rh(_rh) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : rl(set_pd(0.0, f)), rh(set1_pd(0.0)) {}
	finline Vd(const Vd & rhs) : rl(rhs.rl), rh(rhs.rh) {}
	finline Vd & operator=(const Vd & rhs) { rl = rhs.rl; rh = rhs.rh; return *this; }

	finline static Vd broadcast(const double & f) { return Vd(set1_pd(f), set1_pd(f)); }
	finline static Vd broadcast(const double &, const double &) { return Vd(0.0); }	// unused

	finline double operator[](const size_t i) const { return (i <= 1) ? rl[i] : rh[i - 2]; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
	finline void set(const size_t i, const double & f) { if (i <= 1) rl[i] = f; else rh[i - 2] = f; }
#pragma GCC diagnostic pop

	finline bool isZero() const { const bool r = is_zero_pd(rl) && is_zero_pd(rh); return r; }

	finline Vd & operator+=(const Vd & rhs) { rl += rhs.rl; rh += rhs.rh; return *this; }
	finline Vd & operator-=(const Vd & rhs) { rl -= rhs.rl; rh -= rhs.rh; return *this; }
	finline Vd & operator*=(const Vd & rhs) { rl *= rhs.rl; rh *= rhs.rh; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline void shift(const double f) { rh = set_pd(rh[0], rl[1]); rl = set_pd(rl[0], f); }

	finline Vd round() const { return Vd(round_pd(rl), round_pd(rh)); } 

	finline Vd abs() const { return Vd(abs_pd(rl), abs_pd(rh)); }
	finline Vd & max(const Vd & rhs) { rl = max_pd(rl, rhs.rl); rh = max_pd(rh, rhs.rh); return *this; }
	finline double max() const { const double m02 = std::max(rl[0], rh[0]), m13 = std::max(rl[1], rh[1]); return std::max(m02, m13); }

	finline void interleave(Vd &) {}	// unused

	finline static void transpose(Vd vd[4])
	{
		const simd128d a_00_10 = unpacklo_pd(vd[0].rl, vd[1].rl), a_01_11 = unpackhi_pd(vd[0].rl, vd[1].rl);
		const simd128d a_02_12 = unpacklo_pd(vd[0].rh, vd[1].rh), a_03_13 = unpackhi_pd(vd[0].rh, vd[1].rh);
		const simd128d a_20_30 = unpacklo_pd(vd[2].rl, vd[3].rl), a_21_31 = unpackhi_pd(vd[2].rl, vd[3].rl);
		const simd128d a_22_32 = unpacklo_pd(vd[2].rh, vd[3].rh), a_23_33 = unpackhi_pd(vd[2].rh, vd[3].rh);
		vd[0].rl = a_00_10; vd[0].rh = a_20_30;
		vd[1].rl = a_01_11; vd[1].rh = a_21_31;
		vd[2].rl = a_02_12; vd[2].rh = a_22_32;
		vd[3].rl = a_03_13; vd[3].rh = a_23_33;
	}
};
#endif

#if defined(__AVX512F__)
template<>
class Vd<8>
{
private:
	__m512d r;

private:
	constexpr explicit Vd(const __m512d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r(_mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f)) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd(_mm512_set1_pd(f)); }
	finline static Vd broadcast(const double & f_l, const double & f_h) { return Vd(_mm512_set_pd(f_h, f_h, f_h, f_h, f_l, f_l, f_l, f_l)); }

	finline double operator[](const size_t i) const { return r[i]; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
	finline void set(const size_t i, const double & f) { r[i] = f; }
#pragma GCC diagnostic pop

	finline bool isZero() const { return (_mm512_cmp_pd_mask(r, _mm512_setzero_pd(), _CMP_NEQ_OQ) == 0); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline void shift(const double f) { r = _mm512_set_pd(r[6], r[5], r[4], r[3], r[2], r[1], r[0], f); }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

	finline Vd round() const { return Vd(_mm512_roundscale_pd(r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); } 

	finline Vd abs() const { return Vd(_mm512_abs_pd(r)); }
	finline Vd & max(const Vd & rhs) { r = _mm512_max_pd(r, rhs.r); return *this; }
	finline double max() const { return _mm512_reduce_max_pd(r); }

	finline void interleave(Vd & rhs)
	{
		const __m512d t = _mm512_shuffle_f64x2(r, rhs.r, _MM_SHUFFLE(2, 3, 2, 3));
		r = _mm512_shuffle_f64x2(r, rhs.r, _MM_SHUFFLE(0, 1, 0, 1)); rhs.r = t;
	}

	finline static void transpose(Vd vd[8])
	{
		const __m512d r0 = _mm512_unpacklo_pd(vd[0].r, vd[1].r), r1 = _mm512_unpackhi_pd(vd[0].r, vd[1].r);
		const __m512d r2 = _mm512_unpacklo_pd(vd[2].r, vd[3].r), r3 = _mm512_unpackhi_pd(vd[2].r, vd[3].r);
		const __m512d r4 = _mm512_unpacklo_pd(vd[4].r, vd[5].r), r5 = _mm512_unpackhi_pd(vd[4].r, vd[5].r);
		const __m512d r6 = _mm512_unpacklo_pd(vd[6].r, vd[7].r), r7 = _mm512_unpackhi_pd(vd[6].r, vd[7].r);
		const __m512d t0 = _mm512_shuffle_f64x2(r0, r2, _MM_SHUFFLE(2, 0, 2, 0)), t2 = _mm512_shuffle_f64x2(r0, r2, _MM_SHUFFLE(3, 1, 3, 1));
		const __m512d t1 = _mm512_shuffle_f64x2(r1, r3, _MM_SHUFFLE(2, 0, 2, 0)), t3 = _mm512_shuffle_f64x2(r1, r3, _MM_SHUFFLE(3, 1, 3, 1));
		const __m512d t4 = _mm512_shuffle_f64x2(r4, r6, _MM_SHUFFLE(2, 0, 2, 0)), t6 = _mm512_shuffle_f64x2(r4, r6, _MM_SHUFFLE(3, 1, 3, 1));
		const __m512d t5 = _mm512_shuffle_f64x2(r5, r7, _MM_SHUFFLE(2, 0, 2, 0)), t7 = _mm512_shuffle_f64x2(r5, r7, _MM_SHUFFLE(3, 1, 3, 1));
		vd[0].r = _mm512_shuffle_f64x2(t0, t4, _MM_SHUFFLE(2, 0, 2, 0)); vd[4].r = _mm512_shuffle_f64x2(t0, t4, _MM_SHUFFLE(3, 1, 3, 1));
		vd[1].r = _mm512_shuffle_f64x2(t1, t5, _MM_SHUFFLE(2, 0, 2, 0)); vd[5].r = _mm512_shuffle_f64x2(t1, t5, _MM_SHUFFLE(3, 1, 3, 1));
		vd[2].r = _mm512_shuffle_f64x2(t2, t6, _MM_SHUFFLE(2, 0, 2, 0)); vd[6].r = _mm512_shuffle_f64x2(t2, t6, _MM_SHUFFLE(3, 1, 3, 1));
		vd[3].r = _mm512_shuffle_f64x2(t3, t7, _MM_SHUFFLE(2, 0, 2, 0)); vd[7].r = _mm512_shuffle_f64x2(t3, t7, _MM_SHUFFLE(3, 1, 3, 1));
	}

#pragma GCC diagnostic pop
};
#endif

template<size_t N>
class Vcx
{
private:
	Vd<N> re, im;

public:
	finline explicit Vcx() {}
	finline constexpr explicit Vcx(const double & real) : re(real), im(0.0) {}
	finline constexpr Vcx(const Vcx & rhs) : re(rhs.re), im(rhs.im) {}
	finline constexpr explicit Vcx(const Vd<N> & real, const Vd<N> & imag) : re(real), im(imag) {}
	finline Vcx & operator=(const Vcx & rhs) { re = rhs.re; im = rhs.im; return *this; }

	finline static Vcx broadcast(const Complex & z) { return Vcx(Vd<N>::broadcast(z.real), Vd<N>::broadcast(z.imag)); }
	finline static Vcx broadcast(const Complex & z_l, const Complex & z_h)
	{
		return Vcx(Vd<N>::broadcast(z_l.real, z_h.real), Vd<N>::broadcast(z_l.imag, z_h.imag));
	}

	finline Complex operator[](const size_t i) const { return Complex(re[i], im[i]); }
	finline void set(const size_t i, const Complex & z) { re.set(i, z.real); im.set(i, z.imag); }

	finline Vd<N> real() const { return re; }
	finline Vd<N> imag() const { return im; }

	finline bool isZero() const { const bool r = re.isZero() && im.isZero(); return r; }

	finline Vcx & operator+=(const Vcx & rhs) { re += rhs.re; im += rhs.im; return *this; }
	finline Vcx & operator-=(const Vcx & rhs) { re -= rhs.re; im -= rhs.im; return *this; }
	finline Vcx & operator*=(const double & f) { const Vd<N> vf = Vd<N>::broadcast(&f); re *= vf; im *= vf; return *this; }

	finline Vcx operator+(const Vcx & rhs) const { return Vcx(re + rhs.re, im + rhs.im); }
	finline Vcx operator-(const Vcx & rhs) const { return Vcx(re - rhs.re, im - rhs.im); }
	finline Vcx addi(const Vcx & rhs) const { return Vcx(re - rhs.im, im + rhs.re); }
	finline Vcx subi(const Vcx & rhs) const { return Vcx(re + rhs.im, im - rhs.re); }
	finline Vcx sub_i(const Vcx & rhs) const { return Vcx(rhs.im - im, re - rhs.re); }

	finline Vcx operator*(const Vcx & rhs) const { return Vcx(re * rhs.re - im * rhs.im, im * rhs.re + re * rhs.im); }
	finline Vcx operator*(const double & f) const { const Vd<N> vf = Vd<N>::broadcast(f); return Vcx(re * vf, im * vf); }
	finline Vcx mul1i() const { return Vcx(re - im, im + re); }
	finline Vcx mul1mi() const { return Vcx(re + im, im - re); }
	// finline Vcx muli() const { return Vcx(-im, re); }
	// finline Vcx mulmi() const { return Vcx(im, -re); }

	finline Vcx sqr() const { return Vcx(re * re - im * im, (re + re) * im); }

	finline Vcx mulW(const Vcx & rhs) const { return Vcx((re - im * rhs.im) * rhs.re, (im + re * rhs.im) * rhs.re); }
	finline Vcx mulWconj(const Vcx & rhs) const { return Vcx((re + im * rhs.im) * rhs.re, (im - re * rhs.im) * rhs.re); }

	finline void shift(const Vcx & rhs, const bool rotate)
	{
		// f x^n = -f
		re.shift(rotate ? -rhs.im[N - 1] : rhs.re[N - 1]);
#if defined(CYCLO)
		im.shift(rotate ?  rhs.re[N - 1] + rhs.im[N - 1] : rhs.im[N - 1]);
#else
		im.shift(rotate ?  rhs.re[N - 1] : rhs.im[N - 1]);
#endif
	}

	finline Vcx round() const { return Vcx(re.round(), im.round()); }

	finline Vcx abs() const { return Vcx(re.abs(), im.abs()); }
	finline Vcx & max(const Vcx & rhs) { re.max(rhs.re); im.max(rhs.im); return *this; }
	finline double max() const { return std::max(re.max(), im.max()); }

	finline void interleave(Vcx & rhs) { re.interleave(rhs.re); im.interleave(rhs.im); }

	finline static void transpose(Vcx z[N])
	{
		Vd<N> zr[N]; for (size_t i = 0; i < N; ++i) zr[i] = z[i].re;
		Vd<N>::transpose(zr);
		for (size_t i = 0; i < N; ++i) z[i].re = zr[i];

		Vd<N> zi[N]; for (size_t i = 0; i < N; ++i) zi[i] = z[i].im;
		Vd<N>::transpose(zi);
		for (size_t i = 0; i < N; ++i) z[i].im = zi[i];
	}

	finline static void transpose_in(Vcx z[8])
	{
		Vcx zr[8];
		for (size_t k = 0; k < 8 / N; ++k)
		{
			Vcx zb[N]; for (size_t i = 0; i < N; ++i) zb[i] = z[8 / N * i + k];
			transpose(zb);
			for (size_t i = 0; i < N; ++i) zr[N * k + i] = zb[i];
		}

		for (size_t i = 0; i < 8; ++i) z[i] = zr[i];
	}

	finline static void transpose_out(Vcx z[8])
	{
		Vcx zr[8];
		for (size_t k = 0; k < 8 / N; ++k)
		{
			Vcx zb[N]; for (size_t i = 0; i < N; ++i) zb[i] = z[N * k + i];
			transpose(zb);
			for (size_t i = 0; i < N; ++i) zr[8 / N * i + k] = zb[i];
		}
		for (size_t j = 0; j < 8; ++j) z[j] = zr[j];
	}
};

#if defined(CYCLO)
static constexpr double csqrt3_2 = 0.86602540378443864676372317075293618347, c2_sqrt3 = 1.15470053837925152901829756100391491130;
static constexpr Complex cs2pi_1_24 = Complex(0.96592582628906828674974319972889736763, 0.26794919243112270647255365849412763306);
static constexpr Complex cs2pi_1_48 = Complex(0.99144486137381041114455752692856287128, 0.13165249758739585347152645740971710359);
static constexpr Complex cs2pi_7_48 = Complex(0.60876142900872063941609754289816400452, 1.30322537284120575586814900899032094645);
#else
static constexpr double csqrt2_2 = 0.70710678118654752440084436210484903929;
static constexpr Complex cs2pi_1_16 = Complex(0.92387953251128675612818318939678828682, 0.41421356237309504880168872420969807857);
static constexpr Complex cs2pi_1_32 = Complex(0.98078528040323044912618223613423903697, 0.19891236737965800691159762264467622860);
static constexpr Complex cs2pi_5_32 = Complex(0.55557023301960222474283081394853287438, 1.49660576266548901760113513494247691870);
#endif

template<size_t N>
class Vradix4
{
	using Vc = Vcx<N>;

private:
	Vc z[4];

public:
	finline explicit Vradix4(const Vc * const mem, const size_t step)
	{
		for (size_t i = 0; i < 4; ++i) z[i] = mem[i * step];
	}

	finline void store(Vc * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 4; ++i) mem[i * step] = z[i];
	}

	finline explicit Vradix4(const Vc * const mem)	// VSIZE = 8, 4_4
	{
		for (size_t i = 0; i < 4; ++i) z[i] = mem[(4 * i) / 8 + ((4 * i) % 8)];
	}

	finline void store(Vc * const mem) const	// VSIZE = 8, 4_4
	{
		for (size_t i = 0; i < 4; ++i) mem[(4 * i) / 8 + ((4 * i) % 8)] = z[i];
	}

	finline void interleave()
	{
		z[0].interleave(z[1]); z[2].interleave(z[3]);
	}

	finline void forward4e(const Vc & w0, const Vc & w1)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w0), u1 = z[1], u3 = z[3].mulW(w0);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = Vc(u1 + u3).mulW(w1), v3 = Vc(u1 - u3).mulW(w1);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
	}

	finline void forward4o(const Vc & w0, const Vc & w2)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w0), u1 = z[1], u3 = z[3].mulW(w0);
		const Vc v0 = u0.addi(u2), v2 = u0.subi(u2), v1 = u1.addi(u3).mulW(w2), v3 = u1.subi(u3).mulW(w2);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
	}

	finline void backward4e(const Vc & w0, const Vc & w1)
	{
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w1), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w1);
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mulWconj(w0); z[1] = u1.subi(u3); z[3] = u1.addi(u3).mulWconj(w0);
	}

	finline void backward4o(const Vc & w0, const Vc & w2)
	{
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w2), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w2);
		z[0] = u0 + u2; z[2] = u2.sub_i(u0).mulWconj(w0); z[1] = u1.subi(u3); z[3] = u3.subi(u1).mulWconj(w0);
	}

	finline void forward4_0(const Vc & w0)
	{
#if defined(CYCLO)
		const Vd<N> v1_2 = Vd<N>::broadcast(0.5), vsqrt3_2 = Vd<N>::broadcast(csqrt3_2);
		const Vc z0 = z[0], u0 = Vc(z0.real() + z0.imag() * v1_2, z0.imag() * vsqrt3_2);
		const Vc z2 = z[2], u2 = Vc(z2.real() * vsqrt3_2, z2.imag() + z2.real() * v1_2);
		const Vc z1 = z[1], u1 = Vc(z1.real() + z1.imag() * v1_2, z1.imag() * vsqrt3_2);
		const Vc z3 = z[3], u3 = Vc(z3.real() * vsqrt3_2, z3.imag() + z3.real() * v1_2);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = Vc(u1 + u3).mulW(w0), v3 = Vc(u1 - u3).mulW(w0);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
#else
		const Vc u0 = z[0], u2 = z[2].mul1i(), u1 = z[1].mulW(w0), u3 = z[3].mulWconj(w0);
		const Vc v0 = u0 + u2 * csqrt2_2, v2 = u0 - u2 * csqrt2_2, v1 = u1.addi(u3), v3 = u3.addi(u1);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2 + v3; z[3] = v2 - v3;
#endif
	}

	finline void backward4_0(const Vc & w0)
	{
#if defined(CYCLO)
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w0), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w0);
		const Vc z0 = u0 + u2, z2 = u0 - u2, z1 = u1.subi(u3), z3 = u1.addi(u3);
		const Vd<N> v1_sqrt3 = Vd<N>::broadcast(c2_sqrt3 * 0.5), v2_sqrt3 = Vd<N>::broadcast(c2_sqrt3);
		z[0] = Vc(z0.real() - z0.imag() * v1_sqrt3, z0.imag() * v2_sqrt3);
		z[2] = Vc(z2.real() * v2_sqrt3, z2.imag() - z2.real() * v1_sqrt3);
		z[1] = Vc(z1.real() - z1.imag() * v1_sqrt3, z1.imag() * v2_sqrt3);
		z[3] = Vc(z3.real() * v2_sqrt3, z3.imag() - z3.real() * v1_sqrt3);
#else
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = v0 - v1, u2 = v2 + v3, u3 = v2 - v3;
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mul1mi() * csqrt2_2; z[1] = u1.subi(u3).mulWconj(w0); z[3] = u3.subi(u1).mulW(w0);
#endif
	}

	finline static void forward4e(const size_t m, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4 vr(&z[i], m);
			vr.forward4e(w0, w1);
			vr.store(&z[i], m);
		}
	}

	finline static void forward4o(const size_t m, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4 vr(&z[i], m);
			vr.forward4o(w0, w2);
			vr.store(&z[i], m);
		}
	}

	finline static void backward4e(const size_t m, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4 vr(&z[i], m);
			vr.backward4e(w0, w1);
			vr.store(&z[i], m);
		}
	}

	finline static void backward4o(const size_t m, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4 vr(&z[i], m);
			vr.backward4o(w0, w2);
			vr.store(&z[i], m);
		}
	}

	finline static void forward4e(const size_t mi, const size_t stepi, const size_t count, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix4 vr(zi, mi);
				vr.forward4e(w0, w1);
				vr.store(zi, mi);
			}
		}
	}

	finline static void forward4o(const size_t mi, const size_t stepi, const size_t count, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix4 vr(zi, mi);
				vr.forward4o(w0, w2);
				vr.store(zi, mi);
			}
		}
	}

	finline static void backward4e(const size_t mi, const size_t stepi, const size_t count, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix4 vr(zi, mi);
				vr.backward4e(w0, w1);
				vr.store(zi, mi);
			}
		}
	}

	finline static void backward4o(const size_t mi, const size_t stepi, const size_t count, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix4 vr(zi, mi);
				vr.backward4o(w0, w2);
				vr.store(zi, mi);
			}
		}
	}

	finline static void forward4_0(const size_t mi, const size_t stepi, const size_t count, Vc * const z)
	{
		const Vc w0 =
#if defined(CYCLO)
			Vc::broadcast(cs2pi_1_24);
#else
			Vc::broadcast(cs2pi_1_16);
#endif
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix4 vr(zi, mi);
				vr.forward4_0(w0);
				vr.store(zi, mi);
			}
		}
	}

	finline static void backward4_0(const size_t mi, const size_t stepi, const size_t count, Vc * const z)
	{
		const Vc w0 =
#if defined(CYCLO)
			Vc::broadcast(cs2pi_1_24);
#else
			Vc::broadcast(cs2pi_1_16);
#endif
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix4 vr(zi, mi);
				vr.backward4_0(w0);
				vr.store(zi, mi);
			}
		}
	}

	finline static void forward4e_4(Vc * const z, const Vc & w0, const Vc & w1)	// VSIZE = 8
	{
		Vradix4 vr(z);
		vr.interleave();
		vr.forward4e(w0, w1);
		vr.interleave();
		vr.store(z);
	}

	finline static void forward4o_4(Vc * const z, const Vc & w0, const Vc & w2)	// VSIZE = 8
	{
		Vradix4 vr(z);
		vr.interleave();
		vr.forward4o(w0, w2);
		vr.interleave();
		vr.store(z);
	}

	finline static void backward4e_4(Vc * const z, const Vc & w0, const Vc & w1)	// VSIZE = 8
	{
		Vradix4 vr(z);
		vr.interleave();
		vr.backward4e(w0, w1);
		vr.interleave();
		vr.store(z);
	}

	finline static void backward4o_4(Vc * const z, const Vc & w0, const Vc & w2)	// VSIZE = 8
	{
		Vradix4 vr(z);
		vr.interleave();
		vr.backward4o(w0, w2);
		vr.interleave();
		vr.store(z);
	}
};

template<size_t N>
class Vradix8
{
	using Vc = Vcx<N>;

private:
	Vc z[8];

public:
	finline explicit Vradix8(const Vc * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i) z[i] = mem[i * step];
	}

	finline void store(Vc * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i * step] = z[i];
	}

	finline void forward8_0()
	{
#if defined(CYCLO)
		const Vd<N> v1_2 = Vd<N>::broadcast(0.5), vsqrt3_2 = Vd<N>::broadcast(csqrt3_2);
		const Vc z0 = z[0], u0 = Vc(z0.real() + z0.imag() * v1_2, z0.imag() * vsqrt3_2);
		const Vc z4 = z[4], u4 = Vc(z4.real() * vsqrt3_2, z4.imag() + z4.real() * v1_2);
		const Vc z2 = z[2], u2 = Vc(z2.real() + z2.imag() * v1_2, z2.imag() * vsqrt3_2);
		const Vc z6 = z[6], u6 = Vc(z6.real() * vsqrt3_2, z6.imag() + z6.real() * v1_2);
		const Vc z1 = z[1], u1 = Vc(z1.real() + z1.imag() * v1_2, z1.imag() * vsqrt3_2);
		const Vc z5 = z[5], u5 = Vc(z5.real() * vsqrt3_2, z5.imag() + z5.real() * v1_2);
		const Vc z3 = z[3], u3 = Vc(z3.real() + z3.imag() * v1_2, z3.imag() * vsqrt3_2);
		const Vc z7 = z[7], u7 = Vc(z7.real() * vsqrt3_2, z7.imag() + z7.real() * v1_2);
		const Vc w0 = Vc::broadcast(cs2pi_1_24);
		const Vc v0 = u0 + u4, v4 = u0 - u4, v2 = Vc(u2 + u6).mulW(w0), v6 = Vc(u2 - u6).mulW(w0);
		const Vc v1 = u1 + u5, v5 = u1 - u5, v3 = Vc(u3 + u7).mulW(w0), v7 = Vc(u3 - u7).mulW(w0);
		const Vc w1 = Vc::broadcast(cs2pi_1_48), w2 = Vc::broadcast(cs2pi_7_48);
		const Vc s0 = v0 + v2, s2 = v0 - v2, s1 = Vc(v1 + v3).mulW(w1), s3 = Vc(v1 - v3).mulW(w1);
		const Vc s4 = v4.addi(v6), s6 = v4.subi(v6), s5 = v5.addi(v7).mulW(w2), s7 = v5.subi(v7).mulW(w2);
#else
		const Vc w0 = Vc::broadcast(cs2pi_1_16);
		const Vc u0 = z[0], u4 = z[4].mul1i(), u2 = z[2].mulW(w0), u6 = z[6].mul1i().mulW(w0);
		const Vc u1 = z[1], u5 = z[5].mul1i(), u3 = z[3].mulW(w0), u7 = z[7].mul1i().mulW(w0);
		const Vc v0 = u0 + u4 * csqrt2_2, v4 = u0 - u4 * csqrt2_2, v2 = u2 + u6 * csqrt2_2, v6 = u2 - u6 * csqrt2_2;
		const Vc w1 = Vc::broadcast(cs2pi_1_32), w2 = Vc::broadcast(cs2pi_5_32);
		const Vc v1 = Vc(u1 + u5 * csqrt2_2).mulW(w1), v5 = Vc(u1 - u5 * csqrt2_2).mulW(w2);
		const Vc v3 = Vc(u3 + u7 * csqrt2_2).mulW(w1), v7 = Vc(u3 - u7 * csqrt2_2).mulW(w2);
		const Vc s0 = v0 + v2, s2 = v0 - v2, s1 = v1 + v3, s3 = v1 - v3;
		const Vc s4 = v4.addi(v6), s6 = v4.subi(v6), s5 = v5.addi(v7), s7 = v5.subi(v7);
#endif
		z[0] = s0 + s1; z[1] = s0 - s1; z[2] = s2.addi(s3); z[3] = s2.subi(s3);
		z[4] = s4 + s5; z[5] = s4 - s5; z[6] = s6.addi(s7); z[7] = s6.subi(s7);
	}

	finline void backward8_0()
	{
		const Vc s0 = z[0], s1 = z[1], s2 = z[2], s3 = z[3], s4 = z[4], s5 = z[5], s6 = z[6], s7 = z[7];
#if defined(CYCLO)
		const Vc w1 = Vc::broadcast(cs2pi_1_48), w2 = Vc::broadcast(cs2pi_7_48);
		const Vc v0 = s0 + s1, v1 = Vc(s0 - s1).mulWconj(w1), v2 = s2 + s3, v3 = Vc(s2 - s3).mulWconj(w1);
		const Vc v4 = s4 + s5, v5 = Vc(s4 - s5).mulWconj(w2), v6 = s6 + s7, v7 = Vc(s6 - s7).mulWconj(w2);
		const Vc w0 = Vc::broadcast(cs2pi_1_24);
		const Vc u0 = v0 + v2, u2 = (v0 - v2).mulWconj(w0), u4 = v4 + v6, u6 = (v4 - v6).mulWconj(w0);
		const Vc u1 = v1.subi(v3), u3 = v1.addi(v3).mulWconj(w0), u5 = v5.subi(v7), u7 = v5.addi(v7).mulWconj(w0);
		const Vc z0 = u0 + u4, z4 = u0 - u4, z2 = u2.subi(u6), z6 = u2.addi(u6);
		const Vc z1 = u1 + u5, z5 = u1 - u5, z3 = u3.subi(u7), z7 = u3.addi(u7);
		const Vd<N> v1_sqrt3 = Vd<N>::broadcast(c2_sqrt3 * 0.5), v2_sqrt3 = Vd<N>::broadcast(c2_sqrt3);
		z[0] = Vc(z0.real() - z0.imag() * v1_sqrt3, z0.imag() * v2_sqrt3);
		z[4] = Vc(z4.real() * v2_sqrt3, z4.imag() - z4.real() * v1_sqrt3);
		z[2] = Vc(z2.real() - z2.imag() * v1_sqrt3, z2.imag() * v2_sqrt3);
		z[6] = Vc(z6.real() * v2_sqrt3, z6.imag() - z6.real() * v1_sqrt3);
		z[1] = Vc(z1.real() - z1.imag() * v1_sqrt3, z1.imag() * v2_sqrt3);
		z[5] = Vc(z5.real() * v2_sqrt3, z5.imag() - z5.real() * v1_sqrt3);
		z[3] = Vc(z3.real() - z3.imag() * v1_sqrt3, z3.imag() * v2_sqrt3);
		z[7] = Vc(z7.real() * v2_sqrt3, z7.imag() - z7.real() * v1_sqrt3);
#else
		const Vc w1 = Vc::broadcast(cs2pi_1_32), w2 = Vc::broadcast(cs2pi_5_32);
		const Vc v0 = s0 + s1, v1 = Vc(s0 - s1).mulWconj(w1), v2 = s2 + s3, v3 = Vc(s2 - s3).mulWconj(w1);
		const Vc v4 = s4 + s5, v5 = Vc(s4 - s5).mulWconj(w2), v6 = s6 + s7, v7 = Vc(s6 - s7).mulWconj(w2);
		const Vc u0 = v0 + v2, u2 = v0 - v2, u4 = v4 + v6, u6 = v4 - v6;
		const Vc u1 = v1.subi(v3), u3 = v1.addi(v3), u5 = v5.subi(v7), u7 = v5.addi(v7);
		const Vc w0 = Vc::broadcast(cs2pi_1_16);
		z[0] = u0 + u4; z[4] = Vc(u0 - u4).mul1mi() * csqrt2_2; z[2] = u2.subi(u6).mulWconj(w0); z[6] = u6.subi(u2).mulW(w0);
		z[1] = u1 + u5; z[5] = Vc(u1 - u5).mul1mi() * csqrt2_2; z[3] = u3.subi(u7).mulWconj(w0); z[7] = u7.subi(u3).mulW(w0);
#endif
	}

	finline static void forward8_0(const size_t mi, const size_t stepi, const size_t count, Vc * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix8 vr(zi, mi);
				vr.forward8_0();
				vr.store(zi, mi);
			}
		}
	}

	finline static void backward8_0(const size_t mi, const size_t stepi, const size_t count, Vc * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vradix8 vr(zi, mi);
				vr.backward8_0();
				vr.store(zi, mi);
			}
		}
	}
};

}
