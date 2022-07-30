/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <immintrin.h>

#include <omp.h>

#include "transform.h"
#include "fp16_80.h"

struct Complex
{
	double real, imag;

	explicit Complex() {}
	constexpr explicit Complex(const double re, const double im) : real(re), imag(im) {}

	static Complex exp2iPi(const size_t a, const size_t b)
	{
#define	C2PI	6.2831853071795864769252867665590057684L
		const long double alpha = C2PI * (long double)a / (long double)b;
		const double cs = (double)cosl(alpha), sn = (double)sinl(alpha);
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
	explicit Vd() {}
	explicit Vd(const double & f) { r[0] = f; for (size_t i = 1; i < N; ++i) r[i] = 0.0; }
	Vd(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = rhs.r[i]; }
	Vd & operator=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = rhs.r[i]; return *this; }

	static Vd broadcast(const double & f) { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = f; return vd; }
	static Vd broadcast(const double & f_l, const double & f_h)
	{
		Vd vd;
		for (size_t i = 0; i < N / 2; ++i) vd.r[i + 0 * N / 2] = f_l;
		for (size_t i = 0; i < N / 2; ++i) vd.r[i + 1 * N / 2] = f_h;
		return vd;
	}

	void interleave(Vd & rhs) { for (size_t i = 0; i < N / 2; ++i) { std::swap(r[i + N / 2], rhs.r[i]); } }	// N = 8

	double operator[](const size_t i) const { return r[i]; }
	void set(const size_t i, const double & f) { r[i] = f; }

	bool isZero() const { bool zero = true; for (size_t i = 0; i < N; ++i) zero &= (r[i] == 0.0); return zero; }

	// Vd operator-() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = -r[i]; return vd; }

	Vd & operator+=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] += rhs.r[i]; return *this; }
	Vd & operator-=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] -= rhs.r[i]; return *this; }
	Vd & operator*=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] *= rhs.r[i]; return *this; }

	Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	Vd abs() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = std::fabs(r[i]); return vd; }
	Vd round() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = std::round(r[i]); return vd; }

	Vd & max(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = std::max(r[i], rhs.r[i]); return *this; }

	double max() const { double m = r[0]; for (size_t i = 1; i < N; ++i) m = std::max(m, r[i]); return m; }

	void shift(const double f) { for (size_t i = N - 1; i > 0; --i) r[i] = r[i - 1]; r[0] = f; }

	static void transpose(Vd vd[N])
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

#define finline	__attribute__((always_inline))

inline __m128d round_pd(const __m128d rhs)
{
#ifdef __SSE4_1__
	return _mm_round_pd(rhs, _MM_FROUND_TO_NEAREST_INT);
#else // SSE2
	const __m128d signMask = _mm_set1_pd(-0.0), C52 = _mm_set1_pd(4503599627370496.0);  // 2^52
	const __m128d ar = _mm_andnot_pd(signMask, rhs);
	const __m128d ir = _mm_or_pd(_mm_sub_pd(_mm_add_pd(ar, C52), C52), _mm_and_pd(signMask, rhs));
	const __m128d mr = _mm_cmpge_pd(ar, C52);
	return _mm_or_pd(_mm_and_pd(mr, rhs), _mm_andnot_pd(mr, ir));
#endif
}

template<>
class Vd<2>
{
private:
	__m128d r;

private:
	constexpr explicit Vd(const __m128d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r(_mm_set_pd(0.0, f)) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd(_mm_set1_pd(f)); }
	finline static Vd broadcast(const double &, const double &) { return Vd(0.0); }	// unused

	finline void interleave(Vd &) {}	// unused

	finline double operator[](const size_t i) const { return r[i]; }
	finline void set(const size_t i, const double & f) { r[i] = f; }

	finline bool isZero() const { return (_mm_movemask_pd(_mm_cmpneq_pd(r, _mm_setzero_pd())) == 0); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline Vd abs() const { return Vd(_mm_andnot_pd(_mm_set1_pd(-0.0), r)); }
	finline Vd round() const { return Vd(round_pd(r)); } 

	finline Vd & max(const Vd & rhs) { r = _mm_max_pd(r, rhs.r); return *this; }

	finline double max() const { return std::max(r[0], r[1]); }

	finline void shift(const double f) { r = _mm_set_pd(r[0], f); }

	finline static void transpose(Vd vd[2])
	{
		const __m128d t = _mm_unpackhi_pd(vd[0].r, vd[1].r);
		vd[0].r = _mm_unpacklo_pd(vd[0].r, vd[1].r); vd[1].r = t;
	}
};

#ifdef __AVX__
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

	finline void interleave(Vd &) {}	// unused

	finline double operator[](const size_t i) const { return r[i]; }
	finline void set(const size_t i, const double & f) { r[i] = f; }

	finline bool isZero() const { return (_mm256_movemask_pd(_mm256_cmp_pd(r, _mm256_setzero_pd(), _CMP_NEQ_OQ)) == 0); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline Vd abs() const { return Vd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), r)); }
	finline Vd round() const { return Vd(_mm256_round_pd(r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); } 

	finline Vd & max(const Vd & rhs) { r = _mm256_max_pd(r, rhs.r); return *this; }

	finline double max() const { const double m01 = std::max(r[0], r[1]), m23 = std::max(r[2], r[3]); return std::max(m01, m23); }

	finline void shift(const double f) { r = _mm256_set_pd(r[2], r[1], r[0], f); }

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
#endif

#ifdef __AVX512F__
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

	finline void interleave(Vd & rhs)
	{
		const __m512d t = _mm512_shuffle_f64x2(r, rhs.r, _MM_SHUFFLE(2, 3, 2, 3));
		r = _mm512_shuffle_f64x2(r, rhs.r, _MM_SHUFFLE(0, 1, 0, 1)); rhs.r = t;
	}

	finline double operator[](const size_t i) const { return r[i]; }
	finline void set(const size_t i, const double & f) { r[i] = f; }

	finline bool isZero() const { return (_mm512_cmp_pd_mask(r, _mm512_setzero_pd(), _CMP_NEQ_OQ) == 0); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline Vd abs() const { return Vd(_mm512_abs_pd(r)); }
	finline Vd round() const { return Vd(_mm512_roundscale_pd(r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); } 

	finline Vd & max(const Vd & rhs) { r = _mm512_max_pd(r, rhs.r); return *this; }

	finline double max() const { return _mm512_reduce_max_pd(r); }

	finline void shift(const double f) { r = _mm512_set_pd(r[6], r[5], r[4], r[3], r[2], r[1], r[0], f); }

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

	finline void interleave(Vcx & rhs) { re.interleave(rhs.re); im.interleave(rhs.im); }

	finline Complex operator[](const size_t i) const { return Complex(re[i], im[i]); }
	finline void set(const size_t i, const Complex & z) { re.set(i, z.real); im.set(i, z.imag); }

	finline bool isZero() const { return (re.isZero() & im.isZero()); }

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

	finline Vcx abs() const { return Vcx(re.abs(), im.abs()); }
	finline Vcx round() const { return Vcx(re.round(), im.round()); }

	finline Vcx & max(const Vcx & rhs) { re.max(rhs.re); im.max(rhs.im); return *this; }

	finline double max() const { return std::max(re.max(), im.max()); }

	finline void shift(const Vcx & rhs, const bool rotate)
	{
		// f x^n = -f
		re.shift(rotate ? -rhs.im[N - 1] : rhs.re[N - 1]);
		im.shift(rotate ?  rhs.re[N - 1] : rhs.im[N - 1]);
	}

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

static constexpr double csqrt2_2 = 0.707106781186547524400844362104849039284835937688;
static constexpr Complex cs2pi_1_16 = Complex(0.92387953251128675612818318939678828682, 0.41421356237309504880168872420969807857);
static constexpr Complex cs2pi_1_32 = Complex(0.98078528040323044912618223613423903697, 0.19891236737965800691159762264467622860);
static constexpr Complex cs2pi_5_32 = Complex(0.55557023301960222474283081394853287438, 1.49660576266548901760113513494247691870);

template<size_t N>
class Vradix4
{
private:
	using Vc = Vcx<N>;
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
		const Vc u0 = z[0], u2 = z[2].mul1i(), u1 = z[1].mulW(w0), u3 = z[3].mulWconj(w0);
		const Vc v0 = u0 + u2 * csqrt2_2, v2 = u0 - u2 * csqrt2_2, v1 = u1.addi(u3), v3 = u3.addi(u1);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2 + v3; z[3] = v2 - v3;
	}

	finline void backward4_0(const Vc & w0)
	{
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = v0 - v1, u2 = v2 + v3, u3 = v2 - v3;
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mul1mi() * csqrt2_2; z[1] = u1.subi(u3).mulWconj(w0); z[3] = u3.subi(u1).mulW(w0);
	}
};

template<size_t N>
class Vradix8
{
private:
	using Vc = Vcx<N>;
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
		const Vc w0 = Vc::broadcast(cs2pi_1_16);
		const Vc u0 = z[0], u4 = z[4].mul1i(), u2 = z[2].mulW(w0), u6 = z[6].mul1i().mulW(w0);
		const Vc u1 = z[1], u5 = z[5].mul1i(), u3 = z[3].mulW(w0), u7 = z[7].mul1i().mulW(w0);
		const Vc v0 = u0 + u4 * csqrt2_2, v4 = u0 - u4 * csqrt2_2, v2 = u2 + u6 * csqrt2_2, v6 = u2 - u6 * csqrt2_2;
		const Vc w1 = Vc::broadcast(cs2pi_1_32), w2 = Vc::broadcast(cs2pi_5_32);
		const Vc v1 = Vc(u1 + u5 * csqrt2_2).mulW(w1), v5 = Vc(u1 - u5 * csqrt2_2).mulW(w2);
		const Vc v3 = Vc(u3 + u7 * csqrt2_2).mulW(w1), v7 = Vc(u3 - u7 * csqrt2_2).mulW(w2);
		const Vc s0 = v0 + v2, s2 = v0 - v2, s1 = v1 + v3, s3 = v1 - v3;
		const Vc s4 = v4.addi(v6), s6 = v4.subi(v6), s5 = v5.addi(v7), s7 = v5.subi(v7);
		z[0] = s0 + s1; z[1] = s0 - s1; z[2] = s2.addi(s3); z[3] = s2.subi(s3);
		z[4] = s4 + s5; z[5] = s4 - s5; z[6] = s6.addi(s7); z[7] = s6.subi(s7);
	}

	finline void backward8_0()
	{
		const Vc s0 = z[0], s1 = z[1], s2 = z[2], s3 = z[3], s4 = z[4], s5 = z[5], s6 = z[6], s7 = z[7];
		const Vc w1 = Vc::broadcast(cs2pi_1_32), w2 = Vc::broadcast(cs2pi_5_32);
		const Vc v0 = s0 + s1, v1 = Vc(s0 - s1).mulWconj(w1), v2 = s2 + s3, v3 = Vc(s2 - s3).mulWconj(w1);
		const Vc v4 = s4 + s5, v5 = Vc(s4 - s5).mulWconj(w2), v6 = s6 + s7, v7 = Vc(s6 - s7).mulWconj(w2);
		const Vc u0 = v0 + v2, u2 = v0 - v2, u4 = v4 + v6, u6 = v4 - v6;
		const Vc u1 = v1.subi(v3), u3 = v1.addi(v3), u5 = v5.subi(v7), u7 = v5.addi(v7);
		const Vc w0 = Vc::broadcast(cs2pi_1_16);
		z[0] = u0 + u4; z[4] = Vc(u0 - u4).mul1mi() * csqrt2_2; z[2] = u2.subi(u6).mulWconj(w0); z[6] = u6.subi(u2).mulW(w0);
		z[1] = u1 + u5; z[5] = Vc(u1 - u5).mul1mi() * csqrt2_2; z[3] = u3.subi(u7).mulWconj(w0); z[7] = u7.subi(u3).mulW(w0);
	}
};

template<size_t N>
class Vcx8
{
private:
	using Vc = Vcx<N>;
	Vc z[8];

private:
	Vcx8() {}

public:
	finline explicit Vcx8(const Vc * const mem)
	{
		for (size_t i = 0; i < 8; ++i) z[i] = mem[i];
	}

	finline void store(Vc * const mem) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i] = z[i];
	}

	finline explicit Vcx8(const Vc * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			z[i] = mem[(step * i_h + i_l) / N];
		}
	}

	finline void store(Vc * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			mem[(step * i_h + i_l) / N] = z[i];
		}
	}

	finline void transpose_in() { Vc::transpose_in(z); }
	finline void transpose_out() { Vc::transpose_out(z); }

	finline void square4e(const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const Vc s0 = v0.sqr() + v1.sqr().mulW(w), s1 = (v0 + v0) * v1, s2 = v2.sqr() - v3.sqr().mulW(w), s3 = (v2 + v2) * v3;
		z[0] = s0 + s2; z[2] = Vc(s0 - s2).mulWconj(w); z[1] = s1 + s3; z[3] = Vc(s1 - s3).mulWconj(w);
	}

	finline void square4o(const Vc & w)
	{
		const Vc u4 = z[4], u6 = z[6].mulW(w), u5 = z[5], u7 = z[7].mulW(w);
		const Vc v4 = u4.addi(u6), v6 = u4.subi(u6), v5 = u5.addi(u7), v7 = u7.addi(u5);
		const Vc s4 = v5.sqr().mulW(w).subi(v4.sqr()), s5 = (v4 + v4) * v5, s6 = v6.sqr().addi(v7.sqr().mulW(w)), s7 = (v6 + v6) * v7;
		z[4] = s6.addi(s4); z[6] = s4.addi(s6).mulWconj(w); z[5] = s5.subi(s7); z[7] = s7.subi(s5).mulWconj(w);
	}

	finline void mul4_forward(const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		z[0] = u0 + u2; z[2] = u0 - u2; z[1] = u1 + u3; z[3] = u1 - u3;
		const Vc u4 = z[4], u6 = z[6].mulW(w), u5 = z[5], u7 = z[7].mulW(w);
		z[4] = u4.addi(u6); z[6] = u4.subi(u6); z[5] = u5.addi(u7); z[7] = u7.addi(u5);
	}

	finline void mul4(const Vcx8 & rhs, const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const Vc vp0 = rhs.z[0], vp2 = rhs.z[2], vp1 = rhs.z[1], vp3 = rhs.z[3];
		const Vc s0 = v0 * vp0 + Vc(v1 * vp1).mulW(w), s1 = v0 * vp1 + vp0 * v1;
		const Vc s2 = v2 * vp2 - Vc(v3 * vp3).mulW(w), s3 = v2 * vp3 + vp2 * v3;
		z[0] = s0 + s2; z[2] = Vc(s0 - s2).mulWconj(w); z[1] = s1 + s3; z[3] = Vc(s1 - s3).mulWconj(w);

		const Vc u4 = z[4], u6 = z[6].mulW(w), u5 = z[5], u7 = z[7].mulW(w);
		const Vc v4 = u4.addi(u6), v6 = u4.subi(u6), v5 = u5.addi(u7), v7 = u7.addi(u5);
		const Vc vp4 = rhs.z[4], vp6 = rhs.z[6], vp5 = rhs.z[5], vp7 = rhs.z[7];
		const Vc s4 = Vc(v5 * vp5).mulW(w).subi(v4 * vp4), s5 = v4 * vp5 + vp4 * v5;
		const Vc s6 = Vc(v6 * vp6).addi(Vc(v7 * vp7).mulW(w)), s7 = v6 * vp7 + vp6 * v7;
		z[4] = s6.addi(s4); z[6] = s4.addi(s6).mulWconj(w); z[5] = s5.subi(s7); z[7] = s7.subi(s5).mulWconj(w);
	}

	finline Vc mul_carry(const Vc & f_prev, const double g, const double b, const double b_inv, const double t2_n,
						 const double sb, const double sb_inv, const double isb, const double fsb, Vc & err)
	{
		Vc f = f_prev;
		for (size_t i = 0; i < 4; ++i)
		{
			Vc & z0 = z[2 * i + 0]; Vc & z1 = z[2 * i + 1];
			const Vc o = (z0 + z1 * sb) * t2_n, oi = o.round();
			const Vc f_i = f + oi * g;
			err.max(Vc(o - oi).abs());
			const Vc f_o = Vc(f_i * b_inv).round();
			const Vc r = f_i - f_o * b;
			f = f_o;
			const Vc irh = Vc(r * sb_inv).round();
			z0 = (r - irh * isb) - irh * fsb; z1 = irh;
		}
		return f;
	}

	finline void carry(const Vc & f_i, const double b, const double b_inv, const double sb, const double sb_inv, const double isb, const double fsb)
	{
		Vc f = f_i;
		for (size_t i = 0; i < 4 - 1; ++i)
		{
			Vc & z0 = z[2 * i + 0]; Vc & z1 = z[2 * i + 1];
			const Vc o = z0 + z1 * sb, oi = o.round();
			f += oi;
			const Vc f_o = Vc(f * b_inv).round();
			const Vc r = f - f_o * b;
			f = f_o;
			const Vc irh = Vc(r * sb_inv).round();
			z0 = (r - irh * isb) - irh * fsb; z1 = irh;
			if (f.isZero()) break;
		}

		if (!f.isZero())
		{
			Vc & z0 = z[2 * (4 - 1) + 0]; Vc & z1 = z[2 * (4 - 1) + 1];
			const Vc o = z0 + z1 * sb, oi = o.round();
			const Vc r = f + oi;
			const Vc irh = Vc(r * sb_inv).round();
			z0 = (r - irh * isb) - irh * fsb; z1 = irh;
		}
	}
};

template<size_t N, size_t VSIZE>
class transformCPU : public transform
{
private:
	using Vc = Vcx<VSIZE>;
	using Vr4 = Vradix4<VSIZE>;
	using Vr8 = Vradix8<VSIZE>;
	using Vc8 = Vcx8<VSIZE>;

	// Pass 1: n_io Complex (16 bytes), Pass 2/3: N / n_io Complex
	// n_io must be a power of 4, n_io >= 64, n >= 16 * n_io, n >= num_threads * n_io.
	static const size_t n_io = (N <= (1 << 11)) ? 64 : (N <= (1 << 13)) ? 256 : (N <= (1 << 17)) ? 1024 : 4096;
	static const size_t n_io_s = n_io / 4 / 2;
	static const size_t n_io_inv = N / n_io / VSIZE;
	static const size_t n_gap = (VSIZE <= 4) ? 64 : 16 * VSIZE;	// Cache line size is 64 bytes. Alignment is needed if VSIZE > 4.

	finline static constexpr size_t index(const size_t k) { const size_t j = k / n_io, i = k % n_io; return j * (n_io + n_gap / sizeof(Complex)) + i; }

	static const size_t wSize = N / 8 * sizeof(Complex);
	static const size_t wsSize = N / 8 * sizeof(Complex);
	static const size_t zSize = index(N) * sizeof(Complex);
	static const size_t fcSize = 64 * n_io_inv * sizeof(Vc);	// num_threads <= 64

	static const size_t wOffset = 0;
	static const size_t wsOffset = wOffset + wSize;
	static const size_t zOffset = wsOffset + wsSize;
	static const size_t fcOffset = zOffset + zSize;
	static const size_t zpOffset = fcOffset + fcSize;

	const fp16_80 _sqrt_b;

	const size_t _num_threads;
	const double _b, _sb, _isb, _fsb;
	const double _mem_size;
	double _error;
	char * const _mem;

private:
	finline static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	finline static void forward4e(const size_t m, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vr4 vr(&z[i], m);
			vr.forward4e(w0, w1);
			vr.store(&z[i], m);
		}
	}

	finline static void forward4o(const size_t m, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vr4 vr(&z[i], m);
			vr.forward4o(w0, w2);
			vr.store(&z[i], m);
		}
	}

	finline static void backward4e(const size_t m, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vr4 vr(&z[i], m);
			vr.backward4e(w0, w1);
			vr.store(&z[i], m);
		}
	}

	finline static void backward4o(const size_t m, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vr4 vr(&z[i], m);
			vr.backward4o(w0, w2);
			vr.store(&z[i], m);
		}
	}

	template <size_t stepi, size_t count>
	finline static void forward4e(const size_t mi, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.forward4e(w0, w1);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void forward4o(const size_t mi, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.forward4o(w0, w2);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void backward4e(const size_t mi, Vc * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.backward4e(w0, w1);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void backward4o(const size_t mi, Vc * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.backward4o(w0, w2);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void forward4_0(const size_t mi, Vc * const z)
	{
		const Vc w0 = Vc::broadcast(cs2pi_1_16);
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.forward4_0(w0);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void backward4_0(const size_t mi, Vc * const z)
	{
		const Vc w0 = Vc::broadcast(cs2pi_1_16);
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.backward4_0(w0);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void forward8_0(const size_t mi, Vc * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr8 vr(zi, mi);
				vr.forward8_0();
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	finline static void backward8_0(const size_t mi, Vc * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vc * const zi = &z[j + i];
				Vr8 vr(zi, mi);
				vr.backward8_0();
				vr.store(zi, mi);
			}
		}
	}

	finline static void forward4e_4(Vc * const z, const Vc & w0, const Vc & w1)	// VSIZE = 8
	{
		Vr4 vr(z);
		vr.interleave();
		vr.forward4e(w0, w1);
		vr.interleave();
		vr.store(z);
	}

	finline static void forward4o_4(Vc * const z, const Vc & w0, const Vc & w2)	// VSIZE = 8
	{
		Vr4 vr(z);
		vr.interleave();
		vr.forward4o(w0, w2);
		vr.interleave();
		vr.store(z);
	}

	finline static void backward4e_4(Vc * const z, const Vc & w0, const Vc & w1)	// VSIZE = 8
	{
		Vr4 vr(z);
		vr.interleave();
		vr.backward4e(w0, w1);
		vr.interleave();
		vr.store(z);
	}

	finline static void backward4o_4(Vc * const z, const Vc & w0, const Vc & w2)	// VSIZE = 8
	{
		Vr4 vr(z);
		vr.interleave();
		vr.backward4o(w0, w2);
		vr.interleave();
		vr.store(z);
	}

	finline static void forward_out(Vc * const z, const Complex * const w122i)
	{
		static const size_t stepi = index(n_io) / VSIZE;

		size_t s = (N / 4) / n_io / 2; for (; s >= 4 * 2; s /= 4);

		if (s == 4) forward8_0<stepi, 2 * 4 / VSIZE>(index(N / 8) / VSIZE, z);
		else        forward4_0<stepi, 2 * 4 / VSIZE>(index(N / 4) / VSIZE, z);

		for (size_t mi = index((s == 4) ? N / 32 : N / 16) / VSIZE; mi >= stepi; mi /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
				forward4e<stepi, 2 * 4 / VSIZE>(mi, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				forward4o<stepi, 2 * 4 / VSIZE>(mi, &z[k + 1 * 4 * mi], w0, w2);
			}
		}
	}

	finline static void backward_out(Vc * const z, const Complex * const w122i)
	{
		static const size_t stepi = index(n_io) / VSIZE;

		size_t s = (N / 4) / n_io / 2;
		for (size_t mi = stepi; s >= 2; mi *= 4, s /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
				backward4e<stepi, 2 * 4 / VSIZE>(mi, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				backward4o<stepi, 2 * 4 / VSIZE>(mi, &z[k + 1 * 4 * mi], w0, w2);
			}
		}

		if (s == 1) backward8_0<stepi, 2 * 4 / VSIZE>(index(N / 8) / VSIZE, z);
		else        backward4_0<stepi, 2 * 4 / VSIZE>(index(N / 4) / VSIZE, z);
	}

	void pass1(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const z = (Vc *)&_mem[zOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zl = &z[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); forward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); forward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					forward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					forward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					forward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					forward4o_4(&zj[2], w0, w2);
				}
			}

			// square
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zj = &zl[8 * j];
				Vc8 z8(zj);
				z8.transpose_in();
				z8.square4e(wsl[j]);
				z8.store(zj);
			}
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zj = &zl[8 * j];
				Vc8 z8(zj);
				z8.square4o(wsl[j]);
				z8.transpose_out();
				z8.store(zj);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					backward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					backward4o_4(&zj[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					backward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					backward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); backward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); backward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}
		}
	}

	void pass1multiplicand(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const zp = (Vc *)&_mem[zpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zpl = &zp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); forward4e(n_io / 4 / VSIZE, zpl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); forward4o(n_io / 4 / VSIZE, zpl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zpj = &zpl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					forward4e(m, &zpj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					forward4o(m, &zpj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zpj = &zpl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					forward4e_4(&zpj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					forward4o_4(&zpj[2], w0, w2);
				}
			}

			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zpj = &zpl[8 * j];
				Vc8 zp8(zpj);
				zp8.transpose_in();
				zp8.mul4_forward(wsl[j]);
				zp8.store(zpj);
			}
		}
	}

	void pass1mul(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const z = (Vc *)&_mem[zOffset];
		const Vc * const zp = (Vc *)&_mem[zpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zl = &z[index(n_io * l) / VSIZE];
			const Vc * const zpl = &zp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); forward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); forward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					forward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					forward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					forward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					forward4o_4(&zj[2], w0, w2);
				}
			}

			// mul
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zj = &zl[8 * j];
				const Vc * const zpj = &zpl[8 * j];
				Vc8 z8(zj); z8.transpose_in();
				Vc8 zp8(zpj); z8.mul4(zp8, wsl[j]);
				z8.transpose_out();
				z8.store(zj);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					backward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					backward4o_4(&zj[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					backward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					backward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); backward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); backward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}
		}
	}

	double pass2_0(const size_t thread_id, const bool dup)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		Vc * const z = (Vc *)&_mem[zOffset];
		Vc * const fc = (Vc *)&_mem[fcOffset]; Vc * const f = &fc[thread_id * n_io_inv];
		const double b = _b, sb = _sb, isb = _isb, fsb = _fsb;
		const double b_inv = 1.0 / b, sb_inv = 1.0 / sb, g = dup ? 2.0 : 1.0;

		Vc err = Vc(0.0);

		const size_t num_threads = _num_threads;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			Vc * const zl = &z[2 * 4 / VSIZE * lh];

			backward_out(zl, w122i);

			for (size_t j = 0; j < n_io_inv; ++j)
			{
				Vc * const zj = &zl[index(n_io) * j];
				Vc8 z8(zj, index(n_io));
				z8.transpose_in();

				const Vc f_prev = (lh != l_min) ? f[j] : Vc(0.0);
				f[j] = z8.mul_carry(f_prev, g, b, b_inv, 2.0 / N, sb, sb_inv, isb, fsb, err);

				if (lh != l_min) z8.transpose_out();
				z8.store(zj, index(n_io));	// transposed if lh = l_min
			}

			if (lh != l_min) forward_out(zl, w122i);
		}

		return err.max();
	}

	void pass2_1(const size_t thread_id)
	{
		const size_t num_threads = _num_threads;
		const size_t thread_id_prev = ((thread_id != 0) ? thread_id : num_threads) - 1;
		const size_t lh = thread_id * n_io_s / num_threads;	// l_min of pass2

		Vc * const z = (Vc *)&_mem[zOffset]; Vc * const zl = &z[2 * 4 / VSIZE * lh];
		const Vc * const fc = (Vc *)&_mem[fcOffset]; const Vc * const f = &fc[thread_id_prev * n_io_inv];

		const double b = _b, sb = _sb, isb = _isb, fsb = _fsb;
		const double b_inv = 1.0 / b, sb_inv = 1.0 / sb;

		for (size_t j = 0; j < n_io_inv; ++j)
		{
			Vc * const zj = &zl[index(n_io) * j];
			Vc8 z8(zj, index(n_io));	// transposed

			Vc f_prev = f[j];
			if (thread_id == 0) f_prev.shift(f[((j == 0) ? n_io_inv : j) - 1], j == 0);
			z8.carry(f_prev, b, b_inv, sb, sb_inv, isb, fsb);

			z8.transpose_out();
			z8.store(zj, index(n_io));
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		forward_out(zl, w122i);
	}

public:
	transformCPU(const uint32_t b, const size_t num_threads, const size_t num_regs) : transform(N, b),
		_sqrt_b(fp16_80::sqrt(b)), _num_threads(num_threads), _b(b), _sb(double(sqrtl(b))), _isb(_sqrt_b.hi()), _fsb(_sqrt_b.lo()),
		_mem_size(wSize + wsSize + zSize + fcSize + (num_regs - 1) * zSize), _error(0), _mem((char *)_mm_malloc(_mem_size, 2 * 1024 * 1024))
	{
		Complex * const w122i = (Complex *)&_mem[wOffset];
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

		Vc * const ws = (Vc *)&_mem[wsOffset];
		for (size_t j = 0; j < N / 8 / VSIZE; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				ws[j].set(i, Complex::exp2iPi(bitRev(VSIZE * j + i, 2 * (N / 4)) + 1, 8 * (N / 4)));
			}
		}
	}

	virtual ~transformCPU()
	{
		_mm_free((void *)_mem);
	}

	size_t getMemSize() const override { return _mem_size; }

protected:
	void getZi(int32_t * const zi) const override
	{
		const Vc * const z = (Vc *)&_mem[zOffset];

		Vc * const z_copy = (Vc *)_mm_malloc(zSize, 1024);
		for (size_t k = 0; k < index(N) / VSIZE; ++k) z_copy[k] = z[k];

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			backward_out(&z_copy[2 * 4 / VSIZE * lh], w122i);
		}

		const double sb = _sb, n_io_N = double(n_io) / N;

		for (size_t k = 0; k < N / 2; k += VSIZE / 2)
		{
			Vc vc = z_copy[index(2 * k) / VSIZE];
			for (size_t i = 0; i < VSIZE / 2; ++i)
			{
				const Complex z1 = vc[2 * i + 0], z2 = vc[2 * i + 1];
				zi[k + i + 0 * N / 2] = std::lround((z1.real + sb * z2.real) * n_io_N);
				zi[k + i + 1 * N / 2] = std::lround((z1.imag + sb * z2.imag) * n_io_N);
			}
		}

		_mm_free((void *)z_copy);
	}

	void setZi(int32_t * const zi) override
	{
		Vc * const z = (Vc *)&_mem[zOffset];
		const Vd<VSIZE> isb = Vd<VSIZE>::broadcast(_isb), fsb = Vd<VSIZE>::broadcast(_fsb), sb_inv = Vd<VSIZE>::broadcast(1.0 / _sb);

		for (size_t k = 0; k < N / 2; k += VSIZE / 2)
		{
			Vd<VSIZE> r;
			for (size_t i = 0; i < VSIZE / 2; ++i)
			{
				r.set(2 * i + 0, double(zi[k + i + 0 * N / 2]));
				r.set(2 * i + 1, double(zi[k + i + 1 * N / 2]));
			}

			const Vd<VSIZE> irh = Vd<VSIZE>(r * sb_inv).round();
			const Vd<VSIZE> re = (r - irh * isb) - irh * fsb, im = irh;

			Vc vc;
			for (size_t i = 0; i < VSIZE / 2; ++i)
			{
				vc.set(2 * i + 0, Complex(re[2 * i + 0], re[2 * i + 1]));
				vc.set(2 * i + 1, Complex(im[2 * i + 0], im[2 * i + 1]));
			}

			z[index(2 * k) / VSIZE] = vc;
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&z[2 * 4 / VSIZE * lh], w122i);
		}
	}

public:
	void set(const int32_t a) override
	{
		Vc * const z = (Vc *)&_mem[zOffset];
		z[0] = Vc(a);
		for (size_t k = 1; k < index(N) / VSIZE; ++k) z[k] = Vc(0.0);

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&z[2 * 4 / VSIZE * lh], w122i);
		}
	}

	void squareDup(const bool dup) override
	{
		const size_t num_threads = _num_threads;

		double e[num_threads];

		if (num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				pass1(thread_id);
#pragma omp barrier
				e[thread_id] = pass2_0(thread_id, dup);
#pragma omp barrier
				pass2_1(thread_id);
			}
		}
		else
		{
			pass1(0);
			e[0] = pass2_0(0, dup);
			pass2_1(0);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		_error = std::max(_error, err);
	}

	void initMultiplicand(const size_t src) override
	{
		const Vc * const z_src = (Vc *)&_mem[(src == 0) ? zOffset : zpOffset + src * zSize];
		Vc * const zp = (Vc *)&_mem[zpOffset];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zp[k] = z_src[k];

		if (_num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());
				pass1multiplicand(thread_id);
			}
		}
		else
		{
			pass1multiplicand(0);
		}
	}

	void mul() override
	{
		const size_t num_threads = _num_threads;

		double e[num_threads];

		if (num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				pass1mul(thread_id);
#pragma omp barrier
				e[thread_id] = pass2_0(thread_id, false);
#pragma omp barrier
				pass2_1(thread_id);
			}
		}
		else
		{
			pass1mul(0);
			e[0] = pass2_0(0, false);
			pass2_1(0);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		_error = std::max(_error, err);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const Vc * const z_src = (Vc *)&_mem[(src == 0) ? zOffset : zpOffset + src * zSize];
		Vc * const z_dst = (Vc *)&_mem[(dst == 0) ? zOffset : zpOffset + dst * zSize];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) z_dst[k] = z_src[k];
	}

	void setError(const double error) override { _error = error; }
	double getError() const override { return _error; }
};

template<size_t VSIZE>
inline transform * create_transformCPU(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs)
{
	transform * pTransform = nullptr;
	if (n == 10)      pTransform = new transformCPU<(1 << 10), VSIZE>(b, num_threads, num_regs);
	else if (n == 11) pTransform = new transformCPU<(1 << 11), VSIZE>(b, num_threads, num_regs);
	else if (n == 12) pTransform = new transformCPU<(1 << 12), VSIZE>(b, num_threads, num_regs);
	else if (n == 13) pTransform = new transformCPU<(1 << 13), VSIZE>(b, num_threads, num_regs);
	else if (n == 14) pTransform = new transformCPU<(1 << 14), VSIZE>(b, num_threads, num_regs);
	else if (n == 15) pTransform = new transformCPU<(1 << 15), VSIZE>(b, num_threads, num_regs);
	else if (n == 16) pTransform = new transformCPU<(1 << 16), VSIZE>(b, num_threads, num_regs);
	else if (n == 17) pTransform = new transformCPU<(1 << 17), VSIZE>(b, num_threads, num_regs);
	else if (n == 18) pTransform = new transformCPU<(1 << 18), VSIZE>(b, num_threads, num_regs);
	else if (n == 19) pTransform = new transformCPU<(1 << 19), VSIZE>(b, num_threads, num_regs);
	else if (n == 20) pTransform = new transformCPU<(1 << 20), VSIZE>(b, num_threads, num_regs);
	else if (n == 21) pTransform = new transformCPU<(1 << 21), VSIZE>(b, num_threads, num_regs);
	else if (n == 22) pTransform = new transformCPU<(1 << 22), VSIZE>(b, num_threads, num_regs);
	if (pTransform == nullptr) throw std::runtime_error("exponent is not supported");

	return pTransform;
}
