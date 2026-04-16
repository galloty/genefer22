/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

#include "simd128d.h"
#include "simd256d.h"
#include "simd512d.h"

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
class Vd {};

#if defined(__SSE2__) || defined(__ARM_NEON) || (defined(__ARM_FEATURE_SVE) && (__ARM_FEATURE_SVE_BITS == 128))
template<>
class Vd<2>
{
private:
	simd128d r;

private:
	constexpr explicit Vd(const simd128d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r((simd128d){f, 0.0}) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd((simd128d){f, f}); }
	finline static Vd broadcast(const double &, const double &) { return Vd(0.0); }	// unused

	finline double operator[](const size_t i) const { return r[i]; }
	finline void set(const size_t i, const double & f) { r[i] = f; }

	finline bool isZero() const { return is_zero_128d(r); }

	finline Vd operator-() const { return Vd(-r); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline static Vd addmul(const Vd & vd0, const Vd & vd1, const Vd & vd2) { return Vd(addmul_128d(vd0.r, vd1.r, vd2.r)); }
	finline static Vd submul(const Vd & vd0, const Vd & vd1, const Vd & vd2) { return Vd(submul_128d(vd0.r, vd1.r, vd2.r)); }

	finline void shift(const double f) { r = (simd128d){f, r[0]}; }

	finline Vd round() const { return Vd(round_128d(r)); } 

	finline Vd abs() const { return Vd(abs_128d(r)); }
	finline Vd & max(const Vd & rhs) { r = max_128d(r, rhs.r); return *this; }
	finline double max() const { return reduce_max_128d(r); }

	finline void interleave(Vd &) {}	// unused

	finline static void transpose(Vd vd[2]) { transpose_128d(vd[0].r, vd[1].r); }
};
#endif

#if defined(__AVX__) || (defined(__ARM_FEATURE_SVE) && (__ARM_FEATURE_SVE_BITS == 256))
template<>
class Vd<4>
{
private:
	simd256d r;

private:
	constexpr explicit Vd(const simd256d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r((simd256d){f, 0.0, 0.0, 0.0}) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd((simd256d){f, f, f, f}); }
	finline static Vd broadcast(const double &, const double &) { return Vd(0.0); }	// unused

	finline double operator[](const size_t i) const { return r[i]; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
	finline void set(const size_t i, const double & f) { r[i] = f; }
#pragma GCC diagnostic pop

	finline bool isZero() const { return is_zero_256d(r); }

	finline Vd operator-() const { return Vd(-r); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline static Vd addmul(const Vd & vd0, const Vd & vd1, const Vd & vd2) { return Vd(addmul_256d(vd0.r, vd1.r, vd2.r)); }
	finline static Vd submul(const Vd & vd0, const Vd & vd1, const Vd & vd2) { return Vd(submul_256d(vd0.r, vd1.r, vd2.r)); }

	finline void shift(const double f) { r = (simd256d){f, r[0], r[1], r[2]}; }

	finline Vd round() const { return Vd(round_256d(r)); } 

	inline Vd abs() const { return Vd(abs_256d(r)); }
	finline Vd & max(const Vd & rhs) { r = max_256d(r, rhs.r); return *this; }
	finline double max() const { return reduce_max_256d(r); }

	finline void interleave(Vd &) {}	// unused

	finline static void transpose(Vd vd[4]) { transpose_256d(vd[0].r, vd[1].r, vd[2].r, vd[3].r); }
};
#endif

#if defined(__AVX512F__) || (defined(__ARM_FEATURE_SVE) && (__ARM_FEATURE_SVE_BITS == 512))
template<>
class Vd<8>
{
private:
	simd512d r;

private:
	constexpr explicit Vd(const simd512d & _r) : r(_r) {}

public:
	finline explicit Vd() {}
	finline explicit Vd(const double & f) : r((simd512d){f, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) {}
	finline Vd(const Vd & rhs) : r(rhs.r) {}
	finline Vd & operator=(const Vd & rhs) { r = rhs.r; return *this; }

	finline static Vd broadcast(const double & f) { return Vd((simd512d){f, f, f, f, f, f, f, f}); }
	finline static Vd broadcast(const double & f_l, const double & f_h) { return Vd((simd512d){f_l, f_l, f_l, f_l, f_h, f_h, f_h, f_h}); }

	finline double operator[](const size_t i) const { return r[i]; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
	finline void set(const size_t i, const double & f) { r[i] = f; }
#pragma GCC diagnostic pop

	finline bool isZero() const { return is_zero_512d(r); }

	finline Vd operator-() const { return Vd(-r); }

	finline Vd & operator+=(const Vd & rhs) { r += rhs.r; return *this; }
	finline Vd & operator-=(const Vd & rhs) { r -= rhs.r; return *this; }
	finline Vd & operator*=(const Vd & rhs) { r *= rhs.r; return *this; }

	finline Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	finline Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	finline Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }

	finline static Vd addmul(const Vd & vd0, const Vd & vd1, const Vd & vd2) { return Vd(addmul_512d(vd0.r, vd1.r, vd2.r)); }
	finline static Vd submul(const Vd & vd0, const Vd & vd1, const Vd & vd2) { return Vd(submul_512d(vd0.r, vd1.r, vd2.r)); }

	finline void shift(const double f) { r = (simd512d){f, r[0], r[1], r[2], r[3], r[4], r[5], r[6]}; }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

	finline Vd round() const { return Vd(round_512d(r)); }

	finline Vd abs() const { return Vd(abs_512d(r)); }
	finline Vd & max(const Vd & rhs) { r = max_512d(r, rhs.r); return *this; }
	finline double max() const { return reduce_max_512d(r); }

	finline void interleave(Vd & rhs) { interleave_512d(r, rhs.r); }

	finline static void transpose(Vd vd[8]) { { transpose_512d(vd[0].r, vd[1].r, vd[2].r, vd[3].r, vd[4].r, vd[5].r, vd[6].r, vd[7].r); }}

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

	finline Vcx operator+(const Vcx & rhs) const { return Vcx(re + rhs.re, im + rhs.im); }
	finline Vcx operator-(const Vcx & rhs) const { return Vcx(re - rhs.re, im - rhs.im); }
	finline Vcx addi(const Vcx & rhs) const { return Vcx(re - rhs.im, im + rhs.re); }
	finline Vcx subi(const Vcx & rhs) const { return Vcx(re + rhs.im, im - rhs.re); }
	finline Vcx sub_i(const Vcx & rhs) const { return Vcx(rhs.im - im, re - rhs.re); }

	finline Vcx operator*(const Vcx & rhs) const { return Vcx(Vd<N>::submul(re * rhs.re, im, rhs.im), Vd<N>::addmul(im * rhs.re, re,  rhs.im)); }
	finline Vcx mulS(const Vd<N> & vf) const { return Vcx(re * vf, im * vf); }
	finline Vcx mul1i() const { return Vcx(re - im, im + re); }
	finline Vcx mul1mi() const { return Vcx(re + im, im - re); }
	// finline Vcx muli() const { return Vcx(-im, re); }
	// finline Vcx mulmi() const { return Vcx(im, -re); }

	finline Vcx sqr() const { return Vcx(Vd<N>::submul(re * re, im, im), (re + re) * im); }

	finline Vcx mulW(const Vcx & rhs) const { return Vcx(Vd<N>::submul(re, im, rhs.im) * rhs.re, Vd<N>::addmul(im, re, rhs.im) * rhs.re); }
	finline Vcx mulWconj(const Vcx & rhs) const { return Vcx(Vd<N>::addmul(re, im, rhs.im) * rhs.re, Vd<N>::submul(im, re, rhs.im) * rhs.re); }

	finline Vcx addmulS(const Vcx & rhs, const Vd<N> & vf) const { return Vcx(Vd<N>::addmul(re, rhs.re, vf), Vd<N>::addmul(im, rhs.im, vf)); }
	finline Vcx submulS(const Vcx & rhs, const Vd<N> & vf) const { return Vcx(Vd<N>::submul(re, rhs.re, vf), Vd<N>::submul(im, rhs.im, vf)); }

	finline static void fwd2(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = Vcx(Vd<N>::submul(z1.re, z1.im, w.im), Vd<N>::addmul(z1.im, z1.re, w.im));
		z1 = Vcx(Vd<N>::submul(z0.re, t.re, w.re), Vd<N>::submul(z0.im, t.im, w.re));
		z0 = Vcx(Vd<N>::addmul(z0.re, t.re, w.re), Vd<N>::addmul(z0.im, t.im, w.re));
	}

	finline static void fwd2i(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = Vcx(Vd<N>::submul(z1.re, z1.im, w.im), Vd<N>::addmul(z1.im, z1.re, w.im));
		z1 = Vcx(Vd<N>::addmul(z0.re, t.im, w.re), Vd<N>::submul(z0.im, t.re, w.re));
		z0 = Vcx(Vd<N>::submul(z0.re, t.im, w.re), Vd<N>::addmul(z0.im, t.re, w.re));
	}

	finline static void fwd2a(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = Vcx(Vd<N>::submul(z1.re, z1.im, w.im), Vd<N>::addmul(z1.im, z1.re, w.im));
		z1 = Vcx(-Vd<N>::submul(z0.im, t.re, w.re), Vd<N>::addmul(z0.re, t.im, w.re));
		z0 = Vcx(Vd<N>::submul(z0.re, t.im, w.re), Vd<N>::addmul(z0.im, t.re, w.re));
	}

	finline static void bck2(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = z0 - z1; z0 += z1; z1 = t.mulWconj(w);
	}

	// multiplication by w.re is delayed
	finline static void bck2_1(Vcx & z0, Vcx & z1, const Vd<N> & wim)
	{
		const Vcx t = z0 - z1; z0 += z1;
		z1 = Vcx(Vd<N>::addmul(t.re, t.im, wim), Vd<N>::submul(t.im, t.re, wim));
	}

	// z1 is multiplied by w.re
	finline static void bck2i_2(Vcx & z0, Vcx & z1, const Vd<N> & wre, const Vcx & w)
	{
		const Vcx t = Vcx(Vd<N>::submul(z0.re, z1.im, wre), Vd<N>::addmul(z0.im, z1.re, wre));
		z0 = Vcx(Vd<N>::addmul(z0.re, z1.im, wre), Vd<N>::submul(z0.im, z1.re, wre));
		z1 = t.mulWconj(w);
	}

	finline static void bck2_i(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = z1.sub_i(z0); z0 += z1; z1 = t.mulWconj(w);
	}

	// z1 is multiplied by w.re
	finline static void bck2ir_2(Vcx & z0, Vcx & z1, const Vd<N> & wre, const Vcx & w)
	{
		const Vcx tconj = Vcx(Vd<N>::addmul(z0.im, z1.re, wre), Vd<N>::submul(z0.re, z1.im, wre));
		z0 = Vcx(Vd<N>::addmul(z0.re, z1.im, wre), Vd<N>::submul(z0.im, z1.re, wre));
		z1 = Vcx(Vd<N>::submul(tconj.re, tconj.im, w.im) * w.re, -Vd<N>::addmul(tconj.im, tconj.re, w.im) * w.re);
	}

	finline static void bck2a(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = z0.addi(z1); z0 = z1.addi(z0); z1 = t.mulWconj(w);
	}

	finline static void bck2b(Vcx & z0, Vcx & z1, const Vcx & w)
	{
		const Vcx t = z1.subi(z0); z0 = z0.subi(z1); z1 = t.mulWconj(w);
	}

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
	finline explicit Vradix4() {}

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

	Vc & operator [](const size_t i) { return z[i]; }
	Vc operator [](const size_t i) const { return z[i]; }

	finline void interleave()
	{
		z[0].interleave(z[1]); z[2].interleave(z[3]);
	}

	finline void forward4e(const Vc & w0, const Vc & w1)
	{
#if defined(__clang__)	// help clang to generate fma instructions
		// 24 fma
		Vc::fwd2(z[0], z[2], w0); Vc::fwd2(z[1], z[3], w0);
		Vc::fwd2(z[0], z[1], w1); Vc::fwd2i(z[2], z[3], w1);
#else
		// gcc: 24 fma; clang: 8 fma, 8 mul, 16 add
		const Vc u0 = z[0], u2 = z[2].mulW(w0), u1 = z[1], u3 = z[3].mulW(w0);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = Vc(u1 + u3).mulW(w1), v3 = Vc(u1 - u3).mulW(w1);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
#endif
	}

	finline void forward4o(const Vc & w0, const Vc & w2)
	{
#if defined(__clang__)	// help clang to generate fma instructions
		// 24 fma
		Vc::fwd2i(z[0], z[2], w0); Vc::fwd2i(z[1], z[3], w0);
		Vc::fwd2(z[0], z[1], w2); Vc::fwd2i(z[2], z[3], w2);
#else
		// gcc: 24 fma; clang: 8 fma, 8 mul, 16 add
		const Vc u0 = z[0], u2 = z[2].mulW(w0), u1 = z[1], u3 = z[3].mulW(w0);
		const Vc v0 = u0.addi(u2), v2 = u0.subi(u2), v1 = u1.addi(u3).mulW(w2), v3 = u1.subi(u3).mulW(w2);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
#endif
	}

	finline void backward4e(const Vc & w0, const Vc & w1)
	{
#if defined(__clang__)	// help clang to generate fma instructions
		// 12 fma, 6 mul, 12 add
		Vc::bck2(z[0], z[1], w1); Vc::bck2_1(z[2], z[3], w1.imag());
		Vc::bck2(z[0], z[2], w0); Vc::bck2i_2(z[1], z[3], w1.real(), w0);
#else
		// gcc: 12 fma, 6 mul, 12 add; clang: 8 fma, 8 mul, 16 add
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w1), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w1);
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mulWconj(w0); z[1] = u1.subi(u3); z[3] = u1.addi(u3).mulWconj(w0);
#endif
	}

	finline void backward4o(const Vc & w0, const Vc & w2)
	{
#if defined(__clang__)	// help clang to generate fma instructions
		// 12 fma, 6 mul, 12 add
		Vc::bck2(z[0], z[1], w2); Vc::bck2_1(z[2], z[3], w2.imag());
		Vc::bck2_i(z[0], z[2], w0); Vc::bck2ir_2(z[1], z[3], w2.real(), w0);
#else
		// gcc: 12 fma, 6 mul, 12 add; clang: 8 fma, 8 mul, 16 add
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w2), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w2);
		z[0] = u0 + u2; z[2] = u2.sub_i(u0).mulWconj(w0); z[1] = u1.subi(u3); z[3] = u3.subi(u1).mulWconj(w0);
#endif
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
		const Vd<N> vsqrt2_2 = Vd<N>::broadcast(csqrt2_2);
		const Vc v0 = u0.addmulS(u2, vsqrt2_2), v2 = u0.submulS(u2, vsqrt2_2), v1 = u1.addi(u3), v3 = u3.addi(u1);
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
		const Vd<N> vsqrt2_2 = Vd<N>::broadcast(csqrt2_2);
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mul1mi().mulS(vsqrt2_2); z[1] = u1.subi(u3).mulWconj(w0); z[3] = u3.subi(u1).mulW(w0);
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
	finline explicit Vradix8() {}

	finline explicit Vradix8(const Vc * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i) z[i] = mem[i * step];
	}

	finline void store(Vc * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i * step] = z[i];
	}

	Vc & operator [](const size_t i) { return z[i]; }
	Vc operator [](const size_t i) const { return z[i]; }

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
		const Vd<N> vsqrt2_2 = Vd<N>::broadcast(csqrt2_2);
		const Vc v0 = u0.addmulS(u4, vsqrt2_2), v4 = u0.submulS(u4, vsqrt2_2), v2 = u2.addmulS(u6, vsqrt2_2), v6 = u2.submulS(u6, vsqrt2_2);
		const Vc w1 = Vc::broadcast(cs2pi_1_32), w2 = Vc::broadcast(cs2pi_5_32);
		const Vc v1 = u1.addmulS(u5, vsqrt2_2).mulW(w1), v5 = u1.submulS(u5, vsqrt2_2).mulW(w2);
		const Vc v3 = u3.addmulS(u7, vsqrt2_2).mulW(w1), v7 = u3.submulS(u7, vsqrt2_2).mulW(w2);
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
		const Vc w0 = Vc::broadcast(cs2pi_1_16); const Vd<N> vsqrt2_2 = Vd<N>::broadcast(csqrt2_2);
		z[0] = u0 + u4; z[4] = Vc(u0 - u4).mul1mi().mulS(vsqrt2_2); z[2] = u2.subi(u6).mulWconj(w0); z[6] = u6.subi(u2).mulW(w0);
		z[1] = u1 + u5; z[5] = Vc(u1 - u5).mul1mi().mulS(vsqrt2_2); z[3] = u3.subi(u7).mulWconj(w0); z[7] = u7.subi(u3).mulW(w0);
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
