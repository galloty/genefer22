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

template<size_t N>
class Vd
{
private:
	double __attribute__((aligned(8 * N))) r[N];

public:
	explicit Vd() {}
	constexpr explicit Vd(const double & f) { r[0] = f; for (size_t i = 1; i < N; ++i) r[i] = 0; }
	constexpr Vd(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = rhs.r[i]; }
	Vd & operator=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = rhs.r[i]; return *this; }

	static Vd broadcast(const double * const mem) { Vd vd; const double f = mem[0]; for (size_t i = 0; i < N; ++i) vd.r[i] = f; return vd; }
	static Vd broadcast(const double * const mem_l, const double * const mem_h)
	{
		Vd vd;
		const double f_l = mem_l[0];
		for (size_t i = 0; i < N / 2; ++i) vd.r[i] = f_l;
		const double f_h = mem_h[0];
		for (size_t i = 0; i < N / 2; ++i) vd.r[i + N / 2] = f_h;
		return vd;
	}

	const double & operator[](const size_t i) const { return r[i]; }
	double & operator[](const size_t i) { return r[i]; }

	bool isZero() const { bool zero = true; for (size_t i = 0; i < N; ++i) zero &= (r[i] == 0.0); return zero; }

	Vd operator-() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = -r[i]; return vd; }

	Vd & operator+=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] += rhs.r[i]; return *this; }
	Vd & operator-=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] -= rhs.r[i]; return *this; }
	Vd & operator*=(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] *= rhs.r[i]; return *this; }
	Vd & operator*=(const double & f) { for (size_t i = 0; i < N; ++i) r[i] *= f; return *this; }

	Vd operator+(const Vd & rhs) const { Vd vd = *this; vd += rhs; return vd; }
	Vd operator-(const Vd & rhs) const { Vd vd = *this; vd -= rhs; return vd; }
	Vd operator*(const Vd & rhs) const { Vd vd = *this; vd *= rhs; return vd; }
	Vd operator*(const double & f) const { Vd vd = *this; vd *= f; return vd; }

	Vd abs() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = std::fabs(r[i]); return vd; }
	Vd round() const { Vd vd; for (size_t i = 0; i < N; ++i) vd.r[i] = std::rint(r[i]); return vd; }

	Vd & max(const Vd & rhs) { for (size_t i = 0; i < N; ++i) r[i] = std::max(r[i], rhs.r[i]); return *this; }

	double max() const { double m = r[0]; for (size_t i = 1; i < N; ++i) m = std::max(m, r[i]); return m; }
};

template<size_t N>
class __attribute__((aligned(2 * 8  * N))) Vcx
{
private:
	Vd<N> re, im;

public:
	explicit Vcx() {}
	constexpr explicit Vcx(const double & real) : re(real), im(0.0) {}
	constexpr Vcx(const Vcx & rhs) : re(rhs.re), im(rhs.im) {}
	constexpr explicit Vcx(const Vd<N> & real, const Vd<N> & imag) : re(real), im(imag) {}
	Vcx & operator=(const Vcx & rhs) { re = rhs.re; im = rhs.im; return *this; }

	static Vcx broadcast(const Complex * const mem) { return Vcx(Vd<N>::broadcast(&mem->real), Vd<N>::broadcast(&mem->imag)); }
	static Vcx broadcast(const Complex * const mem_l, const Complex * const mem_h)
	{
		return Vcx(Vd<N>::broadcast(&mem_l->real, &mem_h->real), Vd<N>::broadcast(&mem_l->imag, &mem_h->imag));
	}

	static Vcx read(const Complex * const mem)
	{
		Vcx vcx;
		for (size_t i = 0; i < N; ++i)
		{
			vcx.re[i] = mem[i].real; vcx.im[i] = mem[i].imag;
		}
		return vcx;
	}
	void write(Complex * const mem) const
	{
		for (size_t i = 0; i < N; ++i) mem[i] = Complex(re[i], im[i]);
	}

	static Vcx read(const Complex * const mem1, const Complex * const mem2)
	{
		Vcx vcx;
		for (size_t i = 0; i < N / 2; ++i)
		{
			vcx.re[i] = mem1[i].real; vcx.im[i] = mem1[i].imag;
		}
		for (size_t i = 0; i < N / 2; ++i)
		{
			vcx.re[i + N / 2] = mem2[i].real; vcx.im[i + N / 2] = mem2[i].imag;
		}
		return vcx;
	}
	void write(Complex * const mem1, Complex * const mem2) const
	{
		for (size_t i = 0; i < N / 2; ++i) mem1[i] = Complex(re[i], im[i]);
		for (size_t i = 0; i < N / 2; ++i) mem2[i] = Complex(re[i + N / 2], im[i + N / 2]);
	}

	bool isZero() const { return (re.isZero() & im.isZero()); }

	Vcx & operator+=(const Vcx & rhs) { re += rhs.re; im += rhs.im; return *this; }
	Vcx & operator-=(const Vcx & rhs) { re -= rhs.re; im -= rhs.im; return *this; }
	Vcx & operator*=(const double & f) { re *= f; im *= f; return *this; }

	Vcx operator+(const Vcx & rhs) const { return Vcx(re + rhs.re, im + rhs.im); }
	Vcx operator-(const Vcx & rhs) const { return Vcx(re - rhs.re, im - rhs.im); }
	Vcx addi(const Vcx & rhs) const { return Vcx(re - rhs.im, im + rhs.re); }
	Vcx subi(const Vcx & rhs) const { return Vcx(re + rhs.im, im - rhs.re); }
	Vcx sub_i(const Vcx & rhs) const { return Vcx(rhs.im - im, re - rhs.re); }

	Vcx operator*(const Vcx & rhs) const { return Vcx(re * rhs.re - im * rhs.im, im * rhs.re + re * rhs.im); }
	Vcx operator*(const double & f) const { return Vcx(re * f, im * f); }
	Vcx mul1i() const { return Vcx(re - im, im + re); }
	Vcx mul1mi() const { return Vcx(re + im, im - re); }
	Vcx muli() const { return Vcx(-im, re); }
	Vcx mulmi() const { return Vcx(im, -re); }

	Vcx sqr() const { return Vcx(re * re - im * im, (re + re) * im); }

	Vcx mulW(const Vcx & rhs) const { return Vcx((re - im * rhs.im) * rhs.re, (im + re * rhs.im) * rhs.re); }
	Vcx mulWconj(const Vcx & rhs) const { return Vcx((re + im * rhs.im) * rhs.re, (im - re * rhs.im) * rhs.re); }

	Vcx abs() const { return Vcx(re.abs(), im.abs()); }
	Vcx round() const { return Vcx(re.round(), im.round()); }

	Vcx & max(const Vcx & rhs) { re.max(rhs.re); im.max(rhs.im); return *this; }

	double max() const { return std::max(re.max(), im.max()); }

	Vcx shift(const Vcx & rhs, const bool rotate) const
	{
		// f x^n = -f
		Vcx vcx;
		for (size_t i = N - 1; i > 0; --i)
		{
			vcx.re[i] = re[i - 1];
			vcx.im[i] = im[i - 1];
		}
		if (rotate)
		{
			vcx.re[0] = -rhs.im[N - 1];
			vcx.im[0] = rhs.re[N - 1];
		}
		else
		{
			vcx.re[0] = rhs.re[N - 1];
			vcx.im[0] = rhs.im[N - 1];
		}
		return vcx;
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
	explicit Vradix4(const Complex * const mem, const size_t step)
	{
		for (size_t i = 0; i < 4; ++i) z[i] = Vc::read(&mem[i * step]);
	}

	void store(Complex * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 4; ++i) z[i].write(&mem[i * step]);
	}

	explicit Vradix4(const Complex * const mem1, const Complex * const mem2, const size_t step)
	{
		for (size_t i = 0; i < 4; ++i) z[i] = Vc::read(&mem1[i * step], &mem2[i * step]);
	}

	void store(Complex * const mem1, Complex * const mem2, const size_t step) const
	{
		for (size_t i = 0; i < 4; ++i) z[i].write(&mem1[i * step], &mem2[i * step]);
	}

	void forward4e(const Vc & w0, const Vc & w1)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w0), u1 = z[1], u3 = z[3].mulW(w0);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = Vc(u1 + u3).mulW(w1), v3 = Vc(u1 - u3).mulW(w1);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
	}

	void forward4o(const Vc & w0, const Vc & w2)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w0), u1 = z[1], u3 = z[3].mulW(w0);
		const Vc v0 = u0.addi(u2), v2 = u0.subi(u2), v1 = u1.addi(u3).mulW(w2), v3 = u1.subi(u3).mulW(w2);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2.addi(v3); z[3] = v2.subi(v3);
	}

	void backward4e(const Vc & w0, const Vc & w1)
	{
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w1), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w1);
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mulWconj(w0); z[1] = u1.subi(u3); z[3] = u1.addi(u3).mulWconj(w0);
	}

	void backward4o(const Vc & w0, const Vc & w2)
	{
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = Vc(v0 - v1).mulWconj(w2), u2 = v2 + v3, u3 = Vc(v2 - v3).mulWconj(w2);
		z[0] = u0 + u2; z[2] = u2.sub_i(u0).mulWconj(w0); z[1] = u1.subi(u3); z[3] = u3.subi(u1).mulWconj(w0);
	}

	void forward4_0(const Vc & w0)
	{
		const Vc u0 = z[0], u2 = z[2].mul1i(), u1 = z[1].mulW(w0), u3 = z[3].mulWconj(w0);
		const Vc v0 = u0 + u2 * csqrt2_2, v2 = u0 - u2 * csqrt2_2, v1 = u1.addi(u3), v3 = u3.addi(u1);
		z[0] = v0 + v1; z[1] = v0 - v1; z[2] = v2 + v3; z[3] = v2 - v3;
	}

	void backward4_0(const Vc & w0)
	{
		const Vc v0 = z[0], v1 = z[1], v2 = z[2], v3 = z[3];
		const Vc u0 = v0 + v1, u1 = v0 - v1, u2 = v2 + v3, u3 = v2 - v3;
		z[0] = u0 + u2; z[2] = Vc(u0 - u2).mul1mi() * csqrt2_2; z[1] = u1.subi(u3).mulWconj(w0); z[3] = u3.subi(u1).mulW(w0);
	}

	void square4e(const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const Vc s0 = v0.sqr() + v1.sqr().mulW(w), s1 = (v0 + v0) * v1, s2 = v2.sqr() - v3.sqr().mulW(w), s3 = (v2 + v2) * v3;
		z[0] = s0 + s2; z[2] = Vc(s0 - s2).mulWconj(w); z[1] = s1 + s3; z[3] = Vc(s1 - s3).mulWconj(w);
	}

	void square4o(const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		const Vc v0 = u0.addi(u2), v2 = u0.subi(u2), v1 = u1.addi(u3), v3 = u3.addi(u1);
		const Vc s0 = v1.sqr().mulW(w).subi(v0.sqr()), s1 = (v0 + v0) * v1, s2 = v2.sqr().addi(v3.sqr().mulW(w)), s3 = (v2 + v2) * v3;
		z[0] = s2.addi(s0); z[2] = s0.addi(s2).mulWconj(w); z[1] = s1.subi(s3); z[3] = s3.subi(s1).mulWconj(w);
	}
};

template<size_t N>
class Vradix8
{
private:
	using Vc = Vcx<N>;
	Vc z[8];

public:
	explicit Vradix8(const Complex * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i) z[i] = Vc::read(&mem[i * step]);
	}

	void store(Complex * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i) z[i].write(&mem[i * step]);
	}

	void forward8_0()
	{
		const Vc w0 = Vc::broadcast(&cs2pi_1_16);
		const Vc u0 = z[0], u4 = z[4].mul1i(), u2 = z[2].mulW(w0), u6 = z[6].mul1i().mulW(w0);
		const Vc u1 = z[1], u5 = z[5].mul1i(), u3 = z[3].mulW(w0), u7 = z[7].mul1i().mulW(w0);
		const Vc v0 = u0 + u4 * csqrt2_2, v4 = u0 - u4 * csqrt2_2, v2 = u2 + u6 * csqrt2_2, v6 = u2 - u6 * csqrt2_2;
		const Vc w1 = Vc::broadcast(&cs2pi_1_32), w2 = Vc::broadcast(&cs2pi_5_32);;
		const Vc v1 = Vc(u1 + u5 * csqrt2_2).mulW(w1), v5 = Vc(u1 - u5 * csqrt2_2).mulW(w2);
		const Vc v3 = Vc(u3 + u7 * csqrt2_2).mulW(w1), v7 = Vc(u3 - u7 * csqrt2_2).mulW(w2);
		const Vc s0 = v0 + v2, s2 = v0 - v2, s1 = v1 + v3, s3 = v1 - v3;
		const Vc s4 = v4.addi(v6), s6 = v4.subi(v6), s5 = v5.addi(v7), s7 = v5.subi(v7);
		z[0] = s0 + s1; z[1] = s0 - s1; z[2] = s2.addi(s3); z[3] = s2.subi(s3);
		z[4] = s4 + s5; z[5] = s4 - s5; z[6] = s6.addi(s7); z[7] = s6.subi(s7);
	}

	void backward8_0()
	{
		const Vc s0 = z[0], s1 = z[1], s2 = z[2], s3 = z[3], s4 = z[4], s5 = z[5], s6 = z[6], s7 = z[7];
		const Vc w1 = Vc::broadcast(&cs2pi_1_32), w2 = Vc::broadcast(&cs2pi_5_32);;
		const Vc v0 = s0 + s1, v1 = Vc(s0 - s1).mulWconj(w1), v2 = s2 + s3, v3 = Vc(s2 - s3).mulWconj(w1);
		const Vc v4 = s4 + s5, v5 = Vc(s4 - s5).mulWconj(w2), v6 = s6 + s7, v7 = Vc(s6 - s7).mulWconj(w2);
		const Vc u0 = v0 + v2, u2 = v0 - v2, u4 = v4 + v6, u6 = v4 - v6;
		const Vc u1 = v1.subi(v3), u3 = v1.addi(v3), u5 = v5.subi(v7), u7 = v5.addi(v7);
		const Vc w0 = Vc::broadcast(&cs2pi_1_16);
		z[0] = u0 + u4; z[4] = Vc(u0 - u4).mul1mi() * csqrt2_2; z[2] = u2.subi(u6).mulWconj(w0); z[6] = u6.subi(u2).mulW(w0);
		z[1] = u1 + u5; z[5] = Vc(u1 - u5).mul1mi() * csqrt2_2; z[3] = u3.subi(u7).mulWconj(w0); z[7] = u7.subi(u3).mulW(w0);
	}
};

class ComplexITransform
{
public:
	virtual bool isPrime() = 0;
	virtual double squareDup(const bool dup, const size_t num_threads) = 0;
};

template<size_t N, size_t VSIZE>
class CZIT_CPU_vec_mt : public ComplexITransform
{
private:
	static const size_t l_shift = 4;		// l_shift >= VSIZE / 2
	static const size_t n_io = 16 * 4;		// power of 4, n_io >= 64, n_io >= 2 * l_shift, n >= 16 * n_io, n >= num_threads * n_io
	static const size_t n_io_s = n_io / l_shift / 2;
	static const size_t n_io_inv = N / n_io / VSIZE;

	using Vc = Vcx<VSIZE>;
	using Vr4 = Vradix4<VSIZE>;
	using Vr8 = Vradix8<VSIZE>;

	const fp16_80 sqrt_b;

	const uint32_t _b;
	const size_t _num_threads;
	const double _sb, _isb, _fsb;
	Complex * const _w122i;
	Complex * const _ws;
	Complex * const _z;
	Vc * const _f;

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	static constexpr size_t index(const size_t k) { const size_t j = k / n_io, i = k % n_io; return j * (n_io + 4) + i; }

	void reducePos()
	{
		const uint32_t b = _b;
		const double sb = _sb, n_io_N = double(n_io) / N;
		Complex * const z = _z;

		int64_t f = 0;
		for (size_t k = 0; k < 2 * N; k += 2)
		{
			const double o1 = (k < N) ? z[index(k + 0)].real : z[index(k + 0 - N)].imag;
			const double o2 = (k < N) ? z[index(k + 1)].real : z[index(k + 1 - N)].imag;
			const double o = (o1 + sb * o2) * n_io_N;
			f += llrint(o);
			int32_t r = f % b; if (r < 0) r += b;
			f -= r; f /= b;
			if (k < N) z[index(k + 0)] = Complex(double(r), z[index(k + 0)].imag); else z[index(k + 0 - N)] = Complex(z[index(k + 0 - N)].real, double(r));
			if (k < N) z[index(k + 1)] = Complex(0.0, z[index(k + 1)].imag); else z[index(k + 1 - N)] = Complex(z[index(k + 1 - N)].real, 0.0);
		}
		while (f != 0)
		{
			f = -f;
			for (size_t k = 0; k < 2 * N; k += 2)
			{
				const double o = (k < N) ? z[index(k)].real : z[index(k - N)].imag;
				f += llrint(o);
				int32_t r = f % b; if (r < 0) r += b;
				f -= r; f /= b;
				if (k < N) z[index(k)] = Complex(double(r), z[index(k)].imag); else z[index(k - N)] = Complex(z[index(k - N)].real, double(r));
				if (r == 0) break;
			}
		}
	}

	template <size_t stepi, size_t count>
	static void forward4e(const size_t mi, Complex * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += (stepi == 1) ? VSIZE : stepi)
		{
			for (size_t i = 0; i < count; i += (count != 1) ? VSIZE : 1)
			{
				Complex * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.forward4e(w0, w1);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	static void forward4o(const size_t mi, Complex * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += (stepi == 1) ? VSIZE : stepi)
		{
			for (size_t i = 0; i < count; i += (count != 1) ? VSIZE : 1)
			{
				Complex * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.forward4o(w0, w2);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	static void backward4e(const size_t mi, Complex * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += (stepi == 1) ? VSIZE : stepi)
		{
			for (size_t i = 0; i < count; i += (count != 1) ? VSIZE : 1)
			{
				Complex * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.backward4e(w0, w1);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	static void backward4o(const size_t mi, Complex * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += (stepi == 1) ? VSIZE : stepi)
		{
			for (size_t i = 0; i < count; i += (count != 1) ? VSIZE : 1)
			{
				Complex * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.backward4o(w0, w2);
				vr.store(zi, mi);
			}
		}
	}

	static void forward4e_4(Complex * const z, const Vc & w0, const Vc & w1)
	{
		Vr4 vr(z, &z[32], 4);
		vr.forward4e(w0, w1);
		vr.store(z, &z[32], 4);
	}

	static void forward4o_4(Complex * const z, const Vc & w0, const Vc & w2)
	{
		Vr4 vr(z, &z[32], 4);
		vr.forward4o(w0, w2);
		vr.store(z, &z[32], 4);
	}

	static void backward4e_4(Complex * const z, const Vc & w0, const Vc & w1)
	{
		Vr4 vr(z, &z[32], 4);
		vr.backward4e(w0, w1);
		vr.store(z, &z[32], 4);
	}

	static void backward4o_4(Complex * const z, const Vc & w0, const Vc & w2)
	{
		Vr4 vr(z, &z[32], 4);
		vr.backward4o(w0, w2);
		vr.store(z, &z[32], 4);
	}

	template <size_t stepi, size_t count>
	static void forward4_0(const size_t mi, Complex * const z)
	{
		const Vc w0 = Vc::broadcast(&cs2pi_1_16);
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; i += VSIZE)
			{
				Complex * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.forward4_0(w0);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	static void backward4_0(const size_t mi, Complex * const z)
	{
		const Vc w0 = Vc::broadcast(&cs2pi_1_16);
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; i += VSIZE)
			{
				Complex * const zi = &z[j + i];
				Vr4 vr(zi, mi);
				vr.backward4_0(w0);
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	static void forward8_0(const size_t mi, Complex * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; i += VSIZE)
			{
				Complex * const zi = &z[j + i];
				Vr8 vr(zi, mi);
				vr.forward8_0();
				vr.store(zi, mi);
			}
		}
	}

	template <size_t stepi, size_t count>
	static void backward8_0(const size_t mi, Complex * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; i += VSIZE)
			{
				Complex * const zi = &z[j + i];
				Vr8 vr(zi, mi);
				vr.backward8_0();
				vr.store(zi, mi);
			}
		}
	}

	static void square4e(Complex * const z, const Vc & w)
	{
		Vr4 vr(z, VSIZE);
		vr.square4e(w);
		vr.store(z, VSIZE);
	}

	static void square4o(Complex * const z, const Vc & w)
	{
		Vr4 vr(z, VSIZE);
		vr.square4o(w);
		vr.store(z, VSIZE);
	}

	static void forward_out(Complex * const z, const size_t lh, const Complex * const w122i)
	{
		size_t s = (N / 4) / n_io / 2; for (; s >= 4 * 2; s /= 4);

		if (s == 4) forward8_0<index(n_io), 2 * l_shift>(index(N / 8), &z[2 * l_shift * lh]);
		else        forward4_0<index(n_io), 2 * l_shift>(index(N / 4), &z[2 * l_shift * lh]);

		for (size_t mi = index((s == 4) ? N / 32 : N / 16); mi >= index(n_io); mi /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 2 * l_shift * lh + 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(&w[0]), w1 = Vc::broadcast(&w[1]);
				forward4e<index(n_io), 2 * l_shift>(mi, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(&w[2]);
				forward4o<index(n_io), 2 * l_shift>(mi, &z[k + 1 * 4 * mi], w0, w2);
			}
		}
	}

	static void backward_out(Complex * const z, const size_t lh, const Complex * const w122i)
	{
		size_t s = (N / 4) / n_io / 2;
		for (size_t mi = index(n_io); s >= 2; mi *= 4, s /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 2 * l_shift * lh + 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(&w[0]), w1 = Vc::broadcast(&w[1]);
				backward4e<index(n_io), 2 * l_shift>(mi, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(&w[2]);
				backward4o<index(n_io), 2 * l_shift>(mi, &z[k + 1 * 4 * mi], w0, w2);
			}
		}

		if (s == 1) backward8_0<index(n_io), 2 * l_shift>(index(N / 8), &z[2 * l_shift * lh]);
		else        backward4_0<index(n_io), 2 * l_shift>(index(N / 4), &z[2 * l_shift * lh]);
	}

	void pass1(const size_t thread_id)
	{
		const Complex * const w122i = _w122i;
		const Complex * const ws = _ws;
		Complex * const z = _z;

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Complex * const zl = &z[index(n_io * l)];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(&w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(&w[1]); forward4e<1, 1>(n_io / 4, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(&w[2]); forward4o<1, 1>(n_io / 4, zl, w0, w2); }
			}

			for (size_t m = n_io / 16, s = 2; m >= ((VSIZE > 4) ? 16 : 4); m /= 4, s *= 4)
			{
				for (size_t j = 0; j < s; ++j)
				{
					const size_t k = 8 * m * j;
					const Complex * const w = &w122i[(s_io + 3 * l) * s + 3 * j];
					const Vc w0 = Vc::broadcast(&w[0]), w1 = Vc::broadcast(&w[1]);
					forward4e<1, 1>(m, &zl[k + 0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(&w[2]);
					forward4o<1, 1>(m, &zl[k + 1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE > 4)
			{
				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					const size_t k = 32 * j;
					const Complex * const w = &w122i[(s_io + 3 * l) * (n_io / 32) + 3 * j];
					const Vc w0 = Vc::broadcast(&w[0], &w[3]), w1 = Vc::broadcast(&w[1], &w[4]);
					forward4e_4(&zl[k + 0 * 16], w0, w1);
					const Vc w2 = Vc::broadcast(&w[2], &w[5]);
					forward4o_4(&zl[k + 1 * 16], w0, w2);
				}
			}

			// square
			for (size_t j = 0; j < n_io / 8; j += VSIZE)
			{
				const size_t k = 8 * j;
				const Vc ws0 = Vc::read(&ws[l * n_io / 8 + j]);

				Complex ze[4 * VSIZE], zo[4 * VSIZE];
				for (size_t i = 0; i < VSIZE; ++i)
				{
					for (size_t l = 0; l < 4; ++l)
					{
						ze[VSIZE * l + i] = zl[k + 8 * i + l];
						zo[VSIZE * l + i] = zl[k + 8 * i + l + 4];
					}
				}
				square4e(ze, ws0);
				square4o(zo, ws0);
				for (size_t i = 0; i < VSIZE; ++i)
				{
					for (size_t l = 0; l < 4; ++l)
					{
						zl[k + 8 * i + l] = ze[VSIZE * l + i];
						zl[k + 8 * i + l + 4] = zo[VSIZE * l + i];
					}
				}
			}

			if (VSIZE > 4)
			{
				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					const size_t k = 32 * j;
					const Complex * const w = &w122i[(s_io + 3 * l) * (n_io / 32) + 3 * j];
					const Vc w0 = Vc::broadcast(&w[0], &w[3]), w1 = Vc::broadcast(&w[1], &w[4]);
					backward4e_4(&zl[k + 0 * 16], w0, w1);
					const Vc w2 = Vc::broadcast(&w[2], &w[5]);
					backward4o_4(&zl[k + 1 * 16], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE > 4) ? 16 : 4), s = n_io / 4 / m / 2; m <= n_io / 16; m *= 4, s /= 4)
			{
				for (size_t j = 0; j < s; ++j)
				{
					const size_t k = 8 * m * j;
					const Complex * const w = &w122i[(s_io + 3 * l) * s + 3 * j];
					const Vc w0 = Vc::broadcast(&w[0]), w1 = Vc::broadcast(&w[1]);
					backward4e<1, 1>(m, &zl[k + 0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(&w[2]);
					backward4o<1, 1>(m, &zl[k + 1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(&w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(&w[1]); backward4e<1, 1>(n_io / 4, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(&w[2]); backward4o<1, 1>(n_io / 4, zl, w0, w2); }
			}
		}
	}

	double pass2(const size_t thread_id, const bool dup)
	{
		const double b = double(_b);
		const Complex * const w122i = _w122i;
		Complex * const z = _z;
		Vc * const f = _f;
		const double sb = _sb, isb = _isb, fsb = _fsb;

		const double b_inv = 1.0 / b, sb_inv = 1.0 / sb, t2_n = 2.0 / N, g = dup ? 2 : 1;

		Vc err = Vc(0.0);

		const size_t num_threads = _num_threads;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			backward_out(z, lh, w122i);

			// carry_out
			for (size_t j = 0; j < n_io_inv; ++j)
			{
				Complex _z0[l_shift][VSIZE], _z1[l_shift][VSIZE];
				for (size_t i = 0; i < VSIZE; ++i)
				{
					const size_t k = index(n_io) * (VSIZE * j + i) + 2 * l_shift * lh;
					for (size_t l = 0; l < l_shift; ++l)
					{
						_z0[l][i] = z[k + 2 * l + 0]; _z1[l][i] = z[k + 2 * l + 1];
					}
				}
				Vc f_k_f = Vc(0.0);

				for (size_t l = 0; l < l_shift; ++l)
				{
					Vc z0 = Vc::read(_z0[l]), z1 = Vc::read(_z1[l]);

					const Vc o = (z0 + z1 * sb) * t2_n, oi = o.round(), d = Vc(o - oi).abs();
					const Vc f_i = f_k_f + oi * g;
					err.max(d);
					const Vc f_o = Vc(f_i * b_inv).round();
					const Vc r = f_i - f_o * b;
					f_k_f = f_o;
					const Vc irh = Vc(r * sb_inv).round();
					z0 = (r - irh * isb) - irh * fsb;
					z1 = irh;

					z0.write(_z0[l]); z1.write(_z1[l]);
				}

				f[lh * n_io_inv + j] = f_k_f;

				for (size_t i = 0; i < VSIZE; ++i)
				{
					const size_t k = index(n_io) * (VSIZE * j + i) + 2 * l_shift * lh;
					for (size_t l = 0; l < l_shift; ++l)
					{
						z[k + 2 * l + 0] = _z0[l][i]; z[k + 2 * l + 1] = _z1[l][i];
					}
				}
			}
		}

		return err.max();
	}

	void pass3(const size_t thread_id)
	{
		const double b = double(_b);
		const Complex * const w122i = _w122i;
		Complex * const z = _z;
		Vc * const f = _f;
		const double sb = _sb, isb = _isb, fsb = _fsb;

		const double b_inv = 1.0 / b, sb_inv = 1.0 / sb;

		const size_t num_threads = _num_threads;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			// carry_in
			for (size_t j = 0; j < n_io_inv; ++j)
			{
				Complex _z0[l_shift][VSIZE], _z1[l_shift][VSIZE];
				for (size_t i = 0; i < VSIZE; ++i)
				{
					const size_t k = index(n_io) * (VSIZE * j + i) + 2 * l_shift * lh;
					for (size_t l = 0; l < l_shift; ++l)
					{
						_z0[l][i] = z[k + 2 * l + 0]; _z1[l][i] = z[k + 2 * l + 1];
					}
				}

				const size_t lh_prev = ((lh != 0) ? lh : n_io_s) - 1;
				Vc f_j = f[lh_prev * n_io_inv + j];
				if (lh == 0)
				{
					const size_t j_prev = ((j == 0) ? n_io_inv : j) - 1;
					f_j = f_j.shift(f[(n_io_s - 1) * n_io_inv + j_prev], j == 0);
				}

				for (size_t l = 0; l < l_shift - 1; ++l)
				{
					Vc z0 = Vc::read(_z0[l]), z1 = Vc::read(_z1[l]);

					const Vc o = z0 + z1 * sb, oi = o.round();
					f_j += oi;
					const Vc f_o = Vc(f_j * b_inv).round();
					const Vc r = f_j - f_o * b;
					f_j = f_o;
					const Vc irh = Vc(r * sb_inv).round();
					z0 = (r - irh * isb) - irh * fsb;
					z1 = irh;

					z0.write(_z0[l]); z1.write(_z1[l]);
					if (f_j.isZero()) break;
				}

				if (!f_j.isZero())
				{
					Vc z0 = Vc::read(_z0[l_shift - 1]), z1 = Vc::read(_z1[l_shift - 1]);

					const Vc o = z0 + z1 * sb, oi = o.round();
					const Vc r = f_j + oi;
					const Vc irh = Vc(r * sb_inv).round();
					z0 = (r - irh * isb) - irh * fsb;
					z1 = irh;

					z0.write(_z0[l_shift - 1]); z1.write(_z1[l_shift - 1]);
				}

				for (size_t i = 0; i < VSIZE; ++i)
				{
					const size_t k = index(n_io) * (VSIZE * j + i) + 2 * l_shift * lh;
					for (size_t l = 0; l < l_shift; ++l)
					{
						z[k + 2 * l + 0] = _z0[l][i]; z[k + 2 * l + 1] = _z1[l][i];
					}
				}
			}

			forward_out(z, lh, w122i);
		}
	}

public:
	CZIT_CPU_vec_mt(const uint32_t b, const size_t num_threads) : sqrt_b(fp16_80::sqrt(b)), _b(b),
		_num_threads(num_threads), _sb(double(sqrtl(b))), _isb(sqrt_b.hi()), _fsb(sqrt_b.lo()),
		_w122i(new Complex[N / 8]), _ws(new Complex[N / 8]), _z(new Complex[(N / n_io) * (n_io + 4)]), _f(new Vc[N / l_shift / 2 / VSIZE])
	{
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

		Complex * const z = _z;
		z[index(0)] = Complex(2.0, 0.0);
		for (size_t k = 1; k < N; ++k) z[index(k)] = Complex(0.0, 0.0);

		// Complex * const f = _f;
		// for (size_t k = 0; k < N / l_shift / 2; ++k) f[k] = Complex(0.0, 0.0);

		for (size_t lh = 0; lh < n_io / l_shift / 2; ++lh)
		{
			forward_out(z, lh, w122i);
		}
	}

	virtual ~CZIT_CPU_vec_mt()
	{
		delete[] _w122i;
		delete[] _ws;
		delete[] _z;
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

		bool isPrime = ((z[index(0)].real == 1.0) & (z[index(0)].imag == 0.0));
		if (isPrime)
		{
			for (size_t k = 1; k < N; ++k) isPrime &= ((z[index(k)].real == 0.0) & (z[index(k)].imag == 0.0));
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
		if (n == 1024)      t = new CZIT_CPU_vec_mt<1024, 2>(b, num_threads);
		else if (n == 2048) t = new CZIT_CPU_vec_mt<2048, 4>(b, num_threads);
		else if (n == 4096) t = new CZIT_CPU_vec_mt<4096, 8>(b, num_threads);
		else if (n == 8192) t = new CZIT_CPU_vec_mt<8192, 4>(b, num_threads);
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
