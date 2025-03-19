/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <vector>
#include <iostream>
#include <cmath>
#include <ctime>

#include <gmp.h>

typedef uint32_t	uint32;
typedef int32_t		int32;
typedef uint64_t	uint64;
typedef int64_t		int64;

#define	M31		((uint32(1) << 31) - 1)

#define	P1		(127 * (uint32(1) << 24) + 1)
#define	Q1		2164260865u		// p * q = 1 (mod 2^32)
#define ONE1	33554430u		// Montgomery form of 1 is 2^32 (mod p)
#define	RSQ1	402124772u		// (2^32)^2 mod p
#define	H1		167772150u		// Montgomery form of the primitive root 5
#define	I1		66976762u 		// Montgomery form of 5^{(p - 1)/4} = 16711679
#define	IM1		200536044u		// Montgomery form of Montgomery form of I to convert input into Montgomery form

#define	P2		(63 * (uint32(1) << 25) + 1)
#define	Q2		2181038081u
#define ONE2	67108862u
#define	RSQ2	2111798781u
#define	H2		335544310u		// Montgomery form of the primitive root 5
#define	I2		530075385u
#define	IM2		1036950657u

// GF((2^31 - 1)^2)
class GF31
{
private:
	static const uint32 _p = M31;
	uint32 _s[2];
	// a primitive root of order 2^31 which is a root of (0, 1)
	static const uint32 _h_order = uint32(1) << 31;
	static const uint32 _h_0 = 105066u, _h_1 = 333718u;

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= _p) ? _p : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? _p : 0); }

	static uint32 _mul(const uint32 a, const uint32 b)
	{
		const uint64 t = a * uint64(b);
		const uint32 lo = uint32(t) & _p, hi = uint32(t >> 31);
		return _add(hi, lo);
	}

	static uint32 _lshift(const uint32 a, const int s)
	{
		const uint64 t = uint64(a) << s;
		const uint32 lo = uint32(t) & _p, hi = uint32(t >> 31);
		return _add(hi, lo);
	}

	static int32 _get_int(const uint32 a) { return (a >= _p / 2) ? int32(a - _p) : int32(a); }
	static uint32 _set_int(const int32 a) { return (a < 0) ? (uint32(a) + _p) : uint32(a); }

public:
	GF31() {}
	explicit GF31(const uint32 n0, const uint32 n1) { _s[0] = n0; _s[1] = n1; }

	uint32 s0() const { return _s[0]; }
	uint32 s1() const { return _s[1]; }

	void get_int(int32 & i0, int32 & i1) const { i0 = _get_int(_s[0]); i1 = _get_int(_s[1]); }
	GF31 & set_int(const int32 i0, const int32 i1) { _s[0] = _set_int(i0); _s[1] = _set_int(i1); return *this; }

	GF31 add(const GF31 & rhs) const { return GF31(_add(_s[0], rhs._s[0]), _add(_s[1], rhs._s[1])); }
	GF31 sub(const GF31 & rhs) const { return GF31(_sub(_s[0], rhs._s[0]), _sub(_s[1], rhs._s[1])); }
	GF31 addi(const GF31 & rhs) const { return GF31(_sub(_s[0], rhs._s[1]), _add(_s[1], rhs._s[0])); }
	GF31 subi(const GF31 & rhs) const { return GF31(_add(_s[0], rhs._s[1]), _sub(_s[1], rhs._s[0])); }

	GF31 lshift(const int s) const { return GF31(_lshift(_s[0], s), _lshift(_s[1], s)); }
	GF31 muls(const uint32 s) const { return GF31(_mul(_s[0], s), _mul(_s[1], s)); }

	GF31 mul(const GF31 & rhs) const { return GF31(_sub(_mul(_s[0], rhs._s[0]), _mul(_s[1], rhs._s[1])), _add(_mul(_s[1], rhs._s[0]), _mul(_s[0], rhs._s[1]))); }
	GF31 mulconj(const GF31 & rhs) const { return GF31(_add(_mul(_s[0], rhs._s[0]), _mul(_s[1], rhs._s[1])), _sub(_mul(_s[1], rhs._s[0]), _mul(_s[0], rhs._s[1]))); }
	GF31 sqr() const { const uint32 t = _mul(_s[0], _s[1]); return GF31(_sub(_mul(_s[0], _s[0]), _mul(_s[1], _s[1])), _add(t, t)); }

	// 12 mul + 12 mul_hi
	static void forward4(GF31 & z0, GF31 & z1, GF31 & z2, GF31 & z3, const GF31 & w1, const GF31 & w2, const GF31 & w3)
	{
		const GF31 u0 = z0, u2 = z2.mul(w1), u1 = z1.mul(w2), u3 = z3.mul(w3);
		const GF31 v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3), v3 = u1.sub(u3);
		z0 = v0.add(v1); z1 = v0.sub(v1); z2 = v2.addi(v3); z3 = v2.subi(v3);
	}

	static void backward4(GF31 & z0, GF31 & z1, GF31 & z2, GF31 & z3, const GF31 & w1, const GF31 & w2, const GF31 & w3)
	{
		const GF31 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const GF31 v0 = u0.add(u1), v1 = u0.sub(u1), v2 = u2.add(u3), v3 = u3.sub(u2);
		z0 = v0.add(v2); z2 = v0.sub(v2).mulconj(w1); z1 = v1.addi(v3).mulconj(w2); z3 = v1.subi(v3).mulconj(w3);
	}

	static void square22(GF31 & z0, GF31 & z1, GF31 & z2, GF31 & z3, const GF31 & w)
	{
		const GF31 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		z0 = u0.sqr().add(u1.sqr().mul(w)); z1 = u0.mul(u1.add(u1));
		z2 = u2.sqr().sub(u3.sqr().mul(w)); z3 = u2.mul(u3.add(u3));
	}

	static void square4(GF31 & z0, GF31 & z1, GF31 & z2, GF31 & z3, const GF31 & w)
	{
		const GF31 u0 = z0, u2 = z2.mul(w), u1 = z1, u3 = z3.mul(w);
		const GF31 v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3), v3 = u1.sub(u3);
		const GF31 s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
		const GF31 s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
		z0 = s0.add(s2); z2 = s0.sub(s2).mulconj(w); z1 = s1.add(s3); z3 = s1.sub(s3).mulconj(w);
	}

	GF31 pow(const size_t e) const
	{
		if (e == 0) return GF31(1, 0);
		GF31 r = GF31(1, 0), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF31 primroot_n(const uint32 n) { return GF31(_h_0, _h_1).pow(_h_order / n); }
};

template<uint32 P, uint32 Q, uint32 ONE, uint32 RSQ, uint32 H, uint32 I, uint32 IM>
class Zp
{
private:
	uint32 _s[2];

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= P) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? P : 0); }

	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

	static int32 _get_int(const uint32 a) { return (a >= P / 2) ? int32(a - P) : int32(a); }
	static uint32 _set_int(const int32 a) { return (a < 0) ? (uint32(a) + P) : uint32(a); }

	Zp pow(const size_t e) const
	{
		if (e == 0) return Zp(ONE, 0);
		Zp r = Zp(ONE, 0), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r._s[0] = _mul(r._s[0], y._s[0]); y._s[0] = _mul(y._s[0], y._s[0]); }
		r._s[0] = _mul(r._s[0], y._s[0]);
		return r;
	}

public:
	Zp() {}
	explicit Zp(const uint32 n0, const uint32 n1) { _s[0] = n0; _s[1] = n1; }

	uint32 s0() const { return _s[0]; }
	uint32 s1() const { return _s[1]; }

	Zp & set_int(const int32 i0, const int32 i1) { _s[0] = _set_int(i0); _s[1] = _set_int(i1); return *this; }

	Zp swap() const { return Zp(_s[1], _s[0]); }

	Zp add(const Zp & rhs) const { return Zp(_add(_s[0], rhs._s[0]), _add(_s[1], rhs._s[1])); }
	Zp sub(const Zp & rhs) const { return Zp(_sub(_s[0], rhs._s[0]), _sub(_s[1], rhs._s[1])); }

	Zp muls(const uint32 s) const { return Zp(_mul(_s[0], s), _mul(_s[1], s)); }
	Zp muli() const { return muls(I); }

	Zp mul(const Zp & rhs) const { return Zp(_mul(_s[0], rhs._s[0]), _mul(_s[1], rhs._s[1])); }
	Zp sqr() const { return mul(*this); }

	// Conversion into / out of Montgomery form
	// Zp toMonty() const { return Zp(_mul(_s[0], RSQ), _mul(_s[1], RSQ)); }
	// Zp fromMonty() const { return Zp(_mul(_s[0], 1), _mul(_s[1], 1)); }

	Zp & forward2()
	{
		const uint32 u0 = _mul(_s[0], RSQ), u1 = _mul(_s[1], IM);
		_s[0] = _add(u0, u1); _s[1] = _sub(u0, u1);
		return *this;
	}

	Zp & backward2()
	{
		const uint32 u0 = _s[0], u1 = _s[1];
		_s[0] = _add(u0, u1); _s[1] = _mul(_sub(u1, u0), I);
	 	return *this;
	}

	// 16 mul + 16 mul_hi
	static void forward4(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w1, const Zp & w20, const Zp & w21)
	{
		const Zp u0 = z0, u2 = z2.mul(w1), u1 = z1, u3 = z3.mul(w1);
		const Zp v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3).mul(w20), v3 = u1.sub(u3).mul(w21);
		z0 = v0.add(v1); z1 = v0.sub(v1); z2 = v2.add(v3); z3 = v2.sub(v3);
	}

	static void backward4(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & win1, const Zp & win20, const Zp & win21)
	{
		const Zp u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const Zp v0 = u0.add(u1), v1 = u1.sub(u0).mul(win20), v2 = u2.add(u3), v3 = u3.sub(u2).mul(win21);
		z0 = v0.add(v2); z2 = v2.sub(v0).mul(win1); z1 = v1.add(v3); z3 = v3.sub(v1).mul(win1);
	}

	static void forward8_0(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w1, const Zp & w20, const Zp & w21)
	{
		Zp t0 = z0.forward2(), t2 = z2.forward2(), t1 = z1.forward2(), t3 = z3.forward2();
		forward4(t0, t1, t2, t3, w1, w20, w21);
		z0 = t0; z1 = t1; z2 = t2; z3 = t3;
	}

	static void backward8_0(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & win1, const Zp & win20, const Zp & win21)
	{
		Zp t0 = z0, t2 = z2, t1 = z1, t3 = z3;
		backward4(t0, t1, t2, t3, win1, win20, win21);
		z0 = t0.backward2(); z2 = t2.backward2(); z1 = t1.backward2(); z3 = t3.backward2();
	}

	static void square22(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w)
	{
		const Zp u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		z0 = u0.sqr().add(u1.sqr().mul(w)); z1 = u0.mul(u1.add(u1));
		z2 = u2.sqr().sub(u3.sqr().mul(w)); z3 = u2.mul(u3.add(u3));
	}

	static void square4(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w, const Zp & win)
	{
		const Zp u0 = z0, u2 = z2.mul(w), u1 = z1, u3 = z3.mul(w);
		const Zp v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3), v3 = u1.sub(u3);
		const Zp s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
		const Zp s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
		z0 = s0.add(s2); z2 = s2.sub(s0).mul(win); z1 = s1.add(s3); z3 = s3.sub(s1).mul(win);
	}

	Zp pow_mul_sqr(const size_t e) const { Zp r = pow(e); r._s[1] = _mul(r._s[0], _s[1]); return r; }

	static const Zp primroot_n(const uint32 n) { Zp r = Zp(H, 0).pow((P - 1) / n); r._s[1] = _mul(r._s[0], r._s[0]); return r; }
	static uint32 norm(const uint32 n) { return P - (P - 1) / n; }
};

typedef Zp<P1, Q1, ONE1, RSQ1, H1, I1, IM1> Zp1;
typedef Zp<P2, Q2, ONE2, RSQ2, H2, I2, IM2> Zp2;

class Transform
{
private:
	const size_t _size;
	std::vector<GF31> _vz31;
	std::vector<Zp1> _vz1;
	std::vector<GF31> _vwr31;
	std::vector<Zp1> _vwr1;
	const int32 _base;
	const int32 _multiplier;
	const int _snorm31;
	const uint32 _norm1;
	uint64 _fmax;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	void forward4_31(const size_t m, const size_t s)
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF31 w1 = wr[s + j], w2 = wr[n / 2 + s + j], w3 = wr[n + s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				GF31::forward4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], w1, w2, w3);
			}
		}
	}

	void forward4_1(const size_t m, const size_t s)
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		for (size_t j = 0; j < s; ++j)
		{
			const Zp1 w1 = wr[s + j], w20 = wr[n / 2 + s + j], w21 = wr[n + s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				Zp1::forward4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], w1, w20, w21);
			}
		}
	}

	void forward8_1_0()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		const Zp1 w1 = wr[1], w20 = wr[n / 2 + 1], w21 = wr[n + 1];

		for (size_t i = 0; i < n / 4; ++i)
		{
			Zp1::forward8_0(z[i + 0 * n / 4], z[i + 1 * n / 4], z[i + 2 * n / 4], z[i + 3 * n / 4], w1, w20, w21);
		}
	}

	void backward4_31(const size_t m, const size_t s)
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF31 w1 = wr[s + j], w2 = wr[n / 2 + s + j], w3 = wr[n + s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				GF31::backward4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], w1, w2, w3);
			}
		}
	}

	void backward4_1(const size_t m, const size_t s)
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		for (size_t j = 0; j < s; ++j)
		{
			const size_t ji = s - j - 1;
			const Zp1 win1 = wr[s + ji].swap(), win20 = wr[n + s + ji].swap(), win21 = wr[n / 2 + s + ji].swap();

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				Zp1::backward4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], win1, win20, win21);
			}
		}
	}

	void backward8_1_0()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		const Zp1 win1 = wr[1].swap(), win20 = wr[n + 1].swap(), win21 = wr[n / 2 + 1].swap();

		for (size_t i = 0; i < n / 4; ++i)
		{
			Zp1::backward8_0(z[i + 0 * n / 4], z[i + 1 * n / 4], z[i + 2 * n / 4], z[i + 3 * n / 4], win1, win20, win21);
		}
	}

	void square2_31()
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			GF31::square22(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], wr[n / 4 + j]);
		}
	}

	void square2_1()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			Zp1::square22(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], wr[n / 4 + j]);
		}
	}

	void square4_31()
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			GF31::square4(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], wr[n / 4 + j]);
		}
	}

	void square4_1()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const Zp1 w = wr[n / 4 + j], win = wr[n / 4 + n / 4 - j - 1].swap();
			Zp1::square4(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], w, win);
		}
	}

	static void garner2(const GF31 r1, const Zp1 r2, int64 & i0, int64 & i1)
	{
		const uint32 InvP1_M1 = 8421505u;	//, InvP2_M1 = 152183881u;
		const uint64 M31P1 = M31 * uint64(P1);

		GF31 r2_1 = GF31(r2.s0(), r2.s1());	// P1 < M31
		GF31 u12 = r1.sub(r2_1).muls(InvP1_M1);
		const uint64 n0 = r2.s0() + u12.s0() * uint64(P1), n1 = r2.s1() + u12.s1() * uint64(P1);
		i0 = (n0 > M31P1 / 2) ? int64(n0 - M31P1) : int64(n0);
		i1 = (n1 > M31P1 / 2) ? int64(n1 - M31P1) : int64(n1);
	}

	void carry(const bool mul)
	{
		const size_t n = _size;
		GF31 * const z31 = _vz31.data();
		Zp1 * const z1 = _vz1.data();

		const int snorm31 = _snorm31;
		const uint32 norm1 = _norm1;	// Not converted into Montgomery form such that output is converted out of Montgomery form
		const int32 m = _multiplier, base = _base;
		int64 f0 = 0, f1 = 0;
		uint64 fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const GF31 u31 = z31[k].lshift(snorm31);
			const Zp1 u1 = z1[k].muls(norm1);
			int64 l0, l1; garner2(u31, u1, l0, l1);
			if (mul) { l0 *= m; l1 *= m; }
			f0 += l0; f1 += l1;
			const uint64 uf0 = uint64((f0 < 0) ? -f0 : f0), uf1 = uint64((f1 < 0) ? -f1 : f1);
			fmax = (uf0 > fmax) ? uf0 : fmax; fmax = (uf1 > fmax) ? uf1 : fmax;
			int64 r0 = f0 / base, r1 = f1 / base;
			int32 i0 = int32(f0 - r0 * base), i1 = int32(f1 - r1 * base);
			z31[k].set_int(i0, i1); z1[k].set_int(i0, i1);
			f0 = r0; f1 = r1;
		}

		if (fmax > _fmax) _fmax = fmax;

		while ((f0 != 0) || (f1 != 0))
		{
			int64 t = f0; f0 = -f1; f1 = t;	// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				const GF31 u = z31[k];
				int32 i0, i1; u.get_int(i0, i1);
				f0 += i0; f1 += i1;
				int64 r0 = f0 / base, r1 = f1 / base;
				i0 = int32(f0 - r0 * base); i1 = int32(f1 - r1 * base);
				z31[k].set_int(i0, i1); z1[k].set_int(i0, i1);
				f0 = r0; f1 = r1;
				if ((r0 == 0) && (r1 == 0)) break;
			}
		}
	}

public:
	Transform(const uint32_t b, const int n, const uint32_t a)
		: _size(size_t(1) << (n - 1)), _vz31(_size), _vz1(_size), _vwr31(3 * _size / 2), _vwr1(3 * _size / 2),
		_base(int32(b)), _multiplier(int32(a)), _snorm31(31 - n + 2), _norm1(Zp1::norm(uint32(_size)))
	{
		const size_t size = _size;

		GF31 * const wr31 = _vwr31.data();
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const GF31 r_s = GF31::primroot_n(16 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const GF31 w2 = r_s.pow(bitrev(j, 4 * s) + 1), w1 = w2.sqr(), w3 = w1.mul(w2);
				wr31[s + j] = w1; wr31[size / 2 + s + j] = w2; wr31[size + s + j] = w3;
			}
		}

		Zp1 * const wr1 = _vwr1.data();
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp1 r_s = Zp1::primroot_n(16 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const Zp1 w20 = r_s.pow_mul_sqr(bitrev(j, 4 * s) + 1), w1 = w20.sqr(), w21 = w20.muli();
				wr1[s + j] = w1; wr1[size / 2 + s + j] = w20; wr1[size + s + j] = w21;
			}
		}

		GF31 * const z31 = _vz31.data();
		z31[0] = GF31(1, 0); for (size_t k = 1; k < size; ++k) z31[k] = GF31(0, 0);

		Zp1 * const z1 = _vz1.data();
		z1[0] = Zp1(1, 0); for (size_t k = 1; k < size; ++k) z1[k] = Zp1(0, 0);

		_fmax = 0;
	}

public:
	void squareMul(const bool mul)
	{
		const size_t n = _size;

		size_t m = n / 4, s = 1;
		for (; m > 1; m /= 4, s *= 4) forward4_31(m, s);
		if (m == 1) square4_31(); else square2_31();
		for (m = (m == 1) ? 4 : 2, s /= 4; m <= n / 4; m *= 4, s /= 4) backward4_31(m, s);

		forward8_1_0();
		m = n / 16, s = 4;
		for (; m > 1; m /= 4, s *= 4) forward4_1(m, s);
		if (m == 1) square4_1(); else square2_1();
		for (m = (m == 1) ? 4 : 2, s /= 4; m <= n / 16; m *= 4, s /= 4) backward4_1(m, s);
		backward8_1_0();

		carry(mul);
	}

public:
	bool isOne(uint64_t & res64) const
	{
		const size_t n = _size;
		const GF31 * const z31 = _vz31.data();

		std::vector<int64_t> vzi(2 * n);
		int64_t * const zi = vzi.data();

		for (size_t i = 0; i < n; ++i)
		{
			int32_t i0, i1; z31[i].get_int(i0, i1);
			zi[i + 0 * n] = i0; zi[i + 1 * n] = i1;
		}

		const int32_t base = _base;
		int64_t f;
		do
		{
			f = 0;
			for (size_t i = 0; i < 2 * n; ++i)
			{
				f += zi[i];
				int32_t r = int32_t(f % base);
				if (r < 0) r += base;
				zi[i] = r;
				f -= r;
				f /= base;
			}
			zi[0] -= f;        // a[n] = -a[0]
		} while (f != 0);

		bool bOne = (zi[0] == 1);
		if (bOne) for (size_t i = 1; i < 2 * n; ++i) bOne &= (zi[i] == 0);

		uint64_t r64 = 0, b = 1;
		for (size_t i = 0; i < 2 * n; ++i)
		{
			r64 += uint32_t(zi[i]) * b;
			b *= uint32_t(base);
		}
		res64 = r64;

		return bOne;
	}

	__uint128_t fmax() const { return _fmax; }
};

static void check(const uint32_t b, const int n, const uint64_t expectedResidue = 0)
{
	const clock_t start = clock();

	mpz_t exponent; mpz_init(exponent);
	mpz_ui_pow_ui(exponent, b, static_cast<unsigned long int>(1) << n);

	const bool ispoweroftwo = ((b != 0) && ((b & (~b + 1)) == b));
	Transform transform(b, n, ispoweroftwo ? 3 : 2);

	for (int i = static_cast<int>(mpz_sizeinbase(exponent, 2) - 1); i >= 0; --i)
	{
		transform.squareMul(mpz_tstbit(exponent, mp_bitcnt_t(i)) != 0);
	}

	mpz_clear(exponent);

	uint64_t residue;
	const bool isPrime = transform.isOne(residue);

	const float duration = (float)(clock() - start) / CLOCKS_PER_SEC;

	std::cout << b << "^{2^" << n << "} + 1 is ";
	if (isPrime) std::cout << "a probable prime";
	else
	{
		std::cout << "composite (RES = ";
		std::cout.width(16); std::cout.fill('0');
		std::cout << std::internal << std::hex << residue << std::dec << ")";
	}
	std::cout.precision(3);
	std::cout << ", max = " << transform.fmax() * std::pow(2.0, -64) << ", " << duration << " sec.";

	if ((isPrime && (expectedResidue != 0)) || (!isPrime && (expectedResidue != residue))) std::cout << " ERROR!";

	std::cout << std::endl;
}

int main()
{
	// GF31 x Z/p1Z
	check(267384822, 5, 0xa2d24ee1329ebe66ull);
	check(189069622, 6, 0x9ba4c067a5e92b00ull);
	check(133692410, 7, 0x9453e9ce94674a1eull);
	check(94534810, 8, 0x97ee9426b7ea3ee3ull);
	check(66846204, 9, 0x973ab6949fb8caecull);
	check(47267404, 10, 0x1ae50255387901b8ull);
	check(33423102, 11, 0xf0ab429cbbc7ed7dull);
	check(23633702, 12, 0x8054ae714890f6d0ull);
	check(267384578, 5);
	check(189069558, 6);
	check(133691878, 7);
	check(94533980, 8);
	check(66843022, 9);
	check(47264498, 10);
	check(33422122, 11);
	check(23626024, 12);
	check(16679528, 13);
	check(11804938, 14);
	check(8345790, 15);
	check(5905206, 16);
	check(4085818, 17);
	return EXIT_SUCCESS;
}
