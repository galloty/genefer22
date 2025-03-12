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

// GF((2^31 - 1)^2)
class GF31
{
private:
	static const uint32 _p = (uint32(1) << 31) - 1;
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
	GF31 lshift(const int s) const { return GF31(_lshift(_s[0], s), _lshift(_s[1], s)); }
	GF31 muls(const uint32 s) const { return GF31(_mul(_s[0], s), _mul(_s[1], s)); }

	GF31 mul(const GF31 & rhs) const { return GF31(_sub(_mul(_s[0], rhs._s[0]), _mul(_s[1], rhs._s[1])), _add(_mul(_s[1], rhs._s[0]), _mul(_s[0], rhs._s[1]))); }
	GF31 mulconj(const GF31 & rhs) const { return GF31(_add(_mul(_s[0], rhs._s[0]), _mul(_s[1], rhs._s[1])), _sub(_mul(_s[1], rhs._s[0]), _mul(_s[0], rhs._s[1]))); }
	GF31 sqr() const { const uint32 t = _mul(_s[0], _s[1]); return GF31(_sub(_mul(_s[0], _s[0]), _mul(_s[1], _s[1])), _add(t, t)); }

	GF31 addi(const GF31 & rhs) const { return GF31(_sub(_s[0], rhs._s[1]), _add(_s[1], rhs._s[0])); }
	GF31 subi(const GF31 & rhs) const { return GF31(_add(_s[0], rhs._s[1]), _sub(_s[1], rhs._s[0])); }

	static void forward2(GF31 & z0, GF31 & z1, const GF31 & w)
	{
		const GF31 u0 = z0, u1 = z1.mul(w);
		z0 = u0.add(u1); z1 = u0.sub(u1);
	}

	static void backward2(GF31 & z0, GF31 & z1, const GF31 & w)
	{
		const GF31 u0 = z0, u1 = z1;
		z0 = u0.add(u1); z1 = u0.sub(u1).mulconj(w);
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

class Zp1
{
private:
	static const uint32 _p = 127 * (uint32(1) << 24) + 1;
	static const uint32 _q = 2164260865u;	// p * q = 1 (mod 2^32)
	static const uint32 _r2 = 402124772u;	// (2^32)^2 mod p
	static const uint32 _h = 167772150u;	// Montgomery form of the primitive root 5
	static const uint32 _i = 66976762u; 	// Montgomery form of 16711679 = 5^{(p + 1)/4}
	static const uint32 _im = 200536044u;	// Montgomery form of Montgomery form to convert input into Montgomery form
	uint32 _s[2];

	static uint32 _neg(const uint32 a) { return (a == 0) ? 0 : (_p - a); }

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= _p) ? _p : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? _p : 0); }

	static uint32 _mulMonty(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * _q) * uint64(_p)) >> 32);
		return _sub(hi, mp);
	}

	static int32 _get_int(const uint32 a) { return (a >= _p / 2) ? int32(a - _p) : int32(a); }
	static uint32 _set_int(const int32 a) { return (a < 0) ? (uint32(a) + _p) : uint32(a); }

	Zp1 pow(const size_t e) const
	{
		static const uint32 one = -_p * 2u;	// Montgomery form of 1 is 2^32 (mod p)
		if (e == 0) return Zp1(one, 0);
		Zp1 r = Zp1(one, 0), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r._s[0] = _mulMonty(r._s[0], y._s[0]); y._s[0] = _mulMonty(y._s[0], y._s[0]); }
		r._s[0] = _mulMonty(r._s[0], y._s[0]);
		return r;
	}

public:
	Zp1() {}
	explicit Zp1(const uint32 n0, const uint32 n1) { _s[0] = n0; _s[1] = n1; }

	uint32 s0() const { return _s[0]; }
	uint32 s1() const { return _s[1]; }

	Zp1 & set_int(const int32 i0, const int32 i1) { _s[0] = _set_int(i0); _s[1] = _set_int(i1); return *this; }

	Zp1 add(const Zp1 & rhs) const { return Zp1(_add(_s[0], rhs._s[0]), _add(_s[1], rhs._s[1])); }
	Zp1 sub(const Zp1 & rhs) const { return Zp1(_sub(_s[0], rhs._s[0]), _sub(_s[1], rhs._s[1])); }
	Zp1 muls(const uint32 s) const { return Zp1(_mulMonty(_s[0], s), _mulMonty(_s[1], s)); }

	Zp1 mul(const Zp1 & rhs) const { return Zp1(_mulMonty(_s[0], rhs._s[0]), _mulMonty(_s[1], rhs._s[1])); }
	Zp1 sqr() const { return mul(*this); }

	Zp1 neg() const { return Zp1(_neg(_s[0]), _neg(_s[1])); }
	Zp1 swap() const { return Zp1(_s[1], _s[0]); }

	// Conversion into / out of Montgomery form
	// Zp1 toMonty() const { return Zp1(_mulMonty(_s[0], _r2), _mulMonty(_s[1], _r2)); }
	// Zp1 fromMonty() const { return Zp1(_mulMonty(_s[0], 1), _mulMonty(_s[1], 1)); }

	Zp1 & forward2()
	{
		const uint32 u0 = _mulMonty(_s[0], _r2), u1 = _mulMonty(_s[1], _im);
		_s[0] = _add(u0, u1); _s[1] = _sub(u0, u1);
		return *this;
	}

	Zp1 & backward2()
	{
		const uint32 u0 = _s[0], u1 = _s[1];
		_s[0] = _add(u0, u1); _s[1] = _mulMonty(_sub(u1, u0), _i);
	 	return *this;
	}

	static void forward2(Zp1 & z0, Zp1 & z1, const Zp1 & w)
	{
		const Zp1 u0 = z0, u1 = z1.mul(w);
		z0 = u0.add(u1); z1 = u0.sub(u1);
	}

	static void backward2(Zp1 & z0, Zp1 & z1, const Zp1 & wi)
	{
		const Zp1 u0 = z0, u1 = z1;
		z0 = u0.add(u1); z1 = u0.sub(u1).mul(wi);
	}

	Zp1 pow_mul_sqr(const size_t e) const { Zp1 r = pow(e); r._s[1] = _mulMonty(r._s[0], _s[1]); return r; }

	static const Zp1 primroot_n(const uint32 n) { Zp1 r = Zp1(_h, 0).pow((_p - 1) / n); r._s[1] = _mulMonty(r._s[0], r._s[0]); return r; }
	static uint32 norm(const uint32 n) { return _p - (_p - 1) / n; }
};

class Transform
{
private:
	const size_t _size;
	std::vector<GF31> _vz31;
	std::vector<Zp1> _vz1;
	std::vector<GF31> _vwr31;
	std::vector<Zp1> _vwr1, _vwri1;
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

	void forward2_1_0()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();

		for (size_t k = 0; k < n; ++k) z[k].forward2();
	}

	void backward2_1_0()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();

		for (size_t k = 0; k < n; ++k) z[k].backward2();
	}

	void forward2_31(const size_t m, const size_t s)
	{
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF31 w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				GF31::forward2(z[k + 0 * m], z[k + 1 * m], w);
			}
		}
	}

	void forward2_1(const size_t m, const size_t s)
	{
		Zp1 * const z = _vz1.data();
		const Zp1 * const wr = _vwr1.data();

		for (size_t j = 0; j < s; ++j)
		{
			const Zp1 w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				Zp1::forward2(z[k + 0 * m], z[k + 1 * m], w);
			}
		}
	}

	void backward2_31(const size_t m, const size_t s)
	{
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF31 w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				GF31::backward2(z[k + 0 * m], z[k + 1 * m], w);
			}
		}
	}

	void backward2_1(const size_t m, const size_t s)
	{
		Zp1 * const z = _vz1.data();
		const Zp1 * const wri = _vwri1.data();

		for (size_t j = 0; j < s; ++j)
		{
			const Zp1 wi = wri[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				Zp1::backward2(z[k + 0 * m], z[k + 1 * m], wi);
			}
		}
	}

	void forward4_31(const size_t m, const size_t s)
	{
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF31 w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF31 u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w2), u2 = z[k + 2 * m].mul(w1), u3 = z[k + 3 * m].mul(w3);
				const GF31 v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
				z[k + 0 * m] = v0.add(v1); z[k + 1 * m] = v0.sub(v1);
				z[k + 2 * m] = v2.addi(v3); z[k + 3 * m] = v2.subi(v3);
			}
		}
	}

	void backward4_31(const size_t m, const size_t s)
	{
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF31 w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF31 u0 = z[k + 0 * m], u1 = z[k + 1 * m], u2 = z[k + 2 * m], u3 = z[k + 3 * m];
				const GF31 v0 = u0.add(u1), v1 = u0.sub(u1), v2 = u2.add(u3), v3 = u3.sub(u2);
				z[k + 0 * m] = v0.add(v2); z[k + 2 * m] = v0.sub(v2).mulconj(w1);
				z[k + 1 * m] = v1.addi(v3).mulconj(w2); z[k + 3 * m] = v1.subi(v3).mulconj(w3);
			}
		}
	}

	void square_31()
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();

		for (size_t k = 0; k < n; ++k) z[k] = z[k].sqr();
	}

	void square_1()
	{
		const size_t n = _size;
		Zp1 * const z = _vz1.data();

		for (size_t k = 0; k < n; ++k) z[k] = z[k].sqr();
	}

	void square2_31()
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GF31 w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GF31 u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2], u3 = z[k + 3];
			z[k + 0] = u0.sqr().add(u1.sqr().mul(w)); z[k + 1] = u0.mul(u1.add(u1));
			z[k + 2] = u2.sqr().sub(u3.sqr().mul(w)); z[k + 3] = u2.mul(u3.add(u3));
		}
	}

	void square4_31()
	{
		const size_t n = _size;
		GF31 * const z = _vz31.data();
		const GF31 * const wr = _vwr31.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GF31 w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GF31 u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2].mul(w), u3 = z[k + 3].mul(w);
			const GF31 v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
			const GF31 s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
			const GF31 s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
			z[k + 0] = s0.add(s2); z[k + 2] = s0.sub(s2).mulconj(w);
			z[k + 1] = s1.add(s3); z[k + 3] = s1.sub(s3).mulconj(w);
		}
	}

	static void garner2(const GF31 r1, const Zp1 r2, int64 & i0, int64 & i1)
	{
		const uint32 M31 = (uint32(1) << 31) - 1, P1 = 127 * (uint32(1) << 24) + 1, InvP1_M1 = 8421505u;
		const uint64 M31P2 = M31 * uint64(P1);

		GF31 r2_1 = GF31(r2.s0(), r2.s1());
		GF31 u12 = r1.sub(r2_1).muls(InvP1_M1);
		const uint64 n0 = r2.s0() + u12.s0() * uint64(P1), n1 = r2.s1() + u12.s1() * uint64(P1);
		i0 = (n0 > M31P2 / 2) ? int64(n0 - M31P2) : int64(n0);
		i1 = (n1 > M31P2 / 2) ? int64(n1 - M31P2) : int64(n1);
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
		: _size(size_t(1) << (n - 1)), _vz31(_size), _vz1(_size), _vwr31(_size / 1), _vwr1(_size / 1), _vwri1(_size / 1),
		_base(int32(b)), _multiplier(int32(a)), _snorm31(31 - n + 1), _norm1(Zp1::norm(uint32(2 * _size)))
	{
		const size_t size = _size;

		GF31 * const wr31 = _vwr31.data();
		for (size_t s = 1; s < size / 1; s *= 2)
		{
			const GF31 r_s = GF31::primroot_n(2 * 4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr31[s + j] = r_s.pow(bitrev(j, 4 * s) + 1);
			}
		}

		Zp1 * const wr1 = _vwr1.data();
		Zp1 * const wri1 = _vwri1.data();
		for (size_t s = 1; s < size / 1; s *= 2)
		{
			const Zp1 r_s = Zp1::primroot_n(2 * 4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const Zp1 w = r_s.pow_mul_sqr(bitrev(j, 4 * s) + 1);
				wr1[s + j] = w; wri1[s + s - j - 1] = w.neg().swap();
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

		// size_t m = n / 4, s = 1;
		// for (; m > 1; m /= 4, s *= 4) forward4_31(m, s);
		// if (m == 1) square4_31(); else square2_31();
		// for (m = (m == 1) ? 4 : 2, s /= 4; m <= n / 4; m *= 4, s /= 4) backward4_31(m, s);
		// carry(mul);

		size_t m = n / 2, s = 1;
		for (; m >= 1; m /= 2, s *= 2) forward2_31(m, s);
		square_31(); 
		for (m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2) backward2_31(m, s);

		m = n / 2, s = 1;
		forward2_1_0();
		for (; m >= 1; m /= 2, s *= 2) forward2_1(m, s);
		square_1(); 
		for (m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2) backward2_1(m, s);
		backward2_1_0();

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
	check(41688706, 17);

	return EXIT_SUCCESS;
}
