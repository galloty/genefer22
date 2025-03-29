/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <vector>
#include <iostream>
#include <ctime>

#include <gmp.h>

typedef uint32_t	uint32;
typedef int32_t		int32;
typedef uint64_t	uint64;
typedef int64_t		int64;

typedef uint32		uint32_4 __attribute__ ((vector_size(16)));
typedef int32		int32_4 __attribute__ ((vector_size(16)));
typedef uint64		uint64_4 __attribute__ ((vector_size(32)));
typedef int64		int64_4 __attribute__ ((vector_size(32)));

inline uint32 mul_hi(const uint32 lhs, const uint32 rhs) { return uint32((lhs * uint64(rhs)) >> 32); }

inline uint32_4 mul_hi(const uint32_4 lhs, const uint32_4 rhs)
{
	const uint64_4 t = __builtin_convertvector(lhs, uint64_4) * __builtin_convertvector(rhs, uint64_4);
	return __builtin_convertvector(t >> 32, uint32_4);
}

#define	P1		(127 * (uint32(1) << 24) + 1)
#define	Q1		2164260865u		// p * q = 1 (mod 2^32)
#define	R1		33554430u		// 2^32 mod p
#define	RSQ1	402124772u		// (2^32)^2 mod p
#define	H1		100663290u		// Montgomery form of the primitive root 3
#define	IM1		1930170389u		// MF of MF of I = 3^{(p - 1)/4} to convert input into MF
#define	SQRTI1	1626730317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1	856006302u		// MF of i * sqrt(i)

#define	P2		(63 * (uint32(1) << 25) + 1)
#define	Q2		2181038081u
#define	R2		67108862u
#define	RSQ2	2111798781u
#define	H2		335544310u		// MF of the primitive root 5
#define	IM2		1036950657u
#define	SQRTI2	338852760u
#define	ISQRTI2	1090446030u

#define	P3		(15 * (uint32(1) << 27) + 1)
#define	Q3		2281701377u
#define	R3		268435454u
#define	RSQ3	1172168163u
#define	H3		268435390u		// MF of the primitive root 31
#define	IM3		734725699u
#define	SQRTI3	1032137103u
#define	ISQRTI3	1964242958u

#define FWD2(z0, z1, w) { const ZpT t = z1.mul(w); z1 = z0.sub(t); z0 = z0.add(t); }
#define BCK2(z0, z1, win) { const ZpT t = z1.sub(z0); z0 = z0.add(z1), z1 = t.mul(win); }
#define SQR2(z0, z1, w) { const ZpT t = z1.sqr().mul(w); z1 = z0.add(z0).mul(z1); z0 = z0.sqr().add(t); }
#define SQR2N(z0, z1, w) { const ZpT t = z1.sqr().mul(w); z1 = z0.add(z0).mul(z1); z0 = z0.sqr().sub(t); }

#define FWD2v(z0, z1, w) { const Zp4T t = z1.mul(w); z1 = z0.sub(t); z0 = z0.add(t); }
#define BCK2v(z0, z1, win) { const Zp4T t = z1.sub(z0); z0 = z0.add(z1), z1 = t.mul(win); }

template<uint32 P, uint32 Q, uint32 R, uint32 H>
class ZpT
{
private:
	uint32 _n;

public:
	ZpT() {}
	explicit ZpT(const uint32 n) : _n(n) {}

	uint32 get() const { return _n; }

	static void load(const size_t n, ZpT * const zl, const ZpT * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
	static void loadr(const size_t n, ZpT * const zl, const ZpT * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[n - l - 1] = z[l * s]; }

	ZpT add(const ZpT & rhs) const { const uint32 t = _n + rhs._n; return ZpT(t - ((t >= P) ? P : 0)); }
	ZpT sub(const ZpT & rhs) const { const uint32 t = _n - rhs._n; return ZpT(t + ((int32(t) < 0) ? P : 0)); }

	ZpT mul(const ZpT & rhs) const
	{
		const uint64 t = _n * uint64(rhs._n);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = mul_hi(lo * Q, P);
		return ZpT(hi).sub(ZpT(mp));
	}

	ZpT sqr() const { return mul(*this); }

	ZpT pow(const size_t e) const
	{
		if (e == 0) return ZpT(R);	// MF of one is R
		ZpT r = ZpT(R), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		r = r.mul(y);
		return r;
	}

	static const ZpT primroot_n(const uint32 n) { return ZpT(H).pow((P - 1) / n); }
	static ZpT norm(const uint32 n) { return ZpT(P - (P - 1) / n); }
};

template<uint32 P, uint32 Q, uint32 R, uint32 H, uint32 RSQ, uint32 IM, uint32 SQRTI, uint32 ISQRTI>
class Zp4T
{
	using Zp = ZpT<P, Q, R, H>;

private:
	uint32_4 _n;

	static void _forward4(Zp4T z[4], const Zp w1, const Zp w2[2])
	{
		FWD2v(z[0], z[2], w1); FWD2v(z[1], z[3], w1);
		FWD2v(z[0], z[1], w2[0]); FWD2v(z[2], z[3], w2[1]);
	}

	static void _forward4_0(Zp4T z[4])
	{
		z[0] = z[0].toMonty(); z[1] = z[1].toMonty();
		FWD2v(z[0], z[2], Zp(IM)); FWD2v(z[1], z[3], Zp(IM));
		FWD2v(z[0], z[1], Zp(SQRTI)); FWD2v(z[2], z[3], Zp(ISQRTI));
	}

	static void _backward4(Zp4T z[4], const Zp win1, const Zp win2[2])
	{
		BCK2v(z[0], z[1], win2[0]); BCK2v(z[2], z[3], win2[1]);
		BCK2v(z[0], z[2], win1); BCK2v(z[1], z[3], win1);
	}

	static void _square4x2(Zp4T z[2], const Zp w2[2], const Zp win2[2])
	{
		Zp zs[2][4]; z[0].gets(zs[0]); z[1].gets(zs[1]);
		for (size_t i = 0; i < 2; ++i)
		{
			FWD2(zs[i][0], zs[i][2], w2[i]); FWD2(zs[i][1], zs[i][3], w2[i]);
			SQR2(zs[i][0], zs[i][1], w2[i]); SQR2N(zs[i][2], zs[i][3], w2[i]);
			BCK2(zs[i][0], zs[i][2], win2[i]); BCK2(zs[i][1], zs[i][3], win2[i]);
		}
		z[0].sets(zs[0]); z[1].sets(zs[1]);
	}

	static void _square8(Zp4T z[2], const Zp w1, const Zp  win1, const Zp w2[2], const Zp win2[2])
	{
		FWD2v(z[0], z[1], w1);
		_square4x2(z, w2, win2);
		BCK2v(z[0], z[1], win1);
	}

public:
	Zp4T() {}
	explicit Zp4T(const uint32_4 n) : _n(n) {}

	uint32 get(const size_t index) const { return _n[index]; }
	void gets(Zp z[4]) const { for (size_t j = 0; j < 4; ++j) z[j] = Zp(_n[j]); }
	void sets(const Zp z[4]) { for (size_t j = 0; j < 4; ++j) _n[j] = z[j].get(); }

	static void load(const size_t n, Zp4T * const zl, const Zp4T * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
	static void store(const size_t n, Zp4T * const z, const size_t s, const Zp4T * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

	int32 get_int(const size_t index) const { const uint32 n = _n[index]; return (n >= P / 2) ? int32(n - P) : int32(n); }
	Zp4T & set_int(const size_t index, const int32 i) { _n[index] = (i < 0) ? (uint32(i) + P) : uint32(i); return *this; }

	Zp4T add(const Zp4T & rhs) const { const uint32_4 t = _n + rhs._n; return Zp4T(t - ((t >= P) & P)); }
	Zp4T sub(const Zp4T & rhs) const { const uint32_4 t = _n - rhs._n; return Zp4T(t + ((int32_4(t) < 0) & P)); }

	Zp4T mul(const Zp4T & rhs) const
	{
		const uint32_4 lo = _n * rhs._n, hi = mul_hi(_n, rhs._n);
		const uint32_4 mp = mul_hi(lo * Q, uint32_4{P, P, P, P});
		return Zp4T(hi).sub(Zp4T(mp));
	}

	Zp4T mul(const Zp & rhs) const
	{
		const uint32 t = rhs.get();
		return mul(Zp4T(uint32_4{t, t, t, t}));
	}

	// Conversion into / out of Montgomery form
	Zp4T toMonty() const { return mul(Zp(RSQ)); }
	// Zp4T fromMonty() const { return mul(Zp(1)); }

	static void forward4(Zp4T * const z, const Zp * const wr, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Zp w1 = wr[1 * (s + j)];
			Zp w2[2]; Zp::load(2, w2, &wr[2 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				Zp4T zl[4]; Zp4T::load(4, zl, &z[k], m);
				Zp4T::_forward4(zl, w1, w2);
				Zp4T::store(4, &z[k], m, zl);
			}
		}
	}

	static void backward4(Zp4T * const z, const Zp * const wr, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const size_t ji = s - j - 1;
			const Zp win1 = wr[1 * (s + ji)];
			Zp win2[2]; Zp::loadr(2, win2, &wr[2 * (s + ji)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				Zp4T zl[4]; Zp4T::load(4, zl, &z[k], m);
				Zp4T::_backward4(zl, win1, win2);
				Zp4T::store(4, &z[k], m, zl);
			}
		}
	}

	static void forward4_0(Zp4T * const z, const size_t n_16)
	{
		for (size_t i = 0; i < n_16; ++i)
		{
			const size_t k = i;
			Zp4T zl[4]; Zp4T::load(4, zl, &z[k], n_16);
			Zp4T::_forward4_0(zl);
			Zp4T::store(4, &z[k], n_16, zl);
		}
	}

	static void square4x4(Zp4T * const z, const Zp * const wr, const size_t n_16)
	{
		for (size_t j = 0; j < n_16; ++j)
		{
			const size_t ji =  n_16 - j - 1;
			Zp w2[4]; Zp::load(4, w2, &wr[4 * (n_16 + j)], 1);
			Zp win2[4]; Zp::loadr(4, win2, &wr[4 * (n_16 + ji)], 1);

			Zp4T zl[4]; Zp4T::load(4, zl, &z[4 * j], 1);
			Zp4T::_square4x2(&zl[0], &w2[0], &win2[0]);
			Zp4T::_square4x2(&zl[2], &w2[2], &win2[2]);
			Zp4T::store(4, &z[4 * j], 1, zl);
		}
	}

	static void square8x2(Zp4T * const z, const Zp * const wr, const size_t n_16)
	{
		for (size_t j = 0; j < n_16; ++j)
		{
			const size_t ji =  n_16 - j - 1;
			Zp w1[2]; Zp::load(2, w1, &wr[2 * (n_16 + j)], 1);
			Zp win1[2]; Zp::loadr(2, win1, &wr[2 * (n_16 + ji)], 1);
			Zp w2[4]; Zp::load(4, w2, &wr[4 * (n_16 + j)], 1);
			Zp win2[4]; Zp::loadr(4, win2, &wr[4 * (n_16 + ji)], 1);

			Zp4T zl[4]; Zp4T::load(4, zl, &z[4 * j], 1);
			Zp4T::_square8(&zl[0], w1[0], win1[0], &w2[0], &win2[0]);
			Zp4T::_square8(&zl[2], w1[1], win1[1], &w2[2], &win2[2]);
			Zp4T::store(4, &z[4 * j], 1, zl);
		}
	}
};

typedef ZpT<P1, Q1, R1, H1> Zp1;
typedef ZpT<P2, Q2, R2, H2> Zp2;
typedef ZpT<P3, Q3, R3, H3> Zp3;

typedef Zp4T<P1, Q1, R1, H1, RSQ1, IM1, SQRTI1, ISQRTI1> Zp1v;
typedef Zp4T<P2, Q2, R2, H2, RSQ2, IM2, SQRTI2, ISQRTI2> Zp2v;
typedef Zp4T<P3, Q3, R3, H3, RSQ3, IM3, SQRTI3, ISQRTI3> Zp3v;

class Transform
{
private:
	const size_t _size;
	std::vector<Zp1v> _vz1;
	std::vector<Zp2v> _vz2;
	std::vector<Zp3v> _vz3;
	std::vector<Zp1> _vwr1;
	std::vector<Zp2> _vwr2;
	std::vector<Zp3> _vwr3;
	const uint32 _base;
	const int _b_s;
	const uint32 _b_inv;
	const int32 _multiplier;
	const Zp1 _norm1;
	const Zp2 _norm2;
	const Zp3 _norm3;
	__uint128_t _fmax;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	static uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 & a_p)
	{
		const uint32 d = mul_hi(uint32(a >> b_s), b_inv), r = uint32(a) - d * b;
		const bool o = (r >= b);
		a_p = d + (o ? 1 : 0);
		return r - (o ? b : 0);
	}
	
	static int32 reduce64(int64 & f, const uint32 b, const uint32 b_inv, const int b_s)
	{
		const bool s = (f < 0);
		const uint64 t = uint64(s ? -f : f);
		const uint64 t_h = t >> 29;
		const uint32 t_l = (uint32)(t) & ((1u << 29) - 1);
	
		uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32 d_l, r_l = barrett((uint64(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64 d = (uint64(d_h) << 29) | d_l;
	
		f = s ? -int64(d) : int64(d);
		return s ? -int32(r_l) : int32(r_l);
	}
	
	static int32 reduce96(__int128_t & f, const uint32 b, const uint32 b_inv, const int b_s)
	{
		const bool s = (f < 0);
		const __uint128_t t = __uint128_t(s ? -f : f);
		const uint64 t_h = uint64(t >> 29);
		const uint32 t_l = uint32(t) & ((1u << 29) - 1);
	
		uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32 d_l, r_l = barrett((uint64(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64 d = (uint64(d_h) << 29) | d_l;
	
		f = s ? -__int128_t(d) : __int128_t(d);
		return s ? -int32(r_l) : int32(r_l);
	}

	static int64 garner2(const Zp1 r1, const Zp2 r2)
	{
		const uint32 mfInvP2_P1 = 2130706177u;	// Montgomery form of 1 / P2 (mod P1)
		const uint64 P1P2 = P1 * uint64(P2);

		const Zp1 u = r1.sub(Zp1(r2.get())).mul(Zp1(mfInvP2_P1));	// P2 < P1
		const uint64 n = r2.get() + u.get() * uint64(P2);
		return (n > P1P2 / 2) ? int64(n - P1P2) : int64(n);
	}

	static __int128_t garner3(const Zp1 r1, const Zp2 r2, const Zp3 r3)
	{
		// Montgomery form of 1 / Pi (mod Pj)
		const uint32 mfInvP3_P1 = 608773230u, mfInvP2_P1 = 2130706177u, mfInvP3_P2 = 1409286102u;
		const uint64 P2P3 = P2 * uint64(P3);
		const __uint128_t P1P2P3 = P1 * __uint128_t(P2P3);

		const Zp1 u13 = r1.sub(Zp1(r3.get())).mul(Zp1(mfInvP3_P1));	// P3 < P1
		const Zp2 u23 = r2.sub(Zp2(r3.get())).mul(Zp2(mfInvP3_P2));	// P3 < P2
		const Zp1 u123 = u13.sub(Zp1(u23.get())).mul(Zp1(mfInvP2_P1));	// P3 < P1
		const __uint128_t n = __uint128_t(P2P3) * u123.get() + (u23.get() * uint64(P3) + r3.get());
		return (n > P1P2P3 / 2) ? __int128_t(n - P1P2P3) : __int128_t(n);
	}

	void carry(const bool mul)
	{
		const size_t n = _size;
		Zp1v * const z1 = _vz1.data();
		Zp2v * const z2 = _vz2.data();
		Zp3v * const z3 = _vz3.data();

		// Not converted into Montgomery form such that output is converted out of MF
		const Zp1 norm1 = _norm1; const Zp2 norm2 = _norm2; const Zp3 norm3 = _norm3;
		const int32 m = _multiplier; const uint32 b = _base; const int b_s = _b_s; const uint32 b_inv = _b_inv;
		__int128_t f96 = 0;
		__uint128_t fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const Zp1 u1 = Zp1(z1[k / 4].get(k % 4)).mul(norm1);
			const Zp2 u2 = Zp2(z2[k / 4].get(k % 4)).mul(norm2);
			const Zp3 u3 = Zp3(z3[k / 4].get(k % 4)).mul(norm3);
			__int128_t l = garner3(u1, u2, u3);
			if (mul) l *= m;
			f96 += l;
			const __uint128_t uf = __uint128_t((f96 < 0) ? -f96 : f96);
			fmax = (uf > fmax) ? uf : fmax;
			const int32 r = reduce96(f96, b, b_inv, b_s);
			z1[k / 4].set_int(k % 4, r); z2[k / 4].set_int(k % 4, r); z3[k / 4].set_int(k % 4, r);
		}

		if (fmax > _fmax) _fmax = fmax;

		int64 f64 = int64(f96);
		while (f64 != 0)
		{
			f64 = -f64;		// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				f64 += z1[k / 4].get_int(k % 4);
				const int32 r = reduce64(f64, b, b_inv, b_s);
				z1[k / 4].set_int(k % 4, r); z2[k / 4].set_int(k % 4, r); z3[k / 4].set_int(k % 4, r);
				if (f64 == 0) break;
			}
		}
	}

public:
	Transform(const uint32_t b, const int n, const uint32_t a)
		: _size(size_t(1) << n), _vz1(_size / 4), _vz2(_size / 4), _vz3(_size / 4), _vwr1(_size / 2), _vwr2(_size / 2), _vwr3(_size / 2),
		_base(b), _b_s(int(31 - __builtin_clz(b) - 1)), _b_inv(uint32((uint64(1) << (_b_s + 32)) / b)), _multiplier(int32(a)),
		_norm1(Zp1::norm(uint32(_size / 2))), _norm2(Zp2::norm(uint32(_size / 2))), _norm3(Zp3::norm(uint32(_size / 2)))
	{
		const size_t size = _size;

		Zp1 * const wr1 = _vwr1.data();
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp1 r_s = Zp1::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr1[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		Zp2 * const wr2 = _vwr2.data();
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp2 r_s = Zp2::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr2[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		Zp3* const wr3 = _vwr3.data();
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const Zp3 r_s = Zp3::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr3[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		const uint32_4 zero = uint32_4{0, 0, 0, 0}, one = uint32_4{1, 0, 0, 0};
		Zp1v * const z1 = _vz1.data();
		z1[0] = Zp1v(one); for (size_t k = 1; k < size / 4; ++k) z1[k] = Zp1v(zero);
		Zp2v * const z2 = _vz2.data();
		z2[0] = Zp2v(one); for (size_t k = 1; k < size / 4; ++k) z2[k] = Zp2v(zero);
		Zp3v * const z3 = _vz3.data();
		z3[0] = Zp3v(one); for (size_t k = 1; k < size / 4; ++k) z3[k] = Zp3v(zero);

		_fmax = 0;
	}

public:
	void squareMul(const bool mul)
	{
		const size_t n_16 = _size / 16;
		Zp1v * const z1 = _vz1.data();
		Zp2v * const z2 = _vz2.data();
		Zp3v * const z3 = _vz3.data();
		const Zp1 * const wr1 = _vwr1.data();
		const Zp2 * const wr2 = _vwr2.data();
		const Zp3 * const wr3 = _vwr3.data();

		size_t m = n_16, s = 1;
		Zp1v::forward4_0(z1, n_16);
		for (m /= 4, s *= 4; m > 0; m /= 4, s *= 4) Zp1v::forward4(z1, wr1, m, s);
		s /= 4; m = n_16 / s;
		if (m == 2) Zp1v::square8x2(z1, wr1, n_16); else Zp1v::square4x4(z1, wr1, n_16);
		for (; m <= n_16; m *= 4, s /= 4) Zp1v::backward4(z1, wr1, m, s);

		m = n_16, s = 1;
		Zp2v::forward4_0(z2, n_16);
		for (m /= 4, s *= 4; m > 0; m /= 4, s *= 4) Zp2v::forward4(z2, wr2, m, s);
		s /= 4; m = n_16 / s;
		if (m == 2) Zp2v::square8x2(z2, wr2, n_16); else Zp2v::square4x4(z2, wr2, n_16);
		for (; m <= n_16; m *= 4, s /= 4) Zp2v::backward4(z2, wr2, m, s);

		m = n_16, s = 1;
		Zp3v::forward4_0(z3, n_16);
		for (m /= 4, s *= 4; m > 0; m /= 4, s *= 4) Zp3v::forward4(z3, wr3, m, s);
		s /= 4; m = n_16 / s;
		if (m == 2) Zp3v::square8x2(z3, wr3, n_16); else Zp3v::square4x4(z3, wr3, n_16);
		for (; m <= n_16; m *= 4, s /= 4) Zp3v::backward4(z3, wr3, m, s);

		carry(mul);
	}

public:
	bool isOne(uint64_t & res64) const
	{
		const size_t n = _size;
		const Zp1v * const z1 = _vz1.data();

		std::vector<int32_t> vzi(n);
		int32_t * const zi = vzi.data();

		for (size_t i = 0; i < n; ++i) zi[i] = z1[i / 4].get_int(i % 4);

		const int32_t base = int32_t(_base);
		int64_t f;
		do
		{
			f = 0;
			for (size_t i = 0; i < n; ++i)
			{
				f += zi[i];
				int32_t r = int32_t(f % base);
				if (r < 0) r += base;
				zi[i] = r;
				f -= r;
				f /= base;
			}
			zi[0] -= int32_t(f);	// a[n] = -a[0]
		} while (f != 0);

		bool bOne = (zi[0] == 1);
		if (bOne) for (size_t i = 1; i < n; ++i) bOne &= (zi[i] == 0);

		uint64_t r64 = 0, b = 1;
		for (size_t i = 0; i < n; ++i)
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
	std::cout << ", max = " << transform.fmax() / double(__uint128_t(1) << 96) << ", " << duration << " sec.";

	if ((isPrime && (expectedResidue != 0)) || (!isPrime && (expectedResidue != residue))) std::cout << " ERROR!";

	std::cout << std::endl;
}

int main()
{
	// Z/p1Z x Z/p2Z x Z/p3Z
	check(1000000000, 5, 0x7da127b73585c207ull);
	check(1000000000, 6, 0xc37bd72acdea47e1ull);
	check(1000000000, 7, 0xcfa99a35cc4a7179ull);
	check(1000000000, 8, 0x280cd7361db156d7ull);
	check(1000000000, 9, 0x0629e1b97daab9dfull);
	check(1000000000, 10, 0x9983e7718a7d7a4dull);
	check(1000000000, 11, 0x43c6cc5326e5c77full);
	check(1000000000, 12, 0x4e43a5b93273c649ull);
	check(1000000064, 5);
	check(1000000432, 6);
	check(1000000072, 7);
	check(1000000116, 8);
	check(1000000512, 9);
	check(1000003008, 10);
	check(1000002456, 11);
	check(1000005800, 12);
	check(1000001960, 13);
	check(1000032014, 14);
	check(459986590, 15);
	check(456492690, 16);
	check(1000032472, 17);

	// Z/p1Z x Z/p2Z
	// check(265287654, 5, 0x2d5f91935581646full);
	// check(187586700, 6, 0x4ef835cde43b2a6cull);
	// check(132643826, 7, 0x7a4bb40ced568fceull);
	// check(93793350, 8, 0x3c15022022d4fa73ull);
	// check(66321912, 9, 0x356ce9635e490e4dull);
	// check(46896674, 10, 0x076885421c8780cfull);
	// check(33160956, 11, 0x9e2d51da6c7c3f54ull);
	// check(23448336, 12, 0x81cff004ace85caaull);
	// check(265287418, 5);
	// check(187586400, 6);
	// check(132643476, 7);
	// check(93792538, 8);
	// check(66321726, 9);
	// check(46896522, 10);
	// check(33141254, 11);
	// check(23445612, 12);
	// check(16558530, 13);
	// check(11709684, 14);
	// check(8285500, 15);
	// check(5645768, 16);
	// check(4085818, 17);

	return EXIT_SUCCESS;
}
