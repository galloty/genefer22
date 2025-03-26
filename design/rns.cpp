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

#define FWD2(z0, z1, w) { const Zp t = z1.mul(w); z1 = z0.sub(t); z0 = z0.add(t); }
#define BCK2(z0, z1, win) { const Zp t = z1.sub(z0); z0 = z0.add(z1), z1 = t.mul(win); }

#define SQR2(z0, z1, w) { const Zp t = z1.sqr().mul(w); z1 = z0.add(z0).mul(z1); z0 = z0.sqr().add(t); }
#define SQR2N(z0, z1, w) { const Zp t = z1.sqr().mul(w); z1 = z0.add(z0).mul(z1); z0 = z0.sqr().sub(t); }

template<uint32 P, uint32 Q, uint32 R, uint32 RSQ, uint32 H, uint32 IM, uint32 SQRTI, uint32 ISQRTI>
class Zp
{
private:
	uint32 _n;

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= P) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? P : 0); }

	// 2 mul + 2 mul_hi
	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

	static void _load(const size_t n, Zp * const zl, const Zp * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
	static void _loadr(const size_t n, Zp * const zl, const Zp * const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[n - l - 1] = z[l * s]; }
	static void _store(const size_t n, Zp * const z, const size_t s, const Zp * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }

	// static void _forward4(Zp z[4], const Zp w1, const Zp w2[2])
	// {
	// 	FWD2(z[0], z[2], w1); FWD2(z[1], z[3], w1);
	// 	FWD2(z[0], z[1], w2[0]); FWD2(z[2], z[3], w2[1]);
	// }

	// static void _forward4_0(Zp z[4])
	// {
	// 	z[0] = z[0].toMonty(); z[1] = z[1].toMonty();
	// 	FWD2(z[0], z[2], Zp(IM)); FWD2(z[1], z[3], Zp(IM));
	// 	FWD2(z[0], z[1], Zp(SQRTI)); FWD2(z[2], z[3], Zp(ISQRTI));
	// }

	// static void _backward4(Zp z[4], const Zp win1, const Zp win2[2])
	// {
	// 	BCK2(z[0], z[1], win2[0]); BCK2(z[2], z[3], win2[1]);
	// 	BCK2(z[0], z[2], win1); BCK2(z[1], z[3], win1);
	// }

	// static void _square22(Zp z[4], const Zp w)
	// {
	// 	SQR2(z[0], z[1], w); SQR2N(z[2], z[3], w);
	// }

	// static void _square4(Zp z[4], const Zp w, const Zp win)
	// {
	// 	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);
	// 	_square22(z, w);
	// 	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);
	// }

	static void _forward8(Zp z[8], const Zp w1, const Zp w2[2], const Zp w4[4])
	{
		FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
		FWD2(z[0], z[2], w2[0]); FWD2(z[1], z[3], w2[0]); FWD2(z[4], z[6], w2[1]); FWD2(z[5], z[7], w2[1]);
		FWD2(z[0], z[1], w4[0]); FWD2(z[2], z[3], w4[1]); FWD2(z[4], z[5], w4[2]); FWD2(z[6], z[7], w4[3]);
	}

	static void _backward8(Zp z[8], const Zp win1, const Zp win2[2], const Zp win4[4])
	{
		BCK2(z[0], z[1], win4[0]); BCK2(z[2], z[3], win4[1]); BCK2(z[4], z[5], win4[2]); BCK2(z[6], z[7], win4[3]);
		BCK2(z[0], z[2], win2[0]); BCK2(z[1], z[3], win2[0]); BCK2(z[4], z[6], win2[1]); BCK2(z[5], z[7], win2[1]);
		BCK2(z[0], z[4], win1); BCK2(z[2], z[6], win1); BCK2(z[1], z[5], win1); BCK2(z[3], z[7], win1);
	}

	static void _forward8_0(Zp z[4], const Zp w2[2], const Zp w4[4])
	{
		z[0] = z[0].toMonty(); z[1] = z[1].toMonty(); z[2] = z[2].toMonty(); z[3] = z[3].toMonty();
		FWD2(z[0], z[4], Zp(IM)); FWD2(z[2], z[6], Zp(IM)); FWD2(z[1], z[5], Zp(IM)); FWD2(z[3], z[7], Zp(IM));
		FWD2(z[0], z[2], w2[0]); FWD2(z[1], z[3], w2[0]); FWD2(z[4], z[6], w2[1]); FWD2(z[5], z[7], w2[1]);
		FWD2(z[0], z[1], w4[0]); FWD2(z[2], z[3], w4[1]); FWD2(z[4], z[5], w4[2]); FWD2(z[6], z[7], w4[3]);
	}

	static void _square2x4(Zp z[8], const Zp w2[2])
	{
		SQR2(z[0], z[1], w2[0]); SQR2N(z[2], z[3], w2[0]); SQR2(z[4], z[5], w2[1]); SQR2N(z[6], z[7], w2[1]);
	}

	static void _square4x2(Zp z[8], const Zp w2[2], const Zp win2[2])
	{
		FWD2(z[0], z[2], w2[0]); FWD2(z[1], z[3], w2[0]); FWD2(z[4], z[6], w2[1]); FWD2(z[5], z[7], w2[1]);
		_square2x4(z, w2);
		BCK2(z[0], z[2], win2[0]); BCK2(z[1], z[3], win2[0]); BCK2(z[4], z[6], win2[1]); BCK2(z[5], z[7], win2[1]);
	}

	static void _square8(Zp z[8], const Zp w1, const Zp  win1, const Zp w2[2], const Zp win2[2])
	{
		FWD2(z[0], z[4], w1); FWD2(z[2], z[6], w1); FWD2(z[1], z[5], w1); FWD2(z[3], z[7], w1);
		_square4x2(z, w2, win2);
		BCK2(z[0], z[4], win1); BCK2(z[2], z[6], win1); BCK2(z[1], z[5], win1); BCK2(z[3], z[7], win1);
	}

public:
	Zp() {}
	explicit Zp(const uint32 n) : _n(n) {}

	uint32 get() const { return _n; }

	int32 get_int() const { return (_n >= P / 2) ? int32(_n - P) : int32(_n); }
	Zp & set_int(const int32 i) { _n = (i < 0) ? (uint32(i) + P) : uint32(i); return *this; }

	Zp add(const Zp & rhs) const { return Zp(_add(_n, rhs._n)); }
	Zp sub(const Zp & rhs) const { return Zp(_sub(_n, rhs._n)); }
	Zp mul(const Zp & rhs) const { return Zp(_mul(_n, rhs._n)); }
	Zp sqr() const { return mul(*this); }

	// Conversion into / out of Montgomery form
	Zp toMonty() const { return Zp(_mul(_n, RSQ)); }
	// Zp fromMonty() const { return Zp(_mul(_n, 1)); }

	Zp pow(const size_t e) const
	{
		if (e == 0) return Zp(R);	// MF of one is R
		Zp r = Zp(R), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		r = r.mul(y);
		return r;
	}

	static const Zp primroot_n(const uint32 n) { return Zp(H).pow((P - 1) / n); }
	static Zp norm(const uint32 n) { return Zp(P - (P - 1) / n); }

	// static void forward4(Zp * const z, const Zp * const wr, const size_t m, const size_t s)
	// {
	// 	for (size_t j = 0; j < s; ++j)
	// 	{
	// 		const Zp w1 = wr[1 * (s + j)];
	// 		Zp w2[2]; _load(2, w2, &wr[2 * (s + j)], 1);

	// 		for (size_t i = 0; i < m; ++i)
	// 		{
	// 			const size_t k = 4 * m * j + i;
	// 			Zp zl[4]; _load(4, zl, &z[k], m);
	// 			_forward4(zl, w1, w2);
	// 			_store(4, &z[k], m, zl);
	// 		}
	// 	}
	// }

	// static void backward4(Zp * const z, const Zp * const wr, const size_t m, const size_t s)
	// {
	// 	for (size_t j = 0; j < s; ++j)
	// 	{
	// 		const size_t ji = s - j - 1;
	// 		const Zp win1 = wr[1 * (s + ji)];
	// 		Zp win2[2]; _loadr(2, win2, &wr[2 * (s + ji)], 1);

	// 		for (size_t i = 0; i < m; ++i)
	// 		{
	// 			const size_t k = 4 * m * j + i;
	// 			Zp zl[4]; _load(4, zl, &z[k], m);
	// 			Zp::_backward4(zl, win1, win2);
	// 			_store(4, &z[k], m, zl);
	// 		}
	// 	}
	// }

	// static void forward4_0(Zp * const z, const size_t n_4)
	// {
	// 	for (size_t i = 0; i < n_4; ++i)
	// 	{
	// 		const size_t k = i;
	// 		Zp zl[4]; _load(4, zl, &z[k], n_4);
	// 		Zp::_forward4_0(zl);
	// 		_store(4, &z[k], n_4, zl);
	// 	}
	// }

	// static void square22(Zp * const z, const Zp * const wr, const size_t n_4)
	// {
	// 	for (size_t j = 0; j < n_4; ++j)
	// 	{
	// 		Zp zl[4]; _load(4, zl, &z[4 * j], 1);
	// 		Zp::_square22(zl, wr[n_4 + j]);
	// 		_store(4, &z[4 * j], 1, zl);
	// 	}
	// }

	// static void square4(Zp * const z, const Zp * const wr, const size_t n_4)
	// {
	// 	for (size_t j = 0; j < n_4; ++j)
	// 	{
	// 		Zp zl[4]; _load(4, zl, &z[4 * j], 1);
	// 		Zp::_square4(zl, wr[n_4 + j], wr[n_4 + n_4 - j - 1]);
	// 		_store(4, &z[4 * j], 1, zl);
	// 	}
	// }

	static void forward8(Zp * const z, const Zp * const wr, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Zp w1 = wr[1 * (s + j)];
			Zp w2[2]; _load(2, w2, &wr[2 * (s + j)], 1);
			Zp w4[4]; _load(4, w4, &wr[4 * (s + j)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				Zp zl[8]; _load(8, zl, &z[k], m);
				_forward8(zl, w1, w2, w4);
				_store(8, &z[k], m, zl);
			}
		}
	}

	static void backward8(Zp * const z, const Zp * const wr, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const size_t ji = s - j - 1;
			const Zp win1 = wr[1 * (s + ji)];
			Zp win2[2]; _loadr(2, win2, &wr[2 * (s + ji)], 1);
			Zp win4[4]; _loadr(4, win4, &wr[4 * (s + ji)], 1);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				Zp zl[8]; _load(8, zl, &z[k], m);
				Zp::_backward8(zl, win1, win2, win4);
				_store(8, &z[k], m, zl);
			}
		}
	}

	static void forward8_0(Zp * const z, const Zp * const wr, const size_t n_8)
	{
		Zp w2[2]; _load(2, w2, &wr[2], 1);
		Zp w4[4]; _load(4, w4, &wr[4], 1);

		for (size_t i = 0; i < n_8; ++i)
		{
			const size_t k = i;
			Zp zl[8]; _load(8, zl, &z[k], n_8);
			_forward8_0(zl, w2, w4);
			_store(8, &z[k], n_8, zl);
		}
	}

	static void square2x4(Zp * const z, const Zp * const wr, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			Zp w2[2]; _load(2, w2, &wr[2 * n_8 + 2 * j], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			Zp::_square2x4(zl, w2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void square4x2(Zp * const z, const Zp * const wr, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const size_t ji = n_8 - j - 1;
			Zp w2[2]; _load(2, w2, &wr[2 * (n_8 + j)], 1);
			Zp win2[2]; _loadr(2, win2, &wr[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			Zp::_square4x2(zl, w2, win2);
			_store(8, &z[8 * j], 1, zl);
		}
	}

	static void square8(Zp * const z, const Zp * const wr, const size_t n_8)
	{
		for (size_t j = 0; j < n_8; ++j)
		{
			const size_t ji = n_8 - j - 1;
			const Zp w1 = wr[1 * (n_8 + j)];
			Zp w2[2]; _load(2, w2, &wr[2 * (n_8 + j)], 1);
			const Zp win1 = wr[1 * (n_8 + ji)];
			Zp win2[2]; _loadr(2, win2, &wr[2 * (n_8 + ji)], 1);

			Zp zl[8]; _load(8, zl, &z[8 * j], 1);
			Zp::_square8(zl, w1, win1, w2, win2);
			_store(8, &z[8 * j], 1, zl);
		}
	}
};

typedef Zp<P1, Q1, R1, RSQ1, H1, IM1, SQRTI1, ISQRTI1> Zp1;
typedef Zp<P2, Q2, R2, RSQ2, H2, IM2, SQRTI2, ISQRTI2> Zp2;
typedef Zp<P3, Q3, R3, RSQ3, H3, IM3, SQRTI3, ISQRTI3> Zp3;

class Transform
{
private:
	const size_t _size;
	std::vector<Zp1> _vz1;
	std::vector<Zp2> _vz2;
	std::vector<Zp3> _vz3;
	std::vector<Zp1> _vwr1;
	std::vector<Zp2> _vwr2;
	std::vector<Zp3> _vwr3;
	const int32 _base;
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
		Zp1 * const z1 = _vz1.data();
		Zp2 * const z2 = _vz2.data();
		Zp3 * const z3 = _vz3.data();

		// Not converted into Montgomery form such that output is converted out of MF
		const Zp1 norm1 = _norm1; const Zp2 norm2 = _norm2;	const Zp3 norm3 = _norm3;
		const int32 m = _multiplier, base = _base;
		__int128_t f = 0;
		__uint128_t fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const Zp1 u1 = z1[k].mul(norm1);
			const Zp2 u2 = z2[k].mul(norm2);
			const Zp3 u3 = z3[k].mul(norm3);
			// int64 l = garner2(u1, u2);
			__int128_t l = garner3(u1, u2, u3);
			if (mul) l *= m;
			f += l;
			const __uint128_t uf = __uint128_t((f < 0) ? -f : f);
			fmax = (uf > fmax) ? uf : fmax;
			const __int128_t r = f / base;
			const int32 i = int32(f - r * base);
			f = r;
			z1[k].set_int(i); z2[k].set_int(i); z3[k].set_int(i);
		}

		if (fmax > _fmax) _fmax = fmax;

		while (f != 0)
		{
			f = -f;		// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				f += z1[k].get_int();
				const __int128_t r = f / base;
				const int32 i = int32(f - r * base);
				z1[k].set_int(i); z2[k].set_int(i); z3[k].set_int(i);
				f = r;
				if (r == 0) break;
			}
		}
	}

public:
	Transform(const uint32_t b, const int n, const uint32_t a)
		: _size(size_t(1) << n), _vz1(_size), _vz2(_size), _vz3(_size), _vwr1(_size / 2), _vwr2(_size / 2), _vwr3(_size / 2),
		_base(int32(b)), _multiplier(int32(a)),
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

		Zp1 * const z1 = _vz1.data();
		z1[0] = Zp1(1); for (size_t k = 1; k < size; ++k) z1[k] = Zp1(0);

		Zp2 * const z2 = _vz2.data();
		z2[0] = Zp2(1); for (size_t k = 1; k < size; ++k) z2[k] = Zp2(0);

		Zp3 * const z3 = _vz3.data();
		z3[0] = Zp3(1); for (size_t k = 1; k < size; ++k) z3[k] = Zp3(0);

		_fmax = 0;
	}

public:
	void squareMul(const bool mul)
	{
		const size_t n_8 = _size / 8;	// n_4 = _size / 4
		Zp1 * const z1 = _vz1.data();
		Zp2 * const z2 = _vz2.data();
		Zp3 * const z3 = _vz3.data();
		const Zp1 * const wr1 = _vwr1.data();
		const Zp2 * const wr2 = _vwr2.data();
		const Zp3 * const wr3 = _vwr3.data();

		// Zp1::forward4_0(z1, n_4);
		// size_t m = n_4 / 4, s = 4;
		// for (; m > 1; m /= 4, s *= 4) Zp1::forward4(z1, wr1, m, s);
		// if (m == 1) Zp1::square4(z1, wr1, n_4); else Zp1::square22(z1, wr1, n_4);
		// for (m = (m == 1) ? 4 : 2, s /= 4; m <= n_4 / 4; m *= 4, s /= 4) Zp1::backward4(z1, wr1, m, s);
		// Zp1::backward4(z1, wr1, n_4, 1);

		// Zp2::forward4_0(z2, n_4);
		// m = n_4 / 4, s = 4;
		// for (; m > 1; m /= 4, s *= 4) Zp2::forward4(z2, wr2, m, s);
		// if (m == 1) Zp2::square4(z2, wr2, n_4); else Zp2::square22(z2, wr2, n_4);
		// for (m = (m == 1) ? 4 : 2, s /= 4; m <= n_4; m *= 4, s /= 4) Zp2::backward4(z2, wr2, m, s);

		// Zp3::forward4_0(z3, n_4);
		// m = n_4 / 4, s = 4;
		// for (; m > 1; m /= 4, s *= 4) Zp3::forward4(z3, wr3, m, s);
		// if (m == 1) Zp3::square4(z3, wr3, n_4); else Zp3::square22(z3, wr3, n_4);
		// for (m = (m == 1) ? 4 : 2, s /= 4; m <= n_4; m *= 4, s /= 4) Zp3::backward4(z3, wr3, m, s);

		Zp1::forward8_0(z1, wr1, n_8);
		size_t m = n_8 / 8, s = 8;
		for (; m > 1; m /= 8, s *= 8) Zp1::forward8(z1, wr1, m, s);
		if (s == n_8) Zp1::square8(z1, wr1, n_8);
		else if (s == 2 * n_8) Zp1::square4x2(z1, wr1, n_8);
		else if (s == 4 * n_8) Zp1::square2x4(z1, wr1, n_8);
		for (s /= 8, m = n_8 / s; m <= n_8; m *= 8, s /= 8) Zp1::backward8(z1, wr1, m, s);

		Zp2::forward8_0(z2, wr2, n_8);
		m = n_8 / 8, s = 8;
		for (; m > 1; m /= 8, s *= 8) Zp2::forward8(z2, wr2, m, s);
		if (s == n_8) Zp2::square8(z2, wr2, n_8);
		else if (s == 2 * n_8) Zp2::square4x2(z2, wr2, n_8);
		else if (s == 4 * n_8) Zp2::square2x4(z2, wr2, n_8);
		for (s /= 8, m = n_8 / s; m <= n_8; m *= 8, s /= 8) Zp2::backward8(z2, wr2, m, s);

		Zp3::forward8_0(z3, wr3, n_8);
		m = n_8 / 8, s = 8;
		for (; m > 1; m /= 8, s *= 8) Zp3::forward8(z3, wr3, m, s);
		if (s == n_8) Zp3::square8(z3, wr3, n_8);
		else if (s == 2 * n_8) Zp3::square4x2(z3, wr3, n_8);
		else if (s == 4 * n_8) Zp3::square2x4(z3, wr3, n_8);
		for (s /= 8, m = n_8 / s; m <= n_8; m *= 8, s /= 8) Zp3::backward8(z3, wr3, m, s);

		carry(mul);
	}

public:
	bool isOne(uint64_t & res64) const
	{
		const size_t n = _size;
		const Zp1 * const z1 = _vz1.data();

		std::vector<int64_t> vzi(n);
		int64_t * const zi = vzi.data();

		for (size_t i = 0; i < n; ++i) zi[i] = z1[i].get_int();

		const int32_t base = _base;
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
			zi[0] -= f;		// a[n] = -a[0]
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
	return EXIT_SUCCESS;

	// Z/p1Z x Z/p2Z
	check(265287654, 5, 0x2d5f91935581646full);
	check(187586700, 6, 0x4ef835cde43b2a6cull);
	check(132643826, 7, 0x7a4bb40ced568fceull);
	check(93793350, 8, 0x3c15022022d4fa73ull);
	check(66321912, 9, 0x356ce9635e490e4dull);
	check(46896674, 10, 0x076885421c8780cfull);
	check(33160956, 11, 0x9e2d51da6c7c3f54ull);
	check(23448336, 12, 0x81cff004ace85caaull);
	check(265287418, 5);
	check(187586400, 6);
	check(132643476, 7);
	check(93792538, 8);
	check(66321726, 9);
	check(46896522, 10);
	check(33141254, 11);
	check(23445612, 12);
	check(16558530, 13);
	check(11709684, 14);
	check(8285500, 15);
	check(5645768, 16);
	check(4085818, 17);
	return EXIT_SUCCESS;
}
