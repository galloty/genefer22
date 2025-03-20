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

#define	P1		(127 * (uint32(1) << 24) + 1)
#define	Q1		2164260865u		// p * q = 1 (mod 2^32)
#define ONE1	33554430u		// Montgomery form of 1 is 2^32 (mod p)
#define	RSQ1	402124772u		// (2^32)^2 mod p
#define	H1		167772150u		// Montgomery form of the primitive root 5
#define	IM1		200536044u		// Montgomery form of Montgomery form of I = 5^{(p - 1)/4} to convert input into Montgomery form
#define	SQRTI1	856006302u		// Montgomery form of 5^{(p - 1)/8}
#define	ISQRTI1	1626730317u		// i * sqrt(i)

#define	P2		(63 * (uint32(1) << 25) + 1)
#define	Q2		2181038081u
#define ONE2	67108862u
#define	RSQ2	2111798781u
#define	H2		335544310u		// Montgomery form of the primitive root 5
#define	IM2		1036950657u
#define	SQRTI2	338852760u
#define	ISQRTI2	1090446030u

template<uint32 P, uint32 Q, uint32 ONE, uint32 RSQ, uint32 H, uint32 IM, uint32 SQRTI, uint32 ISQRTI>
class Zp
{
private:
	uint32 _n;

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= P) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? P : 0); }

	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

	// 16 mul + 16 mul_hi
	static void _forward4(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w1, const Zp & w20, const Zp & w21)
	{
		const Zp u0 = z0, u2 = z2.mul(w1), u1 = z1, u3 = z3.mul(w1);
		const Zp v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3).mul(w20), v3 = u1.sub(u3).mul(w21);
		z0 = v0.add(v1); z1 = v0.sub(v1); z2 = v2.add(v3); z3 = v2.sub(v3);
	}

	static void _backward4(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & win1, const Zp & win20, const Zp & win21)
	{
		const Zp u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const Zp v0 = u0.add(u1), v1 = u1.sub(u0).mul(win20), v2 = u2.add(u3), v3 = u3.sub(u2).mul(win21);
		z0 = v0.add(v2); z2 = v2.sub(v0).mul(win1); z1 = v1.add(v3); z3 = v3.sub(v1).mul(win1);
	}

	static void _forward4_0(Zp & z0, Zp & z1, Zp & z2, Zp & z3)
	{
		const Zp u0 = z0.toMonty(), u2 = z2.mul(Zp(IM)), u1 = z1.toMonty(), u3 = z3.mul(Zp(IM));
		const Zp v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3).mul(Zp(SQRTI)), v3 = u1.sub(u3).mul(Zp(ISQRTI));
		z0 = v0.add(v1); z1 = v0.sub(v1); z2 = v2.add(v3); z3 = v2.sub(v3);
	}

	static void _square22(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w)
	{
		const Zp u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		z0 = u0.sqr().add(u1.sqr().mul(w)); z1 = u0.mul(u1.add(u1));
		z2 = u2.sqr().sub(u3.sqr().mul(w)); z3 = u2.mul(u3.add(u3));
	}

	static void _square4(Zp & z0, Zp & z1, Zp & z2, Zp & z3, const Zp & w, const Zp & win)
	{
		const Zp u0 = z0, u2 = z2.mul(w), u1 = z1, u3 = z3.mul(w);
		const Zp v0 = u0.add(u2), v2 = u0.sub(u2), v1 = u1.add(u3), v3 = u1.sub(u3);
		const Zp s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
		const Zp s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
		z0 = s0.add(s2); z2 = s2.sub(s0).mul(win); z1 = s1.add(s3); z3 = s3.sub(s1).mul(win);
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
		if (e == 0) return Zp(ONE);
		Zp r = Zp(ONE), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		r = r.mul(y);
		return r;
	}

	static const Zp primroot_n(const uint32 n) { return Zp(H).pow((P - 1) / n); }
	static Zp norm(const uint32 n) { return Zp(P - (P - 1) / n); }

	static void forward4(Zp * const z, const Zp * const wr, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Zp w1 = wr[s + j], w20 = wr[2 * (s + j) + 0], w21 = wr[2 * (s + j) + 1];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				_forward4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], w1, w20, w21);
			}
		}
	}

	static void backward4(Zp * const z, const Zp * const wr, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const size_t ji = s - j - 1;
			const Zp win1 = wr[s + ji], win20 = wr[2 * (s + ji) + 1], win21 = wr[2 * (s + ji) + 0];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				Zp::_backward4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], win1, win20, win21);
			}
		}
	}

	static void forward4_0(Zp * const z, const size_t n_4)
	{
		for (size_t i = 0; i < n_4; ++i)
		{
			const size_t k = i;
			Zp::_forward4_0(z[k + 0 * n_4], z[k + 1 * n_4], z[k + 2 * n_4], z[k + 3 * n_4]);
		}
	}

	static void square2(Zp * const z, const Zp * const wr, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			Zp::_square22(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], wr[n_4 + j]);
		}
	}

	static void square4(Zp * const z, const Zp * const wr, const size_t n_4)
	{
		for (size_t j = 0; j < n_4; ++j)
		{
			Zp::_square4(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], wr[n_4 + j], wr[n_4 + n_4 - j - 1]);
		}
	}
};

typedef Zp<P1, Q1, ONE1, RSQ1, H1, IM1, SQRTI1, ISQRTI1> Zp1;
typedef Zp<P2, Q2, ONE2, RSQ2, H2, IM2, SQRTI2, ISQRTI2> Zp2;

class Transform
{
private:
	const size_t _size;
	std::vector<Zp1> _vz1;
	std::vector<Zp2> _vz2;
	std::vector<Zp1> _vwr1;
	std::vector<Zp2> _vwr2;
	const int32 _base;
	const int32 _multiplier;
	const Zp1 _norm1;
	const Zp2 _norm2;
	uint64 _fmax;

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
		const uint64 P1P2 = P1 * (uint64)(P2);

		Zp1 u12 = r1.sub(Zp1(r2.get())).mul(Zp1(mfInvP2_P1));	// P2 < P1
		const uint64 n = r2.get() + u12.get() * uint64(P2);
		return (n > P1P2 / 2) ? int64(n - P1P2) : int64(n);
	}

	void carry(const bool mul)
	{
		const size_t n = _size;
		Zp1 * const z1 = _vz1.data();
		Zp2 * const z2 = _vz2.data();

		const Zp1 norm1 = _norm1; const Zp2 norm2 = _norm2;	// Not converted into Montgomery form such that output is converted out of Montgomery form
		const int32 m = _multiplier, base = _base;
		int64 f = 0;
		uint64 fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const Zp1 u1 = z1[k].mul(norm1);
			const Zp2 u2 = z2[k].mul(norm2);
			int64 l = garner2(u1, u2);
			if (mul) l *= m;
			f += l;
			const uint64 uf = uint64((f < 0) ? -f : f);
			fmax = (uf > fmax) ? uf : fmax;
			const int64 r = f / base;
			const int32 i = int32(f - r * base);
			f = r;
			z1[k].set_int(i); z2[k].set_int(i);
		}

		if (fmax > _fmax) _fmax = fmax;

		while (f != 0)
		{
			f = -f;		// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				f += z1[k].get_int();
				const int64 r = f / base;
				const int32 i = int32(f - r * base);
				z1[k].set_int(i); z2[k].set_int(i);
				f = r;
				if (r == 0) break;
			}
		}
	}

public:
	Transform(const uint32_t b, const int n, const uint32_t a)
		: _size(size_t(1) << n), _vz1(_size), _vz2(_size), _vwr1(_size / 1), _vwr2(_size / 1),
		_base(int32(b)), _multiplier(int32(a)), _norm1(Zp1::norm(uint32(_size / 2))), _norm2(Zp2::norm(uint32(_size / 2)))
	{
		const size_t size = _size;

		Zp1 * const wr1 = _vwr1.data();
		for (size_t s = 1; s < size / 1; s *= 2)
		{
			const Zp1 r_s = Zp1::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr1[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		Zp2 * const wr2 = _vwr2.data();
		for (size_t s = 1; s < size / 1; s *= 2)
		{
			const Zp2 r_s = Zp2::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr2[s + j] = r_s.pow(bitrev(j, 2 * s) + 1);
			}
		}

		Zp1 * const z1 = _vz1.data();
		z1[0] = Zp1(1); for (size_t k = 1; k < size; ++k) z1[k] = Zp1(0);

		Zp2 * const z2 = _vz2.data();
		z2[0] = Zp2(1); for (size_t k = 1; k < size; ++k) z2[k] = Zp2(0);

		_fmax = 0;
	}

public:
	void squareMul(const bool mul)
	{
		const size_t n_4 = _size / 4;
		Zp1 * const z1 = _vz1.data();
		Zp2 * const z2 = _vz2.data();
		const Zp1 * const wr1 = _vwr1.data();
		const Zp2 * const wr2 = _vwr2.data();

		Zp1::forward4_0(z1, n_4);
		size_t m = n_4 / 4, s = 4;
		for (; m > 1; m /= 4, s *= 4) Zp1::forward4(z1, wr1, m, s);
		if (m == 1) Zp1::square4(z1, wr1, n_4); else Zp1::square2(z1, wr1, n_4);
		for (m = (m == 1) ? 4 : 2, s /= 4; m <= n_4; m *= 4, s /= 4) Zp1::backward4(z1, wr1, m, s);

		Zp2::forward4_0(z2, n_4);
		m = n_4 / 4, s = 4;
		for (; m > 1; m /= 4, s *= 4) Zp2::forward4(z2, wr2, m, s);
		if (m == 1) Zp2::square4(z2, wr2, n_4); else Zp2::square2(z2, wr2, n_4);
		for (m = (m == 1) ? 4 : 2, s /= 4; m <= n_4; m *= 4, s /= 4) Zp2::backward4(z2, wr2, m, s);

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
	std::cout << ", max = " << transform.fmax() * std::pow(2.0, -64) << ", " << duration << " sec.";

	if ((isPrime && (expectedResidue != 0)) || (!isPrime && (expectedResidue != residue))) std::cout << " ERROR!";

	std::cout << std::endl;
}

int main()
{
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
