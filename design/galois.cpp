/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <vector>
#include <ctime>
#include <iostream>

#include <gmp.h>

// Modulo 2^61 - 1
class Zp
{
private:
	static const uint64_t _p = (uint64_t(1) << 61) - 1;
	uint64_t _n;	// 0 <= n <= p

	static uint64_t _add(const uint64_t a, const uint64_t b)
	{
		const uint64_t t = a + b;
		const uint64_t c = (uint32_t(t >> 32) > uint32_t(_p >> 32)) ? _p : 0;	// t > p ?
		return t - c;
	}

	static uint64_t _sub(const uint64_t a, const uint64_t b)
	{
		const uint64_t t = a - b;
		const uint64_t c = (int32_t(t >> 32) < 0) ? _p : 0;	// t < 0 ?
		return t + c;
	}

	static uint64_t _mul(const uint64_t a, const uint64_t b)
	{
		const __uint128_t t = a * __uint128_t(b);
		const uint64_t lo = uint64_t(t) & _p, hi = uint64_t(t >> 61);
		return _add(hi, lo);
	}

public:
	Zp() {}
	explicit Zp(const uint64_t n) : _n(n) {}

	int64_t get_int() const { return (_n >= _p / 2) ? int64_t(_n - _p) : int64_t(_n); }	// if n = p then return 0
	Zp & set_int(const int64_t i) { _n = (i < 0) ? uint64_t(i) + _p : uint64_t(i); return *this; }

	// Zp neg() const { return Zp((_n == 0) ? 0 : _p - _n); }

	Zp add(const Zp & rhs) const { return Zp(_add(_n, rhs._n)); }
	Zp sub(const Zp & rhs) const { return Zp(_sub(_n, rhs._n)); }
	Zp mul(const Zp & rhs) const { return Zp(_mul(_n, rhs._n)); }

	static Zp inv(const uint64_t n) { return Zp((_p + 1) / n); }
};

// GF((2^61 - 1)^2)
class GF
{
private:
	Zp _a, _b;
	// a primitive root of order 2^62 which is a root of (0, 1).
	static const uint64_t _h_order = uint64_t(1) << 62;
	static const uint64_t _h_a = 2147483648ull, _h_b = 1272521237944691271ull;

public:
	GF() {}
	explicit GF(const Zp & a, const Zp & b) : _a(a), _b(b) {}

	const Zp & a() const { return _a; }
	const Zp & b() const { return _b; }

	// GF conj() const { return GF(_a, -_b); }

	GF add(const GF & rhs) const { return GF(_a.add(rhs._a), _b.add(rhs._b)); }
	GF sub(const GF & rhs) const { return GF(_a.sub(rhs._a), _b.sub(rhs._b)); }
	GF muls(const Zp & rhs) const { return GF(_a.mul(rhs), _b.mul(rhs)); }

	GF sqr() const { const Zp t = _a.mul(_b); return GF(_a.mul(_a).sub(_b.mul(_b)), t.add(t)); }
	GF mul(const GF & rhs) const { return GF(_a.mul(rhs._a).sub(_b.mul(rhs._b)), _b.mul(rhs._a).add(_a.mul(rhs._b))); }
	GF mulconj(const GF & rhs) const { return GF(_a.mul(rhs._a).add(_b.mul(rhs._b)), _b.mul(rhs._a).sub(_a.mul(rhs._b))); }

	// GF muli() const { return GF(-_b, _a); }
	GF addi(const GF & rhs) const { return GF(_a.sub(rhs._b), _b.add(rhs._a)); }
	GF subi(const GF & rhs) const { return GF(_a.add(rhs._b), _b.sub(rhs._a)); }

	GF pow(const uint64_t e) const
	{
		if (e == 0) return GF(Zp(1), Zp(0));
		GF r = GF(Zp(1), Zp(0)), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF primroot_n(const uint64_t n) { return GF(Zp(_h_a), Zp(_h_b)).pow(_h_order / n); }
};

class Transform
{
private:
	std::vector<GF> _vz;
	std::vector<GF> _vwr;
	const int32_t _base;
	const int32_t _multiplier;
	const Zp _norm;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	// void forward2(const size_t m, const size_t s)
	// {
	// 	GF * const z = _vz.data();
	// 	const GF * const wr = _vwr.data();

	// 	for (size_t j = 0; j < s; ++j)
	// 	{
	// 		const GF w = wr[s + j];

	// 		for (size_t i = 0; i < m; ++i)
	// 		{
	// 			const size_t k = 2 * m * j + i;
	// 			const GF u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w);
	// 			z[k + 0 * m] = u0.add(u1); z[k + 1 * m] = u0.sub(u1);
	// 		}
	// 	}
	// }

	// void backward2(const size_t m, const size_t s)
	// {
	// 	GF * const z = _vz.data();
	// 	const GF * const wr = _vwr.data();

	// 	for (size_t j = 0; j < s; ++j)
	// 	{
	// 		const GF w = wr[s + j];

	// 		for (size_t i = 0; i < m; ++i)
	// 		{
	// 			const size_t k = 2 * m * j + i;
	// 			const GF u0 = z[k + 0 * m], u1 = z[k + 1 * m];
	// 			z[k + 0 * m] = u0.add(u1); z[k + 1 * m] = u0.sub(u1).mulconj(w);
	// 		}
	// 	}
	// }

	void forward4(const size_t m, const size_t s)
	{
		GF * const z = _vz.data();
		const GF * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w2), u2 = z[k + 2 * m].mul(w1), u3 = z[k + 3 * m].mul(w3);
				const GF v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
				z[k + 0 * m] = v0.add(v1); z[k + 1 * m] = v0.sub(v1);
				z[k + 2 * m] = v2.addi(v3); z[k + 3 * m] = v2.subi(v3);
			}
		}
	}

	void backward4(const size_t m, const size_t s)
	{
		GF * const z = _vz.data();
		const GF * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF u0 = z[k + 0 * m], u1 = z[k + 1 * m], u2 = z[k + 2 * m], u3 = z[k + 3 * m];
				const GF v0 = u0.add(u1), v1 = u0.sub(u1), v2 = u2.add(u3), v3 = u3.sub(u2);
				z[k + 0 * m] = v0.add(v2); z[k + 2 * m] = v0.sub(v2).mulconj(w1);
				z[k + 1 * m] = v1.addi(v3).mulconj(w2); z[k + 3 * m] = v1.subi(v3).mulconj(w3);
			}
		}
	}

	void square2()
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();
		const GF * const wr = _vwr.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GF w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GF u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2], u3 = z[k + 3];
			z[k + 0] = u0.sqr().add(u1.sqr().mul(w)); z[k + 1] = u0.mul(u1.add(u1));
			z[k + 2] = u2.sqr().sub(u3.sqr().mul(w)); z[k + 3] = u2.mul(u3.add(u3));
		}
	}

	void square4()
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();
		const GF * const wr = _vwr.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GF w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GF u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2].mul(w), u3 = z[k + 3].mul(w);
			const GF v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
			const GF s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
			const GF s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
			z[k + 0] = s0.add(s2); z[k + 2] = s0.sub(s2).mulconj(w);
			z[k + 1] = s1.add(s3); z[k + 3] = s1.sub(s3).mulconj(w);
		}
	}

	void carry(const bool mul)
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();

		const Zp norm = _norm;
		const int32_t m = _multiplier, base = _base;
		int64_t fa = 0, fb = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const GF u = z[k].muls(norm);
			int64_t ia = u.a().get_int(), ib = u.b().get_int();
			if (mul) { ia *= m; ib *= m; }
			fa += ia; fb += ib;
			int64_t la = fa / base, lb = fb / base;
			z[k] = GF(Zp().set_int(fa - la * base), Zp().set_int(fb - lb * base));
			fa = la; fb = lb;
		}

		while ((fa != 0) || (fb != 0))
		{
			int64_t t = fa; fa = -fb; fb = t;	// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				const GF u = z[k];
				fa += u.a().get_int(); fb += u.b().get_int();
				int64_t la = fa / base, lb = fb / base;
				z[k] = GF(Zp().set_int(fa - la * base), Zp().set_int(fb - lb * base));
				fa = la; fb = lb;
				if ((la == 0) && (lb == 0)) break;
			}
		}
	}

public:
	Transform(const size_t size, const uint32_t b, const uint32_t a)
		: _vz(size / 2, GF(Zp(0), Zp(0))), _vwr(size / 4), _base(int32_t(b)), _multiplier(int32_t(a)), _norm(Zp::inv(size / 4))
	{
		const size_t n = _vz.size();
		GF * const wr = _vwr.data();

		for (size_t s = 1; s < n / 2; s *= 2)
		{
			const GF r_s = GF::primroot_n(2 * 4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr[s + j] = r_s.pow(bitrev(j, 4 * s) + 1);
			}
		}

		_vz[0] = GF(Zp(1), Zp(0));
	}

public:
	void squareMul(const bool mul)
	{
		const size_t n = _vz.size();

		size_t m = n / 4, s = 1;
		for (; m > 1; m /= 4, s *= 4) forward4(m, s);
		if (m == 1) square4(); else square2();
		for (m = (m == 1) ? 4 : 2, s /= 4; m <= n / 4; m *= 4, s /= 4) backward4(m, s);
		carry(mul);
	}

public:
	bool isOne(uint64_t & res64) const
	{
		const size_t n = _vz.size();
		const GF * const z = _vz.data();

		std::vector<int64_t> vzi(2 * n);
		int64_t * const zi = vzi.data();

		for (size_t i = 0; i < n; ++i)
		{
			zi[i] = z[i].a().get_int();
			zi[i + n] = z[i].b().get_int();
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
};

static void check(const uint32_t b, const int n, const uint64_t expectedResidue = 0)
{
	const clock_t start = clock();

	mpz_t exponent; mpz_init(exponent);
	mpz_ui_pow_ui(exponent, b, static_cast<unsigned long int>(1) << n);

	const bool ispoweroftwo = ((b != 0) && ((b & (~b + 1)) == b));
	Transform transform(1 << n, b, ispoweroftwo ? 3 : 2);

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
	std::cout << " (" << duration << " sec.)";

	if ((isPrime && (expectedResidue != 0)) || (!isPrime && (expectedResidue != residue))) std::cout << " ERROR!";

	std::cout << std::endl;
}

int main()
{
	check(100014, 5);
	check(100234, 6);
	check(10234, 6, 0x831a0082ced5b6e1ull);
	check(100032, 7);
	check(10032, 7, 0x3b2808f35afe61bdull);
	check(5684328, 8);
	check(584328, 8, 0xe6205f75c7438c2eull);
	check(4619000, 9);
	check(419000, 9, 0xfb2b0688cefd4fabull);
	check(3752220, 10);
	check(352220, 10, 0x1e830b3c54de5ef3ull);
	check(3066672, 11);
	check(366672, 11, 0xef2f0357a06a13e3ull);
	check(2485064, 12);
	check(285064, 12, 0x4e349f8254b4e364ull);
	check(2030234, 13);
	check(1651902, 14);
	check(1277444, 15);
	check(857678, 16);
	check(572186, 17);
	check(676754, 18);
	check(475856, 19);

	return EXIT_SUCCESS;
}
