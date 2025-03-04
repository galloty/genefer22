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
	uint64_t _n;

	static uint64_t add(const uint64_t a, const uint64_t b) { return a + b - ((a >= _p - b) ? _p : 0); }
	static uint64_t sub(const uint64_t a, const uint64_t b) { return a - b + ((a < b) ? _p : 0); }
	static uint64_t mul(const uint64_t a, const uint64_t b)
	{
		const __uint128_t t = a * __uint128_t(b);
		const uint64_t lo = uint64_t(t) & _p, hi = uint64_t(t >> 61);
		return add(hi, lo);
	}

public:
	Zp() {}
	explicit Zp(const uint64_t n) : _n(n) {}

	int64_t get_int() const { return (_n >= _p / 2) ? int64_t(_n - _p) : int64_t(_n); }
	Zp & set_int(const int64_t i) { _n = (i < 0) ? _p + uint64_t(i) : uint64_t(i); return *this; }

	Zp operator-() const { return Zp((_n == 0) ? 0 : _p - _n); }

	Zp operator+(const Zp & rhs) const { return Zp(add(_n, rhs._n)); }
	Zp operator-(const Zp & rhs) const { return Zp(sub(_n, rhs._n)); }
	Zp operator*(const Zp & rhs) const { return Zp(mul(_n, rhs._n)); }

	Zp & operator*=(const Zp & rhs) { _n = mul(_n, rhs._n); return *this; }

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

	GF conj() const { return GF(_a, -_b); }

	GF operator+(const GF & rhs) const { return GF(_a + rhs._a, _b + rhs._b); }
	GF operator-(const GF & rhs) const { return GF(_a - rhs._a, _b - rhs._b); }
	GF operator*(const GF & rhs) const { return GF(_a * rhs._a - _b * rhs._b, _a * rhs._b + _b * rhs._a); }
	GF operator*(const Zp & rhs) const { return GF(_a * rhs, _b * rhs); }

	GF & operator*=(const GF & rhs) { *this = *this * rhs; return *this; }

	GF pow(const uint64_t e) const
	{
		if (e == 0) return GF(Zp(1), Zp(0));
		GF r = GF(Zp(1), Zp(0)), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r *= y; y *= y; }
		return r * y;
	}

	static const GF primroot_n(const uint64_t n) { return GF(Zp(_h_a), Zp(_h_b)).pow(_h_order / n); }
};

class Transform
{
private:
	std::vector<GF> _vz;
	std::vector<GF> _vwr;
	const int32_t _base;
	const Zp _multiplier;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	void forward2()
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();
		const GF * const wr = _vwr.data();

		for (size_t m = n / 2, s = 1; m >= 1; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const GF wr_sj = wr[s + j];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = j * 2 * m + i;
					const GF u0 = z[k + 0 * m], u1 = z[k + 1 * m] * wr_sj;
					z[k + 0 * m] = u0 + u1; z[k + 1 * m] = u0 - u1;
				}
			}
		}
	}

	void backward2()
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();
		const GF * const wr = _vwr.data();

		for (size_t m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const GF wr_sj_inv = wr[s + j].conj();

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = j * 2 * m + i;
					const GF u0 = z[k + 0 * m], u1 = z[k + 1 * m];
					z[k + 0 * m] = u0 + u1; z[k + 1 * m] = (u0 - u1) * wr_sj_inv;
				}
			}
		}
	}

	void square()
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();

		for (size_t k = 0; k < n; ++k) z[k] *= z[k];
	}

	void carry(const bool mul)
	{
		const size_t n = _vz.size();
		GF * const z = _vz.data();

		const Zp r = Zp::inv(n), rg = mul ? r * _multiplier : r;

		const int32_t base = _base;
		int64_t fa = 0, fb = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const GF u = z[k] * rg;
			fa += u.a().get_int(); fb += u.b().get_int();
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
				if ((fa == 0) && (fb == 0)) break;
			}
		}
	}

public:
	Transform(const size_t size, const uint32_t b, const uint32_t a)
		: _vz(size / 2, GF(Zp(0), Zp(0))), _vwr(size / 2), _base(int32_t(b)), _multiplier(a)
	{
		const size_t n = _vz.size();
		GF * const wr = _vwr.data();

		for (size_t s = 1; s < n; s *= 2)
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
		forward2();
		square();
		backward2();
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
