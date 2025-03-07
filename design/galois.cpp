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
class Z61
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
		const uint64_t lo = uint64_t(t), hi = uint64_t(t >> 64);
		const uint64_t lo61 = lo & _p, hi61 = (lo >> 61) | (hi << 3);
		return _add(lo61, hi61);
	}

public:
	Z61() {}
	explicit Z61(const uint64_t n) : _n(n) {}

	int64_t get_int() const { return (_n >= _p / 2) ? int64_t(_n - _p) : int64_t(_n); }	// if n = p then return 0
	Z61 & set_int(const int64_t i) { _n = (i < 0) ? uint64_t(i) + _p : uint64_t(i); return *this; }

	// Z61 neg() const { return Z61((_n == 0) ? 0 : _p - _n); }

	Z61 add(const Z61 & rhs) const { return Z61(_add(_n, rhs._n)); }
	Z61 sub(const Z61 & rhs) const { return Z61(_sub(_n, rhs._n)); }
	Z61 mul(const Z61 & rhs) const { return Z61(_mul(_n, rhs._n)); }

	static Z61 inv(const uint32_t n) { return Z61((_p + 1) / n); }
};

// GF((2^61 - 1)^2)
class GF61
{
private:
	Z61 _s0, _s1;
	// a primitive root of order 2^62 which is a root of (0, 1).
	static const uint64_t _h_order = uint64_t(1) << 62;
	static const uint64_t _h_0 = 264036120304204ull, _h_1 = 4677669021635377ull;
public:
	GF61() {}
	explicit GF61(const Z61 & s0, const Z61 & s1) : _s0(s0), _s1(s1) {}

	const Z61 & s0() const { return _s0; }
	const Z61 & s1() const { return _s1; }

	GF61 & set_int(const int64_t i0, const int64_t i1) { _s0.set_int(i0); _s1.set_int(i1); return *this; }

	// GF61 conj() const { return GF61(_s0, -_s1); }

	GF61 add(const GF61 & rhs) const { return GF61(_s0.add(rhs._s0), _s1.add(rhs._s1)); }
	GF61 sub(const GF61 & rhs) const { return GF61(_s0.sub(rhs._s0), _s1.sub(rhs._s1)); }
	GF61 muls(const Z61 & rhs) const { return GF61(_s0.mul(rhs), _s1.mul(rhs)); }

	GF61 sqr() const { const Z61 t = _s0.mul(_s1); return GF61(_s0.mul(_s0).sub(_s1.mul(_s1)), t.add(t)); }
	GF61 mul(const GF61 & rhs) const { return GF61(_s0.mul(rhs._s0).sub(_s1.mul(rhs._s1)), _s1.mul(rhs._s0).add(_s0.mul(rhs._s1))); }
	GF61 mulconj(const GF61 & rhs) const { return GF61(_s0.mul(rhs._s0).add(_s1.mul(rhs._s1)), _s1.mul(rhs._s0).sub(_s0.mul(rhs._s1))); }

	// GF61 muli() const { return GF61(-_s1, _s0); }
	GF61 addi(const GF61 & rhs) const { return GF61(_s0.sub(rhs._s1), _s1.add(rhs._s0)); }
	GF61 subi(const GF61 & rhs) const { return GF61(_s0.add(rhs._s1), _s1.sub(rhs._s0)); }

	GF61 pow(const uint64_t e) const
	{
		if (e == 0) return GF61(Z61(1), Z61(0));
		GF61 r = GF61(Z61(1), Z61(0)), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF61 primroot_n(const uint32_t n) { return GF61(Z61(_h_0), Z61(_h_1)).pow(_h_order / n); }
};

// Modulo 2^31 - 1
class Z31
{
private:
	static const uint32_t _p = (uint32_t(1) << 31) - 1;
	uint32_t _n;	// 0 <= n < p

	static uint32_t _add(const uint32_t a, const uint32_t b)
	{
		const uint32_t t = a + b;
		return t - ((t >= _p) ? _p : 0);
	}

	static uint32_t _sub(const uint32_t a, const uint32_t b)
	{
		const uint32_t t = a - b;
		return t + ((int32_t(t) < 0) ? _p : 0);
	}

	static uint32_t _mul(const uint32_t a, const uint32_t b)
	{
		const uint64_t t = a * uint64_t(b);
		const uint32_t lo = uint32_t(t) & _p, hi = uint32_t(t >> 31);
		return _add(hi, lo);
	}

public:
	Z31() {}
	explicit Z31(const uint32_t n) : _n(n) {}

	int32_t get_int() const { return (_n >= _p / 2) ? int32_t(_n - _p) : int32_t(_n); }
	Z31 & set_int(const int32_t i) { _n = (i < 0) ? uint32_t(i) + _p : uint32_t(i); return *this; }

	// Z31 neg() const { return Z31((_n == 0) ? 0 : _p - _n); }

	Z31 add(const Z31 & rhs) const { return Z31(_add(_n, rhs._n)); }
	Z31 sub(const Z31 & rhs) const { return Z31(_sub(_n, rhs._n)); }
	Z31 mul(const Z31 & rhs) const { return Z31(_mul(_n, rhs._n)); }

	static Z31 inv(const uint32_t n) { return Z31((_p + 1) / n); }
};

// GF((2^31 - 1)^2)
class GF31
{
private:
	Z31 _s0, _s1;
	// a primitive root of order 2^31 which is a root of (0, 1).
	static const uint32_t _h_order = uint32_t(1) << 31;
	static const uint32_t _h_0 = 105066u, _h_1 = 333718u;

public:
	GF31() {}
	explicit GF31(const Z31 & s0, const Z31 & s1) : _s0(s0), _s1(s1) {}

	const Z31 & s0() const { return _s0; }
	const Z31 & s1() const { return _s1; }

	GF31 & set_int(const int32_t i0, const int32_t i1) { _s0.set_int(i0); _s1.set_int(i1); return *this; }

	// GF31 conj() const { return GF31(_s0, -_s1); }

	GF31 add(const GF31 & rhs) const { return GF31(_s0.add(rhs._s0), _s1.add(rhs._s1)); }
	GF31 sub(const GF31 & rhs) const { return GF31(_s0.sub(rhs._s0), _s1.sub(rhs._s1)); }
	GF31 muls(const Z31 & rhs) const { return GF31(_s0.mul(rhs), _s1.mul(rhs)); }

	GF31 sqr() const { const Z31 t = _s0.mul(_s1); return GF31(_s0.mul(_s0).sub(_s1.mul(_s1)), t.add(t)); }
	GF31 mul(const GF31 & rhs) const { return GF31(_s0.mul(rhs._s0).sub(_s1.mul(rhs._s1)), _s1.mul(rhs._s0).add(_s0.mul(rhs._s1))); }
	GF31 mulconj(const GF31 & rhs) const { return GF31(_s0.mul(rhs._s0).add(_s1.mul(rhs._s1)), _s1.mul(rhs._s0).sub(_s0.mul(rhs._s1))); }

	// GF31 muli() const { return GF31(-_s1, _s0); }
	GF31 addi(const GF31 & rhs) const { return GF31(_s0.sub(rhs._s1), _s1.add(rhs._s0)); }
	GF31 subi(const GF31 & rhs) const { return GF31(_s0.add(rhs._s1), _s1.sub(rhs._s0)); }

	GF31 pow(const uint32_t e) const
	{
		if (e == 0) return GF31(Z31(1), Z31(0));
		GF31 r = GF31(Z31(1), Z31(0)), y = *this;
		for (uint32_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF31 primroot_n(const uint32_t n) { return GF31(Z31(_h_0), Z31(_h_1)).pow(_h_order / n); }
};

template<class Field, class GField>
class Transform
{
private:
	std::vector<GField> _vz;
	std::vector<GField> _vwr;
	const int32_t _base;
	const int32_t _multiplier;
	const Field _norm;
	__int64 _fmax;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	void forward2(const size_t m, const size_t s)
	{
		GField * const z = _vz.data();
		const GField * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GField w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const GField u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w);
				z[k + 0 * m] = u0.add(u1); z[k + 1 * m] = u0.sub(u1);
			}
		}
	}

	void backward2(const size_t m, const size_t s)
	{
		GField * const z = _vz.data();
		const GField * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GField w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const GField u0 = z[k + 0 * m], u1 = z[k + 1 * m];
				z[k + 0 * m] = u0.add(u1); z[k + 1 * m] = u0.sub(u1).mulconj(w);
			}
		}
	}

	void forward4(const size_t m, const size_t s)
	{
		GField * const z = _vz.data();
		const GField * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GField w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GField u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w2), u2 = z[k + 2 * m].mul(w1), u3 = z[k + 3 * m].mul(w3);
				const GField v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
				z[k + 0 * m] = v0.add(v1); z[k + 1 * m] = v0.sub(v1);
				z[k + 2 * m] = v2.addi(v3); z[k + 3 * m] = v2.subi(v3);
			}
		}
	}

	void backward4(const size_t m, const size_t s)
	{
		GField * const z = _vz.data();
		const GField * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GField w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GField u0 = z[k + 0 * m], u1 = z[k + 1 * m], u2 = z[k + 2 * m], u3 = z[k + 3 * m];
				const GField v0 = u0.add(u1), v1 = u0.sub(u1), v2 = u2.add(u3), v3 = u3.sub(u2);
				z[k + 0 * m] = v0.add(v2); z[k + 2 * m] = v0.sub(v2).mulconj(w1);
				z[k + 1 * m] = v1.addi(v3).mulconj(w2); z[k + 3 * m] = v1.subi(v3).mulconj(w3);
			}
		}
	}

	void square2()
	{
		const size_t n = _vz.size();
		GField * const z = _vz.data();
		const GField * const wr = _vwr.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GField w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GField u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2], u3 = z[k + 3];
			z[k + 0] = u0.sqr().add(u1.sqr().mul(w)); z[k + 1] = u0.mul(u1.add(u1));
			z[k + 2] = u2.sqr().sub(u3.sqr().mul(w)); z[k + 3] = u2.mul(u3.add(u3));
		}
	}

	void square4()
	{
		const size_t n = _vz.size();
		GField * const z = _vz.data();
		const GField * const wr = _vwr.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GField w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GField u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2].mul(w), u3 = z[k + 3].mul(w);
			const GField v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
			const GField s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
			const GField s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
			z[k + 0] = s0.add(s2); z[k + 2] = s0.sub(s2).mulconj(w);
			z[k + 1] = s1.add(s3); z[k + 3] = s1.sub(s3).mulconj(w);
		}
	}

	void carry(const bool mul)
	{
		const size_t n = _vz.size();
		GField * const z = _vz.data();

		const Field norm = _norm;
		const int32_t m = _multiplier, base = _base;
		int64_t f0 = 0, f1 = 0, fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const GField u = z[k].muls(norm);
			int64_t i0 = u.s0().get_int(), i1 = u.s1().get_int();
			if (mul) { i0 *= m; i1 *= m; }
			f0 += i0; f1 += i1;
			fmax = std::max(fmax, std::max(std::abs(f0), std::abs(f1)));
			int64_t l0 = f0 / base, l1 = f1 / base;
			z[k].set_int(f0 - l0 * base, f1 - l1 * base);
			f0 = l0; f1 = l1;
		}

		_fmax = std::max(_fmax, fmax);

		while ((f0 != 0) || (f1 != 0))
		{
			int64_t t = f0; f0 = -f1; f1 = t;	// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				const GField u = z[k];
				f0 += u.s0().get_int(); f1 += u.s1().get_int();
				int64_t l0 = f0 / base, l1 = f1 / base;
				z[k].set_int(f0 - l0 * base, f1 - l1 * base);
				f0 = l0; f1 = l1;
				if ((l0 == 0) && (l1 == 0)) break;
			}
		}
	}

public:
	Transform(const size_t size, const uint32_t b, const uint32_t a)
		: _vz(size / 2, GField(Field(0), Field(0))), _vwr(size / 4), _base(int32_t(b)), _multiplier(int32_t(a)), _norm(Field::inv(uint32_t(size / 4)))
	{
		const size_t n = _vz.size();
		GField * const wr = _vwr.data();

		for (size_t s = 1; s < n / 2; s *= 2)
		{
			const size_t m = 4 * s;
			const GField r_s = GField::primroot_n(2 * m);
			for (size_t j = 0; j < s; ++j)
			{
				wr[s + j] = r_s.pow(bitrev(j, m) + 1);
			}
		}

		_vz[0] = GField(Field(1), Field(0));

		_fmax = 0;
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
		const GField * const z = _vz.data();

		std::vector<int64_t> vzi(2 * n);
		int64_t * const zi = vzi.data();

		for (size_t i = 0; i < n; ++i)
		{
			zi[i + 0 * n] = z[i].s0().get_int();
			zi[i + 1 * n] = z[i].s1().get_int();
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

	__int64 fmax() const { return _fmax; }
};

static void check(const uint32_t b, const int n, const uint64_t expectedResidue = 0)
{
	const clock_t start = clock();

	mpz_t exponent; mpz_init(exponent);
	mpz_ui_pow_ui(exponent, b, static_cast<unsigned long int>(1) << n);

	const bool ispoweroftwo = ((b != 0) && ((b & (~b + 1)) == b));
	Transform<Z61, GF61> transform(1 << n, b, ispoweroftwo ? 3 : 2);

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
	std::cout << ", max = " << transform.fmax() * 1.0 / __INT64_MAX__ << ", " << duration << " sec.";

	if ((isPrime && (expectedResidue != 0)) || (!isPrime && (expectedResidue != residue))) std::cout << " ERROR!";

	std::cout << std::endl;
}

int main()
{
	// GF31
	// check(5748, 5);
	// check(4058, 6);
	// check(2884, 7);
	// check(1938, 8);
	// check(1342, 9);
	// check(824, 10);
	// check(150, 11);

	// GF61
	check(189812522, 5);
	check(134217660, 6);
	check(10234, 6, 0x831a0082ced5b6e1ull);
	check(94905500, 7);
	check(10032, 7, 0x3b2808f35afe61bdull);
	check(67108840, 8);
	check(584328, 8, 0xe6205f75c7438c2eull);
	check(47452788, 9);
	check(419000, 9, 0xfb2b0688cefd4fabull);
	check(33553366, 10);
	check(352220, 10, 0x1e830b3c54de5ef3ull);
	check(23723612, 11);
	check(366672, 11, 0xef2f0357a06a13e3ull);
	check(16757788, 12);
	check(285064, 12, 0x4e349f8254b4e364ull);
	check(11844594, 13);
	check(8351796, 14);
	check(5761466, 15);
	check(3966304, 16);
	check(2639850, 17);
	check(2042774, 18);
	check(475856, 19);
	check(919444, 20);

	return EXIT_SUCCESS;
}
