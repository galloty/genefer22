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

	static uint64_t _lshift(const uint64_t a, const int s)
	{
		const uint64_t lo = a << s, hi = a >> (64 - s);
		const uint64_t lo61 = lo & _p, hi61 = (lo >> 61) | (hi << 3);
		return _add(lo61, hi61);
	}

public:
	Z61() {}
	explicit Z61(const uint64_t n) : _n(n) {}

	uint64_t get() const { return _n; }
	int64_t get_int() const { return (_n >= _p / 2) ? int64_t(_n - _p) : int64_t(_n); }	// if n = p then return 0
	Z61 & set_int(const int i) { _n = (i < 0) ? uint64_t(i) + _p : uint64_t(i); return *this; }

	// Z61 neg() const { return Z61((_n == 0) ? 0 : _p - _n); }

	Z61 add(const Z61 & rhs) const { return Z61(_add(_n, rhs._n)); }
	Z61 sub(const Z61 & rhs) const { return Z61(_sub(_n, rhs._n)); }
	Z61 mul(const Z61 & rhs) const { return Z61(_mul(_n, rhs._n)); }
	Z61 lshift(const int s) const { return Z61(_lshift(_n, s)); }
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

	GF61 & set_int(const int i0, const int i1) { _s0.set_int(i0); _s1.set_int(i1); return *this; }

	// GF61 conj() const { return GF61(_s0, -_s1); }

	GF61 add(const GF61 & rhs) const { return GF61(_s0.add(rhs._s0), _s1.add(rhs._s1)); }
	GF61 sub(const GF61 & rhs) const { return GF61(_s0.sub(rhs._s0), _s1.sub(rhs._s1)); }
	GF61 lshift(const int s) const { return GF61(_s0.lshift(s), _s1.lshift(s)); }

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

	static uint32_t _lshift(const uint32_t a, const int s)
	{
		const uint64_t t = uint64_t(a) << s;
		const uint32_t lo = uint32_t(t) & _p, hi = uint32_t(t >> 31);
		return _add(hi, lo);
	}

public:
	Z31() {}
	explicit Z31(const uint32_t n) : _n(n) {}

	uint32_t get() const { return _n; }
	int32_t get_int() const { return (_n >= _p / 2) ? int32_t(_n - _p) : int32_t(_n); }
	Z31 & set_int(const int i) { _n = (i < 0) ? uint32_t(i) + _p : uint32_t(i); return *this; }

	// Z31 neg() const { return Z31((_n == 0) ? 0 : _p - _n); }

	Z31 add(const Z31 & rhs) const { return Z31(_add(_n, rhs._n)); }
	Z31 sub(const Z31 & rhs) const { return Z31(_sub(_n, rhs._n)); }
	Z31 mul(const Z31 & rhs) const { return Z31(_mul(_n, rhs._n)); }
	Z31 lshift(const int s) const { return Z31(_lshift(_n, s)); }
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

	GF31 & set_int(const int i0, const int i1) { _s0.set_int(i0); _s1.set_int(i1); return *this; }

	// GF31 conj() const { return GF31(_s0, -_s1); }

	GF31 add(const GF31 & rhs) const { return GF31(_s0.add(rhs._s0), _s1.add(rhs._s1)); }
	GF31 sub(const GF31 & rhs) const { return GF31(_s0.sub(rhs._s0), _s1.sub(rhs._s1)); }
	GF31 lshift(const int s) const { return GF31(_s0.lshift(s), _s1.lshift(s)); }

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

// GF((2^61 - 1)^2) x GF((2^31 - 1)^2)
class GF61_31
{
private:
	GF61 _n61;
	GF31 _n31;

	public:
	GF61_31() {}
	explicit GF61_31(const GF61 & n61, const GF31 & n31) : _n61(n61), _n31(n31) {}
	explicit GF61_31(const unsigned int i) : _n61(Z61(i), Z61(0)), _n31(Z31(i), Z31(0)) {}

	const GF61 & n61() const { return _n61; }
	const GF31 & n31() const { return _n31; }

	GF61_31 & set_int(const int i0, const int i1) { _n61.set_int(i0, i1); _n31.set_int(i0, i1); return *this; }

	// GF61_31 conj() const { return GF61_31(_n61.conj(), _n31.conj()); }

	GF61_31 add(const GF61_31 & rhs) const { return GF61_31(_n61.add(rhs._n61), _n31.add(rhs._n31)); }
	GF61_31 sub(const GF61_31 & rhs) const { return GF61_31(_n61.sub(rhs._n61), _n31.sub(rhs._n31)); }
	GF61_31 lshift(const int s61, const int s31) const { return GF61_31(_n61.lshift(s61), _n31.lshift(s31)); }

	GF61_31 sqr() const { return GF61_31(_n61.sqr(), _n31.sqr()); }
	GF61_31 mul(const GF61_31 & rhs) const { return GF61_31(_n61.mul(rhs._n61), _n31.mul(rhs._n31)); }
	GF61_31 mulconj(const GF61_31 & rhs) const { return GF61_31(_n61.mulconj(rhs._n61), _n31.mulconj(rhs._n31)); }

	// GF61_31 muli() const { return GF61_31(_n61.muli(), _n31.muli()); }
	GF61_31 addi(const GF61_31 & rhs) const { return GF61_31(_n61.addi(rhs._n61), _n31.addi(rhs._n31)); }
	GF61_31 subi(const GF61_31 & rhs) const { return GF61_31(_n61.subi(rhs._n61), _n31.subi(rhs._n31)); }

	GF61_31 pow(const uint64_t e) const
	{
		if (e == 0) return GF61_31(1u);
		GF61_31 r = GF61_31(1u), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF61_31 primroot_n(const uint32_t n) { return GF61_31(GF61::primroot_n(n), GF31::primroot_n(n)); }

	void garner(__int128_t & i_0, __int128_t & i_1) const
	{
		const uint32_t n31_0 = _n31.s0().get(), n31_1 = _n31.s1().get();
		const GF61 n31 = GF61(Z61(n31_0), Z61(n31_1));
		GF61 u = _n61.sub(n31); 
		// The inverse of 2^31 - 1 mod 2^61 - 1 is 2^31 + 1
		u = u.add(u.lshift(31));
		const uint64_t s_0 = u.s0().get(), s_1 = u.s1().get();
		const __uint128_t n_0 = n31_0 +	(__uint128_t(s_0) << 31) - s_0;
		const __uint128_t n_1 = n31_1 + (__uint128_t(s_1) << 31) - s_1;
		static const __uint128_t M61M31 = ((__uint128_t(1) << 61) - 1) * ((uint32_t(1) << 31) - 1);
		i_0 = __int128_t((n_0 > M61M31 / 2) ? (n_0 - M61M31) : n_0);
		i_1 = __int128_t((n_1 > M61M31 / 2) ? (n_1 - M61M31) : n_1);
	}
};

class Transform
{
private:
	std::vector<GF61_31> _vz;
	std::vector<GF61_31> _vwr;
	const int32_t _base;
	const int32_t _multiplier;
	const int _snorm31;
	__uint128_t _fmax;

private:
	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	void forward2(const size_t m, const size_t s)
	{
		GF61_31 * const z = _vz.data();
		const GF61_31 * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF61_31 w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const GF61_31 u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w);
				z[k + 0 * m] = u0.add(u1); z[k + 1 * m] = u0.sub(u1);
			}
		}
	}

	void backward2(const size_t m, const size_t s)
	{
		GF61_31 * const z = _vz.data();
		const GF61_31 * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF61_31 w = wr[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const GF61_31 u0 = z[k + 0 * m], u1 = z[k + 1 * m];
				z[k + 0 * m] = u0.add(u1); z[k + 1 * m] = u0.sub(u1).mulconj(w);
			}
		}
	}

	void forward4(const size_t m, const size_t s)
	{
		GF61_31 * const z = _vz.data();
		const GF61_31 * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF61_31 w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF61_31 u0 = z[k + 0 * m], u1 = z[k + 1 * m].mul(w2), u2 = z[k + 2 * m].mul(w1), u3 = z[k + 3 * m].mul(w3);
				const GF61_31 v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
				z[k + 0 * m] = v0.add(v1); z[k + 1 * m] = v0.sub(v1);
				z[k + 2 * m] = v2.addi(v3); z[k + 3 * m] = v2.subi(v3);
			}
		}
	}

	void backward4(const size_t m, const size_t s)
	{
		GF61_31 * const z = _vz.data();
		const GF61_31 * const wr = _vwr.data();

		for (size_t j = 0; j < s; ++j)
		{
			const GF61_31 w1 = wr[s + j], w2 = wr[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				const GF61_31 u0 = z[k + 0 * m], u1 = z[k + 1 * m], u2 = z[k + 2 * m], u3 = z[k + 3 * m];
				const GF61_31 v0 = u0.add(u1), v1 = u0.sub(u1), v2 = u2.add(u3), v3 = u3.sub(u2);
				z[k + 0 * m] = v0.add(v2); z[k + 2 * m] = v0.sub(v2).mulconj(w1);
				z[k + 1 * m] = v1.addi(v3).mulconj(w2); z[k + 3 * m] = v1.subi(v3).mulconj(w3);
			}
		}
	}

	void square2()
	{
		const size_t n = _vz.size();
		GF61_31 * const z = _vz.data();
		const GF61_31 * const wr = _vwr.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GF61_31 w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GF61_31 u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2], u3 = z[k + 3];
			z[k + 0] = u0.sqr().add(u1.sqr().mul(w)); z[k + 1] = u0.mul(u1.add(u1));
			z[k + 2] = u2.sqr().sub(u3.sqr().mul(w)); z[k + 3] = u2.mul(u3.add(u3));
		}
	}

	void square4()
	{
		const size_t n = _vz.size();
		GF61_31 * const z = _vz.data();
		const GF61_31 * const wr = _vwr.data();

		for (size_t j = 0; j < n / 4; ++j)
		{
			const GF61_31 w = wr[n / 4 + j];

			const size_t k = 4 * j;
			const GF61_31 u0 = z[k + 0], u1 = z[k + 1], u2 = z[k + 2].mul(w), u3 = z[k + 3].mul(w);
			const GF61_31 v0 = u0.add(u2), v1 = u1.add(u3), v2 = u0.sub(u2), v3 = u1.sub(u3);
			const GF61_31 s0 = v0.sqr().add(v1.sqr().mul(w)), s1 = v0.mul(v1.add(v1));
			const GF61_31 s2 = v2.sqr().sub(v3.sqr().mul(w)), s3 = v2.mul(v3.add(v3));
			z[k + 0] = s0.add(s2); z[k + 2] = s0.sub(s2).mulconj(w);
			z[k + 1] = s1.add(s3); z[k + 3] = s1.sub(s3).mulconj(w);
		}
	}

	void carry(const bool mul)
	{
		const size_t n = _vz.size();
		GF61_31 * const z = _vz.data();

		const int snorm31 = _snorm31;
		const int32_t m = _multiplier, base = _base;
		__int128_t f0 = 0, f1 = 0;
		__uint128_t fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const GF61_31 u = z[k].lshift(snorm31 + 61 - 31, snorm31);
			__int128_t i0, i1; u.garner(i0, i1);
			if (mul) { i0 *= m; i1 *= m; }
			f0 += i0; f1 += i1;
			const __uint128_t uf0 = __uint128_t((f0 < 0) ? -f0 : f0), uf1 = __uint128_t((f1 < 0) ? -f1 : f1);
			fmax = (uf0 > fmax) ? uf0 : fmax; fmax = (uf1 > fmax) ? uf1 : fmax;
			__int128_t l0 = f0 / base, l1 = f1 / base;
			z[k].set_int(int(f0 - l0 * base), int(f1 - l1 * base));
			f0 = l0; f1 = l1;
		}

		if (fmax > _fmax) _fmax = fmax;

		while ((f0 != 0) || (f1 != 0))
		{
			__int128_t t = f0; f0 = -f1; f1 = t;	// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				const GF61_31 u = z[k];
				int32_t i0 = u.n31().s0().get_int(), i1 = u.n31().s1().get_int();
				f0 += i0; f1 += i1;
				__int128_t l0 = f0 / base, l1 = f1 / base;
				z[k].set_int(int(f0 - l0 * base), int(f1 - l1 * base));
				f0 = l0; f1 = l1;
				if ((l0 == 0) && (l1 == 0)) break;
			}
		}
	}

public:
	Transform(const uint32_t b, const int n, const uint32_t a)
		: _vz(size_t(1) << (n - 1), GF61_31(0u)), _vwr(size_t(1) << (n - 2)), _base(int32_t(b)), _multiplier(int32_t(a)), _snorm31(31 - n + 2)
	{
		const size_t size = _vz.size();
		GF61_31 * const wr = _vwr.data();

		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const GF61_31 r_s = GF61_31::primroot_n(2 * 4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr[s + j] = r_s.pow(bitrev(j, 4 * s) + 1);
			}
		}

		_vz[0] = GF61_31(1u);

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
		const GF61_31 * const z = _vz.data();

		std::vector<int64_t> vzi(2 * n);
		int64_t * const zi = vzi.data();

		for (size_t i = 0; i < n; ++i)
		{
			const GF31 n31 = z[i].n31();
			zi[i + 0 * n] = n31.s0().get_int();
			zi[i + 1 * n] = n31.s1().get_int();
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
	std::cout << ", max = " << transform.fmax() * std::pow(2.0, -96) << ", " << duration << " sec.";

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
	// check(10234, 6, 0x831a0082ced5b6e1ull);
	// check(10032, 7, 0x3b2808f35afe61bdull);
	// check(584328, 8, 0xe6205f75c7438c2eull);
	// check(419000, 9, 0xfb2b0688cefd4fabull);
	// check(352220, 10, 0x1e830b3c54de5ef3ull);
	// check(366672, 11, 0xef2f0357a06a13e3ull);
	// check(285064, 12, 0x4e349f8254b4e364ull);
	// check(189812522, 5);
	// check(134217660, 6);
	// check(94905500, 7);
	// check(67108840, 8);
	// check(47452788, 9);
	// check(33553366, 10);
	// check(23723612, 11);
	// check(16757788, 12);
	// check(11844594, 13);
	// check(8351796, 14);
	// check(5761466, 15);
	// check(3966304, 16);
	// check(2639850, 17);
	// check(2042774, 18);
	// check(475856, 19);
	// check(919444, 20);

	// GF61 x GF31
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
	check(440025190, 16);
	check(300359914, 17);
	check(45007104, 18);
	check(11937916, 19);
	check(3843236, 20);
	
	return EXIT_SUCCESS;
}
