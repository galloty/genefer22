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

// Z/M_qZ: the prime field of order M_q = 2^q - 1
template<int q, typename uint_t, typename ulong_t>
class ZMq
{
private:
	static const uint_t _p = (uint_t(1) << q) - 1;
	uint_t _n;	// 0 <= n < p

	static uint_t _add(const uint_t a, const uint_t b) { const uint_t t = a + b; return t - ((t >= _p) ? _p : 0); }
	static uint_t _sub(const uint_t a, const uint_t b) { const uint_t t = a - b; return t + ((a < b) ? _p : 0); }
	static uint_t _mul(const uint_t a, const uint_t b) { const ulong_t t = a * ulong_t(b); return _add(uint_t(t) & _p, uint_t(t >> q)); }
	static uint_t _lshift(const uint_t a, const int s) { const ulong_t t = ulong_t(a) << s; return _add(uint_t(t) & _p, uint_t(t >> q)); }

public:
	ZMq() {}
	explicit ZMq(const uint_t n) : _n(n) {}

	static int get_q() { return q; }

	uint_t get() const { return _n; }
 	int32_t get_int() const { return (_n >= _p / 2) ? int32_t(_n - _p) : int32_t(_n); }
 	ZMq & set_int(const int32_t i) { _n = (i < 0) ? uint_t(i) + _p : uint_t(i); return *this; }

	// ZMq neg() const { return ZMq((_n == 0) ? 0 : _p - _n); }

	ZMq operator+(const ZMq & rhs) const { return ZMq(_add(_n, rhs._n)); }
	ZMq operator-(const ZMq & rhs) const { return ZMq(_sub(_n, rhs._n)); }
	ZMq operator*(const ZMq & rhs) const { return ZMq(_mul(_n, rhs._n)); }

	ZMq sqr() const { return ZMq(_mul(_n, _n)); }

	ZMq lshift(const int s) const { return ZMq(_lshift(_n, s)); }
};

// GF(p^2): the field of order p^2, where p = 2^q - 1
// primroot must be a primitive root of order primroot_order and
// such that primroot^{order/8) = 2^{(q - 1)/2} * (1, 1).
template<typename Zp, uint64_t primroot_order, uint64_t primroot_0, uint64_t primroot_1>
class GFp2
{
private:
	Zp _s0, _s1;

	explicit GFp2(const uint64_t n0, const uint64_t n1 = 0) : _s0(n0), _s1(n1) {}

public:
	GFp2() {}
	explicit GFp2(const Zp & s0, const Zp & s1) : _s0(s0), _s1(s1) {}

	const Zp & s0() const { return _s0; }
	const Zp & s1() const { return _s1; }

	void get_int(int32_t & i0, int32_t & i1) const { i0 = _s0.get_int(); i1 = _s1.get_int(); }
	GFp2 & set_int(const int32_t i0, const int32_t i1) { _s0.set_int(i0); _s1.set_int(i1); return *this; }

	GFp2 operator+(const GFp2 & rhs) const { return GFp2(_s0 + rhs._s0, _s1 + rhs._s1); }
	GFp2 operator-(const GFp2 & rhs) const { return GFp2(_s0 - rhs._s0, _s1 - rhs._s1); }

	GFp2 lshift(const int s) const { return GFp2(_s0.lshift(s), _s1.lshift(s)); }

private:
	// GFp2 muli() const { return GFp2(_s1.neg(), _s0); }

	GFp2 addi(const GFp2 & rhs) const { return GFp2(_s0 - rhs._s1, _s1 + rhs._s0); }
	GFp2 subi(const GFp2 & rhs) const { return GFp2(_s0 + rhs._s1, _s1 - rhs._s0); }

	// * sqrt(2)/2 * (1 + i)
	GFp2 mul_R8() const
	{
		const uint8_t log2_sqrt2_2 = uint8_t((Zp::get_q() - 1) / 2);
		return GFp2((_s0 - _s1).lshift(log2_sqrt2_2), (_s1 + _s0).lshift(log2_sqrt2_2));
	}

	// * sqrt(2)/2 * (1 - i)
	GFp2 mul_R8conj() const
	{
		const uint8_t log2_sqrt2_2 = uint8_t((Zp::get_q() - 1) / 2);
		return GFp2((_s0 + _s1).lshift(log2_sqrt2_2), (_s1 - _s0).lshift(log2_sqrt2_2));
	}

	GFp2 sqr() const { const Zp t = _s0 * _s1; return GFp2(_s0.sqr() - _s1.sqr(), t + t); }
	GFp2 mul(const GFp2 & rhs) const { return GFp2(_s0 * rhs._s0 - _s1 * rhs._s1, _s1 * rhs._s0 + _s0 * rhs._s1); }
	GFp2 mulconj(const GFp2 & rhs) const { return GFp2(_s0 * rhs._s0 + _s1 * rhs._s1, _s1 * rhs._s0 - _s0 * rhs._s1); }

	GFp2 pow(const uint64_t e) const
	{
		if (e == 0) return GFp2(1);
		GFp2 r = GFp2(1), y = *this;
		for (uint64_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GFp2 root_nth(const size_t n) { return GFp2(primroot_0, primroot_1).pow(primroot_order / n); }

	static constexpr size_t bitrev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k != 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	static void fwd2(GFp2 & z0, GFp2 & z1, const GFp2 & w)
	{
		const GFp2 u0 = z0, u1 = z1.mul(w);
		z0 = u0 + u1; z1 = u0 - u1;
	}

	static void bck2(GFp2 & z0, GFp2 & z1, const GFp2 & w)
	{
		const GFp2 u0 = z0, u1 = z1;
		z0 = u0 + u1; z1 = (u0 - u1).mulconj(w);
	}

	static void fwd4(GFp2 & z0, GFp2 & z1, GFp2 & z2, GFp2 & z3, const GFp2 & w1, const GFp2 & w2, const GFp2 & w3)
	{
		const GFp2 u0 = z0, u2 = z2.mul(w1), u1 = z1.mul(w2), u3 = z3.mul(w3);
		const GFp2 v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		z0 = v0 + v1; z1 = v0 - v1; z2 = v2.addi(v3); z3 = v2.subi(v3);
	}

	static void bck4(GFp2 & z0, GFp2 & z1, GFp2 & z2, GFp2 & z3, const GFp2 & w1, const GFp2 & w2, const GFp2 & w3)
	{
		const GFp2 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const GFp2 v0 = u0 + u1, v1 = u0 - u1, v2 = u2 + u3, v3 = u2 - u3;
		z0 = v0 + v2; z2 = (v0 - v2).mulconj(w1); z1 = v1.subi(v3).mulconj(w2); z3 = v1.addi(v3).mulconj(w3);
	}

	static void fwd8(GFp2 & z0, GFp2 & z1, GFp2 & z2, GFp2 & z3, GFp2 & z4, GFp2 & z5, GFp2 & z6, GFp2 & z7,
					 const GFp2 & w1, const GFp2 & w2, const GFp2 & w3, const GFp2 & w4, const GFp2 & w5, const GFp2 & w6, const GFp2 & w7)
	{
		const GFp2 u0 = z0, u4 = z4.mul(w1), u2 = z2.mul(w2), u6 = z6.mul(w3), u1 = z1.mul(w4), u5 = z5.mul(w5), u3 = z3.mul(w6), u7 = z7.mul(w7);

		const GFp2 v0 = u0 + u4, v4 = u0 - u4, v2 = u2 + u6, v6 = u2 - u6;
		const GFp2 v1 = u1 + u5, v5 = u1 - u5, v3 = u3 + u7, v7 = u3 - u7;

		const GFp2 t0 = v0 + v2, t2 = v0 - v2, t1 = v1 + v3, t3 = v1 - v3;
		const GFp2 t4 = v4.addi(v6), t6 = v4.subi(v6), t5 = v5.addi(v7).mul_R8(), t7 = v5.subi(v7).mul_R8();

		z0 = t0 + t1; z1 = t0 - t1; z2 = t2.addi(t3); z3 = t2.subi(t3);
		z4 = t4 + t5; z5 = t4 - t5; z6 = t6.addi(t7); z7 = t6.subi(t7);
	}

	static void bck8(GFp2 & z0, GFp2 & z1, GFp2 & z2, GFp2 & z3, GFp2 & z4, GFp2 & z5, GFp2 & z6, GFp2 & z7,
					 const GFp2 & w1, const GFp2 & w2, const GFp2 & w3, const GFp2 & w4, const GFp2 & w5, const GFp2 & w6, const GFp2 & w7)
	{
		const GFp2 u0 = z0, u1 = z1, u2 = z2, u3 = z3, u4 = z4, u5 = z5, u6 = z6, u7 = z7;
		const GFp2 v0 = u0 + u1, v1 = u0 - u1, v2 = u2 + u3, v3 = u2 - u3;
		const GFp2 v4 = u4 + u5, v5 = (u4 - u5).mul_R8conj(), v6 = u6 + u7, v7 = (u6 - u7).mul_R8conj();

		const GFp2 t0 = v0 + v2, t2 = v0 - v2, t1 = v1.subi(v3), t3 = v1.addi(v3);
		const GFp2 t4 = v4 + v6, t6 = v4 - v6, t5 = v5.subi(v7), t7 = v5.addi(v7);

		z0 = t0 + t4; z4 = (t0 - t4).mulconj(w1); z2 = t2.subi(t6).mulconj(w2); z6 = t2.addi(t6).mulconj(w3);
		z1 = (t1 + t5).mulconj(w4); z5 = (t1 - t5).mulconj(w5); z3 = t3.subi(t7).mulconj(w6); z7 = t3.addi(t7).mulconj(w7);
	}

	static void sqr2x2(GFp2 & z0, GFp2 & z1, GFp2 & z2, GFp2 & z3, const GFp2 & w)
	{
		const GFp2 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		z0 = u0.sqr() + u1.sqr().mul(w), z1 = u0.mul(u1 + u1);
		z2 = u2.sqr() - u3.sqr().mul(w), z3 = u2.mul(u3 + u3);
	}

	static void sqr4(GFp2 & z0, GFp2 & z1, GFp2 & z2, GFp2 & z3, const GFp2 & w)
	{
		const GFp2 u0 = z0, u2 = z2.mul(w), u1 = z1, u3 = z3.mul(w);
		const GFp2 v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const GFp2 s0 = v0.sqr() + v1.sqr().mul(w), s1 = v0.mul(v1 + v1);
		const GFp2 s2 = v2.sqr() - v3.sqr().mul(w), s3 = v2.mul(v3 + v3);
		z0 = s0 + s2; z2 = (s0 - s2).mulconj(w); z1 = s1 + s3; z3 = (s1 - s3).mulconj(w);
	}

	static void forward2(GFp2 * const z, const GFp2 * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const GFp2 w1 = w[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				fwd2(z[k + 0 * m], z[k + 1 * m], w1);
			}
		}
	}

	static void backward2(GFp2 * const z, const GFp2 * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const GFp2 w1 = w[s + j];

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				bck2(z[k + 0 * m], z[k + 1 * m], w1);
			}
		}
	}

	static void forward4(GFp2 * const z, const GFp2 * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const GFp2 w1 = w[s + j], w2 = w[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				fwd4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], w1, w2, w3);
			}
		}
	}

	static void backward4(GFp2 * const z, const GFp2 * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const GFp2 w1 = w[s + j], w2 = w[2 * (s + j)], w3 = w1.mul(w2);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 4 * m * j + i;
				bck4(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], w1, w2, w3);
			}
		}
	}

	static void forward8(GFp2 * const z, const GFp2 * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const GFp2 w1 = w[s + j], w2 = w[2 * (s + j)], w4 = w[4 * (s + j)];
			const GFp2 w3 = w1.mul(w2), w5 = w1.mul(w4), w6 = w2.mul(w4), w7 = w3.mul(w4);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				fwd8(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], z[k + 4 * m], z[k + 5 * m], z[k + 6 * m], z[k + 7 * m],
					 w1, w2, w3, w4, w5, w6, w7);
			}
		}
	}

	static void backward8(GFp2 * const z, const GFp2 * const w, const size_t m, const size_t s)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const GFp2 w1 = w[s + j], w2 = w[2 * (s + j)], w4 = w[4 * (s + j)];
			const GFp2 w3 = w1.mul(w2), w5 = w1.mul(w4), w6 = w2.mul(w4), w7 = w3.mul(w4);

			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 8 * m * j + i;
				bck8(z[k + 0 * m], z[k + 1 * m], z[k + 2 * m], z[k + 3 * m], z[k + 4 * m], z[k + 5 * m], z[k + 6 * m], z[k + 7 * m],
					 w1, w2, w3, w4, w5, w6, w7);
			}
		}
	}

	static void square2(GFp2 * const z, const GFp2 * const w, const size_t n)
	{
		for (size_t j = 0; j < n / 4; ++j)
		{
			sqr2x2(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], w[n / 4 + j]);
		}
	}

	static void square4(GFp2 * const z, const GFp2 * const w, const size_t n)
	{
		for (size_t j = 0; j < n / 4; ++j)
		{
			sqr4(z[4 * j + 0], z[4 * j + 1], z[4 * j + 2], z[4 * j + 3], w[n / 4 + j]);
		}
	}

public:
	static void init_twiddle_factors(GFp2 * const w, const size_t n)
	{
		for (size_t s = 1; s < n; s *= 2)
		{
			const GFp2 r_s = root_nth(2 * 4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				w[s + j] = r_s.pow(bitrev(j, 4 * s) + 1);
			}
		}
	}

	static void init(GFp2 * const z, const size_t n, const uint32_t a)
	{
		z[0] = GFp2(a); for (size_t k = 1; k < n; ++k) z[k] = GFp2(0);
	}

	static void square(GFp2 * const z, const GFp2 * const w, const size_t n)
	{
		size_t m = n / 2, s = 1;

		// for (; m >= 2; m /= 2, s *= 2) forward2(z, w, m, s);
		// square2(z, w, n);
		// for (m = 2, s /= 2; s >= 1; m *= 2, s /= 2) backward2(z, w, m, s);

		// for (; m >= 4; m /= 4, s *= 4) forward4(z, w, m / 2, s);
		// if (m == 1) square2(z, w, n); else square4(z, w, n);
		// for (m *= 4, s /= 4; s >= 1; m *= 4, s /= 4) backward4(z, w, m / 2, s);

		for (; m >= 8; m /= 8, s *= 8) forward8(z, w, m / 4, s);
		if (m == 4) { forward4(z, w, 2, n / 8); square2(z, w, n); backward4(z, w, 2, n / 8); }
		else if (m == 2) square4(z, w, n);
		else /*if (m == 1)*/ square2(z, w, n);
		for (m *= 8, s /= 8; s >= 1; m *= 8, s /= 8) backward8(z, w, m / 4, s);
	}
};

using Z61 = ZMq<61, uint64_t, __uint128_t>;
using Z31 = ZMq<31, uint32_t, uint64_t>;

using GF61 = GFp2<Z61, uint64_t(1) << 31, 2187240813972ull, 5388952866649ull>;
using GF31 = GFp2<Z31, uint64_t(1) << 31, 12683u, 91810u>;

class Transform
{
private:
	const size_t _n;
	const int32_t _base;
	const int32_t _multiplier;
	const int _snorm31;
	__uint128_t _fmax;
	std::vector<GF61> _vz61;
	std::vector<GF31> _vz31;
	std::vector<GF61> _vw61;
	std::vector<GF31> _vw31;

private:
	void garner(const GF61 & u61, const GF31 & u31, __int128_t & i_0, __int128_t & i_1) const
	{
		const uint32_t n31_0 = u31.s0().get(), n31_1 = u31.s1().get();
		const GF61 n31 = GF61(Z61(n31_0), Z61(n31_1));
		GF61 u = u61 - n31; 
		// The inverse of 2^31 - 1 mod 2^61 - 1 is 2^31 + 1
		u = u + u.lshift(31);
		const uint64_t s_0 = u.s0().get(), s_1 = u.s1().get();
		const __uint128_t n_0 = n31_0 +	(__uint128_t(s_0) << 31) - s_0;
		const __uint128_t n_1 = n31_1 + (__uint128_t(s_1) << 31) - s_1;
		static const __uint128_t M61M31 = ((__uint128_t(1) << 61) - 1) * ((uint32_t(1) << 31) - 1);
		i_0 = __int128_t((n_0 > M61M31 / 2) ? (n_0 - M61M31) : n_0);
		i_1 = __int128_t((n_1 > M61M31 / 2) ? (n_1 - M61M31) : n_1);
	}

	void carry(const bool mul)
	{
		const size_t n = _vz61.size();
		GF61 * const z61 = _vz61.data();
		GF31 * const z31 = _vz31.data();

		const int snorm31 = _snorm31;
		const int32_t m = _multiplier, base = _base;
		__int128_t f0 = 0, f1 = 0;
		__uint128_t fmax = 0;

		for (size_t k = 0; k < n; ++k)
		{
			const GF61 u61 = z61[k].lshift(snorm31 + 61 - 31);
			const GF31 u31 = z31[k].lshift(snorm31);
			__int128_t i0, i1; garner(u61, u31, i0, i1);
			if (mul) { i0 *= m; i1 *= m; }
			f0 += i0; f1 += i1;
			const __uint128_t uf0 = __uint128_t((f0 < 0) ? -f0 : f0), uf1 = __uint128_t((f1 < 0) ? -f1 : f1);
			fmax = (uf0 > fmax) ? uf0 : fmax; fmax = (uf1 > fmax) ? uf1 : fmax;
			const __int128_t l0 = f0 / base, l1 = f1 / base;
			const int32_t r0 = int32_t(f0 - l0 * base), r1 = int32_t(f1 - l1 * base);
			z61[k].set_int(r0, r1); z31[k].set_int(r0, r1);
			f0 = l0; f1 = l1;
		}

		if (fmax > _fmax) _fmax = fmax;

		while ((f0 != 0) || (f1 != 0))
		{
			__int128_t t = f0; f0 = -f1; f1 = t;	// a_n = -a_0

			for (size_t k = 0; k < n; ++k)
			{
				const GF31 u = z31[k];
				int32_t i0, i1; u.get_int(i0, i1);
				f0 += i0; f1 += i1;
				const __int128_t l0 = f0 / base, l1 = f1 / base;
				const int32_t r0 = int32_t(f0 - l0 * base), r1 = int32_t(f1 - l1 * base);
				z61[k].set_int(r0, r1); z31[k].set_int(r0, r1);
				f0 = l0; f1 = l1;
				if ((l0 == 0) && (l1 == 0)) break;
			}
		}
	}

public:
	Transform(const uint32_t b, const int m, const uint32_t a)
		: _n(size_t(1) << (m - 1)), _base(int32_t(b)), _multiplier(int32_t(a)), _snorm31(31 - m + 2), _fmax(0),
		  _vz61(_n), _vz31(_n), _vw61(_n), _vw31(_n)
	{
		GF61::init_twiddle_factors(_vw61.data(), _n);
		GF31::init_twiddle_factors(_vw31.data(), _n);
	}

	void init(const uint32_t a)
	{
		const size_t n = _vz61.size();

		GF61::init(_vz61.data(), n, a);
		GF31::init(_vz31.data(), n, a);
	}

	void square_mul(const bool mul)
	{
		const size_t n = _vz61.size();

		GF61::square(_vz61.data(), _vw61.data(), n);
		GF31::square(_vz31.data(), _vw31.data(), n);

		carry(mul);
	}

public:
	bool equal_one(uint64_t & res64) const
	{
		const size_t n = _vz31.size();
		const GF31 * const z = _vz31.data();

		std::vector<int32_t> vzi(2 * n);
		int32_t * const zi = vzi.data();
		for (size_t i = 0; i < n; ++i) z[i].get_int(zi[i + 0 * n], zi[i + 1 * n]);

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
			zi[0] -= f;		// a[n] = -a[0]
		} while (f != 0);

		bool b_one = (zi[0] == 1);
		if (b_one) for (size_t i = 1; i < 2 * n; ++i) b_one &= (zi[i] == 0);

		uint64_t r64 = 0, b = 1;
		for (size_t i = 0; i < 2 * n; ++i)
		{
			r64 += uint32_t(zi[i]) * b;
			b *= uint32_t(base);
		}
		res64 = r64;

		return b_one;
	}

	__uint128_t fmax() const { return _fmax; }
};

static void check(const uint32_t b, const int m, const uint64_t expected_residue = 0)
{
	const clock_t start = clock();

	mpz_t exponent; mpz_init(exponent);
	mpz_ui_pow_ui(exponent, b, 1u << m);

	const bool is_power_of_two = ((b != 0) && ((b & (~b + 1)) == b));
	Transform transform(b, m, is_power_of_two ? 3 : 2);

	transform.init(1);
	for (int i = int(mpz_sizeinbase(exponent, 2) - 1); i >= 0; --i)
	{
		transform.square_mul(mpz_tstbit(exponent, mp_bitcnt_t(i)) != 0);
	}

	mpz_clear(exponent);

	uint64_t residue;
	const bool is_prp = transform.equal_one(residue);

	const float duration = (float)(clock() - start) / CLOCKS_PER_SEC;

	std::cout << b << "^{2^" << m << "} + 1 is ";
	if (is_prp) std::cout << "a probable prime";
	else
	{
		std::cout << "composite (RES = ";
		std::cout.width(16); std::cout.fill('0');
		std::cout << std::internal << std::hex << residue << std::dec << ")";
	}
	std::cout.precision(3);
	std::cout << ", max = " << transform.fmax() * std::pow(2.0, -96) << ", " << duration << " sec.";

	if ((is_prp && (expected_residue != 0)) || (!is_prp && (expected_residue != residue))) std::cout << " ERROR!";

	std::cout << std::endl;
}

int main()
{
	// The condition are b < (2^31 - 1)/2 and n * (b-1)^2 < (2^31 - 1)*(2^61 - 1)/2
	// We have 2^31 * (1G - 1)^2 < (2^31 - 1)*(2^61 - 1)/2

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
	check(460026578, 15);
	check(584797994, 16);
	check(1000032472, 17);
	check(100013182, 18);
	check(23865586, 19);
	check(5336284, 20);
	check(2524190, 21);
	
	return EXIT_SUCCESS;
}
