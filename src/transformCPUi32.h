/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <immintrin.h>

#include "transform.h"

#define finline	__attribute__((always_inline))

#define P1		4253024257u		// 507 * 2^23 + 1
#define P2		4194304001u		// 125 * 2^25 + 1
#define P3		4076863489u		// 243 * 2^24 + 1
#define P1_inv	(uint64_t(-1) / P1 - (uint64_t(1) << 32))
#define P2_inv	(uint64_t(-1) / P2 - (uint64_t(1) << 32))
#define P3_inv	(uint64_t(-1) / P3 - (uint64_t(1) << 32))

template <uint32_t p, uint32_t p_inv, uint32_t prRoot>
class Zp
{
private:
	uint32_t _n;

public:
	finline Zp() {}
	finline explicit Zp(const uint32_t n) : _n(n) {}
	finline explicit Zp(const int32_t i) : _n((i < 0) ? p - uint32_t(-i) : uint32_t(i)) {}

	finline uint32_t get() const { return _n; }
	finline int32_t getInt() const { return (_n > p / 2) ? int32_t(_n - p) : int32_t(_n); }

	finline Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

	finline Zp operator+(const Zp & rhs) const
	{
		const uint32_t c = (_n >= p - rhs._n) ? p : 0;
		return Zp(_n + rhs._n - c);
	}

	finline Zp operator-(const Zp & rhs) const
	{
		const uint32_t c = (_n < rhs._n) ? p : 0;
		return Zp(_n - rhs._n + c);
	}

	finline Zp operator*(const Zp & rhs) const
	{
		return Zp(uint32_t((_n * uint64_t(rhs._n)) % p));
	}

	finline Zp & operator+=(const Zp & rhs) { *this = *this + rhs; return *this; }
	finline Zp & operator-=(const Zp & rhs) { *this = *this - rhs; return *this; }
	finline Zp & operator*=(const Zp & rhs) { *this = *this * rhs; return *this; }

	Zp pow(const uint32_t e) const
	{
		if (e == 0) return Zp(1);

		Zp r = Zp(1), y = *this;
		for (uint32_t i = e; i > 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;

		return r;
	}

	static Zp norm(const uint32_t n) { return -Zp((p - 1) / n); }
	static const Zp prRoot_n(const uint32_t n) { return Zp(prRoot).pow((p - 1) / n); }
};

typedef Zp<P1, P1_inv, 5> Zp1;
typedef Zp<P2, P2_inv, 3> Zp2;
typedef Zp<P3, P3_inv, 7> Zp3;

class RNS
{
private:
	Zp1 _r1;
	Zp2 _r2;
	Zp3 _r3;

public:
	finline RNS() {}
	finline explicit RNS(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3) : _r1(r1), _r2(r2), _r3(r3) {}
	finline explicit RNS(const int32_t i) : _r1(i), _r2(i), _r3(i) {}

	finline Zp1 r1() const { return _r1; }
	finline Zp2 r2() const { return _r2; }
	finline Zp3 r3() const { return _r3; }

	RNS pow(const uint32_t e) const { return RNS(_r1.pow(e), _r2.pow(e), _r3.pow(e)); }

	static const RNS prRoot_n(const uint32_t n) { return RNS(Zp1::prRoot_n(n), Zp2::prRoot_n(n), Zp3::prRoot_n(n)); }
};

template <uint32_t p, uint32_t p_inv, uint32_t prRoot>
class Zp4
{
	using Zpp = Zp<p, p_inv, prRoot>;

private:
	__v8su _n0123;
	static constexpr __v8su p0123 = { p, 0, p, 0, p, 0, p, 0 };
	static constexpr __v8su p0123_inv = { p_inv, 0, p_inv, 0, p_inv, 0, p_inv, 0 };
	static constexpr __v8su p0123_2 = { p / 2, 0, p / 2, 0, p / 2, 0, p / 2, 0 };

public:
	Zp4() {}
	finline explicit Zp4(const __v8su & n0123) : _n0123(n0123) {}
	finline explicit Zp4(const Zpp & z) : _n0123(__v8su(_mm256_set1_epi32(int(z.get())))) {}
	finline explicit Zp4(const __v4di & n0123) : _n0123(__v8su(n0123) + (__v8su(n0123 < __v4di{0, 0, 0, 0}) & p0123)) {}
	finline explicit Zp4(const Zpp & z0, const Zpp & z1, const Zpp & z2, const Zpp & z3)
	{
		_n0123[0] = z0.get(); _n0123[2] = z1.get(); _n0123[4] = z2.get(); _n0123[6] = z3.get();
	}

	finline __v8su get() const { return _n0123; }
	finline __v8si getInt() const { return __v8si(_n0123 - ((_n0123 > p0123_2) & p0123)); }

	finline Zpp operator[](const size_t i) const { return Zpp(_n0123[2 * i]); }

	finline Zp4 operator-() const { return Zp4((_n0123 != 0) & (p0123 - _n0123)); }

	finline Zp4 operator+(const Zp4 & rhs) const
	{
		const __v8su c = (_n0123 >= p0123 - rhs._n0123) & p0123;
		return Zp4(_n0123 + rhs._n0123 - c);
	}

	finline Zp4 operator-(const Zp4 & rhs) const
	{
		const __v8su c = (_n0123 < rhs._n0123) & p0123;
		return Zp4(_n0123 - rhs._n0123 + c);
	}

	finline Zp4 operator*(const Zp4 & rhs) const
	{
		// const uint64_t m = _n * uint64_t(rhs._n)
		const __v4du m = __v4du(_mm256_mul_epu32(__m256i(_n0123), __m256i(rhs._n0123)));

		// uint64_t q = uint32_t(m >> 32) * uint64_t(p_inv) + m;
		const __v8su m_32 = __v8su(_mm256_bsrli_epi128(__m256i(m), 4));
		const __v4du q = __v4du(_mm256_add_epi64(_mm256_mul_epu32(__m256i(m_32), __m256i(p0123_inv)), __m256i(m)));

		// uint32_t r = uint32_t(m) - (1 + uint32_t(q >> 32)) * p;
		const __v8su q_32 = __v8su(_mm256_bsrli_epi128(__m256i(q), 4));
		__v8su r = __v8su(m) - q_32 * p0123 - p0123;

		// if (r > uint32_t(q)) r += p;
		r += (r > __v8su(q)) & p0123;
		// if (r >= p) r -= p ;
		r -= (r >= p0123) & p0123;

		return Zp4(r);
	}

	finline Zp4 & operator+=(const Zp4 & rhs) { *this = *this + rhs; return *this; }
	finline Zp4 & operator-=(const Zp4 & rhs) { *this = *this - rhs; return *this; }
	finline Zp4 & operator*=(const Zp4 & rhs) { *this = *this * rhs; return *this; }

	finline void forward_4(const Zpp & w1, const Zpp & w2, const Zpp & w3)
	{
		const Zpp u0 = (*this)[0], u2 = (*this)[2] * w1, u1 = (*this)[1], u3 = (*this)[3] * w1;
		const Zpp v0 = u0 + u2, v1 = (u1 + u3) * w2, v2 = u0 - u2, v3 = (u1 - u3) * w3;
		*this = Zp4(v0 + v1, v0 - v1, v2 + v3, v2 - v3);
	}

	finline void backward_4(const Zpp & wi1, const Zpp & wi2, const Zpp & wi3)
	{
		const Zpp u0 = (*this)[0], u2 = (*this)[2], u1 = (*this)[1], u3 = (*this)[3];
		const Zpp v0 = u0 + u1, v1 = Zpp(u0 - u1) * wi2, v2 = u2 + u3, v3 = Zpp(u2 - u3) * wi3;
		*this = Zp4(v0 + v2, v1 + v3, Zpp(v0 - v2) * wi1, Zpp(v1 - v3) * wi1);
	}
};

typedef Zp4<P1, P1_inv, 5> Zp4_1;
typedef Zp4<P2, P2_inv, 3> Zp4_2;
typedef Zp4<P3, P3_inv, 7> Zp4_3;

class RNS4
{
private:
	Zp4_1 _r1;
	Zp4_2 _r2;
	Zp4_3 _r3;

public:
	finline explicit RNS4() {}
	finline explicit RNS4(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3) : _r1(r1), _r2(r2), _r3(r3) {}
	finline explicit RNS4(const Zp4_1 & r1, const Zp4_2 & r2, const Zp4_3 & r3) : _r1(r1), _r2(r2), _r3(r3) {}
	finline explicit RNS4(const RNS & s0, const RNS & s1, const RNS & s2, const RNS & s3)
		: _r1(s0.r1(), s1.r1(), s2.r1(), s3.r1()), _r2(s0.r2(), s1.r2(), s2.r2(), s3.r2()), _r3(s0.r3(), s1.r3(), s2.r3(), s3.r3()) {}

	finline Zp4_1 r1() const { return _r1; }
	finline Zp4_2 r2() const { return _r2; }
	finline Zp4_3 r3() const { return _r3; }

	finline RNS4 operator-() const { return RNS4(-_r1, -_r2, -_r3); }

	finline RNS4 operator+(const RNS4 & rhs) const { return RNS4(_r1 + rhs._r1, _r2 + rhs._r2, _r3 + rhs._r3); }
	finline RNS4 operator-(const RNS4 & rhs) const { return RNS4(_r1 - rhs._r1, _r2 - rhs._r2, _r3 - rhs._r3); }
	finline RNS4 operator*(const RNS4 & rhs) const { return RNS4(_r1 * rhs._r1, _r2 * rhs._r2, _r3 * rhs._r3); }

	finline RNS4 & operator+=(const RNS4 & rhs) { *this = *this + rhs; return *this; }
	finline RNS4 & operator-=(const RNS4 & rhs) { *this = *this - rhs; return *this; }
	finline RNS4 & operator*=(const RNS4 & rhs) { *this = *this * rhs; return *this; }

	finline void forward_4(const RNS & w1, const RNS & w2, const RNS & w3)
	{
		_r1.forward_4(w1.r1(), w2.r1(), w3.r1());
		_r2.forward_4(w1.r2(), w2.r2(), w3.r2());
		_r3.forward_4(w1.r3(), w2.r3(), w3.r3());
	}

	finline void backward_4(const RNS & wi1, const RNS & wi2, const RNS & wi3)
	{
		_r1.backward_4(wi1.r1(), wi2.r1(), wi3.r1());
		_r2.backward_4(wi1.r2(), wi2.r2(), wi3.r2());
		_r3.backward_4(wi1.r3(), wi2.r3(), wi3.r3());
	}
};

class int64_4
{
private:
	__v4di _n0123;

public:
	int64_4() {}
	finline explicit int64_4(const __v4di & n0123) : _n0123(n0123) {}
	finline explicit int64_4(const int64_t n) : _n0123(__v4di(_mm256_set1_epi64x((long long)(n)))) {}
	finline explicit int64_4(const __v8si & n0123) : _n0123(__v4di(_mm256_setr_epi64x(n0123[0], n0123[2], n0123[4], n0123[6]))) {}

	finline __v4di get() const { return _n0123; }

	finline static int64_4 select(const __v4du & mask, const int64_4 & a, const int64_4 & b)
	{
		return int64_4(__v4di(_mm256_blendv_epi8(__m256i(b._n0123), __m256i(a._n0123), __m256i(mask))));
	}

	finline bool isZero() const { return (_mm256_movemask_epi8(_mm256_cmpeq_epi64(__m256i(_n0123), _mm256_setzero_si256())) == -1); }
	finline __v4du is_neg() const { return __v4du(int64_4(*this < int64_4(0)).get()); }

	finline int64_4 operator<(const int64_4 & rhs) const { return int64_4(__v4di(_n0123 < rhs._n0123)); }

	finline int64_4 operator&(const int64_4 & rhs) const { return int64_4(_n0123 & rhs._n0123); }

	finline int64_4 sign() const { return (*this < int64_4(0)) & int64_4(-1); }

	finline int64_4 operator-() const { return int64_4(-_n0123); }

	finline int64_4 operator+(const int64_4 & rhs) const { return int64_4(_n0123 + rhs._n0123); }
	finline int64_4 operator-(const int64_4 & rhs) const { return int64_4(_n0123 - rhs._n0123); }

	finline int64_4 & operator+=(const int64_4 & rhs) { *this = *this + rhs; return *this; }

	finline void rotate() { _n0123 = __v4di(_mm256_setr_epi64x(-_n0123[3], _n0123[0], _n0123[1], _n0123[2])); }
};

class uint64_4
{
private:
	__v4du _n0123;
	static constexpr __v8su mask32 = { (unsigned int)(-1), 0, (unsigned int)(-1), 0, (unsigned int)(-1), 0, (unsigned int)(-1), 0 };

public:
	uint64_4() {}
	finline explicit uint64_4(const __v4du & n0123) : _n0123(n0123) {}
	finline explicit uint64_4(const uint32_t n) : _n0123(__v4du(_mm256_set1_epi64x((long long)(n)))) {}
	finline explicit uint64_4(const uint64_t n) : _n0123(__v4du(_mm256_set1_epi64x((long long)(n)))) {}
	finline explicit uint64_4(const __v8su & n0123) : _n0123(__v4du(_mm256_and_si256(__m256i(n0123), __m256i(mask32)))) {}
	finline explicit uint64_4(const int64_4 & rhs) : _n0123(__v4du(rhs.get())) {}

	finline __v4du get() const { return _n0123; }

	finline static uint64_4 select(const __v4du & mask, const uint64_4 & a, const uint64_4 & b)
	{
		return uint64_4(__v4du(_mm256_blendv_epi8(__m256i(b._n0123), __m256i(a._n0123), __m256i(mask))));
	}

	finline int64_4 u2i() const { return int64_4(__v4di(_n0123)); }

	finline uint64_4 cast32() const { return uint64_4(__v4du(_mm256_and_si256(__m256i(_n0123), __m256i(mask32)))); }

	finline uint64_4 operator==(const uint64_4 & rhs) const { return uint64_4(__v4du(_n0123 == rhs._n0123)); }
	finline uint64_4 operator!=(const uint64_4 & rhs) const { return uint64_4(__v4du(_n0123 != rhs._n0123)); }
	finline uint64_4 operator<(const uint64_4 & rhs) const { return uint64_4(__v4du(_n0123 < rhs._n0123)); }
	finline uint64_4 operator>(const uint64_4 & rhs) const { return uint64_4(__v4du(_n0123 > rhs._n0123)); }
	finline uint64_4 operator>=(const uint64_4 & rhs) const { return uint64_4(__v4du(_n0123 >= rhs._n0123)); }

	finline uint64_4 operator&(const uint64_4 & rhs) const { return uint64_4(_n0123 & rhs._n0123); }
	finline uint64_4 operator|(const uint64_4 & rhs) const { return uint64_4(_n0123 | rhs._n0123); }

	finline uint64_4 operator-() const { return uint64_4(-_n0123); }

	finline uint64_4 operator+(const uint64_4 & rhs) const { return uint64_4(_n0123 + rhs._n0123); }
	finline uint64_4 operator-(const uint64_4 & rhs) const { return uint64_4(_n0123 - rhs._n0123); }

	finline uint64_4 operator*(const uint64_4 & rhs) const
	{
		return uint64_4(__v4du(_mm256_mul_epu32(__m256i(_n0123), __m256i(rhs._n0123))));
	}

	finline uint64_4 operator>>(const int s) const { return uint64_4(_n0123 >> s); }
	finline uint64_4 operator<<(const int s) const { return uint64_4(_n0123 << s); }

	finline static uint64_4 abs(const int64_4 & rhs, __v4du & neg)
	{
		neg = rhs.is_neg();
		return uint64_4(int64_4::select(neg, -rhs, rhs));
	}
};

class int96_4
{
private:
	uint64_4 _lo;
	int64_4 _hi;

public:
	finline int96_4(const uint64_4 & lo, const int64_4 & hi) : _lo(lo), _hi(hi) {}
	finline int96_4(const int64_t n) : _lo(uint64_t(n)), _hi((n < 0) ? int64_t(-1) : int64_t(0)) {}
	finline int96_4(const int64_4 & n) : _lo(uint64_4(n)), _hi(n.sign()) {}

	finline uint64_4 lo() const { return _lo; }
	finline int64_4 hi() const { return _hi; }
	finline int64_4 get64() const { return _lo.u2i(); }

	finline static int96_4 select(const __v4du & mask, const int96_4 & a, const int96_4 & b)
	{
		const uint64_4 lo = uint64_4::select(mask, a.lo(), b.lo());
		const int64_4 hi = int64_4::select(mask, a.hi(), b.hi());
		return int96_4(lo, hi);
	}

	finline __v4du is_neg() const { return _hi.is_neg(); }

	finline int96_4 operator-() const
	{
		const uint64_4 c = (_lo != uint64_4(0u)) & uint64_4(1u);
		return int96_4(-_lo, -_hi - c.u2i());
	}

	finline int96_4 operator+(const int96_4 & rhs) const
	{
		const uint64_4 lo = _lo + rhs._lo;
		const uint64_4 c = (lo < _lo) & uint64_4(1u);
		return int96_4(lo, _hi + rhs._hi + c.u2i());
	}

	finline int96_4 & operator+=(const int96_4 & rhs) { *this = *this + rhs; return *this; }
};

class uint96_4
{
private:
	uint64_4 _lo, _hi;

public:
	finline uint96_4(const uint64_4 & lo, const uint64_4 & hi) : _lo(lo), _hi(hi) {}
	finline uint96_4(const int96_4 & rhs) : _lo(rhs.lo()), _hi(rhs.hi()) {}

	finline uint64_4 lo() const { return _lo; }
	finline uint64_4 hi() const { return _hi; }

	finline int96_4 u2i() const { return int96_4(_lo, _hi.u2i()); }

	finline uint64_4 is_greater(const uint96_4 & rhs) const { return (_hi > rhs._hi) | ((_hi == rhs._hi) & (_lo > rhs._lo)); }

	finline uint96_4 operator&(const uint64_4 & mask) const { return uint96_4(_lo & mask, _hi & mask); }

	finline uint96_4 operator+(const uint64_4 & rhs) const
	{
		const uint64_4 lo = _lo + rhs;
		const uint64_4 c = (lo < rhs) & uint64_4(1u);
		return uint96_4(lo, _hi + c);
	}

	finline uint96_4 operator-(const uint96_4 & rhs) const
	{
		const uint64_4 c = (_lo < rhs._lo) & uint64_4(1u);
		return uint96_4(_lo - rhs._lo, _hi - rhs._hi - c);
	}

	finline static uint96_4 mul_64_32(const uint64_4 & x, const uint64_4 & y)
	{
		const uint64_4 l = x * y, h = (x >> 32) * y + (l >> 32);
		return uint96_4((h << 32) | l.cast32(), h >> 32);
	}

	finline static uint96_4 abs(const int96_4 & rhs, __v4du & neg)
	{
		neg = rhs.is_neg();
		return uint96_4(int96_4::select(neg, -rhs, rhs));
	}
};

class transformCPUi32 : public transform
{
private:
	const size_t _num_threads, _num_regs;
	const size_t _mem_size, _cache_size;
	const RNS4 _norm;
	const uint32_t _b, _b_inv;
	const int _b_s;
	RNS4 * const _z;
	RNS4 * const _wr;
	RNS4 * const _zp;

private:
	finline static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	finline static uint64_4 barrett(const uint64_4 a, const uint32_t b, const uint32_t b_inv, const int b_s, uint64_4 & a_p)
	{
		// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.
		// b < 2^31, alpha = 2^29 => a < 2^29 b
		// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31
		// b_inv = [2^{s + 32} / b]
		// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31
		// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].
		// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])
		// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,
		// 0 <= h < 1 + 1/2 + 1/2 => h = 1.

		const uint64_4 d = ((a >> b_s) * uint64_4(b_inv)) >> 32, r = a - d * uint64_4(b);
		const uint64_4 o = (r >= uint64_4(b));
		a_p = d + (o & uint64_4(1u));
		return r - (o & uint64_4(b));
	}

	finline static RNS4 reduce64(int64_4 & f, const uint32_t b, const uint32_t b_inv, const int b_s)
	{
		// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
		// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b

		__v4du s; const uint64_4 t = uint64_4::abs(f, s);
		const uint64_4 t_h = t >> 29;
		const uint64_4 t_l = t & uint64_4((uint32_t(1) << 29) - 1);

		uint64_4 d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint64_4 d_l, r_l = barrett((r_h << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64_4 d = (d_h << 29) | d_l;
		const int64_4 ri = r_l.u2i(), di = d.u2i();

		const int64_4 rs = int64_4::select(s, -ri, ri);
		f = int64_4::select(s, -di, di);

		return RNS4(Zp4_1(rs.get()), Zp4_2(rs.get()), Zp4_3(rs.get()));
	}

	finline static RNS4 reduce96(int96_4 & f, const uint32_t b, const uint32_t b_inv, const int b_s)
	{
		__v4du s; const uint96_4 t = uint96_4::abs(f, s);
		const uint64_4 t_h = (t.hi() << (64 - 29)) | (t.lo() >> 29);
		const uint64_4 t_l = t.lo() & uint64_4((uint32_t(1) << 29) - 1);

		uint64_4 d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint64_4 d_l, r_l = barrett((r_h << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64_4 d = (d_h << 29) | d_l;
		const int64_4 ri = r_l.u2i(), di = d.u2i();

		const int64_4 rs = int64_4::select(s, -ri, ri);
		f = int96_4(int64_4::select(s, -di, di));

		return RNS4(Zp4_1(rs.get()), Zp4_2(rs.get()), Zp4_3(rs.get()));
	}

	finline static int96_4 garner3(const RNS4 & s)
	{
		const Zp1 invP2_P1 = Zp1(1822724754u);		// 1 / P2 mod P1
		const Zp1 invP3_P1 = Zp1(607574918u);		// 1 / P3 mod P1
		const Zp2 invP3_P2 = Zp2(2995931465u);		// 1 / P3 mod P2
		const uint96_4 P1P2P3 = uint96_4(uint64_4(15383592652180029441ull), uint64_4(3942432002u));
		const uint96_4 P1P2P3_2 = uint96_4(uint64_4(7691796326090014720ull), uint64_4(1971216001u));

		const auto r3 = s.r3().get();
		const Zp4_1 u13 = (s.r1() - Zp4_1(r3)) * Zp4_1(invP3_P1);
		const Zp4_2 u23 = (s.r2() - Zp4_2(r3)) * Zp4_2(invP3_P2);
		const Zp4_1 u123 = (u13 - Zp4_1(u23.get())) * Zp4_1(invP2_P1);

		const uint64_4 t = uint64_4(u23.get()) * uint64_4(P3) + uint64_4(r3);
		const uint96_4 n = uint96_4::mul_64_32(uint64_4(P2 * uint64_t(P3)), uint64_4(u123.get())) + t;

		const uint96_4 r = n - (P1P2P3 & n.is_greater(P1P2P3_2));
		return r.u2i();
	}

public:
	transformCPUi32(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs) : transform(1 << n, n, b, EKind::NTT3cpu),
		_num_threads(num_threads), _num_regs(num_regs),
		_mem_size((size_t(1) << n) / 4 * (num_regs + 2) * sizeof(RNS4)), _cache_size((size_t(1) << n) / 4 * sizeof(RNS4)),
		_norm(Zp1::norm(uint32_t(1) << (n - 1)), Zp2::norm(uint32_t(1) << (n - 1)), Zp3::norm(uint32_t(1) << (n - 1))),
		_b(b), _b_inv(uint32_t((uint64_t(1) << ((int(31 - __builtin_clz(b) - 1)) + 32)) / b)), _b_s(int(31 - __builtin_clz(b) - 1)),
		_z(new RNS4[(size_t(1) << n) / 4 * num_regs]), _wr(new RNS4[2 * (size_t(1) << n) / 4]), _zp(new RNS4[(size_t(1) << n) / 4])
	{
		const size_t size_4 = (size_t(1) << n) / 4;
		RNS4 * const wr = _wr;
		RNS4 * const wri = &wr[size_4];

		for (size_t s_4 = 1; s_4 < size_4 / 2; s_4 *= 2)
		{
			const size_t s = 4 * s_4, m = 4 * s;
			const RNS prRoot_m = RNS::prRoot_n(uint32_t(m));

			for (size_t i = 0; i < s_4; ++i)
			{
				size_t e[4]; for (size_t j = 0; j < 4; ++j) e[j] = bitRev(i + j * s_4, 2 * s) + 1;
				RNS wrsi[4]; for (size_t j = 0; j < 4; ++j) wrsi[j] = prRoot_m.pow(uint32_t(e[j]));
				wr[s_4 + i] = RNS4(wrsi[0], wrsi[1], wrsi[2], wrsi[3]);
				wri[s_4 + (s_4 - i - 1)] = -RNS4(wrsi[3], wrsi[2], wrsi[1], wrsi[0]);
			}
		}

		const size_t s_4 = size_4 / 4, s = 4 * s_4, m = 4 * (2 * s);
		const RNS prRoot_m = RNS::prRoot_n(uint32_t(m));
		for (size_t i = 0; i < s_4; ++i)
		{
			size_t e[4]; for (size_t j = 0; j < 4; ++j) e[j] = bitRev(i + j * s_4, 2 * s) + 1;
			RNS wrsi[4]; for (size_t j = 0; j < 4; ++j) wrsi[j] = prRoot_m.pow(uint32_t(e[j]));
			wr[2 * s_4 + i] = RNS4(wrsi[0], wrsi[1], wrsi[2], wrsi[3]);
		}
	}

	virtual ~transformCPUi32()
	{
		delete[] _z;
		delete[] _wr;
		delete[] _zp;
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return _cache_size; }

private:
	finline static void forward_2(RNS4 * const z, const size_t k, const size_t m, const RNS4 & w1)
	{
		RNS4 & z0 = z[k + 0 * m]; RNS4 & z1 = z[k + 1 * m]; RNS4 & z2 = z[k + 2 * m]; RNS4 & z3 = z[k + 3 * m];
		const RNS4 u0 = z0, u2 = z2 * w1, u1 = z1, u3 = z3 * w1;
		z0 = u0 + u2; z2 = u0 - u2; z1 = u1 + u3; z3 = u1 - u3;
	}

	finline static void forward_4(RNS4 * const z, const size_t k, const size_t m, const RNS4 & w1, const RNS4 & w2, const RNS4 & w3)
	{
		RNS4 & z0 = z[k + 0 * m]; RNS4 & z1 = z[k + 1 * m]; RNS4 & z2 = z[k + 2 * m]; RNS4 & z3 = z[k + 3 * m];
		const RNS4 u0 = z0, u2 = z2 * w1, u1 = z1, u3 = z3 * w1;
		const RNS4 v0 = u0 + u2, v1 = (u1 + u3) * w2, v2 = u0 - u2, v3 = (u1 - u3) * w3;
		z0 = v0 + v1; z1 = v0 - v1; z2 = v2 + v3; z3 = v2 - v3;
	}

	finline static void backward_4(RNS4 * const z, const size_t k, const size_t m, const RNS4 & wi1, const RNS4 & wi2, const RNS4 & wi3)
	{
		RNS4 & z0 = z[k + 0 * m]; RNS4 & z1 = z[k + 1 * m]; RNS4 & z2 = z[k + 2 * m]; RNS4 & z3 = z[k + 3 * m];
		const RNS4 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const RNS4 v0 = u0 + u1, v1 = RNS4(u0 - u1) * wi2, v2 = u2 + u3, v3 = RNS4(u2 - u3) * wi3;
		z0 = v0 + v2; z2 = RNS4(v0 - v2) * wi1; z1 = v1 + v3; z3 = RNS4(v1 - v3) * wi1;
	}

	finline static void square_22(RNS4 * const z, const size_t k, const RNS4 & w0)
	{
		RNS4 & z0 = z[k + 0]; RNS4 & z1 = z[k + 1]; RNS4 & z2 = z[k + 2]; RNS4 & z3 = z[k + 3];
		const RNS4 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const RNS4 t1 = u1 * w0, t3 = u3 * w0;
		z0 = u0 * u0 + t1 * t1; z1 = (u0 + u0) * u1; z2 = u2 * u2 - t3 * t3; z3 = (u2 + u2) * u3;
	}

	finline static void square_4(RNS4 * const z, const size_t k, const RNS4 & w0, const RNS4 & w1, const RNS4 & wi1)
	{
		RNS4 & z0 = z[k + 0]; RNS4 & z1 = z[k + 1]; RNS4 & z2 = z[k + 2]; RNS4 & z3 = z[k + 3];
		const RNS4 u0 = z0, u2 = z2 * w1, u1 = z1, u3 = z3 * w1;
		const RNS4 v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const RNS4 t1 = v1 * w0, t3 = v3 * w0;
		const RNS4 s0 = v0 * v0 + t1 * t1, s1 = (v0 + v0) * v1;
		const RNS4 s2 = v2 * v2 - t3 * t3, s3 = (v2 + v2) * v3;
		z0 = s0 + s2; z2 = RNS4(s0 - s2) * wi1; z1 = s1 + s3; z3 = RNS4(s1 - s3) * wi1;
	}

	finline static void mul_22(RNS4 * const z, const RNS4 * const zp, const size_t k, const RNS4 & w0)
	{
		RNS4 & z0 = z[k + 0]; RNS4 & z1 = z[k + 1]; RNS4 & z2 = z[k + 2]; RNS4 & z3 = z[k + 3];
		const RNS4 u0 = z0, u1 = z1, u2 = z2, u3 = z3;
		const RNS4 & zp0 = zp[k + 0]; const RNS4 & zp1 = zp[k + 1]; const RNS4 & zp2 = zp[k + 2]; const RNS4 & zp3 = zp[k + 3];
		const RNS4 u0p = zp0, u1p = zp1, u2p = zp2, u3p = zp3;
		const RNS4 v1 = u1 * w0, v3 = u3 * w0, v1p = u1p * w0, v3p = u3p * w0;;
		z0 = u0 * u0p + v1 * v1p; z1 = u0 * u1p + u0p * u1;
		z2 = u2 * u2p - v3 * v3p; z3 = u2 * u3p + u2p * u3;
	}

	finline static void mul_4(RNS4 * const z, const RNS4 * const zp, const size_t k, const RNS4 & w0, const RNS4 & w1, const RNS4 & wi1)
	{
		RNS4 & z0 = z[k + 0]; RNS4 & z1 = z[k + 1]; RNS4 & z2 = z[k + 2]; RNS4 & z3 = z[k + 3];
		const RNS4 u0 = z0, u2 = z2 * w1, u1 = z1, u3 = z3 * w1;
		const RNS4 v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const RNS4 & zp0 = zp[k + 0]; const RNS4 & zp1 = zp[k + 1]; const RNS4 & zp2 = zp[k + 2]; const RNS4 & zp3 = zp[k + 3];
		const RNS4 v0p = zp0, v1p = zp1, v2p = zp2, v3p = zp3;
		const RNS4 t1 = v1 * w0, t3 = v3 * w0, t1p = v1p * w0, t3p = v3p * w0;
		const RNS4 s0 = v0 * v0p + t1 * t1p, s1 = v0 * v1p + v0p * v1;
		const RNS4 s2 = v2 * v2p - t3 * t3p, s3 = v2 * v3p + v2p * v3;
		z0 = s0 + s2; z2 = RNS4(s0 - s2) * wi1; z1 = s1 + s3; z3 = RNS4(s1 - s3) * wi1;
	}

	finline static void forward0(RNS4 * const z, const size_t m)
	{
		const RNS w1 = RNS(Zp1(476884782), Zp2(809539273), Zp3(-1544030998));
		const RNS w2 = RNS(Zp1(-713375721), Zp2(-1890888817), Zp3(-2001741866));
		const RNS w3 = RNS(Zp1(-1361655604), Zp2(781924903), Zp3(-339917925));
		for (size_t i = 0; i < m; ++i) z[i].forward_4(w1, w2, w3);
	}

	finline static void backward0(RNS4 * const z, const size_t m)
	{
		const RNS wi1 = RNS(Zp1(-476884782), Zp2(-809539273), Zp3(1544030998));
		const RNS wi3 = RNS(Zp1(713375721), Zp2(1890888817), Zp3(2001741866));
		const RNS wi2 = RNS(Zp1(1361655604), Zp2(-781924903), Zp3(339917925));
		for (size_t i = 0; i < m; ++i) z[i].backward_4(wi1, wi2, wi3);
	}

	finline static void forward4(RNS4 * const z, const RNS4 * const wr, const size_t m, const size_t s_4, const size_t j)
	{
		for (size_t i = 0; i < m; ++i)
		{
			forward_4(z, 4 * m * j + i, m, wr[s_4 + j], wr[2 * (s_4 + j) + 0], wr[2 * (s_4 + j) + 1]);
		}
	}

	finline static void backward4(RNS4 * const z, const RNS4 * const wri, const size_t m, const size_t s_4, const size_t j)
	{
		for (size_t i = 0; i < m; ++i)
		{
			backward_4(z, 4 * m * j + i, m, wri[s_4 + j], wri[2 * (s_4 + j) + 0], wri[2 * (s_4 + j) + 1]);
		}
	}

	finline static void forward2(RNS4 * const z, const RNS4 * const wr, const size_t m, const size_t s_4, const size_t j0)
	{
		for (size_t i = 0, j = j0; i < m; ++i, ++j) forward_2(z, 4 * j, 1, wr[s_4 + j]);
	}

	finline static void square22(RNS4 * const z, const RNS4 * const wr, const size_t m, const size_t s_4, const size_t j0)
	{
		for (size_t i = 0, j = j0; i < m; ++i, ++j) square_22(z, 4 * j, wr[2 * s_4 + j]);
	}

	finline static void square4(RNS4 * const z, const RNS4 * const wr, const RNS4 * const wri, const size_t m, const size_t s_4, const size_t j0)
	{
		for (size_t i = 0, j = j0; i < m; ++i, ++j) square_4(z, 4 * j, wr[2 * s_4 + j], wr[s_4 + j], wri[s_4 + j]);
	}

	finline static void mul22(RNS4 * const z, const RNS4 * const zp, const RNS4 * const wr, const size_t m, const size_t s_4, const size_t j0)
	{
		for (size_t i = 0, j = j0; i < m; ++i, ++j) mul_22(z, zp, 4 * j, wr[2 * s_4 + j]);
	}

	finline static void mul4(RNS4 * const z, const RNS4 * const zp, const RNS4 * const wr, const RNS4 * const wri, const size_t m, const size_t s_4, const size_t j0)
	{
		for (size_t i = 0, j = j0; i < m; ++i, ++j) mul_4(z, zp, 4 * j, wr[2 * s_4 + j], wr[s_4 + j], wri[s_4 + j]);
	}

	static void forward(RNS4 * const z, const RNS4 * const wr, const size_t mr, const size_t sr_4, const size_t jr)
	{
		if (mr >= 32)
		{
			forward4(z, wr, mr, sr_4, jr);
			for (size_t l = 0; l < 4; ++l) forward(z, wr, mr / 4, 4 * sr_4, 4 * jr + l);
		}
		else
		{
			size_t m = mr;
			for (size_t s = 1; m >= 2; m /= 4, s *= 4)
			{
				for (size_t j = 0; j < s; ++j) forward4(z, wr, m, s * sr_4, s * jr + j);
			}

			if (m == 1) forward2(z, wr, mr, mr * sr_4, mr * jr);
		}
	}

	static void square(RNS4 * const z, const RNS4 * const wr, const RNS4 * const wri, const size_t mr, const size_t sr_4, const size_t jr)
	{
		if (mr >= 32)		// 32 KB / sizeof(RNS)
		{
			forward4(z, wr, mr, sr_4, jr);
			for (size_t l = 0; l < 4; ++l) square(z, wr, wri, mr / 4, 4 * sr_4, 4 * jr + l);
			backward4(z, wri, mr, sr_4, jr);
		}
		else
		{
			size_t m = mr;
			for (size_t s = 1; m >= 2; m /= 4, s *= 4)
			{
				for (size_t j = 0; j < s; ++j) forward4(z, wr, m, s * sr_4, s * jr + j);
			}

			if (m == 0) square22(z, wr, mr, mr * sr_4, mr * jr); else square4(z, wr, wri, mr, mr * sr_4, mr * jr);

			m = (m == 0) ? 2 : 4;
			for (size_t s = mr / m; m <= mr; m *= 4, s /= 4)
			{
				for (size_t j = 0; j < s; ++j) backward4(z, wri, m, s * sr_4, s * jr + j);
			}
		}
	}

	static void mul(RNS4 * const z, const RNS4 * const zp, const RNS4 * const wr, const RNS4 * const wri, const size_t mr, const size_t sr_4, const size_t jr)
	{
		if (mr >= 32)
		{
			forward4(z, wr, mr, sr_4, jr);
			for (size_t l = 0; l < 4; ++l) mul(z, zp, wr, wri, mr / 4, 4 * sr_4, 4 * jr + l);
			backward4(z, wri, mr, sr_4, jr);
		}
		else
		{
			size_t m = mr;
			for (size_t s = 1; m >= 2; m /= 4, s *= 4)
			{
				for (size_t j = 0; j < s; ++j) forward4(z, wr, m, s * sr_4, s * jr + j);
			}

			if (m == 0) mul22(z, zp, wr, mr, mr * sr_4, mr * jr); else mul4(z, zp, wr, wri, mr, mr * sr_4, mr * jr);

			m = (m == 0) ? 2 : 4;
			for (size_t s = mr / m; m <= mr; m *= 4, s /= 4)
			{
				for (size_t j = 0; j < s; ++j) backward4(z, wri, m, s * sr_4, s * jr + j);
			}
		}
	}

	void baseMod(const size_t n, RNS4 * const z, const bool dup = false)
	{
		const RNS4 norm = _norm;
		const uint32_t b = _b, b_inv = _b_inv;
		const int b_s = _b_s;

		int96_4 f96 = int96_4(0);

		for (size_t k = 0; k < n; ++k)
		{
			int96_4 l = garner3(z[k] * norm);
			if (dup) l += l;
			f96 += l;
			z[k] = reduce96(f96, b, b_inv, b_s);
		}

		int64_4 f64 = f96.get64();

		while (!f64.isZero())
		{
			f64.rotate();	// a_0 = -a_n
			for (size_t k = 0; k < n; ++k)
			{
				const int64_4 l = int64_4(z[k].r1().getInt());
				f64 += l;
				z[k] = reduce64(f64, b, b_inv, b_s);
				if (f64.isZero()) return;
			}
		}
	}

protected:
	void getZi(int32_t * const zi) const override
	{
		const size_t size_4 = getSize() / 4;

		RNS4 * const z = _z;
		for (size_t k = 0; k < size_4; ++k)
		{
			const Zp4_1 r1 = z[k].r1();
			for (size_t i = 0; i < 4; ++i) zi[i * size_4 + k] = r1[i].getInt();
		}
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size_4 = getSize() / 4;

		RNS4 * const z = _z;
		for (size_t k = 0; k < size_4; ++k)
		{
			int32_t zik[4]; for (size_t i = 0; i < 4; ++i) zik[i] = zi[i * size_4 + k];
			const Zp4_1 r1 = Zp4_1(Zp1(zik[0]), Zp1(zik[1]), Zp1(zik[2]), Zp1(zik[3]));
			const Zp4_2 r2 = Zp4_2(Zp2(zik[0]), Zp2(zik[1]), Zp2(zik[2]), Zp2(zik[3]));
			const Zp4_3 r3 = Zp4_3(Zp3(zik[0]), Zp3(zik[1]), Zp3(zik[2]), Zp3(zik[3]));
			z[k] = RNS4(r1, r2, r3);
		}
	}

public:
	bool readContext(file & cFile, const size_t num_regs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != int(getKind())) return false;

		const size_t size_4 = getSize() / 4;
		if (!cFile.read(reinterpret_cast<char *>(_z), sizeof(RNS4) * size_4 * num_regs)) return false;
		return true;
	}

	void saveContext(file & cFile, const size_t num_regs) const override
	{
		const int kind = int(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t size_4 = getSize() / 4;
		if (!cFile.write(reinterpret_cast<const char *>(_z), sizeof(RNS4) * size_4 * num_regs)) return;
	}

	void set(const int32_t a) override
	{
		const size_t size_4 = getSize() / 4;

		RNS4 * const z = _z;
		z[0] = RNS4(Zp4_1(Zp1(a), Zp1(0), Zp1(0), Zp1(0)), Zp4_2(Zp2(a), Zp2(0), Zp2(0), Zp2(0)), Zp4_3(Zp3(a), Zp3(0), Zp3(0), Zp3(0)));
		for (size_t k = 1; k < size_4; ++k) z[k] = RNS4(Zp1(0), Zp2(0), Zp3(0));
	}

	void squareDup(const bool dup) override
	{
		const size_t size_4 = getSize() / 4;
		const RNS4 * const wr = _wr;
		RNS4 * const z = _z;

		forward0(z, size_4);
		square(z, wr, &wr[size_4], size_4 / 4, 1, 0);
		backward0(z, size_4);

		baseMod(size_4, z, dup);
	}

	void initMultiplicand(const size_t src) override
	{
		const size_t size_4 = getSize() / 4;
		const RNS4 * const z = _z;
		RNS4 * const zp = _zp;

		for (size_t k = 0; k < size_4; ++k) zp[k] = z[k + src * size_4];

		forward0(zp, size_4);
		forward(zp, _wr, size_4 / 4, 1, 0);
	}

	void mul() override
	{
		const size_t size_4 = getSize() / 4;
		const RNS4 * const wr = _wr;
		RNS4 * const z = _z;

		forward0(z, size_4);
		mul(z, _zp, wr, &wr[size_4], size_4 / 4, 1, 0);
		backward0(z, size_4);

		baseMod(size_4, z);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const size_t size_4 = getSize() / 4;
		RNS4 * const z = _z;

		for (size_t k = 0; k < size_4; ++k) z[k + dst * size_4] = z[k + src * size_4];
	}
};
