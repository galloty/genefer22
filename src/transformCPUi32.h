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

template <uint32_t p, uint32_t p_inv, uint32_t prRoot>
class Zp
{
private:
	uint32_t _n;

public:
	Zp() {}
	explicit Zp(const uint32_t n) : _n(n) {}
	explicit Zp(const int32_t i) : _n((i < 0) ? p - uint32_t(-i) : uint32_t(i)) {}

	uint32_t get() const { return _n; }
	int32_t getInt() const { return (_n > p / 2) ? int32_t(_n - p) : int32_t(_n); }

	Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

	Zp operator+(const Zp & rhs) const
	{
		const uint32_t c = (_n >= p - rhs._n) ? p : 0;
		return Zp(_n + rhs._n - c);
	}

	Zp operator-(const Zp & rhs) const
	{
		const uint32_t c = (_n < rhs._n) ? p : 0;
		return Zp(_n - rhs._n + c);
	}

	Zp operator*(const Zp & rhs) const
	{
		return Zp(uint32_t((_n * uint64_t(rhs._n)) % p));
	}

	Zp & operator+=(const Zp & rhs) { *this = *this + rhs; return *this; }
	Zp & operator-=(const Zp & rhs) { *this = *this - rhs; return *this; }
	Zp & operator*=(const Zp & rhs) { *this = *this * rhs; return *this; }

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

template <uint32_t p, uint32_t p_inv, uint32_t prRoot>
class Zp4
{
	using Zpp = Zp<p, p_inv, prRoot>;

private:
	Zpp _z[4];

public:
	Zp4() {}
	// explicit Zp4(const uint32_t n[4]) { for (size_t i = 0; i < 4; ++i) _n[i] = n[i]; }
	explicit Zp4(const Zpp & z)
	{
		_z[0] = _z[1] = _z[2] = _z[3] = z;
	}
	explicit Zp4(const Zpp & z0, const Zpp & z1, const Zpp & z2, const Zpp & z3)
	{
		_z[0] = z0; _z[1] = z1; _z[2] = z2; _z[3] = z3;
	}

	// explicit Zp4(const int32_t i) : _n((i < 0) ? p - uint32_t(-i) : uint32_t(i)) {}

	finline Zpp operator[](const size_t i) const { return _z[i]; }

	// uint32_t get() const { return _n; }
	// int32_t getInt() const { return (_n > p / 2) ? int32_t(_n - p) : int32_t(_n); }

	Zp4 operator-() const { return Zp4(-_z[0], -_z[1], -_z[2], -_z[3]); }

	Zp4 operator+(const Zp4 & rhs) const { return Zp4(_z[0] + rhs._z[0], _z[1] + rhs._z[1], _z[2] + rhs._z[2], _z[3] + rhs._z[3]); }
	Zp4 operator-(const Zp4 & rhs) const { return Zp4(_z[0] - rhs._z[0], _z[1] - rhs._z[1], _z[2] - rhs._z[2], _z[3] - rhs._z[3]); }
	Zp4 operator*(const Zp4 & rhs) const { return Zp4(_z[0] * rhs._z[0], _z[1] * rhs._z[1], _z[2] * rhs._z[2], _z[3] * rhs._z[3]); }

	Zp4 & operator+=(const Zp4 & rhs) { *this = *this + rhs; return *this; }
	Zp4 & operator-=(const Zp4 & rhs) { *this = *this - rhs; return *this; }
	Zp4 & operator*=(const Zp4 & rhs) { *this = *this * rhs; return *this; }
};

#define P1		4253024257u		// 507 * 2^23 + 1
#define P2		4194304001u		// 125 * 2^25 + 1
#define P3		4076863489u		// 243 * 2^24 + 1
#define P1_inv	(uint64_t(-1) / P1 - (uint64_t(1) << 32))
#define P2_inv	(uint64_t(-1) / P2 - (uint64_t(1) << 32))
#define P3_inv	(uint64_t(-1) / P3 - (uint64_t(1) << 32))

typedef Zp<P1, P1_inv, 5> Zp1;
typedef Zp<P2, P2_inv, 3> Zp2;
typedef Zp<P3, P3_inv, 7> Zp3;

typedef Zp4<P1, P1_inv, 5> Zp4_1;
typedef Zp4<P2, P2_inv, 3> Zp4_2;
typedef Zp4<P3, P3_inv, 7> Zp4_3;

class RNS
{
private:
	__v8su _r123;
	static constexpr __v8su P123 = { P1, 0, P2, 0, P3, 0, 0, 0 };
	static constexpr __v8su P123_inv = { P1_inv, 0, P2_inv, 0, P3_inv, 0, 0, 0 };
	static constexpr __v8su P123_2 = { P1 / 2, 0, P2 / 2, 0, P3 / 2, 0, 0, 0 };

private:
	constexpr explicit RNS(const __v8su & r123) : _r123(r123) {}

public:
	RNS() {}
	explicit RNS(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3) { _r123[0] = r1.get(); _r123[2] = r2.get(); _r123[4] = r3.get(); }
	explicit RNS(const int32_t i)
	{
		_r123 = (__v8su)_mm256_set1_epi32(i);
		_r123 += (__v8su)_mm256_set1_epi32((i < 0) ? -1 : 0) & P123;
	}

	Zp1 r1() const { return Zp1(_r123[0]); }
	Zp2 r2() const { return Zp2(_r123[2]); }
	Zp3 r3() const { return Zp3(_r123[4]); }

	RNS operator-() const { return RNS((_r123 != 0) & (P123 - _r123)); }

	RNS operator+(const RNS & rhs) const
	{
		const __v8su c = (_r123 >= P123 - rhs._r123) & P123;
		return RNS(_r123 + rhs._r123 - c);
	}

	RNS operator-(const RNS & rhs) const
	{
		const __v8su c = (_r123 < rhs._r123) & P123;
		return RNS(_r123 - rhs._r123 + c);
	}

	RNS operator*(const RNS & rhs) const
	{
		// const uint64_t m = _n * uint64_t(rhs._n)
		const __v8su m = (__v8su)_mm256_mul_epu32((__m256i)_r123, (__m256i)rhs._r123);

		// uint64_t q = uint32_t(m >> 32) * uint64_t(p_inv) + m;
		const __v8si mask_32 = { 1, 1, 3, 3, 5, 5, 7, 7 };
		const __v8su m_32 = __builtin_shuffle(m, mask_32);
		const __v8su q = (__v8su)_mm256_add_epi64(_mm256_mul_epu32((__m256i)m_32, (__m256i)P123_inv), (__m256i)m);

		// uint32_t r = uint32_t(m) - (1 + uint32_t(q >> 32)) * p;
		const __v8su q_32 = __builtin_shuffle(q, mask_32);
		__v8su r = m - q_32 * P123 - P123;

		// if (r > uint32_t(q)) r += p;
		r += (r > q) & P123;
		// if (r >= p) r -= p ;
		r -= (r >= P123) & P123;

		return RNS(r);
	}

	RNS & operator+=(const RNS & rhs) { *this = *this + rhs; return *this; }
	RNS & operator-=(const RNS & rhs) { *this = *this - rhs; return *this; }
	RNS & operator*=(const RNS & rhs) { *this = *this * rhs; return *this; }

	RNS pow(const uint32_t e) const { return RNS(r1().pow(e), r2().pow(e), r3().pow(e)); }

	static RNS norm(const uint32_t n) { return RNS(Zp1::norm(n), Zp2::norm(n), Zp3::norm(n)); }
	static const RNS prRoot_n(const uint32_t n) { return RNS(Zp1::prRoot_n(n), Zp2::prRoot_n(n), Zp3::prRoot_n(n)); }
};

class RNS4
{
private:
	Zp4_1 _z1;
	Zp4_2 _z2;
	Zp4_3 _z3;

private:
	// constexpr explicit RNS(const __v8su & r123) : _r123(r123) {}
	explicit RNS4(const Zp4_1 & z1, const Zp4_2 & z2, const Zp4_3 & z3) : _z1(z1), _z2(z2), _z3(z3) {}

public:
	finline explicit RNS4() {}
	finline explicit RNS4(const RNS & s)
	{
		_z1 = Zp4_1(s.r1());
		_z2 = Zp4_2(s.r2());
		_z3 = Zp4_3(s.r3());
	}
	finline explicit RNS4(const RNS & s0, const RNS & s1, const RNS & s2, const RNS & s3)
	{
		_z1 = Zp4_1(s0.r1(), s1.r1(), s2.r1(), s3.r1());
		_z2 = Zp4_2(s0.r2(), s1.r2(), s2.r2(), s3.r2());
		_z3 = Zp4_3(s0.r3(), s1.r3(), s2.r3(), s3.r3());
	}
	// explicit RNS(const int32_t i)
	// {
	// 	_r123 = (__v8su)_mm256_set1_epi32(i);
	// 	_r123 += (__v8su)_mm256_set1_epi32((i < 0) ? -1 : 0) & P123;
	// }

	finline RNS operator[](const size_t i) const
	{
		return RNS(_z1[i], _z2[i], _z3[i]);
	}
	// Zp1 r1() const { return Zp1(_r123[0]); }
	// Zp2 r2() const { return Zp2(_r123[2]); }
	// Zp3 r3() const { return Zp3(_r123[4]); }

	RNS4 operator-() const { return RNS4(-_z1, -_z2, -_z3); }

	RNS4 operator+(const RNS4 & rhs) const { return RNS4(_z1 + rhs._z1, _z2 + rhs._z2, _z3 + rhs._z3); }
	RNS4 operator-(const RNS4 & rhs) const { return RNS4(_z1 - rhs._z1, _z2 - rhs._z2, _z3 - rhs._z3); }
	RNS4 operator*(const RNS4 & rhs) const { return RNS4(_z1 * rhs._z1, _z2 * rhs._z2, _z3 * rhs._z3); }

	RNS4 & operator+=(const RNS4 & rhs) { *this = *this + rhs; return *this; }
	RNS4 & operator-=(const RNS4 & rhs) { *this = *this - rhs; return *this; }
	RNS4 & operator*=(const RNS4 & rhs) { *this = *this * rhs; return *this; }

	finline void forward_4(const RNS & w1, const RNS & w2, const RNS & w3)
	{
		const RNS u0 = (*this)[0], u2 = (*this)[2] * w1, u1 = (*this)[1], u3 = (*this)[3] * w1;
		const RNS v0 = u0 + u2, v1 = (u1 + u3) * w2, v2 = u0 - u2, v3 = (u1 - u3) * w3;
		*this = RNS4(v0 + v1, v0 - v1, v2 + v3, v2 - v3);
	}

	finline void backward_4(const RNS & wi1, const RNS & wi2, const RNS & wi3)
	{
		const RNS u0 = (*this)[0], u2 = (*this)[2], u1 = (*this)[1], u3 = (*this)[3];
		const RNS v0 = u0 + u1, v1 = RNS(u0 - u1) * wi2, v2 = u2 + u3, v3 = RNS(u2 - u3) * wi3;
		*this = RNS4(v0 + v2, v1 + v3, RNS(v0 - v2) * wi1, RNS(v1 - v3) * wi1);
	}
};


class int96
{
private:
	uint64_t _lo;
	int32_t  _hi;

public:
	int96() {}
	int96(const uint64_t lo, const int32_t hi) : _lo(lo), _hi(hi) {}
	int96(const int64_t n) : _lo(uint64_t(n)), _hi((n < 0) ? -1 : 0) {}

	uint64_t lo() const { return _lo; }
	int32_t hi() const { return _hi; }

	bool is_neg() const { return (_hi < 0); }

	int96 operator-() const
	{
		const int32_t c = (_lo != 0) ? 1 : 0;
		return int96(-_lo, -_hi - c);
	}

	int96 & operator+=(const int96 & rhs)
	{
		const uint64_t lo = _lo + rhs._lo;
		const int32_t c = (lo < rhs._lo) ? 1 : 0;
		_lo = lo; _hi += rhs._hi + c;
		return *this;
	}
};

class uint96
{
private:
	uint64_t _lo;
	uint32_t _hi;

public:
	uint96(const uint64_t lo, const uint32_t hi) : _lo(lo), _hi(hi) {}
	uint96(const int96 & rhs) : _lo(rhs.lo()), _hi(uint32_t(rhs.hi())) {}

	uint64_t lo() const { return _lo; }
	uint32_t hi() const { return _hi; }

	int96 u2i() const { return int96(_lo, int32_t(_hi)); }

	bool is_greater(const uint96 & rhs) const { return (_hi > rhs._hi) || ((_hi == rhs._hi) && (_lo > rhs._lo)); }

	uint96 operator+(const uint64_t & rhs) const
	{
		const uint64_t lo = _lo + rhs;
		const uint32_t c = (lo < rhs) ? 1 : 0;
		return uint96(lo, _hi + c);
	}

	int96 operator-(const uint96 & rhs) const
	{
		const uint32_t c = (_lo < rhs._lo) ? 1 : 0;
		return int96(_lo - rhs._lo, int32_t(_hi - rhs._hi - c));
	}

	static uint96 mul_64_32(const uint64_t x, const uint32_t y)
	{
		const uint64_t l = uint64_t(uint32_t(x)) * y, h = (x >> 32) * y + (l >> 32);
		return uint96((h << 32) | uint32_t(l), uint32_t(h >> 32));
	}

	static uint96 abs(const int96 & rhs)
	{
		return uint96(rhs.is_neg() ? -rhs : rhs);
	}
};

class int96_4
{
private:
	int96 _n[4];

public:
	// int96_4() {}
	// int96(const uint64_t lo, const int32_t hi) : _lo(lo), _hi(hi) {}

	finline explicit int96_4(const int96 & n0, const int96 & n1, const int96 & n2, const int96 & n3)
	{
		_n[0] = n0; _n[1] = n1; _n[2] = n2; _n[3] = n3;
	}
	int96_4(const int64_t n)
	{
		for (size_t i = 0; i < 4; ++i) _n[i] = int96(n);
	}

	// uint64_t lo() const { return _lo; }
	// int32_t hi() const { return _hi; }

	finline int96 & operator[](const size_t i) { return _n[i]; }

	// bool is_neg() const { return (_hi < 0); }

	// int96 operator-() const
	// {
	// 	const int32_t c = (_lo != 0) ? 1 : 0;
	// 	return int96(-_lo, -_hi - c);
	// }

	int96_4 & operator+=(const int96_4 & rhs)
	{
		for (size_t i = 0; i < 4; ++i) _n[i] += rhs._n[i];
		return *this;
	}
};

class transformCPUi32 : public transform
{
private:
	const size_t _num_threads, _num_regs;
	const size_t _mem_size, _cache_size;
	const RNS _norm;
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

	finline static uint32_t barrett(const uint64_t a, const uint32_t b, const uint32_t b_inv, const int b_s, uint32_t & a_p)
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

		const uint32_t d = uint32_t((uint32_t(a >> b_s) * uint64_t(b_inv)) >> 32), r = uint32_t(a) - d * b;
		const bool o = (r >= b);
		a_p = o ? d + 1 : d;
		return o ? r - b : r;
	}

	finline static int32_t reduce64(int64_t & f, const uint32_t b, const uint32_t b_inv, const int b_s)
	{
		// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
		// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
		const uint64_t t = uint64_t(std::abs(f));
		const uint64_t t_h = t >> 29;
		const uint32_t t_l = uint32_t(t) & ((uint32_t(1) << 29) - 1);

		uint32_t d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32_t d_l, r_l = barrett((uint64_t(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64_t d = (uint64_t(d_h) << 29) | d_l;

		const bool s = (f < 0);
		f = s ? -int64_t(d) : int64_t(d);
		const int32_t r = s ? -int32_t(r_l) : int32_t(r_l);
		return r;
	}

	finline static int32_t reduce96(int96 & f, const uint32_t b, const uint32_t b_inv, const int b_s)
	{
		const uint96 t = uint96::abs(f);
		const uint64_t t_h = (uint64_t(t.hi()) << (64 - 29)) | (t.lo() >> 29);
		const uint32_t t_l = uint32_t(t.lo()) & ((uint32_t(1) << 29) - 1);

		uint32_t d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32_t d_l, r_l = barrett((uint64_t(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64_t d = (uint64_t(d_h) << 29) | d_l;

		const bool s = f.is_neg();
		f = int96(s ? -int64_t(d) : int64_t(d));
		const int32_t r = s ? -int32_t(r_l) : int32_t(r_l);
		return r;
	}

	finline static int96 garner3(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3)
	{
		const uint32_t invP2_P1	= 1822724754u;		// 1 / P2 mod P1
		const uint32_t invP3_P1	= 607574918u;		// 1 / P3 mod P1
		const uint32_t invP3_P2	= 2995931465u;		// 1 / P3 mod P2
		const uint64_t P1P2P3l = 15383592652180029441ull;
		const uint32_t P1P2P3h = 3942432002u;
		const uint64_t P1P2P3_2l = 7691796326090014720ull;
		const uint32_t P1P2P3_2h = 1971216001u;

		const uint32_t r3i = r3.get();
		const Zp1 u13 = (r1 - Zp1(r3i)) * Zp1(invP3_P1);
		const Zp2 u23 = (r2 - Zp2(r3i)) * Zp2(invP3_P2);
		const uint32_t u23i = u23.get();
		const Zp1 u123 = (u13 - Zp1(u23i)) * Zp1(invP2_P1);
		const uint96 n = uint96::mul_64_32(P2 * uint64_t(P3), u123.get()) + (u23i * uint64_t(P3) + r3i);
		const uint96 P1P2P3 = uint96(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96(P1P2P3_2l, P1P2P3_2h);
		const int96 r = n.is_greater(P1P2P3_2) ? (n - P1P2P3) : n.u2i();
		return r;
	}

public:
	transformCPUi32(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs) : transform(1 << n, n, b, EKind::NTT3cpu),
		_num_threads(num_threads), _num_regs(num_regs),
		_mem_size((size_t(1) << n) / 4 * (num_regs + 2) * sizeof(RNS4)), _cache_size((size_t(1) << n) / 4 * sizeof(RNS4)),
		_norm(RNS::norm(uint32_t(1) << (n - 1))), _b(b), _b_inv(uint32_t((uint64_t(1) << ((int(31 - __builtin_clz(b) - 1)) + 32)) / b)), _b_s(int(31 - __builtin_clz(b) - 1)),
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
		const RNS norm = _norm;
		const uint32_t b = _b, b_inv = _b_inv;
		const int b_s = _b_s;

		int96_4 f96 = int96_4(0);

		for (size_t k = 0; k < n; ++k)
		{
			RNS zk[4]; for (size_t i = 0; i < 4; ++i) zk[i] = z[k][i];
			int96_4 l = int96_4(
				garner3(zk[0].r1() * norm.r1(), zk[0].r2() * norm.r2(), zk[0].r3() * norm.r3()),
				garner3(zk[1].r1() * norm.r1(), zk[1].r2() * norm.r2(), zk[1].r3() * norm.r3()),
				garner3(zk[2].r1() * norm.r1(), zk[2].r2() * norm.r2(), zk[2].r3() * norm.r3()),
				garner3(zk[3].r1() * norm.r1(), zk[3].r2() * norm.r2(), zk[3].r3() * norm.r3())
			);
			if (dup) l += l;
			f96 += l;
			int32_t r[4];
			for (size_t i = 0; i < 4; ++i) r[i] = reduce96(f96[i], b, b_inv, b_s);
			z[k] = RNS4(RNS(r[0]), RNS(r[1]), RNS(r[2]), RNS(r[3]));
		}

		int64_t f[4];
		for (size_t i = 0; i < 4; ++i) f[i] = int64_t(f96[i].lo());

		while (!((f[0] == 0) & (f[1] == 0) & (f[2] == 0) & (f[3] == 0)))
		{
			const int64_t t = f[3]; f[3] = f[2]; f[2] = f[1]; f[1] = f[0]; f[0] = -t;	// a_0 = -a_n
			for (size_t k = 0; k < n; ++k)
			{
				RNS zk[4]; for (size_t i = 0; i < 4; ++i) zk[i] = z[k][i];
				for (size_t i = 0; i < 4; ++i) f[i] += zk[i].r1().getInt();
				int32_t r[4];
				for (size_t i = 0; i < 4; ++i) r[i] = reduce64(f[i], b, b_inv, b_s);
				z[k] = RNS4(RNS(r[0]), RNS(r[1]), RNS(r[2]), RNS(r[3]));
				if ((f[0] == 0) & (f[1] == 0) & (f[2] == 0) & (f[3] == 0)) return;
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
			const RNS4 zk = z[k];
			zi[0 * size_4 + k] = zk[0].r1().getInt();
			zi[1 * size_4 + k] = zk[1].r1().getInt();
			zi[2 * size_4 + k] = zk[2].r1().getInt();
			zi[3 * size_4 + k] = zk[3].r1().getInt();
		}
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size_4 = getSize() / 4;

		RNS4 * const z = _z;
		for (size_t k = 0; k < size_4; ++k)
		{
			z[k] = RNS4(RNS(zi[0 * size_4 + k]), RNS(zi[1 * size_4 + k]), RNS(zi[2 * size_4 + k]), RNS(zi[3 * size_4 + k]));
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
		z[0] = RNS4(RNS(a), RNS(0), RNS(0), RNS(0));
		for (size_t k = 1; k < size_4; ++k) z[k] = RNS4(RNS(0), RNS(0), RNS(0), RNS(0));
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
