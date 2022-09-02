/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "transform.h"

template <uint32_t p, uint32_t p_inv, uint32_t prRoot>
class Zp
{
private:
	uint32_t _n;

private:
	static uint32_t _addMod(const uint32_t lhs, const uint32_t rhs)
	{
		const uint32_t c = (lhs >= p - rhs) ? p : 0;
		return lhs + rhs - c;
	}

	static uint32_t _subMod(const uint32_t lhs, const uint32_t rhs)
	{
		const uint32_t c = (lhs < rhs) ? p : 0;
		return lhs - rhs + c;
	}

	static uint32_t _mulMod(const uint32_t lhs, const uint32_t rhs)
	{
		// Improved division by invariant integers, Niels Moller and Torbjorn Granlund, Algorithm 4.
		const uint64_t m = lhs * uint64_t(rhs), q = (m >> 32) * p_inv + m;
		uint32_t r = uint32_t(m) - (1 + uint32_t(q >> 32)) * p;
		if (r > uint32_t(q)) r += p;
		return (r >= p) ? r - p : r;
	}
 
public:
	Zp() {}
	explicit Zp(const uint32_t n) : _n(n) {}
	explicit Zp(const int32_t i) : _n((i < 0) ? i + p : i) {}

	uint32_t get() const { return _n; }
	int32_t getInt() const { return (_n > p / 2) ? int32_t(_n - p) : int32_t(_n); }

	Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

	Zp operator+(const Zp & rhs) const { return Zp(_addMod(_n, rhs._n)); }
	Zp operator-(const Zp & rhs) const { return Zp(_subMod(_n, rhs._n)); }
	Zp operator*(const Zp & rhs) const { return Zp(_mulMod(_n, rhs._n)); }

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

#define P1			4253024257u		// 507 * 2^23 + 1
#define P2			4194304001u		// 125 * 2^25 + 1
#define P3			4076863489u		// 243 * 2^24 + 1

typedef Zp<P1, uint32_t(uint64_t(-1) / P1 - (uint64_t(1) << 32)), 5> Zp1;
typedef Zp<P2, uint32_t(uint64_t(-1) / P2 - (uint64_t(1) << 32)), 3> Zp2;
typedef Zp<P3, uint32_t(uint64_t(-1) / P3 - (uint64_t(1) << 32)), 7> Zp3;

class RNS
{
private:
	Zp1 _r1; Zp2 _r2; Zp3 _r3;

private:
	explicit RNS(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3) : _r1(r1), _r2(r2), _r3(r3) {}

public:
	RNS() {}
	explicit RNS(const int32_t i) { _r1 = Zp1(i); _r2 = Zp2(i); _r3 = Zp3(i); }

	Zp1 r1() const { return _r1; }
	Zp2 r2() const { return _r2; }
	Zp3 r3() const { return _r3; }

	RNS operator-() const { return RNS(-_r1, -_r2, -_r3); }

	RNS operator+(const RNS & rhs) const { return RNS(_r1 + rhs._r1, _r2 + rhs._r2, _r3 + rhs._r3); }
	RNS operator-(const RNS & rhs) const { return RNS(_r1 - rhs._r1, _r2 - rhs._r2, _r3 - rhs._r3); }
	RNS operator*(const RNS & rhs) const { return RNS(_r1 * rhs._r1, _r2 * rhs._r2, _r3 * rhs._r3); }

	RNS & operator+=(const RNS & rhs) { *this = *this + rhs; return *this; }
	RNS & operator-=(const RNS & rhs) { *this = *this - rhs; return *this; }
	RNS & operator*=(const RNS & rhs) { *this = *this * rhs; return *this; }

	RNS pow(const uint32_t e) const { return RNS(_r1.pow(e), _r2.pow(e), _r3.pow(e)); }

	static RNS norm(const uint32_t n) { return RNS(Zp1::norm(n), Zp2::norm(n), Zp3::norm(n)); }
	static const RNS prRoot_n(const uint32_t n) { return RNS(Zp1::prRoot_n(n), Zp2::prRoot_n(n), Zp3::prRoot_n(n)); }
};

typedef RNS		RNS_W;

struct uint96
{
	uint64_t s0;
	uint32_t s1;
};

struct int96
{
	uint64_t s0;
	int32_t  s1;
};

inline uint96 uint96_set(const uint64_t s0, const uint32_t s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }
inline int96 int96_set_si(const int64_t n) { int96 r; r.s0 = uint64_t(n); r.s1 = (n < 0) ? -1 : 0; return r; }

inline uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = uint32_t(x.s1); return r; }
inline int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = int32_t(x.s1); return r; }

inline bool int96_is_neg(const int96 x) { return (x.s1 < 0); }
inline bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

inline int96 int96_neg(const int96 x)
{
	const int32_t c = (x.s0 != 0) ? 1 : 0;
	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;
	return r;
}

inline int96 int96_add(const int96 x, const int96 y)
{
	const uint64_t s0 = x.s0 + y.s0;
	const int32_t c = (s0 < y.s0) ? 1 : 0;
	int96 r; r.s0 = s0; r.s1 = x.s1 + y.s1 + c;
	return r;
}

inline uint96 uint96_add_64(const uint96 x, const uint64_t y)
{
	const uint64_t s0 = x.s0 + y;
	const uint32_t c = (s0 < y) ? 1 : 0;
	uint96 r; r.s0 = s0; r.s1 = x.s1 + c;
	return r;
}

inline int96 uint96_subi(const uint96 x, const uint96 y)
{
	const uint32_t c = (x.s0 < y.s0) ? 1 : 0;
	int96 r; r.s0 = x.s0 - y.s0; r.s1 = int32_t(x.s1 - y.s1 - c);
	return r;
}

inline uint96 uint96_mul_64_32(const uint64_t x, const uint32_t y)
{
	const uint64_t l = uint64_t(uint32_t(x)) * y, h = (x >> 32) * y + (l >> 32);
	uint96 r; r.s0 = (h << 32) | uint32_t(l); r.s1 = uint32_t(h >> 32);
	return r;
}

inline uint96 int96_abs(const int96 x)
{
	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;
	return int96_u(t);
}

class transformCPUi32 : public transform
{
private:
	const size_t _num_threads, _num_regs;
	const size_t _mem_size, _cache_size;
	const RNS _norm;
	const uint32_t _b, _b_inv;
	const int _b_s;
	RNS * const _z;
	RNS_W * const _wr;
	RNS * const _zp;

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

	static uint32_t barrett(const uint64_t a, const uint32_t b, const uint32_t b_inv, const int b_s, uint32_t & a_p)
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

	static int32_t reduce64(int64_t & f, const uint32_t b, const uint32_t b_inv, const int b_s)
	{
		// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
		// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
		const uint64_t t = std::abs(f);
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

protected:
	static int32_t reduce96(int96 & f, const uint32_t b, const uint32_t b_inv, const int b_s)
	{
		const uint96 t = int96_abs(f);
		const uint64_t t_h = (uint64_t(t.s1) << (64 - 29)) | (t.s0 >> 29);
		const uint32_t t_l = uint32_t(t.s0) & ((uint32_t(1) << 29) - 1);

		uint32_t d_h, r_h = barrett(t_h, b, b_inv, b_s, d_h);
		uint32_t d_l, r_l = barrett((uint64_t(r_h) << 29) | t_l, b, b_inv, b_s, d_l);
		const uint64_t d = (uint64_t(d_h) << 29) | d_l;

		const bool s = int96_is_neg(f);
		f = int96_set_si(s ? -int64_t(d) : int64_t(d));
		const int32_t r = s ? -int32_t(r_l) : int32_t(r_l);
		return r;
	}

	static int96 garner3(const Zp1 & r1, const Zp2 & r2, const Zp3 & r3)
	{
		const uint32_t invP2_P1	= 1822724754u;		// 1 / P2 mod P1
		const uint32_t invP3_P1	= 607574918u;		// 1 / P3 mod P1
		const uint32_t invP3_P2	= 2995931465u;		// 1 / P3 mod P2
		const uint64_t P1P2P3l = 15383592652180029441ull;
		const uint32_t P1P2P3h = 3942432002u;
		const uint64_t P1P2P3_2l = 7691796326090014720ull;
		const uint32_t P1P2P3_2h = 1971216001u;

		const Zp1 u13 = (r1 - Zp1(r3.get())) * Zp1(invP3_P1);
		const Zp2 u23 = (r2 - Zp2(r3.get())) * Zp2(invP3_P2);
		const Zp1 u123 = (u13 - Zp1(u23.get())) * Zp1(invP2_P1);
		const uint96 n = uint96_add_64(uint96_mul_64_32(P2 * uint64_t(P3), u123.get()), u23.get() * uint64_t(P3) + r3.get());
		const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);
		const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);
		return r;
	}

public:
	transformCPUi32(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs) : transform(1 << n, n, b, EKind::NTT3cpu),
		_num_threads(num_threads), _num_regs(num_regs),
		_mem_size((size_t(1) << n) * (num_regs + 2) * sizeof(RNS)), _cache_size((size_t(1) << n) * sizeof(RNS)),
		_norm(RNS::norm(uint32_t(1) << n)), _b(b), _b_inv(uint32_t((uint64_t(1) << ((int(31 - __builtin_clz(b) - 1)) + 32)) / b)), _b_s(int(31 - __builtin_clz(b) - 1)),
		_z(new RNS[(size_t(1) << n) * num_regs]), _wr(new RNS_W[2 * (size_t(1) << n)]), _zp(new RNS[size_t(1) << n])
	{
		const size_t size = (size_t(1) << n);

		RNS_W * const wr = _wr;
		RNS_W * const wri = &wr[size];
		for (size_t s = 1; s < size; s *= 2)
		{
			const size_t m = 4 * s;
			const RNS prRoot_m = RNS::prRoot_n(uint32_t(m));

			for (size_t i = 0; i < s; ++i)
			{
				const size_t e = bitRev(i, 2 * s) + 1;
				const RNS wrsi = prRoot_m.pow(uint32_t(e));
				wr[s + i] = wrsi; wri[s + s - i - 1] = -wrsi;
			}
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
	static void forward(const size_t n, RNS * const z, const RNS_W * const wr)
	{
		for (size_t m = n / 2, s = 1; m >= 1; m /= 2, s *= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const RNS w = wr[s + j];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const RNS u0 = z[k + 0 * m], u1 = z[k + 1 * m] * w;
					z[k + 0 * m] = u0 + u1; z[k + 1 * m] = u0 - u1;
				}
			}
		}
	}

	static void backward(const size_t n, RNS * const z, const RNS_W * const wri)
	{
		for (size_t m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const RNS wi = wri[s + j];

				for (size_t i = 0; i < m; ++i)
				{
					const size_t k = 2 * m * j + i;
					const RNS u0 = z[k + 0 * m], u1 = z[k + 1 * m];
					z[k + 0 * m] = u0 + u1; z[k + 1 * m] = RNS(u0 - u1) * wi;
				}
			}
		}
	}

	static void mul(const size_t n, RNS * const z, const RNS * const zp)
	{
		for (size_t k = 0; k < n; ++k) z[k] *= zp[k];
	}

	void baseMod(const size_t n, RNS * const z, const bool dup = false)
	{
		const RNS norm = _norm;
		const uint32_t b = _b, b_inv = _b_inv;
		const int b_s = _b_s;

		int96 f96 = int96_set_si(0);
		for (size_t k = 0; k < n; ++k)
		{
			const RNS zn = z[k] * norm;
			int96 l = garner3(zn.r1(), zn.r2(), zn.r3());
			if (dup) l = int96_add(l, l);
			f96 = int96_add(f96, l);
			const int32_t r = reduce96(f96, b, b_inv, b_s);
			z[k] = RNS(r);
		}

		int64_t f = int64_t(f96.s0);

		while (f != 0)
		{
			f = -f;		// a_0 = -a_n
			for (size_t k = 0; k < n; ++k)
			{
				f += z[k].r1().getInt();
				const int32_t r = reduce64(f, b, b_inv, b_s);
				z[k] = RNS(r);
				if (f == 0) break;
			}
		}
	}

protected:
	void getZi(int32_t * const zi) const override
	{
		const size_t size = getSize();

		RNS * const z = _z;
		for (size_t i = 0; i < size; ++i) zi[i] = z[i].r1().getInt();
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size = getSize();

		RNS * const z = _z;
		for (size_t i = 0; i < size; ++i) z[i] = RNS(zi[i]);
	}

public:
	bool readContext(file & cFile, const size_t num_regs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != int(getKind())) return false;

		const size_t size = getSize();
		if (!cFile.read(reinterpret_cast<char *>(_z), sizeof(RNS) * size * num_regs)) return false;
		return true;
	}

	void saveContext(file & cFile, const size_t num_regs) const override
	{
		const int kind = int(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t size = getSize();
		if (!cFile.write(reinterpret_cast<const char *>(_z), sizeof(RNS) * size * num_regs)) return;
	}

	void set(const int32_t a) override
	{
		const size_t size = getSize();

		RNS * const z = _z;
		z[0] = RNS(a);
		for (size_t i = 1; i < size; ++i) z[i] = RNS(0);
	}

	void squareDup(const bool dup) override
	{
		const size_t size = getSize();
		const RNS_W * const wr = _wr;
		RNS * const z = _z;

		forward(size, z, wr);
		mul(size, z, z);
		backward(size, z, &wr[size]);
		baseMod(size, z, dup);
	}

	void initMultiplicand(const size_t src) override
	{
		const size_t size = getSize();
		const RNS * const z = _z;
		RNS * const zp = _zp;

		for (size_t k = 0; k < size; ++k) zp[k] = z[k + src * size];

		forward(size, zp, _wr);
	}

	void mul() override
	{
		const size_t size = getSize();
		const RNS_W * const wr = _wr;
		RNS * const z = _z;

		forward(size, z, wr);
		mul(size, z, _zp);
		backward(size, z, &wr[size]);
		baseMod(size, z);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const size_t size = getSize();
		RNS * const z = _z;

		for (size_t k = 0; k < size; ++k) z[k + dst * size] = z[k + src * size];
	}
};
