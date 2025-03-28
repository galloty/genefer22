/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#if __OPENCL_VERSION__ >= 120
	#define INLINE	static inline
#else
	#define INLINE
#endif

#ifndef NSIZE
#define NSIZE		65536
#define	LNSZ		16
#define	SNORM31		16
#define	NORM1		2130673921u
#define	NORM2		2112897025u
#define	WOFFSET_1	98304
#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1
#define CHUNK64		4
#define CHUNK256	2
#define CHUNK1024	1
#define MAX_WORK_GROUP_SIZE	256
#endif

typedef uint	sz_t;
typedef uint	uint32;
typedef int		int32;
typedef ulong	uint64;
typedef long	int64;
typedef uint2	uint32_2;
typedef int2	int32_2;
typedef ulong2	uint64_2;
typedef long2	int64_2;

INLINE uint2 swap(const uint2 lhs) { return (uint2)(lhs.s1, lhs.s0); }

INLINE uint32 _addmod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 t = lhs + rhs;
	return t - ((t >= p) ? p : 0);
}

INLINE uint32 _submod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 t = lhs - rhs;
	return t + (((int32)(t) < 0) ? p : 0);
}

INLINE uint32 _mulmod(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 q)
{
	const uint64 t = lhs * (uint64)(rhs);
	const uint32 lo = (uint32)(t), hi = (uint32)(t >> 32);
	const uint32 mp = mul_hi(lo * q, p);
	return _submod(hi, mp, p);
}

INLINE int32 _get_int(const uint32 n, const uint32 p) { return (n >= p / 2) ? (int32)(n - p) : (int32)(n); }
INLINE uint32 _set_int(const int32 i, const uint32 p) { return (i < 0) ? ((uint32)(i) + p) : (uint32)(i); }

// --- Z/(2^31 - 1)Z ---

#define M31		0x7fffffffu

INLINE uint32 _add31(const uint32 lhs, const uint32 rhs) { return _addmod(lhs, rhs, M31); }
INLINE uint32 _sub31(const uint32 lhs, const uint32 rhs) { return _submod(lhs, rhs, M31); }

INLINE uint32 _mul31(const uint32 lhs, const uint32 rhs)
{
	const uint64 t = lhs * (uint64)(rhs);
	const uint32 lo = (uint32)(t) & M31, hi = (uint32)(t >> 31);
	return _add31(lo, hi);
}

INLINE uint32 _lshift31(const uint32 lhs, const int s)
{
	const uint64 t = (uint64)(lhs) << s;
	const uint32 lo = (uint32)(t) & M31, hi = (uint32)(t >> 31);
	return _add31(hi, lo);
}

INLINE int32 _get_int31(const uint32 n) { return _get_int(n, M31); }
INLINE uint32 _set_int31(const int32 i) { return _set_int(i, M31); }

// --- GF((2^31 - 1)^2) ---

typedef uint2	GF31;

INLINE int32_2 get_int31(const GF31 n) { return (int32_2)(_get_int31(n.s0), _get_int31(n.s1)); }
INLINE GF31 set_int31(const int32_2 i) { return (GF31)(_set_int31(i.s0), _set_int31(i.s1)); }

INLINE GF31 add31(const GF31 lhs, const GF31 rhs) { return (GF31)(_add31(lhs.s0, rhs.s0), _add31(lhs.s1, rhs.s1)); }
INLINE GF31 sub31(const GF31 lhs, const GF31 rhs) { return (GF31)(_sub31(lhs.s0, rhs.s0), _sub31(lhs.s1, rhs.s1)); }
INLINE GF31 addi31(const GF31 lhs, const GF31 rhs)  { return (GF31)(_sub31(lhs.s0, rhs.s1), _add31(lhs.s1, rhs.s0)); }
INLINE GF31 subi31(const GF31 lhs, const GF31 rhs)  { return (GF31)(_add31(lhs.s0, rhs.s1), _sub31(lhs.s1, rhs.s0)); }

INLINE GF31 lshift31(const GF31 lhs, const int s) { return (GF31)(_lshift31(lhs.s0, s), _lshift31(lhs.s1, s)); }
INLINE GF31 muls31(const GF31 lhs, const uint32 s) { return (GF31)(_mul31(lhs.s0, s), _mul31(lhs.s1, s)); }

INLINE GF31 mul31(const GF31 lhs, const GF31 rhs)
{
	return (GF31)(_sub31(_mul31(lhs.s0, rhs.s0), _mul31(lhs.s1, rhs.s1)), _add31(_mul31(lhs.s1, rhs.s0), _mul31(lhs.s0, rhs.s1)));
}
INLINE GF31 mulconj31(const GF31 lhs, const GF31 rhs)
{
	return (GF31)(_add31(_mul31(lhs.s0, rhs.s0), _mul31(lhs.s1, rhs.s1)), _sub31(_mul31(lhs.s1, rhs.s0), _mul31(lhs.s0, rhs.s1)));
}
INLINE GF31 sqr31(const GF31 lhs)
{
	const uint32 t = _mul31(lhs.s0, lhs.s1); return (GF31)(_sub31(_mul31(lhs.s0, lhs.s0), _mul31(lhs.s1, lhs.s1)), _add31(t, t));
}

// --- Z/(127*2^24 + 1)Z ---

#define P1		2130706433u
#define Q1		2164260865u		// p * q = 1 (mod 2^32)
#define RSQ1	402124772u		// (2^32)^2 mod p
#define I1		66976762u	 	// Montgomery form of 5^{(p - 1)/4} = 16711679
#define IM1		200536044u		// Montgomery form of I1 to convert input into Montgomery form

INLINE uint32 _add1(const uint32 lhs, const uint32 rhs) { return _addmod(lhs, rhs, P1); }
INLINE uint32 _sub1(const uint32 lhs, const uint32 rhs) { return _submod(lhs, rhs, P1); }
INLINE uint32 _mul1(const uint32 lhs, const uint32 rhs) { return _mulmod(lhs, rhs, P1, Q1); }
INLINE int32 _get_int1(const uint32 n) { return _get_int(n, P1); }
INLINE uint32 _set_int1(const int32 i) { return _set_int(i, P1); }

// --- pair of Z/p1Z ---

typedef uint2	Zp1;

INLINE int32_2 get_int1(const Zp1 n) { return (int32_2)(_get_int1(n.s0), _get_int1(n.s1)); }
INLINE Zp1 set_int1(const int32_2 i) { return (Zp1)(_set_int1(i.s0), _set_int1(i.s1)); }

INLINE Zp1 add1(const Zp1 lhs, const Zp1 rhs) { return (Zp1)(_add1(lhs.s0, rhs.s0), _add1(lhs.s1, rhs.s1)); }
INLINE Zp1 sub1(const Zp1 lhs, const Zp1 rhs) { return (Zp1)(_sub1(lhs.s0, rhs.s0), _sub1(lhs.s1, rhs.s1)); }

INLINE Zp1 muls1(const Zp1 lhs, const uint32 s) { return (Zp1)(_mul1(lhs.s0, s), _mul1(lhs.s1, s)); }

INLINE Zp1 mul1(const Zp1 lhs, const Zp1 rhs) { return (Zp1)(_mul1(lhs.s0, rhs.s0), _mul1(lhs.s1, rhs.s1)); }
INLINE Zp1 sqr1(const Zp1 lhs) { return mul1(lhs, lhs); }

INLINE Zp1 forward2_1(const Zp1 lhs)
{
	const uint32 u0 = _mul1(lhs.s0, RSQ1), u1 = _mul1(lhs.s1, IM1);
	return (Zp1)(_add1(u0, u1), _sub1(u0, u1));
}

INLINE Zp1 backward2_1(const Zp1 lhs)
{
	const uint32 u0 = lhs.s0, u1 = lhs.s1;
	return (Zp1)(_add1(u0, u1), _mul1(_sub1(u1, u0), I1));
}

// --- Z/(63*2^25 + 1)Z ---

#define P2		2113929217u
#define Q2		2181038081u
#define RSQ2	2111798781u
#define I2		530075385u
#define IM2		1036950657u

INLINE uint32 _add2(const uint32 lhs, const uint32 rhs) { return _addmod(lhs, rhs, P2); }
INLINE uint32 _sub2(const uint32 lhs, const uint32 rhs) { return _submod(lhs, rhs, P2); }
INLINE uint32 _mul2(const uint32 lhs, const uint32 rhs) { return _mulmod(lhs, rhs, P2, Q2); }
INLINE int32 _get_int2(const uint32 n) { return _get_int(n, P2); }
INLINE uint32 _set_int2(const int32 i) { return _set_int(i, P2); }

// --- pair of Z/p2Z ---

typedef uint2	Zp2;

INLINE int32_2 get_int2(const Zp2 n) { return (int32_2)(_get_int2(n.s0), _get_int2(n.s1)); }
INLINE Zp2 set_int2(const int32_2 i) { return (Zp2)(_set_int2(i.s0), _set_int2(i.s1)); }

INLINE Zp2 add2(const Zp2 lhs, const Zp2 rhs) { return (Zp2)(_add2(lhs.s0, rhs.s0), _add2(lhs.s1, rhs.s1)); }
INLINE Zp2 sub2(const Zp2 lhs, const Zp2 rhs) { return (Zp2)(_sub2(lhs.s0, rhs.s0), _sub2(lhs.s1, rhs.s1)); }

INLINE Zp2 muls2(const Zp2 lhs, const uint32 s) { return (Zp2)(_mul2(lhs.s0, s), _mul2(lhs.s1, s)); }

INLINE Zp2 mul2(const Zp2 lhs, const Zp2 rhs) { return (Zp2)(_mul2(lhs.s0, rhs.s0), _mul2(lhs.s1, rhs.s1)); }
INLINE Zp2 sqr2(const Zp2 lhs) { return mul2(lhs, lhs); }

INLINE Zp2 forward2_2(const Zp2 lhs)
{
	const uint32 u0 = _mul2(lhs.s0, RSQ2), u1 = _mul2(lhs.s1, IM2);
	return (Zp2)(_add2(u0, u1), _sub2(u0, u1));
}

INLINE Zp2 backward2_2(const Zp2 lhs)
{
	const uint32 u0 = lhs.s0, u1 = lhs.s1;
	return (Zp2)(_add2(u0, u1), _mul2(_sub2(u1, u0), I2));
}

// --- transform/macro GF31 ---

// 12 mul + 12 mul_hi
#define FORWARD_4_31(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w2, w3) \
	{ \
	const GF31 u0 = zi0, u2 = mul31(zi2, w1), u1 = mul31(zi1, w2), u3 = mul31(zi3, w3); \
	const GF31 v0 = add31(u0, u2), v2 = sub31(u0, u2), v1 = add31(u1, u3), v3 = sub31(u1, u3); \
	zo0 = add31(v0, v1); zo1 = sub31(v0, v1); zo2 = addi31(v2, v3); zo3 = subi31(v2, v3); \
	}

#define BACKWARD_4_31(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w2, w3) \
	{ \
	const GF31 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \
	const GF31 v0 = add31(u0, u1), v1 = sub31(u0, u1), v2 = add31(u2, u3), v3 = sub31(u3, u2); \
	zo0 = add31(v0, v2); zo2 = mulconj31(sub31(v0, v2), w1); \
	zo1 = mulconj31(addi31(v1, v3), w2); zo3 = mulconj31(subi31(v1, v3), w3); \
	}

#define SQUARE_22_31(z0, z1, z2, z3, w) \
	{ \
	const GF31 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add31(sqr31(u0), mul31(sqr31(u1), w)); z1 = mul31(add31(u0, u0), u1); \
	z2 = sub31(sqr31(u2), mul31(sqr31(u3), w)); z3 = mul31(add31(u2, u2), u3); \
	}

#define SQUARE_4_31(z0, z1, z2, z3, w, win) \
	{ \
	const GF31 u0 = z0, u2 = mul31(z2, w), u1 = z1, u3 = mul31(z3, w); \
	const GF31 v0 = add31(u0, u2), v2 = sub31(u0, u2), v1 = add31(u1, u3), v3 = sub31(u1, u3); \
	const GF31 s0 = add31(sqr31(v0), mul31(sqr31(v1), w)), s1 = mul31(add31(v0, v0), v1); \
	const GF31 s2 = sub31(sqr31(v2), mul31(sqr31(v3), w)), s3 = mul31(add31(v2, v2), v3); \
	z0 = add31(s0, s2); z2 = mulconj31(sub31(s0, s2), w); \
	z1 = add31(s1, s3); z3 = mulconj31(sub31(s1, s3), w); \
	}

#define FWD_2_31(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \
	{ \
	const GF31 u0 = zi0, u2 = mul31(zi2, w), u1 = zi1, u3 = mul31(zi3, w); \
	zo0 = add31(u0, u2); zo2 = sub31(u0, u2); zo1 = add31(u1, u3); zo3 = sub31(u1, u3); \
	}

#define MUL_22_31(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
	{ \
	const GF31 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \
	const GF31 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add31(mul31(u0, u0p), mul31(mul31(u1, u1p), w)); \
	z1 = add31(mul31(u0, u1p), mul31(u0p, u1)); \
	z2 = sub31(mul31(u2, u2p), mul31(mul31(u3, u3p), w)); \
	z3 = add31(mul31(u2, u3p), mul31(u2p, u3)); \
	}

#define MUL_4_31(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w, win) \
	{ \
	const GF31 u0 = z0, u2 = mul31(z2, w), u1 = z1, u3 = mul31(z3, w); \
	const GF31 v0 = add31(u0, u2), v2 = sub31(u0, u2), v1 = add31(u1, u3), v3 = sub31(u1, u3); \
	const GF31 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \
	const GF31 s0 = add31(mul31(v0, v0p), mul31(mul31(v1, v1p), w)); \
	const GF31 s1 = add31(mul31(v0, v1p), mul31(v0p, v1)); \
	const GF31 s2 = sub31(mul31(v2, v2p), mul31(mul31(v3, v3p), w)); \
	const GF31 s3 = add31(mul31(v2, v3p), mul31(v2p, v3)); \
	z0 = add31(s0, s2); z2 = mulconj31(sub31(s0, s2), w); \
	z1 = add31(s1, s3); z3 = mulconj31(sub31(s1, s3), w); \
	}

// --- transform/macro Zp1 ---

// 16 mul + 16 mul_hi
#define FORWARD_4_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
	{ \
	const Zp1 u0 = zi0, u2 = mul1(zi2, w1), u1 = zi1, u3 = mul1(zi3, w1); \
	const Zp1 v0 = add1(u0, u2), v2 = sub1(u0, u2), v1 = mul1(add1(u1, u3), w20), v3 = mul1(sub1(u1, u3), w21); \
	zo0 = add1(v0, v1); zo1 = sub1(v0, v1); zo2 = add1(v2, v3); zo3 = sub1(v2, v3); \
	}

#define BACKWARD_4_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
	{ \
	const Zp1 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \
	const Zp1 v0 = add1(u0, u1), v1 = mul1(sub1(u1, u0), win20), v2 = add1(u2, u3), v3 = mul1(sub1(u3, u2), win21); \
	zo0 = add1(v0, v2); zo2 = mul1(sub1(v2, v0), win1); zo1 = add1(v1, v3); zo3 = mul1(sub1(v3, v1), win1); \
	}

#define FORWARD_8_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
	{ \
	const Zp1 t0 = forward2_1(zi0), t2 = forward2_1(zi2), t1 = forward2_1(zi1), t3 = forward2_1(zi3); \
	FORWARD_4_1(t0, t1, t2, t3, zo0, zo1, zo2, zo3, w1, w20, w21); \
	}

#define BACKWARD_8_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
	{ \
	Zp1 t0, t1, t2, t3; \
	BACKWARD_4_1(zi0, zi1, zi2, zi3, t0, t1, t2, t3, win1, win20, win21); \
	zo0 = backward2_1(t0); zo2 = backward2_1(t2); zo1 = backward2_1(t1); zo3 = backward2_1(t3); \
	}

#define SQUARE_22_1(z0, z1, z2, z3, w) \
	{ \
	const Zp1 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add1(sqr1(u0), mul1(sqr1(u1), w)); z1 = mul1(add1(u0, u0), u1); \
	z2 = sub1(sqr1(u2), mul1(sqr1(u3), w)); z3 = mul1(add1(u2, u2), u3); \
	}

#define SQUARE_4_1(z0, z1, z2, z3, w, win) \
	{ \
	const Zp1 u0 = z0, u2 = mul1(z2, w), u1 = z1, u3 = mul1(z3, w); \
	const Zp1 v0 = add1(u0, u2), v2 = sub1(u0, u2), v1 = add1(u1, u3), v3 = sub1(u1, u3); \
	const Zp1 s0 = add1(sqr1(v0), mul1(sqr1(v1), w)), s1 = mul1(add1(v0, v0), v1); \
	const Zp1 s2 = sub1(sqr1(v2), mul1(sqr1(v3), w)), s3 = mul1(add1(v2, v2), v3); \
	z0 = add1(s0, s2); z2 = mul1(sub1(s2, s0), win); \
	z1 = add1(s1, s3); z3 = mul1(sub1(s3, s1), win); \
	}

#define FWD_2_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \
	{ \
	const Zp1 u0 = zi0, u2 = mul1(zi2, w), u1 = zi1, u3 = mul1(zi3, w); \
	zo0 = add1(u0, u2); zo2 = sub1(u0, u2); zo1 = add1(u1, u3); zo3 = sub1(u1, u3); \
	}

#define MUL_22_1(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
	{ \
	const Zp1 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \
	const Zp1 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add1(mul1(u0, u0p), mul1(mul1(u1, u1p), w)); \
	z1 = add1(mul1(u0, u1p), mul1(u0p, u1)); \
	z2 = sub1(mul1(u2, u2p), mul1(mul1(u3, u3p), w)); \
	z3 = add1(mul1(u2, u3p), mul1(u2p, u3)); \
	}

#define MUL_4_1(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w, win) \
	{ \
	const Zp1 u0 = z0, u2 = mul1(z2, w), u1 = z1, u3 = mul1(z3, w); \
	const Zp1 v0 = add1(u0, u2), v2 = sub1(u0, u2), v1 = add1(u1, u3), v3 = sub1(u1, u3); \
	const Zp1 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \
	const Zp1 s0 = add1(mul1(v0, v0p), mul1(mul1(v1, v1p), w)); \
	const Zp1 s1 = add1(mul1(v0, v1p), mul1(v0p, v1)); \
	const Zp1 s2 = sub1(mul1(v2, v2p), mul1(mul1(v3, v3p), w)); \
	const Zp1 s3 = add1(mul1(v2, v3p), mul1(v2p, v3)); \
	z0 = add1(s0, s2); z2 = mul1(sub1(s2, s0), win); \
	z1 = add1(s1, s3); z3 = mul1(sub1(s3, s1), win); \
	}

// --- transform/macro Zp2 ---

// 16 mul + 16 mul_hi
#define FORWARD_4_2(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
	{ \
	const Zp2 u0 = zi0, u2 = mul2(zi2, w1), u1 = zi1, u3 = mul2(zi3, w1); \
	const Zp2 v0 = add2(u0, u2), v2 = sub2(u0, u2), v1 = mul2(add2(u1, u3), w20), v3 = mul2(sub2(u1, u3), w21); \
	zo0 = add2(v0, v1); zo1 = sub2(v0, v1); zo2 = add2(v2, v3); zo3 = sub2(v2, v3); \
	}

#define BACKWARD_4_2(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
	{ \
	const Zp2 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \
	const Zp2 v0 = add2(u0, u1), v1 = mul2(sub2(u1, u0), win20), v2 = add2(u2, u3), v3 = mul2(sub2(u3, u2), win21); \
	zo0 = add2(v0, v2); zo2 = mul2(sub2(v2, v0), win1); zo1 = add2(v1, v3); zo3 = mul2(sub2(v3, v1), win1); \
	}

#define FORWARD_8_2(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
	{ \
	const Zp2 t0 = forward2_2(zi0), t2 = forward2_2(zi2), t1 = forward2_2(zi1), t3 = forward2_2(zi3); \
	FORWARD_4_2(t0, t1, t2, t3, zo0, zo1, zo2, zo3, w1, w20, w21); \
	}

#define BACKWARD_8_2(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
	{ \
	Zp2 t0, t1, t2, t3; \
	BACKWARD_4_2(zi0, zi1, zi2, zi3, t0, t1, t2, t3, win1, win20, win21); \
	zo0 = backward2_2(t0); zo2 = backward2_2(t2); zo1 = backward2_2(t1); zo3 = backward2_2(t3); \
	}

#define SQUARE_22_2(z0, z1, z2, z3, w) \
	{ \
	const Zp2 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add2(sqr2(u0), mul2(sqr2(u1), w)); z1 = mul2(add2(u0, u0), u1); \
	z2 = sub2(sqr2(u2), mul2(sqr2(u3), w)); z3 = mul2(add2(u2, u2), u3); \
	}

#define SQUARE_4_2(z0, z1, z2, z3, w, win) \
	{ \
	const Zp2 u0 = z0, u2 = mul2(z2, w), u1 = z1, u3 = mul2(z3, w); \
	const Zp2 v0 = add2(u0, u2), v2 = sub2(u0, u2), v1 = add2(u1, u3), v3 = sub2(u1, u3); \
	const Zp2 s0 = add2(sqr2(v0), mul2(sqr2(v1), w)), s1 = mul2(add2(v0, v0), v1); \
	const Zp2 s2 = sub2(sqr2(v2), mul2(sqr2(v3), w)), s3 = mul2(add2(v2, v2), v3); \
	z0 = add2(s0, s2); z2 = mul2(sub2(s2, s0), win); \
	z1 = add2(s1, s3); z3 = mul2(sub2(s3, s1), win); \
	}

#define FWD_2_2(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \
	{ \
	const Zp2 u0 = zi0, u2 = mul2(zi2, w), u1 = zi1, u3 = mul2(zi3, w); \
	zo0 = add2(u0, u2); zo2 = sub2(u0, u2); zo1 = add2(u1, u3); zo3 = sub2(u1, u3); \
	}

#define MUL_22_2(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
	{ \
	const Zp2 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \
	const Zp2 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add2(mul2(u0, u0p), mul2(mul2(u1, u1p), w)); \
	z1 = add2(mul2(u0, u1p), mul2(u0p, u1)); \
	z2 = sub2(mul2(u2, u2p), mul2(mul2(u3, u3p), w)); \
	z3 = add2(mul2(u2, u3p), mul2(u2p, u3)); \
	}

#define MUL_4_2(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w, win) \
	{ \
	const Zp2 u0 = z0, u2 = mul2(z2, w), u1 = z1, u3 = mul2(z3, w); \
	const Zp2 v0 = add2(u0, u2), v2 = sub2(u0, u2), v1 = add2(u1, u3), v3 = sub2(u1, u3); \
	const Zp2 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \
	const Zp2 s0 = add2(mul2(v0, v0p), mul2(mul2(v1, v1p), w)); \
	const Zp2 s1 = add2(mul2(v0, v1p), mul2(v0p, v1)); \
	const Zp2 s2 = sub2(mul2(v2, v2p), mul2(mul2(v3, v3p), w)); \
	const Zp2 s3 = add2(mul2(v2, v3p), mul2(v2p, v3)); \
	z0 = add2(s0, s2); z2 = mul2(sub2(s2, s0), win); \
	z1 = add2(s1, s3); z3 = mul2(sub2(s3, s1), win); \
	}

// --- set pair M31/P1 ---

#ifdef W123
#define DECLARE_VAR_INDEXW(j) \
	const sz_t j0 = 3 * j + 0, j1 = 3 * j + 1, j2 = 3 * j + 2;
#define DECLARE_VAR_INDEXWIN(ji) \
	const sz_t ji0 = 3 * ji + 0, ji1 = 3 * ji + 1, ji2 = 3 * ji + 2;
#define DECLARE_VAR_INDEXWS() \
	const sz_t ws_offset = NSIZE / 2;
#else
#define DECLARE_VAR_INDEXW(j) \
	const sz_t j0 = j, j1 = NSIZE / 2 + j, j2 = NSIZE + j;
#define DECLARE_VAR_INDEXWIN(ji) \
	const sz_t ji0 = ji, ji1 = NSIZE / 2 + ji, ji2 = NSIZE + ji;
#define DECLARE_VAR_INDEXWS() \
	const sz_t ws_offset = 0;
#endif

#define DECLARE_VAR_W(j) \
	DECLARE_VAR_INDEXW(j); \
	const Zp1 w1_0 = w[j0], w2_0 = w[j1], w3_0 = w[j2]; \
	const Zp2 w1_1 = w[WOFFSET_1 + j0], w2_1 = w[WOFFSET_1 + j1], w3_1 = w[WOFFSET_1 + j2];
	// const GF31 w1_1 = w[WOFFSET_1 + j0], w2_1 = w[WOFFSET_1 + j1], w3_1 = w[WOFFSET_1 + j2];

#define DECLARE_VAR_WIN(j, ji) \
	DECLARE_VAR_INDEXWIN(ji); \
	const Zp1 wi1_0 = swap(w[ji0]), wi3_0 = swap(w[ji1]), wi2_0 = swap(w[ji2]); \
	const Zp2 wi1_1 = swap(w[WOFFSET_1 + ji0]), wi3_1 = swap(w[WOFFSET_1 + ji1]), wi2_1 = swap(w[WOFFSET_1 + ji2]);
	// DECLARE_VAR_INDEXW(j);
	// const GF31 wi1_1 = w[WOFFSET_1 + j0], wi2_1 = w[WOFFSET_1 + j1], wi3_1 = w[WOFFSET_1 + j2];

#define DECLARE_VAR_WS(j) \
	DECLARE_VAR_INDEXWS(); \
	const Zp1 w_0 = w[ws_offset + j]; \
	const Zp2 w_1 = w[WOFFSET_1 + ws_offset + j];
	// const GF31 w_1 = w[WOFFSET_1 + ws_offset + j];

#define DECLARE_VAR_WINS(ji) \
	const Zp1 wi_0 = swap(w[ws_offset + ji]); \
	const Zp2 wi_1 = swap(w[WOFFSET_1 + ws_offset + ji]);
	// const GF31 wi_1 = 0;

#define FORWARD_4_a		FORWARD_4_1
#define FORWARD_4_0_a	FORWARD_8_1
#define BACKWARD_4_a	BACKWARD_4_1
#define BACKWARD_4_0_a	BACKWARD_8_1
#define SQUARE_22_a		SQUARE_22_1
#define SQUARE_4_a		SQUARE_4_1
#define FWD_2_a			FWD_2_1
#define MUL_22_a		MUL_22_1
#define MUL_4_a			MUL_4_1

// #define FORWARD_4_b		FORWARD_4_31
// #define FORWARD_4_0_b	FORWARD_4_31
// #define BACKWARD_4_b	BACKWARD_4_31
// #define BACKWARD_4_0_b	BACKWARD_4_31
// #define SQUARE_22_b		SQUARE_22_31
// #define SQUARE_4_b		SQUARE_4_31
// #define FWD_2_b			FWD_2_31
// #define MUL_22_b		MUL_22_31
// #define MUL_4_b			MUL_4_31

#define FORWARD_4_b		FORWARD_4_2
#define FORWARD_4_0_b	FORWARD_8_2
#define BACKWARD_4_b	BACKWARD_4_2
#define BACKWARD_4_0_b	BACKWARD_8_2
#define SQUARE_22_b		SQUARE_22_2
#define SQUARE_4_b		SQUARE_4_2
#define FWD_2_b			FWD_2_2
#define MUL_22_b		MUL_22_2
#define MUL_4_b			MUL_4_2

// --- transform/inline global mem ---

INLINE void forward_4io(const sz_t m, __global uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j)
{
	DECLARE_VAR_W(j);
	FORWARD_4_a(z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, w1_0, w2_0, w3_0);
	FORWARD_4_b(z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, w1_1, w2_1, w3_1);
}

INLINE void forward_4io_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t m = NSIZE / 4;
	DECLARE_VAR_W(1);
	FORWARD_4_0_a(z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, w1_0, w2_0, w3_0);
	FORWARD_4_0_b(z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, w1_1, w2_1, w3_1);
}

INLINE void backward_4io(const sz_t m, __global uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	DECLARE_VAR_WIN(j, ji);
	BACKWARD_4_a(z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, wi1_0, wi2_0, wi3_0);
	BACKWARD_4_b(z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, wi1_1, wi2_1, wi3_1);
}

INLINE void backward_4io_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t m = NSIZE / 4;
	DECLARE_VAR_WIN(1, 1);
	BACKWARD_4_0_a(z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, z[0 * m].s01, z[1 * m].s01, z[2 * m].s01, z[3 * m].s01, wi1_0, wi2_0, wi3_0);
	BACKWARD_4_0_b(z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, z[0 * m].s23, z[1 * m].s23, z[2 * m].s23, z[3 * m].s23, wi1_1, wi2_1, wi3_1);
}

INLINE void square_22io(__global uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j)
{
	DECLARE_VAR_WS(j);
	SQUARE_22_a(z[0].s01, z[1].s01, z[2].s01, z[3].s01, w_0);
	SQUARE_22_b(z[0].s23, z[1].s23, z[2].s23, z[3].s23, w_1);
}

INLINE void square_4io(__global uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	DECLARE_VAR_WS(j);
	DECLARE_VAR_WINS(ji);
	SQUARE_4_a(z[0].s01, z[1].s01, z[2].s01, z[3].s01, w_0, wi_0);
	SQUARE_4_b(z[0].s23, z[1].s23, z[2].s23, z[3].s23, w_1, wi_1);
}

INLINE void fwd_2io(__global uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j)
{
	DECLARE_VAR_WS(j);
	FWD_2_a(z[0].s01, z[1].s01, z[2].s01, z[3].s01, z[0].s01, z[1].s01, z[2].s01, z[3].s01, w_0);
	FWD_2_b(z[0].s23, z[1].s23, z[2].s23, z[3].s23, z[0].s23, z[1].s23, z[2].s23, z[3].s23, w_1);
}

INLINE void mul_22io(__global uint4 * restrict const z, const __global uint4 * restrict const zp,
	__global const uint2 * restrict const w, const sz_t j)
{
	DECLARE_VAR_WS(j);
	MUL_22_a(z[0].s01, z[1].s01, z[2].s01, z[3].s01, zp[0].s01, zp[1].s01, zp[2].s01, zp[3].s01, w_0);
	MUL_22_b(z[0].s23, z[1].s23, z[2].s23, z[3].s23, zp[0].s23, zp[1].s23, zp[2].s23, zp[3].s23, w_1);
}

INLINE void mul_4io(__global uint4 * restrict const z, const __global uint4 * restrict const zp,
	__global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	DECLARE_VAR_WS(j);
	DECLARE_VAR_WINS(ji);
	MUL_4_a(z[0].s01, z[1].s01, z[2].s01, z[3].s01, zp[0].s01, zp[1].s01, zp[2].s01, zp[3].s01, w_0, wi_0);
	MUL_4_b(z[0].s23, z[1].s23, z[2].s23, z[3].s23, zp[0].s23, zp[1].s23, zp[2].s23, zp[3].s23, w_1, wi_1);
}

// --- transform/inline local & global mem ---

INLINE void forward_4(const sz_t m, __local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j)
{
	DECLARE_VAR_W(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4_a(Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, w1_0, w2_0, w3_0);
	FORWARD_4_b(Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, w1_1, w2_1, w3_1);
}

INLINE void forward_4i(const sz_t ml, __local uint4 * restrict const Z, const sz_t mg,
	__global const uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j)
{
	__global const uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_W(j);
	FORWARD_4_a(z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, w1_0, w2_0, w3_0);
	FORWARD_4_b(z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, w1_1, w2_1, w3_1);
}

INLINE void forward_4i_0(const sz_t ml, __local uint4 * restrict const Z, const sz_t mg,
	__global const uint4 * restrict const z, __global const uint2 * restrict const w)
{
	__global const uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_W(1);
	FORWARD_4_0_a(z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, w1_0, w2_0, w3_0);
	FORWARD_4_0_b(z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, w1_1, w2_1, w3_1);
}

INLINE void forward_4o(const sz_t mg, __global uint4 * restrict const z, const sz_t ml,
	__local const uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j)
{
	__global uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_W(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4_a(Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, w1_0, w2_0, w3_0);
	FORWARD_4_b(Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, w1_1, w2_1, w3_1);
}

INLINE void backward_4(const sz_t m, __local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	DECLARE_VAR_WIN(j, ji);
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_a(Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, wi1_0, wi2_0, wi3_0);
	BACKWARD_4_b(Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, wi1_1, wi2_1, wi3_1);
}

INLINE void backward_4i(const sz_t ml, __local uint4 * restrict const Z, const sz_t mg,
	__global const uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	__global const uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_WIN(j, ji);
	BACKWARD_4_a(z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, wi1_0, wi2_0, wi3_0);
	BACKWARD_4_b(z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, wi1_1, wi2_1, wi3_1);
}

INLINE void backward_4o(const sz_t mg, __global uint4 * restrict const z, const sz_t ml,
	__local const uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	__global uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_WIN(j, ji);
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_a(Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, wi1_0, wi2_0, wi3_0);
	BACKWARD_4_b(Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, wi1_1, wi2_1, wi3_1);
}

INLINE void backward_4o_0(const sz_t mg, __global uint4 * restrict const z, const sz_t ml,
	__local const uint4 * restrict const Z, __global const uint2 * restrict const w)
{
	__global uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_WIN(1, 1);
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_0_a(Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, wi1_0, wi2_0, wi3_0);
	BACKWARD_4_0_b(Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, wi1_1, wi2_1, wi3_1);
}

INLINE void square_22(__local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j)
{
	DECLARE_VAR_WS(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_22_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, w_0);
	SQUARE_22_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, w_1);
}

INLINE void square_4(__local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	DECLARE_VAR_WS(j);
	DECLARE_VAR_WINS(ji);
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_4_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, w_0, wi_0);
	SQUARE_4_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, w_1, wi_1);
}

INLINE void write_4(const sz_t mg, __global uint4 * restrict const z, __local const uint4 * restrict const Z)
{
	__global uint4 * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];
}

INLINE void fwd2_write_4(const sz_t mg, __global uint4 * restrict const z, __local const uint4 * restrict const Z,
	__global const uint2 * restrict const w, const sz_t j)
{
	__global uint4 * const z2mg = &z[2 * mg];
	DECLARE_VAR_WS(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	FWD_2_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, w_0);
	FWD_2_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, w_1);
}

INLINE void mul_22(__local uint4 * restrict const Z, const sz_t mg, __global const uint4 * restrict const z,
	__global const uint2 * restrict const w, const sz_t j)
{
	__global const uint4 * const z2mg = &z[2 * mg];
	const uint4 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	DECLARE_VAR_WS(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_22_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, z0p.s01, z1p.s01, z2p.s01, z3p.s01, w_0);
	MUL_22_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, z0p.s23, z1p.s23, z2p.s23, z3p.s23, w_1);
}

INLINE void mul_4(__local uint4 * restrict const Z, const sz_t mg, __global const uint4 * restrict const z,
	__global const uint2 * restrict const w, const sz_t j, const sz_t ji)
{
	__global const uint4 * const z2mg = &z[2 * mg];
	const uint4 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	DECLARE_VAR_WS(j);
	DECLARE_VAR_WINS(ji);
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_4_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, z0p.s01, z1p.s01, z2p.s01, z3p.s01, w_0, wi_0);
	MUL_4_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, z0p.s23, z1p.s23, z2p.s23, z3p.s23, w_1, wi_1);
}

// --- transform without local mem ---

__kernel
void forward4(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx >> lm, k = 3 * (j << lm) + idx;
	forward_4io((sz_t)(1) << lm, &z[k], w, s + j);
}

__kernel
void backward4(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx >> lm, k = 3 * (j << lm) + idx;
	backward_4io((sz_t)(1) << lm, &z[k], w, s + j, s + s - j - 1);
}

__kernel
void forward4_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t k = idx;
	forward_4io_0(&z[k], w);
}

__kernel
void backward4_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t k = idx;
	backward_4io_0(&z[k], w);
}

__kernel
void square22(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	square_22io(&z[k], w, NSIZE / 4 + j);
}

__kernel
void square4(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	square_4io(&z[k], w, NSIZE / 4 +  j, NSIZE / 4 + NSIZE / 4 - j - 1);
}

__kernel
void fwd4p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	fwd_2io(&z[k], w, NSIZE / 4 + j);
}

__kernel
void mul22(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	mul_22io(&z[k], &zp[k], w, NSIZE / 4 + j);
}

__kernel
void mul4(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	mul_4io(&z[k], &zp[k], w, NSIZE / 4 + j, NSIZE / 4 + NSIZE / 4 - j - 1);
}

// --- transform ---

#define DECLARE_VAR(B_N, CHUNK_N) \
	__local uint4 Z[4 * B_N * CHUNK_N]; \
	\
	/* threadIdx < B_N */ \
	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \
	__local uint4 * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	sz_t sj = s + idx_m, sji = s + s - idx_m - 1;

#define DECLARE_VAR_FORWARD() \
	__global uint4 * restrict const zi = &z[ki]; \
	__global uint4 * restrict const zo = &z[ko];

#define DECLARE_VAR_BACKWARD() \
	__global uint4 * restrict const zi = &z[ko]; \
	__global uint4 * restrict const zo = &z[ki];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);

#define FORWARD_I_0(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_0(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, w, sj / 1, sji / 1);

// -----------------

#define B_64	(64 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void forward64(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void forward64_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 6; const unsigned int s = 64 / 4;

	FORWARD_I_0(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void backward64(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4, sji / 4);
	backward_4o(B_64 << lm, zo, B_64 * CHUNK64, &Z[i], w, sj / B_64, sji / B_64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void backward64_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 6; const unsigned int s = 64 / 4;

	BACKWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4, sji / 4);
	backward_4o_0(B_64 << lm, zo, B_64 * CHUNK64, &Z[i], w);
}

// -----------------

#define B_256	(256 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void forward256(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void forward256_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 8; const unsigned int s = 256 / 4;

	FORWARD_I_0(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void backward256(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16, sji / 16);
	backward_4o(B_256 << lm, zo, B_256 * CHUNK256, &Z[i], w, sj / B_256, sji / B_256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void backward256_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 8; const unsigned int s = 256 / 4;

	BACKWARD_I(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16, sji / 16);
	backward_4o_0(B_256 << lm, zo, B_256 * CHUNK256, &Z[i], w);
}

// -----------------

#define B_1024	(1024 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 10; const unsigned int s = 1024 / 4;

	FORWARD_I_0(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void backward1024(__global uint4 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16, sji / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64, sji / 64);
	backward_4o(B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], w, sj / B_1024, sji / B_1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void backward1024_0(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 10; const unsigned int s = 1024 / 4;

	BACKWARD_I(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16, sji / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64, sji / 64);
	backward_4o_0(B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], w);
}

// -----------------

#define DECLARE_VAR_32() \
	__local uint4 Z[32 * BLK32]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k32 = (sz_t)get_group_id(0) * 32 * BLK32, i = (sz_t)get_local_id(0); \
	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \
	\
	__global uint4 * restrict const zk = &z[k32 + i32 + i8]; \
	__local uint4 * const Z32 = &Z[i32]; \
	__local uint4 * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \
	__local uint4 * const Zi2 = &Z32[i2]; \
	__local uint4 * const Z4 = &Z32[4 * i8];

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void square32(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4o(8, zk, 8, Zi8, w, j / 8, ji / 8);
}

#define DECLARE_VAR_64() \
	__local uint4 Z[64 * BLK64]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k64 = (sz_t)get_group_id(0) * 64 * BLK64, i = (sz_t)get_local_id(0); \
	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \
	\
	__global uint4 * restrict const zk = &z[k64 + i64 + i16]; \
	__local uint4 * const Z64 = &Z[i64]; \
	__local uint4 * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \
	__local uint4 * const Zi4 = &Z64[i4]; \
	__local uint4 * const Z4 = &Z64[4 * i16];

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void square64(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w, j, ji);
	backward_4(4, Zi4, w, j / 4, ji / 4);
	backward_4o(16, zk, 16, Zi16, w, j / 16, ji / 16);
}

#define DECLARE_VAR_128() \
	__local uint4 Z[128 * BLK128]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k128 = (sz_t)get_group_id(0) * 128 * BLK128, i = (sz_t)get_local_id(0); \
	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \
	\
	__global uint4 * restrict const zk = &z[k128 + i128 + i32]; \
	__local uint4 * const Z128 = &Z[i128]; \
	__local uint4 * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \
	__local uint4 * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \
	__local uint4 * const Zi2 = &Z128[i2]; \
	__local uint4 * const Z4 = &Z128[4 * i32];

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void square128(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4(8, Zi8, w, j / 8, ji / 8);
	backward_4o(32, zk, 32, Zi32, w, j / 32, ji / 32);
}

// if BLK256 != 1 then const sz_t i256 = (i & (sz_t)~(256 / 4 - 1)) * 4, i64 = i % (256 / 4);
// if BLK256 = 1 then const sz_t i256 = 0, i64 = i;
#define DECLARE_VAR_256() \
	__local uint4 Z[256 * BLK256]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k256 = (sz_t)get_group_id(0) * 256 * BLK256, i = (sz_t)get_local_id(0); \
	const sz_t i256 = 0, i64 = i; \
	\
	__global uint4 * restrict const zk = &z[k256 + i256 + i64]; \
	__local uint4 * const Z256 = &Z[i256]; \
	__local uint4 * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \
	__local uint4 * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \
	__local uint4 * const Zi4 = &Z256[i4]; \
	__local uint4 * const Z4 = &Z256[4 * i64];

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void square256(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w, j, ji);
	backward_4(4, Zi4, w, j / 4, ji / 4);
	backward_4(16, Zi16, w, j / 16, ji / 16);
	backward_4o(64, zk, 64, Zi64, w, j / 64, ji / 64);
}

#define DECLARE_VAR_512() \
	__local uint4 Z[512]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k512 = (sz_t)get_group_id(0) * 512, i128 = (sz_t)get_local_id(0); \
	\
	__global uint4 * restrict const zk = &z[k512 + i128]; \
	__local uint4 * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \
	__local uint4 * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \
	__local uint4 * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \
	__local uint4 * const Zi2 = &Z[i2]; \
	__local uint4 * const Z4 = &Z[4 * i128];

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void square512(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4(8, Zi8, w, j / 8, ji / 8);
	backward_4(32, Zi32, w, j / 32, ji / 32);
	backward_4o(128, zk, 128, Zi128, w, j / 128, ji / 128);
}

#define DECLARE_VAR_1024() \
	__local uint4 Z[1024]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k1024 = (sz_t)get_group_id(0) * 1024, i256 = (sz_t)get_local_id(0); \
	\
	__global uint4 * restrict const zk = &z[k1024 + i256]; \
	__local uint4 * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \
	__local uint4 * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \
	__local uint4 * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \
	__local uint4 * const Zi4 = &Z[i4]; \
	__local uint4 * const Z4 = &Z[4 * i256];

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void square1024(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w, j, ji);
	backward_4(4, Zi4, w, j / 4, ji / 4);
	backward_4(16, Zi16, w, j / 16, ji / 16);
	backward_4(64, Zi64, w, j / 64, ji / 64);
	backward_4o(256, zk, 256, Zi256, w, j / 256, ji / 256);
}

/*#define DECLARE_VAR_2048() \
	__local uint4 Z[2048]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k2048 = (sz_t)get_group_id(0) * 2048, i512 = (sz_t)get_local_id(0); \
	\
	__global uint4 * restrict const zk = &z[k2048 + i512]; \
	__local uint4 * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \
	__local uint4 * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \
	__local uint4 * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \
	__local uint4 * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \
	__local uint4 * const Zi2 = &Z[i2]; \
	__local uint4 * const Z4 = &Z[4 * i512];

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void square2048(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4(8, Zi8, w, j / 8, ji / 8);
	backward_4(32, Zi32, w, j / 32, ji / 32);
	backward_4(128, Zi128, w, j / 128, ji / 128);
	backward_4o(512, zk, 512, Zi512, w, j / 512, ji / 51);
}*/

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(8, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2_write_4(16, zk, Z4, w, j);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(32, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2_write_4(64, zk, Z4, w, j);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void fwd512p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(128, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void fwd1024p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2_write_4(256, zk, Z4, w, j);
}

/*__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void fwd2048p(__global uint4 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(512, zk, Z4);
}*/

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void mul32(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_32();
	__global const uint4 * restrict const zpk = &zp[k32 + i32 + i8];

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 8, zpk, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4o(8, zk, 8, Zi8, w, j / 8, ji / 8);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void mul64(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_64();
	__global const uint4 * restrict const zpk = &zp[k64 + i64 + i16];

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 16, zpk, w, j, ji);
	backward_4(4, Zi4, w, j / 4, ji / 4);
	backward_4o(16, zk, 16, Zi16, w, j / 16, ji / 16);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void mul128(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_128();
	__global const uint4 * restrict const zpk = &zp[k128 + i128 + i32];

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 32, zpk, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4(8, Zi8, w, j / 8, ji / 8);
	backward_4o(32, zk, 32, Zi32, w, j / 32, ji / 32);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void mul256(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_256();
	__global const uint4 * restrict const zpk = &zp[k256 + i256 + i64];

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 64, zpk, w, j, ji);
	backward_4(4, Zi4, w, j / 4, ji / 4);
	backward_4(16, Zi16, w, j / 16, ji / 16);
	backward_4o(64, zk, 64, Zi64, w, j / 64, ji / 64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void mul512(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_512();
	__global const uint4 * restrict const zpk = &zp[k512 + i128];

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 128, zpk, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4(8, Zi8, w, j / 8, ji / 8);
	backward_4(32, Zi32, w, j / 32, ji / 32);
	backward_4o(128, zk, 128, Zi128, w, j / 128, ji / 128);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void mul1024(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_1024();
	__global const uint4 * restrict const zpk = &zp[k1024 + i256];

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 256, zpk, w, j, ji);
	backward_4(4, Zi4, w, j / 4, ji / 4);
	backward_4(16, Zi16, w, j / 16, ji / 16);
	backward_4(64, Zi64, w, j / 64, ji / 64);
	backward_4o(256, zk, 256, Zi256, w, j / 256, ji / 256);
}

/*__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void mul2048(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_2048();
	__global const uint4 * restrict const zpk = &zp[k2048 + i512];

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 512, zpk, w, j);
	backward_4(2, Zi2, w, j / 2, ji / 2);
	backward_4(8, Zi8, w, j / 8, ji / 8);
	backward_4(32, Zi32, w, j / 32, ji / 32);
	backward_4(128, Zi128, w, j / 128, ji / 128);
	backward_4o(512, zk, 512, Zi512, w, j / 512, ji / 512);
}*/

// -----------------

INLINE uint32_2 barrett(const uint64_2 a, const uint32 b, const uint32 b_inv, const int b_s, uint32_2 * a_p)
{
	// Using notations of Modular SIMD arithmetic in Mathemagix, Joris van der Hoeven, Grégoire Lecerf, Guillaume Quintin, 2014, HAL.
	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.
	// b < 2^31, alpha = 2^29 => a < 2^29 b
	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31
	// b_inv = [2^{s + 32} / b]
	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31
	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].
	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])
	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,
	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.

	const uint64_2 as = a >> b_s;
	const uint32_2 d = mul_hi((uint32_2)(as.s0, as.s1), b_inv), r = (uint32_2)(a.s0, a.s1) - d * b;
	const bool o0 = (r.s0 >= b), o1 = (r.s1 >= b);
	*a_p = d + (uint32_2)(o0 ? 1 : 0, o1 ? 1 : 0);
	return r - (uint32_2)(o0 ? b : 0, o1 ? b : 0);
}

INLINE uint64_2 shl29or(const uint32_2 h, const uint32_2 l)
{
	return (uint64_2)(((uint64)(h.s0) << 29) | l.s0, ((uint64)(h.s1) << 29) | l.s1);
}

INLINE int32_2 reduce64(int64_2 * f, const uint32 b, const uint32 b_inv, const int b_s)
{
	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
	const uint64_2 t = abs(*f);
	const uint64_2 t_h = t >> 29;
	const uint32_2 t_l = (uint32_2)((uint32)(t.s0), (uint32)(t.s1)) & ((1u << 29) - 1);

	uint32_2 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint32_2 d_l, r_l = barrett(shl29or(r_h, t_l), b, b_inv, b_s, &d_l);
	const uint64_2 d = shl29or(d_h, d_l);

	const bool b0 = ((*f).s0 < 0), b1 = ((*f).s1 < 0);
	*f = (int64_2)(b0 ? -(int64)(d.s0) : (int64)(d.s0), b1 ? -(int64)(d.s1) : (int64)(d.s1));
	return (int32_2)(b0 ? -(int32)(r_l.s0) : (int32)(r_l.s0), b1 ? -(int32)(r_l.s1) : (int32)(r_l.s1));
}

INLINE int64_2 garner2_31_1(const GF31 r1, const Zp1 r2)
{
	const uint32 InvP1_M1 = 8421505u;	// 1 / P1 (mod M1)
	const uint64 M31P1 = M31 * (uint64)(P1);
	GF31 r2_1 = (GF31)(r2);	// P1 < M31
	GF31 u12 = muls31(sub31(r1, r2_1), InvP1_M1);
	const uint64 n0 = r2.s0 + u12.s0 * (uint64)(P1), n1 = r2.s1 + u12.s1 * (uint64)(P1);
	return (int64_2)((n0 > M31P1 / 2) ? (int64)(n0 - M31P1) : (int64)(n0), (n1 > M31P1 / 2) ? (int64)(n1 - M31P1) : (int64)(n1));
}

INLINE int64_2 garner2_31_2(const GF31 r1, const Zp1 r2)
{
	const uint32 InvP2_M1 = 152183881u;	// 1 / P2 (mod M1)
	const uint64 M31P2 = M31 * (uint64)(P2);
	GF31 r2_1 = (GF31)(r2);	// P2 < M31
	GF31 u12 = muls31(sub31(r1, r2_1), InvP2_M1);
	const uint64 n0 = r2.s0 + u12.s0 * (uint64)(P2), n1 = r2.s1 + u12.s1 * (uint64)(P2);
	return (int64_2)((n0 > M31P2 / 2) ? (int64)(n0 - M31P2) : (int64)(n0), (n1 > M31P2 / 2) ? (int64)(n1 - M31P2) : (int64)(n1));
}

INLINE int64_2 garner2_1_2(const Zp1 r1, const Zp2 r2)
{
	const uint32 mfInvP2_P1 = 2130706177u;	// Montgomery form of 1 / P2 (mod P1)
	const uint64 P1P2 = P1 * (uint64)(P2);
	Zp1 u12 = muls1(sub1(r1, (Zp1)(r2)), mfInvP2_P1);	// P2 < P1
	const uint64 n0 = r2.s0 + u12.s0 * (uint64)(P2), n1 = r2.s1 + u12.s1 * (uint64)(P2);
	return (int64_2)((n0 > P1P2 / 2) ? (int64)(n0 - P1P2) : (int64)(n0), (n1 > P1P2 / 2) ? (int64)(n1 - P1P2) : (int64)(n1));
}

__kernel
void normalize1(__global uint4 * restrict const z, __global long2 * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const unsigned int blk = abs(sblk);
	__global uint4 * restrict const zi = &z[blk * idx];

	prefetch(zi, (size_t)blk);

	int64_2 f = 0;

	sz_t j = 0;
	do
	{
		const Zp1 u1 = muls1(zi[j].s01, NORM1);
		// const GF31 u31 = lshift31(zi[j].s23, SNORM31);
		const Zp2 u2 = muls2(zi[j].s23, NORM2);
		int64_2 l = garner2_1_2(u1, u2);
		if (sblk < 0) l += l;
		f += l;
		const int32_2 r = reduce64(&f, b, b_inv, b_s);
		zi[j].s01 = set_int1(r);
		zi[j].s23 = set_int2(r);

		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	if (i == 0) { const int64 t = f.s0; f.s0 = -f.s1; f.s1 = t; }	// a_n = -a_0
	c[i] = (long2)(f);
}

__kernel
void normalize2(__global uint4 * restrict const z, __global const long2 * restrict const c, 
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global uint4 * restrict const zi = &z[blk * idx];

	int64_2 f = (int64_2)(c[idx]);

	sz_t j = 0;
	do
	{
		const int32_2 i = get_int1(zi[j].s01);
		f += (int64_2)(i.s0, i.s1);
		const int32_2 r = reduce64(&f, b, b_inv, b_s);
		zi[j].s01 = set_int1(r);
		zi[j].s23 = set_int2(r);

		if ((f.s0 == 0) && (f.s1 == 0)) return;
		++j;
	} while (j != blk - 1);

	const int32_2 f32 = (int32_2)((int32)(f.s0), (int32)(f.s1));
	zi[blk - 1].s01 = add1(zi[blk - 1].s01, set_int1(f32));
	zi[blk - 1].s23 = add2(zi[blk - 1].s23, set_int2(f32));
}

__kernel
void mulscalar(__global uint4 * restrict const z, __global long2 * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global uint4 * restrict const zi = &z[blk * idx];

	prefetch(zi, (size_t)blk);

	int64_2 f = 0;

	sz_t j = 0;
	do
	{
		int64_2 l = garner2_1_2(zi[j].s01, zi[j].s23) * a;
		f += l;
		const int32_2 r = reduce64(&f, b, b_inv, b_s);
		zi[j].s01 = set_int1(r);
		zi[j].s23 = set_int2(r);

		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	if (i == 0) { const int64 t = f.s0; f.s0 = -f.s1; f.s1 = t; }	// a_n = -a_0
	c[i] = (long2)(f);
}

__kernel
void set(__global uint4 * restrict const z, const unsigned int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const uint32 ai = (idx == 0) ? a : 0;
	z[idx] = (uint4)(ai, 0, ai, 0);
}

__kernel
void copy(__global uint4 * restrict const z, const unsigned int dst, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[dst + idx] = z[src + idx];
}

__kernel
void copyp(__global uint4 * restrict const zp, __global const uint4 * restrict const z, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	zp[idx] = z[src + idx];
}
