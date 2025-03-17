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
#define	ZOFFSET_1	196608
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

// --- Z/(2^31 - 1)Z ---

#define	M31		0x7fffffffu

INLINE uint32 _add31(const uint32 lhs, const uint32 rhs)
{
	const uint32 t = lhs + rhs;
	return t - ((t >= M31) ? M31 : 0);
}

INLINE uint32 _sub31(const uint32 lhs, const uint32 rhs)
{
	const uint32 t = lhs - rhs;
	return t + (((int32)(t) < 0) ? M31 : 0);
}

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

INLINE int32 _get_int31(const uint32 n) { return (n >= M31 / 2) ? (int32)(n - M31) : (int32)(n); }
INLINE uint32 _set_int31(const int32 i) { return (i < 0) ? ((uint32)(i) + M31) : (uint32)(i); }

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

#define	P1		2130706433u
#define	Q1		2164260865u		// p * q = 1 (mod 2^32)

INLINE uint32 _add1(const uint32 lhs, const uint32 rhs)
{
	const uint32 t = lhs + rhs;
	return t - ((t >= P1) ? P1 : 0);
}

INLINE uint32 _sub1(const uint32 lhs, const uint32 rhs)
{
	const uint32 t = lhs - rhs;
	return t + (((int32)(t) < 0) ? P1 : 0);
}

INLINE uint32 _mul1(const uint32 lhs, const uint32 rhs)
{
	const uint64 t = lhs * (uint64)(rhs);
	const uint32 lo = (uint32)(t), hi = (uint32)(t >> 32);
	const uint32 mp = mul_hi(lo * Q1, P1);
	return _sub1(hi, mp);
}

INLINE uint32 _set_int1(const int32 i) { return (i < 0) ? ((uint32)(i) + P1) : (uint32)(i); }

// --- pair of Z/p1Z ---

typedef uint2	Zp1;

INLINE Zp1 set_int1(const int32_2 i) { return (Zp1)(_set_int1(i.s0), _set_int1(i.s1)); }

INLINE Zp1 swap1(const Zp1 lhs) { return (Zp1)(lhs.s1, lhs.s0); }

INLINE Zp1 add1(const Zp1 lhs, const Zp1 rhs) { return (Zp1)(_add1(lhs.s0, rhs.s0), _add1(lhs.s1, rhs.s1)); }
INLINE Zp1 sub1(const Zp1 lhs, const Zp1 rhs) { return (Zp1)(_sub1(lhs.s0, rhs.s0), _sub1(lhs.s1, rhs.s1)); }

INLINE Zp1 muls1(const Zp1 lhs, const uint32 s) { return (Zp1)(_mul1(lhs.s0, s), _mul1(lhs.s1, s)); }
INLINE Zp1 muli1(const Zp1 lhs) { const uint32 _i = 66976762u; return muls1(lhs, _i); } 	// Montgomery form of 5^{(p + 1)/4} = 16711679

INLINE Zp1 mul1(const Zp1 lhs, const Zp1 rhs) { return (Zp1)(_mul1(lhs.s0, rhs.s0), _mul1(lhs.s1, rhs.s1)); }
INLINE Zp1 sqr1(const Zp1 lhs) { return mul1(lhs, lhs); }

INLINE Zp1 forward2(const Zp1 lhs)
{
	const uint32 _r2 = 402124772u;	// (2^32)^2 mod p
	const uint32 _im = 200536044u;	// Montgomery form of Montgomery form to convert input into Montgomery form
	const uint32 u0 = _mul1(lhs.s0, _r2), u1 = _mul1(lhs.s1, _im);
	return (Zp1)(_add1(u0, u1), _sub1(u0, u1));
}

INLINE Zp1 backward2(const Zp1 lhs)
{
	const uint32 _i = 66976762u; 	// Montgomery form of 5^{(p + 1)/4} = 16711679
	const uint32 u0 = lhs.s0, u1 = lhs.s1;
	return (Zp1)(_add1(u0, u1), _mul1(_sub1(u1, u0), _i));
}

// --- transform/inline GF31 ---

// 12 mul + 12 mul_hi
#define FORWARD_4_31(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w2, w3) \
	const GF31 u0 = zi0, u2 = mul31(zi2, w1), u1 = mul31(zi1, w2), u3 = mul31(zi3, w3); \
	const GF31 v0 = add31(u0, u2), v2 = sub31(u0, u2), v1 = add31(u1, u3), v3 = sub31(u1, u3); \
	zo0 = add31(v0, v1); zo1 = sub31(v0, v1); zo2 = addi31(v2, v3); zo3 = subi31(v2, v3);

#define BACKWARD_4_31(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w2, w3) \
	const GF31 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \
	const GF31 v0 = add31(u0, u1), v1 = sub31(u0, u1), v2 = add31(u2, u3), v3 = sub31(u3, u2); \
	zo0 = add31(v0, v2); zo2 = mulconj31(sub31(v0, v2), w1); \
	zo1 = mulconj31(addi31(v1, v3), w2); zo3 = mulconj31(subi31(v1, v3), w3);

#define SQUARE_22_31(z0, z1, z2, z3, w) \
	const GF31 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add31(sqr31(u0), mul31(sqr31(u1), w)); z1 = mul31(add31(u0, u0), u1); \
	z2 = sub31(sqr31(u2), mul31(sqr31(u3), w)); z3 = mul31(add31(u2, u2), u3);

#define SQUARE_4_31(z0, z1, z2, z3, w) \
	const GF31 u0 = z0, u2 = mul31(z2, w), u1 = z1, u3 = mul31(z3, w); \
	const GF31 v0 = add31(u0, u2), v2 = sub31(u0, u2), v1 = add31(u1, u3), v3 = sub31(u1, u3); \
	const GF31 s0 = add31(sqr31(v0), mul31(sqr31(v1), w)), s1 = mul31(add31(v0, v0), v1); \
	const GF31 s2 = sub31(sqr31(v2), mul31(sqr31(v3), w)), s3 = mul31(add31(v2, v2), v3); \
	z0 = add31(s0, s2); z2 = mulconj31(sub31(s0, s2), w); \
	z1 = add31(s1, s3); z3 = mulconj31(sub31(s1, s3), w);

#define FWD_2_31(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \
	const GF31 u0 = zi0, u2 = mul31(zi2, w), u1 = zi1, u3 = mul31(zi3, w); \
	zo0 = add31(u0, u2); zo2 = sub31(u0, u2); zo1 = add31(u1, u3); zo3 = sub31(u1, u3);

#define MUL_22_31(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
	const GF31 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \
	const GF31 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add31(mul31(u0, u0p), mul31(mul31(u1, u1p), w)); \
	z1 = add31(mul31(u0, u1p), mul31(u0p, u1)); \
	z2 = sub31(mul31(u2, u2p), mul31(mul31(u3, u3p), w)); \
	z3 = add31(mul31(u2, u3p), mul31(u2p, u3));

#define MUL_4_31(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
	const GF31 u0 = z0, u2 = mul31(z2, w), u1 = z1, u3 = mul31(z3, w); \
	const GF31 v0 = add31(u0, u2), v2 = sub31(u0, u2), v1 = add31(u1, u3), v3 = sub31(u1, u3); \
	const GF31 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \
	const GF31 s0 = add31(mul31(v0, v0p), mul31(mul31(v1, v1p), w)); \
	const GF31 s1 = add31(mul31(v0, v1p), mul31(v0p, v1)); \
	const GF31 s2 = sub31(mul31(v2, v2p), mul31(mul31(v3, v3p), w)); \
	const GF31 s3 = add31(mul31(v2, v3p), mul31(v2p, v3)); \
	z0 = add31(s0, s2); z2 = mulconj31(sub31(s0, s2), w); \
	z1 = add31(s1, s3); z3 = mulconj31(sub31(s1, s3), w); \


INLINE void forward_4io_31(const sz_t m, __global GF31 * restrict const z, __global const GF31 * restrict const w, const sz_t j)
{
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	FORWARD_4_31(z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], w1, w2, w3);
}

INLINE void backward_4io_31(const sz_t m, __global GF31 * restrict const z, __global const GF31 * restrict const w, const sz_t j)
{
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	BACKWARD_4_31(z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], w1, w2, w3);
}

INLINE void square_22io_31(__global GF31 * restrict const z, const GF31 w)
{
	SQUARE_22_31(z[0], z[1], z[2], z[3], w);
}

INLINE void square_4io_31(__global GF31 * restrict const z, const GF31 w)
{
	SQUARE_4_31(z[0], z[1], z[2], z[3], w);
}

INLINE void fwd_2io_31(__global GF31 * restrict const z, const GF31 w)
{
	FWD_2_31(z[0], z[1], z[2], z[3], z[0], z[1], z[2], z[3], w);
}

INLINE void mul_22io_31(__global GF31 * restrict const z, const __global GF31 * restrict const zp, const GF31 w)
{
	MUL_22_31(z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w);
}

INLINE void mul_4io_31(__global GF31 * restrict const z, const __global GF31 * restrict const zp, const GF31 w)
{
	MUL_4_31(z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w);
}

// -----------------

INLINE void forward_4_31(const sz_t m, __local GF31 * restrict const Z, __global const GF31 * restrict const w, const sz_t j)
{
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4_31(Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], w1, w2, w3);
}

INLINE void forward_4i_31(const sz_t ml, __local GF31 * restrict const Z, const sz_t mg,
	__global const GF31 * restrict const z, __global const GF31 * restrict const w, const sz_t j)
{
	__global const GF31 * const z2mg = &z[2 * mg];
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	FORWARD_4_31(z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], w1, w2, w3);
}

INLINE void forward_4o_31(const sz_t mg, __global GF31 * restrict const z, const sz_t ml,
	__local const GF31 * restrict const Z, __global const GF31 * restrict const w, const sz_t j)
{
	__global GF31 * const z2mg = &z[2 * mg];
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4_31(Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], w1, w2, w3);
}

INLINE void backward_4_31(const sz_t m, __local GF31 * restrict const Z, __global const GF31 * restrict const w, const sz_t j)
{
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_31(Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], w1, w2, w3);
}

INLINE void backward_4i_31(const sz_t ml, __local GF31 * restrict const Z, const sz_t mg,
	__global const GF31 * restrict const z, __global const GF31 * restrict const w, const sz_t j)
{
	__global const GF31 * const z2mg = &z[2 * mg];
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	BACKWARD_4_31(z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], w1, w2, w3);
}

INLINE void backward_4o_31(const sz_t mg, __global GF31 * restrict const z, const sz_t ml,
	__local const GF31 * restrict const Z, __global const GF31 * restrict const w, const sz_t j)
{
	__global GF31 * const z2mg = &z[2 * mg];
	const GF31 w1 = w[j], w2 = w[NSIZE / 2 + j], w3 = w[NSIZE + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_31(Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], w1, w2, w3);
}

INLINE void square_22_31(__local GF31 * restrict const Z, const GF31 w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_22_31(Z[0], Z[1], Z[2], Z[3], w);
}

INLINE void square_4_31(__local GF31 * restrict const Z, const GF31 w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_4_31(Z[0], Z[1], Z[2], Z[3], w);
}

INLINE void write_4_31(const sz_t mg, __global GF31 * restrict const z, __local const GF31 * restrict const Z)
{
	__global GF31 * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];
}

INLINE void fwd2_write_4_31(const sz_t mg, __global GF31 * restrict const z, __local const GF31 * restrict const Z, const GF31 w)
{
	__global GF31 * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	FWD_2_31(Z[0], Z[1], Z[2], Z[3], z[0], z[mg], z2mg[0], z2mg[mg], w);
}

INLINE void mul_22_31(__local GF31 * restrict const Z, const sz_t mg, __global const GF31 * restrict const z, const GF31 w)
{
	__global const GF31 * const z2mg = &z[2 * mg];
	const GF31 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_22_31(Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w);
}

INLINE void mul_4_31(__local GF31 * restrict const Z, const sz_t mg, __global const GF31 * restrict const z, const GF31 w)
{
	__global const GF31 * const z2mg = &z[2 * mg];
	const GF31 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_4_31(Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w);
}

// --- transform/inline Zp1 ---

// 16 mul + 16 mul_hi
#define FORWARD_4_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
	const Zp1 u0 = zi0, u2 = mul1(zi2, w1), u1 = zi1, u3 = mul1(zi3, w1); \
	const Zp1 v0 = add1(u0, u2), v2 = sub1(u0, u2), v1 = mul1(add1(u1, u3), w20), v3 = mul1(sub1(u1, u3), w21); \
	zo0 = add1(v0, v1); zo1 = sub1(v0, v1); zo2 = add1(v2, v3); zo3 = sub1(v2, v3);

#define BACKWARD_4_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
	const Zp1 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \
	const Zp1 v0 = add1(u0, u1), v1 = mul1(sub1(u1, u0), win20), v2 = add1(u2, u3), v3 = mul1(sub1(u3, u2), win21); \
	zo0 = add1(v0, v2); zo2 = mul1(sub1(v2, v0), win1); zo1 = add1(v1, v3); zo3 = mul1(sub1(v3, v1), win1);

#define FORWARD_8_1_0(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
	const Zp1 t0 = forward2(zi0), t2 = forward2(zi2), t1 = forward2(zi1), t3 = forward2(zi3); \
	FORWARD_4_1(t0, t1, t2, t3, zo0, zo1, zo2, zo3, w1, w20, w21);

#define BACKWARD_8_1_0(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
	Zp1 t0, t1, t2, t3; \
	BACKWARD_4_1(zi0, zi1, zi2, zi3, t0, t1, t2, t3, win1, win20, win21); \
	zo0 = backward2(t0); zo2 = backward2(t2); zo1 = backward2(t1); zo3 = backward2(t3);

#define SQUARE_22_1(z0, z1, z2, z3, w) \
	const Zp1 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add1(sqr1(u0), mul1(sqr1(u1), w)); z1 = mul1(add1(u0, u0), u1); \
	z2 = sub1(sqr1(u2), mul1(sqr1(u3), w)); z3 = mul1(add1(u2, u2), u3);

#define SQUARE_4_1(z0, z1, z2, z3, w, win) \
	const Zp1 u0 = z0, u2 = mul1(z2, w), u1 = z1, u3 = mul1(z3, w); \
	const Zp1 v0 = add1(u0, u2), v2 = sub1(u0, u2), v1 = add1(u1, u3), v3 = sub1(u1, u3); \
	const Zp1 s0 = add1(sqr1(v0), mul1(sqr1(v1), w)), s1 = mul1(add1(v0, v0), v1); \
	const Zp1 s2 = sub1(sqr1(v2), mul1(sqr1(v3), w)), s3 = mul1(add1(v2, v2), v3); \
	z0 = add1(s0, s2); z2 = mul1(sub1(s2, s0), win); \
	z1 = add1(s1, s3); z3 = mul1(sub1(s3, s1), win); \

#define FWD_2_1(zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \
	const Zp1 u0 = zi0, u2 = mul1(zi2, w), u1 = zi1, u3 = mul1(zi3, w); \
	zo0 = add1(u0, u2); zo2 = sub1(u0, u2); zo1 = add1(u1, u3); zo3 = sub1(u1, u3);

#define MUL_22_1(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
	const Zp1 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \
	const Zp1 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = add1(mul1(u0, u0p), mul1(mul1(u1, u1p), w)); \
	z1 = add1(mul1(u0, u1p), mul1(u0p, u1)); \
	z2 = sub1(mul1(u2, u2p), mul1(mul1(u3, u3p), w)); \
	z3 = add1(mul1(u2, u3p), mul1(u2p, u3));

#define MUL_4_1(z0, z1, z2, z3, z0p, z1p, z2p, z3p, w, win) \
	const Zp1 u0 = z0, u2 = mul1(z2, w), u1 = z1, u3 = mul1(z3, w); \
	const Zp1 v0 = add1(u0, u2), v2 = sub1(u0, u2), v1 = add1(u1, u3), v3 = sub1(u1, u3); \
	const Zp1 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \
	const Zp1 s0 = add1(mul1(v0, v0p), mul1(mul1(v1, v1p), w)); \
	const Zp1 s1 = add1(mul1(v0, v1p), mul1(v0p, v1)); \
	const Zp1 s2 = sub1(mul1(v2, v2p), mul1(mul1(v3, v3p), w)); \
	const Zp1 s3 = add1(mul1(v2, v3p), mul1(v2p, v3)); \
	z0 = add1(s0, s2); z2 = mul1(sub1(s2, s0), win); \
	z1 = add1(s1, s3); z3 = mul1(sub1(s3, s1), win); \


INLINE void forward_4io_1(const sz_t m, __global Zp1 * restrict const z, __global const Zp1 * restrict const w, const sz_t j)
{
	const Zp1 w1 = w[j], w20 = w[NSIZE / 2 + j], w21 = w[NSIZE + j];
	FORWARD_4_1(z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], w1, w20, w21);
}

INLINE void backward_4io_1(const sz_t m, __global Zp1 * restrict const z, __global const Zp1 * restrict const w, const sz_t ji)
{
	const Zp1 win1 = swap1(w[ji]), win21 = swap1(w[NSIZE / 2 + ji]), win20 = swap1(w[NSIZE + ji]);
	BACKWARD_4_1(z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], win1, win20, win21);
}

INLINE void forward_8io_1_0(__global Zp1 * restrict const z, __global const Zp1 * restrict const w)
{
	const Zp1 w1 = w[1], w20 = w[NSIZE / 2 + 1], w21 = w[NSIZE + 1];
	FORWARD_8_1_0(z[0 * NSIZE / 4], z[1 * NSIZE / 4], z[2 * NSIZE / 4], z[3 * NSIZE / 4],
		z[0 * NSIZE / 4], z[1 * NSIZE / 4], z[2 * NSIZE / 4], z[3 * NSIZE / 4], w1, w20, w21);
}

INLINE void backward_8io_1_0(__global Zp1 * restrict const z, __global const Zp1 * restrict const w)
{
	const Zp1 win1 = swap1(w[1]), win21 = swap1(w[NSIZE / 2 + 1]), win20 = swap1(w[NSIZE + 1]);
	BACKWARD_8_1_0(z[0 * NSIZE / 4], z[1 * NSIZE / 4], z[2 * NSIZE / 4], z[3 * NSIZE / 4],
		z[0 * NSIZE / 4], z[1 * NSIZE / 4], z[2 * NSIZE / 4], z[3 * NSIZE / 4], win1, win20, win21);
}

INLINE void square_22io_1(__global Zp1 * restrict const z, const Zp1 w)
{
	SQUARE_22_1(z[0], z[1], z[2], z[3], w);
}

INLINE void square_4io_1(__global Zp1 * restrict const z, const Zp1 w, const Zp1 win)
{
	SQUARE_4_1(z[0], z[1], z[2], z[3], w, win);
}

INLINE void fwd_2io_1(__global Zp1 * restrict const z, const Zp1 w)
{
	FWD_2_1(z[0], z[1], z[2], z[3], z[0], z[1], z[2], z[3], w)
}

INLINE void mul_22io_1(__global Zp1 * restrict const z, const __global Zp1 * restrict const zp, const Zp1 w)
{
	MUL_22_1(z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w);
}

INLINE void mul_4io_1(__global Zp1 * restrict const z, const __global Zp1 * restrict const zp, const Zp1 w, const Zp1 win)
{
	MUL_4_1(z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w, win);
}

// -----------------

INLINE void forward_4_1(const sz_t m, __local Zp1 * restrict const Z, __global const Zp1 * restrict const w, const sz_t j)
{
	const Zp1 w1 = w[j], w20 = w[NSIZE / 2 + j], w21 = w[NSIZE + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4_1(Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], w1, w20, w21);
}

INLINE void forward_4i_1(const sz_t ml, __local Zp1 * restrict const Z, const sz_t mg,
	__global const Zp1 * restrict const z, __global const Zp1 * restrict const w, const sz_t j)
{
	__global const Zp1 * const z2mg = &z[2 * mg];
	const Zp1 w1 = w[j], w20 = w[NSIZE / 2 + j], w21 = w[NSIZE + j];
	FORWARD_4_1(z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], w1, w20, w21);
}

INLINE void forward_4o_1(const sz_t mg, __global Zp1 * restrict const z, const sz_t ml,
	__local const Zp1 * restrict const Z, __global const Zp1 * restrict const w, const sz_t j)
{
	__global Zp1 * const z2mg = &z[2 * mg];
	const Zp1 w1 = w[j], w20 = w[NSIZE / 2 + j], w21 = w[NSIZE + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4_1(Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], w1, w20, w21);
}

INLINE void backward_4_1(const sz_t m, __local Zp1 * restrict const Z, __global const Zp1 * restrict const w, const sz_t ji)
{
	const Zp1 win1 = swap1(w[ji]), win21 = swap1(w[NSIZE / 2 + ji]), win20 = swap1(w[NSIZE + ji]);
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_1(Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], win1, win20, win21);
}

INLINE void backward_4i_1(const sz_t ml, __local Zp1 * restrict const Z, const sz_t mg,
	__global const Zp1 * restrict const z, __global const Zp1 * restrict const w, const sz_t ji)
{
	__global const Zp1 * const z2mg = &z[2 * mg];
	const Zp1 win1 = swap1(w[ji]), win21 = swap1(w[NSIZE / 2 + ji]), win20 = swap1(w[NSIZE + ji]);
	BACKWARD_4_1(z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], win1, win20, win21);
}

INLINE void backward_4o_1(const sz_t mg, __global Zp1 * restrict const z, const sz_t ml,
	__local const Zp1 * restrict const Z, __global const Zp1 * restrict const w, const sz_t ji)
{
	__global Zp1 * const z2mg = &z[2 * mg];
	const Zp1 win1 = swap1(w[ji]), win21 = swap1(w[NSIZE / 2 + ji]), win20 = swap1(w[NSIZE + ji]);
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4_1(Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], win1, win20, win21);
}

INLINE void forward_8i_1_0(const sz_t ml, __local Zp1 * restrict const Z, const sz_t mg,
	__global const Zp1 * restrict const z, __global const Zp1 * restrict const w)
{
	__global const Zp1 * const z2mg = &z[2 * mg];
	const Zp1 w1 = w[1], w20 = w[NSIZE / 2 + 1], w21 = w[NSIZE + 1];
	FORWARD_8_1_0(z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], w1, w20, w21);
}

INLINE void backward_8o_1_0(const sz_t mg, __global Zp1 * restrict const z, const sz_t ml,
	__local const Zp1 * restrict const Z, __global const Zp1 * restrict const w)
{
	__global Zp1 * const z2mg = &z[2 * mg];
	const Zp1 win1 = swap1(w[1]), win21 = swap1(w[NSIZE / 2 + 1]), win20 = swap1(w[NSIZE + 1]);
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_8_1_0(Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], win1, win20, win21);
}

INLINE void square_22_1(__local Zp1 * restrict const Z, const Zp1 w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_22_1(Z[0], Z[1], Z[2], Z[3], w);
}

INLINE void square_4_1(__local Zp1 * restrict const Z, const Zp1 w, const Zp1 win)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_4_1(Z[0], Z[1], Z[2], Z[3], w, win);
}

INLINE void write_4_1(const sz_t mg, __global Zp1 * restrict const z, __local const Zp1 * restrict const Z)
{
	__global Zp1 * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];
}

INLINE void fwd2_write_4_1(const sz_t mg, __global Zp1 * restrict const z, __local const Zp1 * restrict const Z, const Zp1 w)
{
	__global Zp1 * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	FWD_2_1(Z[0], Z[1], Z[2], Z[3], z[0], z[mg], z2mg[0], z2mg[mg], w);
}

INLINE void mul_22_1(__local Zp1 * restrict const Z, const sz_t mg, __global const Zp1 * restrict const z, const Zp1 w)
{
	__global const Zp1 * const z2mg = &z[2 * mg];
	const Zp1 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_22_1(Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w);
}

INLINE void mul_4_1(__local Zp1 * restrict const Z, const sz_t mg, __global const Zp1 * restrict const z, const Zp1 w, const Zp1 win)
{
	__global const Zp1 * const z2mg = &z[2 * mg];
	const Zp1 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_4_1(Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w, win);
}

// --- transform ---

__kernel
void forward4(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx >> lm, k = 3 * (j << lm) + idx;
	forward_4io_31((sz_t)(1) << lm, &z[k], w, s + j);
	forward_4io_1((sz_t)(1) << lm, &z[ZOFFSET_1 + k], &w[WOFFSET_1], s + j);
}

__kernel
void backward4(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx >> lm, k = 3 * (j << lm) + idx;
	backward_4io_31((sz_t)(1) << lm, &z[k], w, s + j);
	backward_4io_1((sz_t)(1) << lm, &z[ZOFFSET_1 + k], &w[WOFFSET_1], s + s - j - 1);
}

__kernel
void forward4_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t k = idx;
	forward_4io_31(NSIZE / 4, &z[k], w, 1);
	forward_8io_1_0(&z[ZOFFSET_1 + k], &w[WOFFSET_1]);
}

__kernel
void backward4_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t k = idx;
	backward_4io_31(NSIZE / 4, &z[k], w, 1);
	backward_8io_1_0(&z[ZOFFSET_1 + k], &w[WOFFSET_1]);
}

__kernel
void square22(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	square_22io_31(&z[k], w[NSIZE / 4 + j]);
	square_22io_1(&z[ZOFFSET_1 + k], w[WOFFSET_1 + NSIZE / 4 + j]);
}

__kernel
void square4(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	square_4io_31(&z[k], w[NSIZE / 4 + j]);
	square_4io_1(&z[ZOFFSET_1 + k], w[WOFFSET_1 + NSIZE / 4 + j], swap1(w[WOFFSET_1 + NSIZE / 4 + NSIZE / 4 - j - 1]));
}

__kernel
void fwd4p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	fwd_2io_31(&z[k], w[NSIZE / 4 + j]);
	fwd_2io_1(&z[ZOFFSET_1 + k], w[WOFFSET_1 + NSIZE / 4 + j]);
}

__kernel
void mul22(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	mul_22io_31(&z[k], &zp[k], w[NSIZE / 4 + j]);
	mul_22io_1(&z[ZOFFSET_1 + k], &zp[ZOFFSET_1 + k], w[WOFFSET_1 + NSIZE / 4 + j]);
}

__kernel
void mul4(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx, k = 4 * idx;
	mul_4io_31(&z[k], &zp[k], w[NSIZE / 4 + j]);
	mul_4io_1(&z[ZOFFSET_1 + k], &zp[ZOFFSET_1 + k], w[WOFFSET_1 + NSIZE / 4 + j], swap1(w[WOFFSET_1 + NSIZE / 4 + NSIZE / 4 - j - 1]));
}

// -----------------

#define DECLARE_VAR(B_N, CHUNK_N) \
	__local uint2 Z[4 * B_N * CHUNK_N]; \
	\
	/* threadIdx < B_N */ \
	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \
	__local uint2 * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	sz_t sj = s + idx_m, sji = s + s - idx_m - 1;

#define DECLARE_VAR_FORWARD() \
	__global GF31 * restrict const zi31 = &z[ki]; \
	__global GF31 * restrict const zo31 = &z[ko]; \
	__global Zp1 * restrict const zi1 = &zi31[ZOFFSET_1]; \
	__global Zp1 * restrict const zo1 = &zo31[ZOFFSET_1];

#define DECLARE_VAR_BACKWARD() \
	__global GF31 * restrict const zi31 = &z[ko]; \
	__global GF31 * restrict const zo31 = &z[ki]; \
	__global Zp1 * restrict const zi1 = &zi31[ZOFFSET_1]; \
	__global Zp1 * restrict const zo1 = &zo31[ZOFFSET_1];

#define FORWARD_I_31(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_31(B_N * CHUNK_N, &Z[i], B_N << lm, zi31, w, sj / B_N);

#define FORWARD_I_1(B_N, CHUNK_N) \
	forward_4i_1(B_N * CHUNK_N, &Z[i], B_N << lm, zi1, &w[WOFFSET_1], sj / B_N);

#define FORWARD_I_1_0(B_N, CHUNK_N) \
	forward_8i_1_0(B_N * CHUNK_N, &Z[i], B_N << lm, zi1, &w[WOFFSET_1]);

#define FORWARD_O_31(CHUNK_N) \
	forward_4o_31((sz_t)1 << lm, zo31, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], w, sj / 1);

#define FORWARD_O_1(CHUNK_N) \
	forward_4o_1((sz_t)1 << lm, zo1, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &w[WOFFSET_1], sj / 1);

#define BACKWARD_I_31(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i_31(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi31, w, sj / 1);

#define BACKWARD_I_1(B_N, CHUNK_N) \
	backward_4i_1(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi1, &w[WOFFSET_1], sji / 1);

#define BACKWARD_O_31(B_N, CHUNK_N) \
	backward_4o_31(B_N << lm, zo31, B_N * CHUNK_N, &Z[i], w, sj / B_N);

#define BACKWARD_O_1(B_N, CHUNK_N) \
	backward_4o_1(B_N << lm, zo1, B_N * CHUNK_N, &Z[i], &w[WOFFSET_1], sji / B_N);

#define BACKWARD_O_1_0(B_N, CHUNK_N) \
	backward_8o_1_0(B_N << lm, zo1, B_N * CHUNK_N, &Z[i], &w[WOFFSET_1]);

// -----------------

#define B_64	(64 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void forward64(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I_31(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4_31(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	FORWARD_O_31(CHUNK64);

	barrier(CLK_LOCAL_MEM_FENCE);

	FORWARD_I_1(B_64, CHUNK64);
	forward_4_1(4 * CHUNK64, &Zi[CHUNK64 * k4], &w[WOFFSET_1], sj / 4);
	FORWARD_O_1(CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void forward64_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 6; const unsigned int s = 64 / 4;

	FORWARD_I_31(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4_31(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	FORWARD_O_31(CHUNK64);

	barrier(CLK_LOCAL_MEM_FENCE);

	FORWARD_I_1_0(B_64, CHUNK64);
	forward_4_1(4 * CHUNK64, &Zi[CHUNK64 * k4], &w[WOFFSET_1], sj / 4);
	FORWARD_O_1(CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void backward64(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I_31(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4_31(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	BACKWARD_O_31(B_64, CHUNK64);

	barrier(CLK_LOCAL_MEM_FENCE);

	BACKWARD_I_1(B_64, CHUNK64);
	backward_4_1(4 * CHUNK64, &Zi[CHUNK64 * k4], &w[WOFFSET_1], sji / 4);
	BACKWARD_O_1(B_64, CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#endif
void backward64_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 6; const unsigned int s = 64 / 4;

	BACKWARD_I_31(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4_31(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	BACKWARD_O_31(B_64, CHUNK64);

	barrier(CLK_LOCAL_MEM_FENCE);

	BACKWARD_I_1(B_64, CHUNK64);
	backward_4_1(4 * CHUNK64, &Zi[CHUNK64 * k4], &w[WOFFSET_1], sji / 4);
	BACKWARD_O_1_0(B_64, CHUNK64);
}

// -----------------

#define B_256	(256 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void forward256(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I_31(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4_31(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4_31(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	FORWARD_O_31(CHUNK256);

	barrier(CLK_LOCAL_MEM_FENCE);

	FORWARD_I_1(B_256, CHUNK256);
	forward_4_1(16 * CHUNK256, &Zi[CHUNK256 * k16], &w[WOFFSET_1], sj / 16);
	forward_4_1(4 * CHUNK256, &Zi[CHUNK256 * k4], &w[WOFFSET_1], sj / 4);
	FORWARD_O_1(CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void forward256_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 8; const unsigned int s = 256 / 4;

	FORWARD_I_31(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4_31(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4_31(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	FORWARD_O_31(CHUNK256);

	barrier(CLK_LOCAL_MEM_FENCE);

	FORWARD_I_1_0(B_256, CHUNK256);
	forward_4_1(16 * CHUNK256, &Zi[CHUNK256 * k16], &w[WOFFSET_1], sj / 16);
	forward_4_1(4 * CHUNK256, &Zi[CHUNK256 * k4], &w[WOFFSET_1], sj / 4);
	FORWARD_O_1(CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void backward256(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I_31(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4_31(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4_31(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	BACKWARD_O_31(B_256, CHUNK256);

	barrier(CLK_LOCAL_MEM_FENCE);

	BACKWARD_I_1(B_256, CHUNK256);
	backward_4_1(4 * CHUNK256, &Zi[CHUNK256 * k4], &w[WOFFSET_1], sji / 4);
	backward_4_1(16 * CHUNK256, &Zi[CHUNK256 * k16], &w[WOFFSET_1], sji / 16);
	BACKWARD_O_1(B_256, CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#endif
void backward256_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 8; const unsigned int s = 256 / 4;

	BACKWARD_I_31(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4_31(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4_31(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	BACKWARD_O_31(B_256, CHUNK256);

	barrier(CLK_LOCAL_MEM_FENCE);

	BACKWARD_I_1(B_256, CHUNK256);
	backward_4_1(4 * CHUNK256, &Zi[CHUNK256 * k4], &w[WOFFSET_1], sji / 4);
	backward_4_1(16 * CHUNK256, &Zi[CHUNK256 * k16], &w[WOFFSET_1], sji / 16);
	BACKWARD_O_1_0(B_256, CHUNK256);
}

// -----------------

#define B_1024	(1024 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I_31(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4_31(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4_31(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4_31(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	FORWARD_O_31(CHUNK1024);

	barrier(CLK_LOCAL_MEM_FENCE);

	FORWARD_I_1(B_1024, CHUNK1024);
	forward_4_1(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &w[WOFFSET_1], sj / 64);
	forward_4_1(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &w[WOFFSET_1], sj / 16);
	forward_4_1(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &w[WOFFSET_1], sj / 4);
	FORWARD_O_1(CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 10; const unsigned int s = 1024 / 4;

	FORWARD_I_31(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4_31(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4_31(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4_31(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	FORWARD_O_31(CHUNK1024);

	barrier(CLK_LOCAL_MEM_FENCE);

	FORWARD_I_1_0(B_1024, CHUNK1024);
	forward_4_1(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &w[WOFFSET_1], sj / 64);
	forward_4_1(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &w[WOFFSET_1], sj / 16);
	forward_4_1(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &w[WOFFSET_1], sj / 4);
	FORWARD_O_1(CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void backward1024(__global uint2 * restrict const z, __global const uint2 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I_31(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4_31(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4_31(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4_31(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	BACKWARD_O_31(B_1024, CHUNK1024);

	barrier(CLK_LOCAL_MEM_FENCE);

	BACKWARD_I_1(B_1024, CHUNK1024);
	backward_4_1(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &w[WOFFSET_1], sji / 4);
	backward_4_1(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &w[WOFFSET_1], sji / 16);
	backward_4_1(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &w[WOFFSET_1], sji / 64);
	BACKWARD_O_1(B_1024, CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#endif
void backward1024_0(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	const int lm = LNSZ - 10; const unsigned int s = 1024 / 4;

	BACKWARD_I_31(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4_31(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4_31(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4_31(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	BACKWARD_O_31(B_1024, CHUNK1024);

	barrier(CLK_LOCAL_MEM_FENCE);

	BACKWARD_I_1(B_1024, CHUNK1024);
	backward_4_1(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &w[WOFFSET_1], sji / 4);
	backward_4_1(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &w[WOFFSET_1], sji / 16);
	backward_4_1(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &w[WOFFSET_1], sji / 64);
	BACKWARD_O_1_0(B_1024, CHUNK1024);
}

// -----------------

#define DECLARE_VAR_32() \
	__local uint2 Z[32 * BLK32]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k32 = (sz_t)get_group_id(0) * 32 * BLK32, i = (sz_t)get_local_id(0); \
	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \
	\
	__global GF31 * restrict const zk31 = &z[k32 + i32 + i8]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Z32 = &Z[i32]; \
	__local uint2 * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \
	__local uint2 * const Zi2 = &Z32[i2]; \
	__local uint2 * const Z4 = &Z32[4 * i8];

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void square32(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i_31(8, Zi8, 8, zk31, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	square_22_31(Z4, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4o_31(8, zk31, 8, Zi8, w, j / 8);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(8, Zi8, 8, zk1, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	square_22_1(Z4, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4o_1(8, zk1, 8, Zi8, &w[WOFFSET_1], ji / 8);
}

#define DECLARE_VAR_64() \
	__local uint2 Z[64 * BLK64]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k64 = (sz_t)get_group_id(0) * 64 * BLK64, i = (sz_t)get_local_id(0); \
	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \
	\
	__global GF31 * restrict const zk31 = &z[k64 + i64 + i16]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Z64 = &Z[i64]; \
	__local uint2 * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \
	__local uint2 * const Zi4 = &Z64[i4]; \
	__local uint2 * const Z4 = &Z64[4 * i16];

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void square64(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i_31(16, Zi16, 16, zk31, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	square_4_31(Z4, w[j]);
	backward_4_31(4, Zi4, w, j / 4);
	backward_4o_31(16, zk31, 16, Zi16, w, j / 16);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(16, Zi16, 16, zk1, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	square_4_1(Z4, w[WOFFSET_1 + j], swap1(w[WOFFSET_1 + ji]));
	backward_4_1(4, Zi4, &w[WOFFSET_1], ji / 4);
	backward_4o_1(16, zk1, 16, Zi16, &w[WOFFSET_1], ji / 16);
}

#define DECLARE_VAR_128() \
	__local uint2 Z[128 * BLK128]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k128 = (sz_t)get_group_id(0) * 128 * BLK128, i = (sz_t)get_local_id(0); \
	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \
	\
	__global GF31 * restrict const zk31 = &z[k128 + i128 + i32]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Z128 = &Z[i128]; \
	__local uint2 * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \
	__local uint2 * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \
	__local uint2 * const Zi2 = &Z128[i2]; \
	__local uint2 * const Z4 = &Z128[4 * i32];

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void square128(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i_31(32, Zi32, 32, zk31, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	square_22_31(Z4, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4_31(8, Zi8, w, j / 8);
	backward_4o_31(32, zk31, 32, Zi32, w, j / 32);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(32, Zi32, 32, zk1, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	square_22_1(Z4, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4_1(8, Zi8, &w[WOFFSET_1], ji / 8);
	backward_4o_1(32, zk1, 32, Zi32, &w[WOFFSET_1], ji / 32);
}

#define DECLARE_VAR_256() \
	__local uint2 Z[256 * BLK256]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k256 = (sz_t)get_group_id(0) * 256 * BLK256, i = (sz_t)get_local_id(0); \
	const sz_t i256 = (i & (sz_t)~(256 / 4 - 1)) * 4, i64 = i % (256 / 4); \
	\
	__global GF31 * restrict const zk31 = &z[k256 + i256 + i64]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Z256 = &Z[i256]; \
	__local uint2 * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \
	__local uint2 * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \
	__local uint2 * const Zi4 = &Z256[i4]; \
	__local uint2 * const Z4 = &Z256[4 * i64];

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void square256(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_256();

	forward_4i_31(64, Zi64, 64, zk31, w, j / 64);
	forward_4_31(16, Zi16, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	square_4_31(Z4, w[j]);
	backward_4_31(4, Zi4, w, j / 4);
	backward_4_31(16, Zi16, w, j / 16);
	backward_4o_31(64, zk31, 64, Zi64, w, j / 64);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(64, Zi64, 64, zk1, &w[WOFFSET_1], j / 64);
	forward_4_1(16, Zi16, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	square_4_1(Z4, w[WOFFSET_1 + j], swap1(w[WOFFSET_1 + ji]));
	backward_4_1(4, Zi4, &w[WOFFSET_1], ji / 4);
	backward_4_1(16, Zi16, &w[WOFFSET_1], ji / 16);
	backward_4o_1(64, zk1, 64, Zi64, &w[WOFFSET_1], ji / 64);
}

#define DECLARE_VAR_512() \
	__local uint2 Z[512]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k512 = (sz_t)get_group_id(0) * 512, i128 = (sz_t)get_local_id(0); \
	\
	__global GF31 * restrict const zk31 = &z[k512 + i128]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \
	__local uint2 * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \
	__local uint2 * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \
	__local uint2 * const Zi2 = &Z[i2]; \
	__local uint2 * const Z4 = &Z[4 * i128];

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void square512(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_512();

	forward_4i_31(128, Zi128, 128, zk31, w, j / 128);
	forward_4_31(32, Zi32, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	square_22_31(Z4, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4_31(8, Zi8, w, j / 8);
	backward_4_31(32, Zi32, w, j / 32);
	backward_4o_31(128, zk31, 128, Zi128, w, j / 128);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(128, Zi128, 128, zk1, &w[WOFFSET_1], j / 128);
	forward_4_1(32, Zi32, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	square_22_1(Z4, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4_1(8, Zi8, &w[WOFFSET_1], ji / 8);
	backward_4_1(32, Zi32, &w[WOFFSET_1], ji / 32);
	backward_4o_1(128, zk1, 128, Zi128, &w[WOFFSET_1], ji / 128);
}

#define DECLARE_VAR_1024() \
	__local uint2 Z[1024]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k1024 = (sz_t)get_group_id(0) * 1024, i256 = (sz_t)get_local_id(0); \
	\
	__global GF31 * restrict const zk31 = &z[k1024 + i256]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \
	__local uint2 * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \
	__local uint2 * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \
	__local uint2 * const Zi4 = &Z[i4]; \
	__local uint2 * const Z4 = &Z[4 * i256];

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void square1024(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_1024();

	forward_4i_31(256, Zi256, 256, zk31, w, j / 256);
	forward_4_31(64, Zi64, w, j / 64);
	forward_4_31(16, Zi16, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	square_4_31(Z4, w[j]);
	backward_4_31(4, Zi4, w, j / 4);
	backward_4_31(16, Zi16, w, j / 16);
	backward_4_31(64, Zi64, w, j / 64);
	backward_4o_31(256, zk31, 256, Zi256, w, j / 256);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(256, Zi256, 256, zk1, &w[WOFFSET_1], j / 256);
	forward_4_1(64, Zi64, &w[WOFFSET_1], j / 64);
	forward_4_1(16, Zi16, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	square_4_1(Z4, w[WOFFSET_1 + j], swap1(w[WOFFSET_1 + ji]));
	backward_4_1(4, Zi4, &w[WOFFSET_1], ji / 4);
	backward_4_1(16, Zi16, &w[WOFFSET_1], ji / 16);
	backward_4_1(64, Zi64, &w[WOFFSET_1], ji / 64);
	backward_4o_1(256, zk1, 256, Zi256, &w[WOFFSET_1], ji / 256);
}

#define DECLARE_VAR_2048() \
	__local uint2 Z[2048]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \
	\
	const sz_t k2048 = (sz_t)get_group_id(0) * 2048, i512 = (sz_t)get_local_id(0); \
	\
	__global GF31 * restrict const zk31 = &z[k2048 + i512]; \
	__global Zp1 * restrict const zk1 = &zk31[ZOFFSET_1]; \
	__local uint2 * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \
	__local uint2 * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \
	__local uint2 * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \
	__local uint2 * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \
	__local uint2 * const Zi2 = &Z[i2]; \
	__local uint2 * const Z4 = &Z[4 * i512];

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void square2048(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_2048();

	forward_4i_31(512, Zi512, 512, zk31, w, j / 512);
	forward_4_31(128, Zi128, w, j / 128);
	forward_4_31(32, Zi32, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	square_22_31(Z4, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4_31(8, Zi8, w, j / 8);
	backward_4_31(32, Zi32, w, j / 32);
	backward_4_31(128, Zi128, w, j / 128);
	backward_4o_31(512, zk31, 512, Zi512, w, j / 512);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(512, Zi512, 512, zk1, &w[WOFFSET_1], j / 512);
	forward_4_1(128, Zi128, &w[WOFFSET_1], j / 128);
	forward_4_1(32, Zi32, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	square_22_1(Z4, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4_1(8, Zi8, &w[WOFFSET_1], ji / 8);
	backward_4_1(32, Zi32, &w[WOFFSET_1], ji / 32);
	backward_4_1(128, Zi128, &w[WOFFSET_1], ji / 128);
	backward_4o_1(512, zk1, 512, Zi512, &w[WOFFSET_1], ji / 512);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i_31(8, Zi8, 8, zk31, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	write_4_31(8, zk31, Z4);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(8, Zi8, 8, zk1, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	write_4_1(8, zk1, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i_31(16, Zi16, 16, zk31, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	fwd2_write_4_31(16, zk31, Z4, w[j]);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(16, Zi16, 16, zk1, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	fwd2_write_4_1(16, zk1, Z4, w[WOFFSET_1 + j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i_31(32, Zi32, 32, zk31, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	write_4_31(32, zk31, Z4);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(32, Zi32, 32, zk1, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	write_4_1(32, zk1, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_256();

	forward_4i_31(64, Zi64, 64, zk31, w, j / 64);
	forward_4_31(16, Zi16, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	fwd2_write_4_31(64, zk31, Z4, w[j]);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(64, Zi64, 64, zk1, &w[WOFFSET_1], j / 64);
	forward_4_1(16, Zi16, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	fwd2_write_4_1(64, zk1, Z4, w[WOFFSET_1 + j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void fwd512p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_512();

	forward_4i_31(128, Zi128, 128, zk31, w, j / 128);
	forward_4_31(32, Zi32, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	write_4_31(128, zk31, Z4);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(128, Zi128, 128, zk1, &w[WOFFSET_1], j / 128);
	forward_4_1(32, Zi32, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	write_4_1(128, zk1, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void fwd1024p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_1024();

	forward_4i_31(256, Zi256, 256, zk31, w, j / 256);
	forward_4_31(64, Zi64, w, j / 64);
	forward_4_31(16, Zi16, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	fwd2_write_4_31(256, zk31, Z4, w[j]);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(256, Zi256, 256, zk1, &w[WOFFSET_1], j / 256);
	forward_4_1(64, Zi64, &w[WOFFSET_1], j / 64);
	forward_4_1(16, Zi16, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	fwd2_write_4_1(256, zk1, Z4, w[WOFFSET_1 + j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void fwd2048p(__global uint2 * restrict const z, __global const uint2 * restrict const w)
{
	DECLARE_VAR_2048();

	forward_4i_31(512, Zi512, 512, zk31, w, j / 512);
	forward_4_31(128, Zi128, w, j / 128);
	forward_4_31(32, Zi32, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	write_4_31(512, zk31, Z4);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(512, Zi512, 512, zk1, &w[WOFFSET_1], j / 512);
	forward_4_1(128, Zi128, &w[WOFFSET_1], j / 128);
	forward_4_1(32, Zi32, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	write_4_1(512, zk1, Z4);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void mul32(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_32();
	__global const GF31 * restrict const zpk31 = &zp[k32 + i32 + i8];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(8, Zi8, 8, zk31, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	mul_22_31(Z4, 8, zpk31, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4o_31(8, zk31, 8, Zi8, w, j / 8);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(8, Zi8, 8, zk1, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	mul_22_1(Z4, 8, zpk1, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4o_1(8, zk1, 8, Zi8, &w[WOFFSET_1], ji / 8);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void mul64(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_64();
	__global const GF31 * restrict const zpk31 = &zp[k64 + i64 + i16];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(16, Zi16, 16, zk31, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	mul_4_31(Z4, 16, zpk31, w[j]);
	backward_4_31(4, Zi4, w, j / 4);
	backward_4o_31(16, zk31, 16, Zi16, w, j / 16);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(16, Zi16, 16, zk1, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	mul_4_1(Z4, 16, zpk1, w[WOFFSET_1 + j], swap1(w[WOFFSET_1 + ji]));
	backward_4_1(4, Zi4, &w[WOFFSET_1], ji / 4);
	backward_4o_1(16, zk1, 16, Zi16, &w[WOFFSET_1], ji / 16);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void mul128(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_128();
	__global const GF31 * restrict const zpk31 = &zp[k128 + i128 + i32];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(32, Zi32, 32, zk31, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	mul_22_31(Z4, 32, zpk31, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4_31(8, Zi8, w, j / 8);
	backward_4o_31(32, zk31, 32, Zi32, w, j / 32);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(32, Zi32, 32, zk1, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	mul_22_1(Z4, 32, zpk1, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4_1(8, Zi8, &w[WOFFSET_1], ji / 8);
	backward_4o_1(32, zk1, 32, Zi32, &w[WOFFSET_1], ji / 32);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void mul256(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_256();
	__global const GF31 * restrict const zpk31 = &zp[k256 + i256 + i64];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(64, Zi64, 64, zk31, w, j / 64);
	forward_4_31(16, Zi16, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	mul_4_31(Z4, 64, zpk31, w[j]);
	backward_4_31(4, Zi4, w, j / 4);
	backward_4_31(16, Zi16, w, j / 16);
	backward_4o_31(64, zk31, 64, Zi64, w, j / 64);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(64, Zi64, 64, zk1, &w[WOFFSET_1], j / 64);
	forward_4_1(16, Zi16, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	mul_4_1(Z4, 64, zpk1, w[WOFFSET_1 + j], swap1(w[WOFFSET_1 + ji]));
	backward_4_1(4, Zi4, &w[WOFFSET_1], ji / 4);
	backward_4_1(16, Zi16, &w[WOFFSET_1], ji / 16);
	backward_4o_1(64, zk1, 64, Zi64, &w[WOFFSET_1], ji / 64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void mul512(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_512();
	__global const GF31 * restrict const zpk31 = &zp[k512 + i128];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(128, Zi128, 128, zk31, w, j / 128);
	forward_4_31(32, Zi32, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	mul_22_31(Z4, 128, zpk31, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4_31(8, Zi8, w, j / 8);
	backward_4_31(32, Zi32, w, j / 32);
	backward_4o_31(128, zk31, 128, Zi128, w, j / 128);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(128, Zi128, 128, zk1, &w[WOFFSET_1], j / 128);
	forward_4_1(32, Zi32, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	mul_22_1(Z4, 128, zpk1, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4_1(8, Zi8, &w[WOFFSET_1], ji / 8);
	backward_4_1(32, Zi32, &w[WOFFSET_1], ji / 32);
	backward_4o_1(128, zk1, 128, Zi128, &w[WOFFSET_1], ji / 128);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void mul1024(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_1024();
	__global const GF31 * restrict const zpk31 = &zp[k1024 + i256];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(256, Zi256, 256, zk31, w, j / 256);
	forward_4_31(64, Zi64, w, j / 64);
	forward_4_31(16, Zi16, w, j / 16);
	forward_4_31(4, Zi4, w, j / 4);
	mul_4_31(Z4, 256, zpk31, w[j]);
	backward_4_31(4, Zi4, w, j / 4);
	backward_4_31(16, Zi16, w, j / 16);
	backward_4_31(64, Zi64, w, j / 64);
	backward_4o_31(256, zk31, 256, Zi256, w, j / 256);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(256, Zi256, 256, zk1, &w[WOFFSET_1], j / 256);
	forward_4_1(64, Zi64, &w[WOFFSET_1], j / 64);
	forward_4_1(16, Zi16, &w[WOFFSET_1], j / 16);
	forward_4_1(4, Zi4, &w[WOFFSET_1], j / 4);
	mul_4_1(Z4, 256, zpk1, w[WOFFSET_1 + j], swap1(w[WOFFSET_1 + ji]));
	backward_4_1(4, Zi4, &w[WOFFSET_1], ji / 4);
	backward_4_1(16, Zi16, &w[WOFFSET_1], ji / 16);
	backward_4_1(64, Zi64, &w[WOFFSET_1], ji / 64);
	backward_4o_1(256, zk1, 256, Zi256, &w[WOFFSET_1], ji / 256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void mul2048(__global uint2 * restrict const z, __global const uint2 * restrict const zp, __global const uint2 * restrict const w)
{
	DECLARE_VAR_2048();
	__global const GF31 * restrict const zpk31 = &zp[k2048 + i512];
	__global const Zp1 * restrict const zpk1 = &zpk31[ZOFFSET_1];

	forward_4i_31(512, Zi512, 512, zk31, w, j / 512);
	forward_4_31(128, Zi128, w, j / 128);
	forward_4_31(32, Zi32, w, j / 32);
	forward_4_31(8, Zi8, w, j / 8);
	forward_4_31(2, Zi2, w, j / 2);
	mul_22_31(Z4, 512, zpk31, w[j]);
	backward_4_31(2, Zi2, w, j / 2);
	backward_4_31(8, Zi8, w, j / 8);
	backward_4_31(32, Zi32, w, j / 32);
	backward_4_31(128, Zi128, w, j / 128);
	backward_4o_31(512, zk31, 512, Zi512, w, j / 512);

	barrier(CLK_LOCAL_MEM_FENCE);

	forward_4i_1(512, Zi512, 512, zk1, &w[WOFFSET_1], j / 512);
	forward_4_1(128, Zi128, &w[WOFFSET_1], j / 128);
	forward_4_1(32, Zi32, &w[WOFFSET_1], j / 32);
	forward_4_1(8, Zi8, &w[WOFFSET_1], j / 8);
	forward_4_1(2, Zi2, &w[WOFFSET_1], j / 2);
	mul_22_1(Z4, 512, zpk1, w[WOFFSET_1 + j]);
	backward_4_1(2, Zi2, &w[WOFFSET_1], ji / 2);
	backward_4_1(8, Zi8, &w[WOFFSET_1], ji / 8);
	backward_4_1(32, Zi32, &w[WOFFSET_1], ji / 32);
	backward_4_1(128, Zi128, &w[WOFFSET_1], ji / 128);
	backward_4o_1(512, zk1, 512, Zi512, &w[WOFFSET_1], ji / 512);
}

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

INLINE int64_2 garner2(const GF31 r1, const Zp1 r2)
{
	const uint32 InvP1_M1 = 8421505u;	// 1 / P1 (mod M1)
	const uint64 M31P1 = M31 * (uint64)(P1);
	GF31 r2_1 = (GF31)(r2);	// P1 < M31
	GF31 u12 = muls31(sub31(r1, r2_1), InvP1_M1);
	const uint64 n0 = r2.s0 + u12.s0 * (uint64)(P1), n1 = r2.s1 + u12.s1 * (uint64)(P1);
	return (int64_2)((n0 > M31P1 / 2) ? (int64)(n0 - M31P1) : (int64)(n0), (n1 > M31P1 / 2) ? (int64)(n1 - M31P1) : (int64)(n1));
}

__kernel
void normalize1(__global uint2 * restrict const z, __global long2 * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const unsigned int blk = abs(sblk);
	__global GF31 * restrict const zi31 = &z[blk * idx];
	__global Zp1 * restrict const zi1 = &z[ZOFFSET_1 + blk * idx];

	prefetch(zi31, (size_t)blk); prefetch(zi1, (size_t)blk);

	int64_2 f = 0;

	sz_t j = 0;
	do
	{
		const GF31 u31 = lshift31(zi31[j], SNORM31);
		const Zp1 u1 = muls1(zi1[j], NORM1);
		int64_2 l = garner2(u31, u1);
		if (sblk < 0) l += l;
		f += l;
		const int32_2 r = reduce64(&f, b, b_inv, b_s);
		zi31[j] = set_int31(r);
		zi1[j] = set_int1(r);

		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	if (i == 0) { const int64 t = f.s0; f.s0 = -f.s1; f.s1 = t; }	// a_n = -a_0
	c[i] = (long2)(f);
}

__kernel
void normalize2(__global uint2 * restrict const z, __global const long2 * restrict const c, 
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global GF31 * restrict const zi31 = &z[blk * idx];
	__global Zp1 * restrict const zi1 = &z[ZOFFSET_1 + blk * idx];

	int64_2 f = (int64_2)(c[idx]);

	sz_t j = 0;
	do
	{
		const GF31 u31 = zi31[j];
		const int32_2 i = get_int31(u31);
		f += (int64_2)(i.s0, i.s1);
		const int32_2 r = reduce64(&f, b, b_inv, b_s);
		zi31[j] = set_int31(r);
		zi1[j] = set_int1(r);

		if ((f.s0 == 0) && (f.s1 == 0)) return;
		++j;
	} while (j != blk - 1);

	const int32_2 f32 = (int32_2)((int32)(f.s0), (int32)(f.s1));
	zi31[blk - 1] = add31(zi31[blk - 1], set_int31(f32));
	zi1[blk - 1] = add1(zi1[blk - 1], set_int1(f32));
}

__kernel
void mulscalar(__global uint2 * restrict const z, __global long2 * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global GF31 * restrict const zi31 = &z[blk * idx];
	__global Zp1 * restrict const zi1 = &z[ZOFFSET_1 + blk * idx];

	prefetch(zi31, (size_t)blk); prefetch(zi1, (size_t)blk);

	int64_2 f = 0;

	sz_t j = 0;
	do
	{
		int64_2 l = garner2(zi31[j], zi1[j]) * a;
		f += l;;
		const int32_2 r = reduce64(&f, b, b_inv, b_s);
		zi31[j] = set_int31(r);
		zi1[j] = set_int1(r);

		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	if (i == 0) { const int64 t = f.s0; f.s0 = -f.s1; f.s1 = t; }	// a_n = -a_0
	c[i] = (long2)(f);
}

__kernel
void set(__global uint2 * restrict const z, const unsigned int a, const unsigned int offset)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const uint32 ai = (idx == 0) ? a : 0;
	z[idx + offset] = (uint2)(ai, 0);
}

__kernel
void copy(__global uint2 * restrict const z, const unsigned int dst, const unsigned int src, const unsigned int offset)
{
	const sz_t idx = (sz_t)get_global_id(0) + offset;
	z[dst + idx] = z[src + idx];
}

__kernel
void copyp(__global uint2 * restrict const zp, __global const uint2 * restrict const z, const unsigned int src, const unsigned int offset)
{
	const sz_t idx = (sz_t)get_global_id(0) + offset;
	zp[idx] = z[src + idx];
}
