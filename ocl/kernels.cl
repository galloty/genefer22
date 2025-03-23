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

#ifdef __NV_CL_C_VERSION
	#define PTX_ASM	1
#endif

#ifndef NSIZE
#define NSIZE		4096u
#define	LNSZ		12
#define	RNSSIZE		2
#define NORM1		2129666049u
#define NORM2		2112897025u
#define NORM3		2012774401u
#define WOFFSET		2048u
#define BLK32		32
#define BLK64		16
#define BLK128		8
#define BLK256		4
#define BLK512		2
#define CHUNK64		16
#define CHUNK256	4
#define CHUNK1024	1
#define MAX_WORK_GROUP_SIZE	256
#endif

typedef uint	sz_t;
typedef uint	uint32;
typedef int		int32;
typedef ulong	uint64;
typedef long	int64;
typedef uint2	uint32_2;
typedef uint4	uint32_4;

// --- Z/(127*2^24 + 1)Z ---

#define	P1		2130706433u
#define	Q1		2164260865u		// p * q = 1 (mod 2^32)
// #define	R1		33554430u		// 2^32 mod p
#define	RSQ1	402124772u		// (2^32)^2 mod p
// #define	H1		100663290u		// Montgomery form of the primitive root 3
#define	IM1		1930170389u		// MF of MF of I = 3^{(p - 1)/4} to convert input into MF
#define	SQRTI1	1626730317u		// MF of 3^{(p - 1)/8}
#define	ISQRTI1	856006302u		// MF of i * sqrt(i)

// --- Z/(63*2^25 + 1)Z ---

#define	P2		2113929217u
#define	Q2		2181038081u
// #define	R2		67108862u
#define	RSQ2	2111798781u
// #define	H2		335544310u		// MF of the primitive root 5
#define	IM2		1036950657u
#define	SQRTI2	338852760u
#define	ISQRTI2	1090446030u

// --- Z/(15*2^27 + 1)Z ---

#define	P3		2013265921u
#define	Q3		2281701377u
// #define	R3		268435454u
#define	RSQ3	1172168163u
// #define	H3		268435390u		// MF of the primitive root 31
#define	IM3		734725699u
#define	SQRTI3	1032137103u
#define	ISQRTI3	1964242958u

// ---

#define	PQ1		(uint32_2)(P1, Q1)
#define	PQ2		(uint32_2)(P2, Q2)
#define	PQ3		(uint32_2)(P3, Q3)

__constant uint32_2 g_pq[3] = { PQ1, PQ2, PQ3 };
__constant uint32_4 g_f0[3] = { (uint32_4)(RSQ1, IM1, SQRTI1, ISQRTI1), (uint32_4)(RSQ2, IM2, SQRTI2, ISQRTI2), (uint32_4)(RSQ3, IM3, SQRTI3, ISQRTI3) };

INLINE uint32 addmod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 t = lhs + rhs;
	return t - ((t >= p) ? p : 0);
}

INLINE uint32 submod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 t = lhs - rhs;
	return t + (((int32)(t) < 0) ? p : 0);
}

INLINE uint32 mulmod(const uint32 lhs, const uint32 rhs, const uint32_2 pq)
{
	const uint64 t = lhs * (uint64)(rhs);
	const uint32 lo = (uint32)(t), hi = (uint32)(t >> 32);
	const uint32 mp = mul_hi(lo * pq.s1, pq.s0);
	return submod(hi, mp, pq.s0);
}

INLINE uint32 sqrmod(const uint32 lhs, const uint32_2 pq) { return mulmod(lhs, lhs, pq); }

INLINE int32 get_int(const uint32 n, const uint32 p) { return (n >= p / 2) ? (int32)(n - p) : (int32)(n); }	// ? 2n >= p ?
INLINE uint32 set_int(const int32 i, const uint32 p) { return (i < 0) ? ((uint32)(i) + p) : (uint32)(i); }

// --- uint96/int96 ---

typedef struct { uint64 s0; uint32 s1; } uint96;
typedef struct { uint64 s0; int32 s1; } int96;

INLINE int96 int96_set_si(const int64 n) { int96 r; r.s0 = (uint64)(n); r.s1 = (n < 0) ? -1 : 0; return r; }
INLINE uint96 uint96_set(const uint64 s0, const int32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }

INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int32)(x.s1); return r; }
INLINE uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint32)(x.s1); return r; }

INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }

INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

INLINE int96 int96_neg(const int96 x)
{
	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - ((x.s0 != 0) ? 1 : 0);
	return r;
}

INLINE uint96 int96_abs(const int96 x)
{
	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;
	return int96_u(t);
}

INLINE int96 int96_add(const int96 x, const int96 y)
{
	int96 r;
#ifdef PTX_ASM
	asm volatile ("add.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y.s0));
	asm volatile ("addc.s32 %0, %1, %2;" : "=r" (r.s1) : "r" (x.s1), "r" (y.s1));
#else
	const uint64 s0 = x.s0 + y.s0;
	r.s0 = s0; r.s1 = x.s1 + y.s1 + ((s0 < y.s0) ? 1 : 0);
#endif
	return r;
}

INLINE uint96 uint96_add_64(const uint96 x, const ulong y)
{
	uint96 r;
#ifdef PTX_ASM
	asm volatile ("add.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (r.s1) : "r" (x.s1));
#else
	const uint64 s0 = x.s0 + y;
	r.s0 = s0; r.s1 = x.s1 + ((s0 < y) ? 1 : 0);
#endif
	return r;
}

INLINE int96 uint96_subi(const uint96 x, const uint96 y)
{
	int96 r;
#ifdef PTX_ASM
	asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y.s0));
	asm volatile ("subc.s32 %0, %1, %2;" : "=r" (r.s1) : "r" (x.s1), "r" (y.s1));
#else
	r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - ((x.s0 < y.s0) ? 1 : 0));
#endif
	return r;
}

INLINE uint96 uint96_mul_64_32(const uint64 x, const uint32 y)
{
	const uint64 l = (uint32)(x) * (uint64)(y), h = (x >> 32) * y + (l >> 32);
	uint96 r; r.s0 = (h << 32) | (uint32)(l); r.s1 = (uint32)(h >> 32);
	return r;
}

// --- transform/macro ---

// 16 mul + 16 mul_hi
#define FORWARD_4(pq, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0 = zi0, u2 = mulmod(zi2, w1, pq), u1 = zi1, u3 = mulmod(zi3, w1, pq); \
	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p); \
	const uint32 v1 = mulmod(addmod(u1, u3, p), w20, pq), v3 = mulmod(submod(u1, u3, p), w21, pq); \
	zo0 = addmod(v0, v1, p); zo1 = submod(v0, v1, p); zo2 = addmod(v2, v3, p); zo3 = submod(v2, v3, p); \
}

#define BACKWARD_4(pq, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \
	const uint32 v0 = addmod(u0, u1, p), v1 = mulmod(submod(u1, u0, p), win20, pq); \
	const uint32 v2 = addmod(u2, u3, p), v3 = mulmod(submod(u3, u2, p), win21, pq); \
	zo0 = addmod(v0, v2, p); zo2 = mulmod(submod(v2, v0, p), win1, pq); \
	zo1 = addmod(v1, v3, p); zo3 = mulmod(submod(v3, v1, p), win1, pq); \
}

#define FORWARD_4_0(pq, f0, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3) \
{ \
	const uint32 p = pq.s0, rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3; \
	const uint32 u0 = mulmod(zi0, rsq, pq), u2 = mulmod(zi2, im, pq); \
	const uint32 u1 = mulmod(zi1, rsq, pq), u3 = mulmod(zi3, im, pq); \
	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p); \
	const uint32 v1 = mulmod(addmod(u1, u3, p), sqrti, pq), v3 = mulmod(submod(u1, u3, p), isqrti, pq); \
	zo0 = addmod(v0, v1, p); zo1 = submod(v0, v1, p); zo2 = addmod(v2, v3, p); zo3 = submod(v2, v3, p); \
}

#define SQUARE_22(pq, z0, z1, z2, z3, w) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = addmod(sqrmod(u0, pq), mulmod(sqrmod(u1, pq), w, pq), p); z1 = mulmod(addmod(u0, u0, p), u1, pq); \
	z2 = submod(sqrmod(u2, pq), mulmod(sqrmod(u3, pq), w, pq), p); z3 = mulmod(addmod(u2, u2, p), u3, pq); \
}

#define SQUARE_4(pq, z0, z1, z2, z3, w, win) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0 = z0, u2 = mulmod(z2, w, pq), u1 = z1, u3 = mulmod(z3, w, pq); \
	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p), v1 = addmod(u1, u3, p), v3 = submod(u1, u3, p); \
	const uint32 s0 = addmod(sqrmod(v0, pq), mulmod(sqrmod(v1, pq), w, pq), p); \
	const uint32 s1 = mulmod(addmod(v0, v0, p), v1, pq); \
	const uint32 s2 = submod(sqrmod(v2, pq), mulmod(sqrmod(v3, pq), w, pq), p); \
	const uint32 s3 = mulmod(addmod(v2, v2, p), v3, pq); \
	z0 = addmod(s0, s2, p); z2 = mulmod(submod(s2, s0, p), win, pq); \
	z1 = addmod(s1, s3, p); z3 = mulmod(submod(s3, s1, p), win, pq); \
}

#define FWD_2(pq, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0 = zi0, u2 = mulmod(zi2, w, pq), u1 = zi1, u3 = mulmod(zi3, w, pq); \
	zo0 = addmod(u0, u2, p); zo2 = submod(u0, u2, p); zo1 = addmod(u1, u3, p); zo3 = submod(u1, u3, p); \
}

#define MUL_22(pq, z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \
	const uint32 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \
	z0 = addmod(mulmod(u0, u0p, pq), mulmod(mulmod(u1, u1p, pq), w, pq), p); \
	z1 = addmod(mulmod(u0, u1p, pq), mulmod(u0p, u1, pq), p); \
	z2 = submod(mulmod(u2, u2p, pq), mulmod(mulmod(u3, u3p, pq), w, pq), p); \
	z3 = addmod(mulmod(u2, u3p, pq), mulmod(u2p, u3, pq), p); \
}

#define MUL_4(pq, z0, z1, z2, z3, z0p, z1p, z2p, z3p, w, win) \
{ \
	const uint32 p = pq.s0; \
	const uint32 u0 = z0, u2 = mulmod(z2, w, pq), u1 = z1, u3 = mulmod(z3, w, pq); \
	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p), v1 = addmod(u1, u3, p), v3 = submod(u1, u3, p); \
	const uint32 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \
	const uint32 s0 = addmod(mulmod(v0, v0p, pq), mulmod(mulmod(v1, v1p, pq), w, pq), p); \
	const uint32 s1 = addmod(mulmod(v0, v1p, pq), mulmod(v0p, v1, pq), p); \
	const uint32 s2 = submod(mulmod(v2, v2p, pq), mulmod(mulmod(v3, v3p, pq), w, pq), p); \
	const uint32 s3 = addmod(mulmod(v2, v3p, pq), mulmod(v2p, v3, pq), p); \
	z0 = addmod(s0, s2, p); z2 = mulmod(submod(s2, s0, p), win, pq); \
	z1 = addmod(s1, s3, p); z3 = mulmod(submod(s3, s1, p), win, pq); \
}

// --- transform/inline global mem ---

INLINE void forward_4io(const uint32_2 pq, const sz_t m, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j)
{
	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];
	FORWARD_4(pq, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], w1, w20, w21);
}

INLINE void backward_4io(const uint32_2 pq, const sz_t m, __global uint * restrict const z, __global const uint * restrict const w, const sz_t ji)
{
	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];
	BACKWARD_4(pq, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], win1, win20, win21);
}

INLINE void forward_4io_0(const uint32_2 pq, const uint32_4 f0,	__global uint * restrict const z)
{
	const sz_t m = NSIZE / 4;
	FORWARD_4_0(pq, f0, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m]);
}

INLINE void square_22io(const uint32_2 pq, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j)
{
	SQUARE_22(pq, z[0], z[1], z[2], z[3], w[j]);
}

INLINE void square_4io(const uint32_2 pq, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j, const sz_t ji)
{
	SQUARE_4(pq, z[0], z[1], z[2], z[3], w[j], w[ji]);
}

INLINE void fwd_2io(const uint32_2 pq, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j)
{
	FWD_2(pq, z[0], z[1], z[2], z[3], z[0], z[1], z[2], z[3], w[j]);
}

INLINE void mul_22io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp,
	__global const uint * restrict const w, const sz_t j)
{
	MUL_22(pq, z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w[j]);
}

INLINE void mul_4io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp,
	__global const uint * restrict const w, const sz_t j, const sz_t ji)
{
	MUL_4(pq, z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w[j], w[ji]);
}

// --- transform/inline local & global mem ---

INLINE void forward_4(const uint32_2 pq, const sz_t m, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t j)
{
	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4(pq, Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], w1, w20, w21);
}

INLINE void forward_4i(const uint32_2 pq, const sz_t ml, __local uint * restrict const Z, const sz_t mg,
	__global const uint * restrict const z, __global const uint * restrict const w, const sz_t j)
{
	__global const uint * const z2mg = &z[2 * mg];
	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];
	FORWARD_4(pq, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], w1, w20, w21);
}

INLINE void forward_4i_0(const uint32_2 pq, const uint32_4 f0, const sz_t ml, __local uint * restrict const Z, const sz_t mg,
	__global const uint * restrict const z, __global const uint * restrict const w)
{
	__global const uint * const z2mg = &z[2 * mg];
	FORWARD_4_0(pq, f0, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml]);
}

INLINE void forward_4o(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, const sz_t ml,
	__local const uint * restrict const Z, __global const uint * restrict const w, const sz_t j)
{
	__global uint * const z2mg = &z[2 * mg];
	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	FORWARD_4(pq, Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], w1, w20, w21);
}

INLINE void backward_4(const uint32_2 pq, const sz_t m, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t ji)
{
	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4(pq, Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], win1, win20, win21);
}

INLINE void backward_4i(const uint32_2 pq, const sz_t ml, __local uint * restrict const Z, const sz_t mg,
	__global const uint * restrict const z, __global const uint * restrict const w, const sz_t ji)
{
	__global const uint * const z2mg = &z[2 * mg];
	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];
	BACKWARD_4(pq, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], win1, win20, win21);
}

INLINE void backward_4o(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, const sz_t ml,
	__local const uint * restrict const Z, __global const uint * restrict const w, const sz_t ji)
{
	__global uint * const z2mg = &z[2 * mg];
	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];
	barrier(CLK_LOCAL_MEM_FENCE);
	BACKWARD_4(pq, Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], win1, win20, win21);
}

INLINE void square_22(const uint32_2 pq, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t j)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_22(pq, Z[0], Z[1], Z[2], Z[3], w[j]);
}

INLINE void square_4(const uint32_2 pq, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t j, const sz_t ji)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	SQUARE_4(pq, Z[0], Z[1], Z[2], Z[3], w[j], w[ji]);
}

INLINE void write_4(const sz_t mg, __global uint * restrict const z, __local const uint * restrict const Z)
{
	__global uint * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];
}

INLINE void fwd2_write_4(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, __local const uint * restrict const Z,
	__global const uint * restrict const w, const sz_t j)
{
	__global uint * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	FWD_2(pq, Z[0], Z[1], Z[2], Z[3], z[0], z[mg], z2mg[0], z2mg[mg], w[j]);
}

INLINE void mul_22(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg, __global const uint * restrict const z,
	__global const uint * restrict const w, const sz_t j)
{
	__global const uint * const z2mg = &z[2 * mg];
	const uint z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_22(pq, Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w[j]);
}

INLINE void mul_4(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg, __global const uint * restrict const z,
	__global const uint * restrict const w, const sz_t j, const sz_t ji)
{
	__global const uint * const z2mg = &z[2 * mg];
	const uint z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	MUL_4(pq, Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w[j], w[ji]);
}

// --- transform/macro ---

#define DECLARE_VAR_REG() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LNSZ - 2), mid = gid & ~((NSIZE / 4) - 1), id = gid %  (NSIZE / 4); \
	const uint32_2 pq = g_pq[lid]; \
	__global uint * restrict const z = &zg[4 * mid]; \
	__global const uint * restrict const w = &wg[lid * WOFFSET];

#define DECLARE_VARP_REG() \
	__global const uint * restrict const zp = &zpg[4 * mid];

// --- transform without local mem ---

__kernel
void forward4(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	DECLARE_VAR_REG();
	const sz_t j = id >> lm, k = 3 * (j << lm) + id;
	forward_4io(pq, (sz_t)(1) << lm, &z[k], w, s + j);
}

__kernel
void backward4(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	DECLARE_VAR_REG();
	const sz_t j = id >> lm, k = 3 * (j << lm) + id;
	backward_4io(pq, (sz_t)(1) << lm, &z[k], w, s + s - j - 1);
}

__kernel
void forward4_0(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t k = id;
	forward_4io_0(pq, g_f0[lid], &z[k]);
}

__kernel
void square22(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id;
	square_22io(pq, &z[k], w, NSIZE / 4 + j);
}

__kernel
void square4(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id;
	square_4io(pq, &z[k], w, NSIZE / 4 +  j, NSIZE / 4 + NSIZE / 4 - j - 1);
}

__kernel
void fwd4p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id;
	fwd_2io(pq, &z[k], w, NSIZE / 4 + j);
}

__kernel
void mul22(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t j = id, k = 4 * id;
	mul_22io(pq, &z[k], &zp[k], w, NSIZE / 4 + j);
}

__kernel
void mul4(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t j = id, k = 4 * id;
	mul_4io(pq, &z[k], &zp[k], w, NSIZE / 4 + j, NSIZE / 4 + NSIZE / 4 - j - 1);
}

// --- transform ---

#define DECLARE_VAR(B_N, CHUNK_N) \
	/* threadIdx < B_N */ \
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \
	const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = group_id * CHUNK_N + chunk_idx; \
	__local uint * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	const sz_t sj = s + idx_m, sji = s + s - idx_m - 1;

#define DECLARE_VAR_FORWARD() \
	__global uint * restrict const zi = &z[ki]; \
	__global uint * restrict const zo = &z[ko];

#define DECLARE_VAR_BACKWARD() \
	__global uint * restrict const zi = &z[ko]; \
	__global uint * restrict const zo = &z[ki];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(pq, B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);

#define FORWARD_I_0(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_0(pq, g_f0[lid], B_N * CHUNK_N, &Z[i], B_N << lm, zi, w);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, w, sji / 1);

// -----------------

#define B_64	(64 / 4)

#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
#define ATTR_64() \
	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))
#else
#define ATTR_64()
#endif

#define FORWARD_64() \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);

INLINE void _forward64(__global uint * restrict const zg, __global const uint * restrict const wg,
	__local uint * const Z, const int lm, const unsigned int s)
{
	FORWARD_I(B_64, CHUNK64);
	FORWARD_64();
}

INLINE void _backward64(__global uint * restrict const zg, __global const uint * restrict const wg,
	__local uint * const Z, const int lm, const unsigned int s)
{
	BACKWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sji / 4);
	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], w, sji / B_64);
}

__kernel
ATTR_64()
void forward64(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local uint Z[4 * B_64 * CHUNK64];
	_forward64(zg, wg, Z, lm, s);
}

__kernel
ATTR_64()
void forward64_0(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	const int lm = LNSZ - 6; const unsigned int s = 64 / 4;
	__local uint Z[4 * B_64 * CHUNK64];
	FORWARD_I_0(B_64, CHUNK64);
	FORWARD_64();
}

__kernel
ATTR_64()
void backward64(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local uint Z[4 * B_64 * CHUNK64];
	_backward64(zg, wg, Z, lm, s);
}

// -----------------

#define B_256	(256 / 4)

#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
#define ATTR_256() \
	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))
#else
#define ATTR_256()
#endif

#define FORWARD_256() \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);

INLINE void _forward256(__global uint * restrict const zg, __global const uint * restrict const wg,
	__local uint * const Z, const int lm, const unsigned int s)
{
	FORWARD_I(B_256, CHUNK256);
	FORWARD_256();
}

INLINE void _backward256(__global uint * restrict const zg, __global const uint * restrict const wg,
	__local uint * const Z, const int lm, const unsigned int s)
{
	BACKWARD_I(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sji / 16);
	backward_4o(pq, B_256 << lm, zo, B_256 * CHUNK256, &Z[i], w, sji / B_256);
}

__kernel
ATTR_256()
void forward256(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local uint Z[4 * B_256 * CHUNK256];
	_forward256(zg, wg, Z, lm, s);
}

__kernel
ATTR_256()
void forward256_0(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	const int lm = LNSZ - 8; const unsigned int s = 256 / 4;
	__local uint Z[4 * B_256 * CHUNK256];
	FORWARD_I_0(B_256, CHUNK256);
	FORWARD_256();
}

__kernel
ATTR_256()
void backward256(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local uint Z[4 * B_256 * CHUNK256];
	_backward256(zg, wg, Z, lm, s);
}

// -----------------

#define B_1024	(1024 / 4)

#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
#define ATTR_1024() \
	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))
#else
#define ATTR_1024()
#endif

#define FORWARD_1024() \
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 ); \
	forward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64); \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	forward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);

INLINE void _forward1024(__global uint * restrict const zg, __global const uint * restrict const wg,
	__local uint * const Z, const int lm, const unsigned int s)
{
	FORWARD_I(B_1024, CHUNK1024);
	FORWARD_1024();
}

INLINE void _backward1024(__global uint * restrict const zg, __global const uint * restrict const wg,
	__local uint * const Z, const int lm, const unsigned int s)
{
	BACKWARD_I(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sji / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sji / 64);
	backward_4o(pq, B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], w, sji / B_1024);
}

__kernel
ATTR_1024()
void forward1024(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local uint Z[4 * B_1024 * CHUNK1024];
	_forward1024(zg, wg, Z, lm, s);
}

__kernel
ATTR_1024()
void forward1024_0(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	const int lm = LNSZ - 10; const unsigned int s = 1024 / 4;
	__local uint Z[4 * B_1024 * CHUNK1024];
	FORWARD_I_0(B_1024, CHUNK1024);
	FORWARD_1024();
}

__kernel
ATTR_1024()
void backward1024(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local uint Z[4 * B_1024 * CHUNK1024];
	_backward1024(zg, wg, Z, lm, s);
}

// -----------------

#define DEFINE_KERNEL_FORWARD(m, n) \
	__kernel \
	ATTR_##m() \
	void forward##m##_##n(__global uint * restrict const zg, __global const uint * restrict const wg) \
	{ \
		__local uint Z[4 * B_##m * CHUNK##m]; \
		_forward##m(zg, wg, Z, n, (NSIZE / 4) >> n); \
	}

#define DEFINE_KERNEL_BACKWARD(m, n) \
	__kernel \
	ATTR_##m() \
	void backward##m##_##n(__global uint * restrict const zg, __global const uint * restrict const wg) \
	{ \
		__local uint Z[4 * B_##m * CHUNK##m]; \
		_backward##m(zg, wg, Z, n, (NSIZE / 4) >> n); \
	}

#if LNSZ % 2 != 0

DEFINE_KERNEL_FORWARD(64, 5);
DEFINE_KERNEL_BACKWARD(64, 5);

#if LNSZ >= 19

DEFINE_KERNEL_FORWARD(64, 7);
DEFINE_KERNEL_BACKWARD(64, 7);
DEFINE_KERNEL_FORWARD(256, 5);
DEFINE_KERNEL_BACKWARD(256, 5);

#endif

#else // LNSZ % 2 == 0

DEFINE_KERNEL_FORWARD(64, 6);
DEFINE_KERNEL_BACKWARD(64, 6);

#if LNSZ >= 20

DEFINE_KERNEL_FORWARD(64, 8);
DEFINE_KERNEL_BACKWARD(64, 8);
DEFINE_KERNEL_FORWARD(256, 6);
DEFINE_KERNEL_BACKWARD(256, 6);

#endif

#endif

// -----------------

#define DECLARE_VAR_32() \
	__local uint Z[32 * BLK32]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (32 / 4 * BLK32), group_id = id / (32 / 4 * BLK32); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i32 = (local_id & ~(32 / 4 - 1)) * 4, i8 = local_id % (32 / 4); \
	const sz_t k32 = group_id * 32 * BLK32 + i32 + i8; \
	\
	__global uint * restrict const zk = &z[k32]; \
	__local uint * const Z32 = &Z[i32]; \
	__local uint * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & ~(4 * 2 - 1)) + (i8 % 2); \
	__local uint * const Zi2 = &Z32[i2]; \
	__local uint * const Z4 = &Z32[4 * i8];

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void square32(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_32();

	forward_4i(pq, 8, Zi8, 8, zk, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	square_22(pq, Z4, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4o(pq, 8, zk, 8, Zi8, w, ji / 8);
}

#define DECLARE_VAR_64() \
	__local uint Z[64 * BLK64]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (64 / 4 * BLK64), group_id = id / (64 / 4 * BLK64); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i64 = (local_id & ~(64 / 4 - 1)) * 4, i16 = local_id % (64 / 4); \
	const sz_t k64 = group_id * 64 * BLK64 + i64 + i16; \
	\
	__global uint * restrict const zk = &z[k64]; \
	__local uint * const Z64 = &Z[i64]; \
	__local uint * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & ~(4 * 4 - 1)) + (i16 % 4); \
	__local uint * const Zi4 = &Z64[i4]; \
	__local uint * const Z4 = &Z64[4 * i16];

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void square64(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_64();

	forward_4i(pq, 16, Zi16, 16, zk, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	square_4(pq, Z4, w, j, ji);
	backward_4(pq, 4, Zi4, w, ji / 4);
	backward_4o(pq, 16, zk, 16, Zi16, w, ji / 16);
}

#define DECLARE_VAR_128() \
	__local uint Z[128 * BLK128]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (128 / 4 * BLK128), group_id = id / (128 / 4 * BLK128); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i128 = (local_id & ~(128 / 4 - 1)) * 4, i32 = local_id % (128 / 4); \
	const sz_t k128 = group_id * 128 * BLK128 + i128 + i32; \
	\
	__global uint * restrict const zk = &z[k128]; \
	__local uint * const Z128 = &Z[i128]; \
	__local uint * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & ~(4 * 8 - 1)) + (i32 % 8); \
	__local uint * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & ~(4 * 2 - 1)) + (i32 % 2); \
	__local uint * const Zi2 = &Z128[i2]; \
	__local uint * const Z4 = &Z128[4 * i32];

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void square128(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_128();

	forward_4i(pq, 32, Zi32, 32, zk, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	square_22(pq, Z4, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4(pq, 8, Zi8, w, ji / 8);
	backward_4o(pq, 32, zk, 32, Zi32, w, ji / 32);
}

#define DECLARE_VAR_256() \
	__local uint Z[256 * BLK256]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (256 / 4 * BLK256), group_id = id / (256 / 4 * BLK256); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i256 = (local_id & ~(256 / 4 - 1)) * 4, i64 = local_id % (256 / 4); \
	const sz_t k256 = group_id * 256 * BLK256 + i256 + i64; \
	\
	__global uint * restrict const zk = &z[k256]; \
	__local uint * const Z256 = &Z[i256]; \
	__local uint * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & ~(4 * 16 - 1)) + (i64 % 16); \
	__local uint * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & ~(4 * 4 - 1)) + (i64 % 4); \
	__local uint * const Zi4 = &Z256[i4]; \
	__local uint * const Z4 = &Z256[4 * i64];

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void square256(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_256();

	forward_4i(pq, 64, Zi64, 64, zk, w, j / 64);
	forward_4(pq, 16, Zi16, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	square_4(pq, Z4, w, j, ji);
	backward_4(pq, 4, Zi4, w, ji / 4);
	backward_4(pq, 16, Zi16, w, ji / 16);
	backward_4o(pq, 64, zk, 64, Zi64, w, ji / 64);
}

// if BLK512 != 1 then const sz_t i512 = (i & ~(512 / 4 - 1)) * 4, i128 = i % (512 / 4);
// if BLK512 = 1 then const sz_t i512 = 0, i128 = i;
#define DECLARE_VAR_512() \
	__local uint Z[512 * BLK512]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (512 / 4 * BLK512), group_id = id / (512 / 4 * BLK512); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i512 = (local_id & ~(512 / 4 - 1)) * 4, i128 = local_id % (512 / 4); \
	const sz_t k512 = group_id * 512 * BLK512 + i512 + i128; \
	\
	__global uint * restrict const zk = &z[k512]; \
	__local uint * const Z512 = &Z[i512]; \
	__local uint * const Zi128 = &Z512[i128]; \
	const sz_t i32 = ((4 * i128) & ~(4 * 32 - 1)) + (i128 % 32); \
	__local uint * const Zi32 = &Z512[i32]; \
	const sz_t i8 = ((4 * i128) & ~(4 * 8 - 1)) + (i128 % 8); \
	__local uint * const Zi8 = &Z512[i8]; \
	const sz_t i2 = ((4 * i128) & ~(4 * 2 - 1)) + (i128 % 2); \
	__local uint * const Zi2 = &Z512[i2]; \
	__local uint * const Z4 = &Z512[4 * i128];

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void square512(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_512();

	forward_4i(pq, 128, Zi128, 128, zk, w, j / 128);
	forward_4(pq, 32, Zi32, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	square_22(pq, Z4, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4(pq, 8, Zi8, w, ji / 8);
	backward_4(pq, 32, Zi32, w, ji / 32);
	backward_4o(pq, 128, zk, 128, Zi128, w, ji / 128);
}

#define DECLARE_VAR_1024() \
	__local uint Z[1024]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (1024 / 4), group_id = id / (1024 / 4); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i256 = local_id, k1024 = group_id * 1024 + i256; \
	\
	__global uint * restrict const zk = &z[k1024]; \
	__local uint * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i256) & ~(4 * 64 - 1)) + (i256 % 64); \
	__local uint * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i256) & ~(4 * 16 - 1)) + (i256 % 16); \
	__local uint * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i256) & ~(4 * 4 - 1)) + (i256 % 4); \
	__local uint * const Zi4 = &Z[i4]; \
	__local uint * const Z4 = &Z[4 * i256];

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void square1024(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_1024();

	forward_4i(pq, 256, Zi256, 256, zk, w, j / 256);
	forward_4(pq, 64, Zi64, w, j / 64);
	forward_4(pq, 16, Zi16, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	square_4(pq, Z4, w, j, ji);
	backward_4(pq, 4, Zi4, w, ji / 4);
	backward_4(pq, 16, Zi16, w, ji / 16);
	backward_4(pq, 64, Zi64, w, ji / 64);
	backward_4o(pq, 256, zk, 256, Zi256, w, ji / 256);
}

#define DECLARE_VAR_2048() \
	__local uint Z[2048]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (2048 / 4), group_id = id / (2048 / 4); \
	const sz_t j = NSIZE / 4 + id, ji = NSIZE / 4 + NSIZE / 4 - id - 1; \
	\
	const sz_t i512 = local_id, k2048 = group_id * 2048 + i512; \
	\
	__global uint * restrict const zk = &z[k2048]; \
	__local uint * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & ~(4 * 128 - 1)) + (i512 % 128); \
	__local uint * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & ~(4 * 32 - 1)) + (i512 % 32); \
	__local uint * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & ~(4 * 8 - 1)) + (i512 % 8); \
	__local uint * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & ~(4 * 2 - 1)) + (i512 % 2); \
	__local uint * const Zi2 = &Z[i2]; \
	__local uint * const Z4 = &Z[4 * i512];

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void square2048(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_2048();

	forward_4i(pq, 512, Zi512, 512, zk, w, j / 512);
	forward_4(pq, 128, Zi128, w, j / 128);
	forward_4(pq, 32, Zi32, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	square_22(pq, Z4, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4(pq, 8, Zi8, w, ji / 8);
	backward_4(pq, 32, Zi32, w, ji / 32);
	backward_4(pq, 128, Zi128, w, ji / 128);
	backward_4o(pq, 512, zk, 512, Zi512, w, ji / 512);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_32();

	forward_4i(pq, 8, Zi8, 8, zk, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	write_4(8, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_64();

	forward_4i(pq, 16, Zi16, 16, zk, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	fwd2_write_4(pq, 16, zk, Z4, w, j);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_128();

	forward_4i(pq, 32, Zi32, 32, zk, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	write_4(32, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_256();

	forward_4i(pq, 64, Zi64, 64, zk, w, j / 64);
	forward_4(pq, 16, Zi16, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	fwd2_write_4(pq, 64, zk, Z4, w, j);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void fwd512p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_512();

	forward_4i(pq, 128, Zi128, 128, zk, w, j / 128);
	forward_4(pq, 32, Zi32, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	write_4(128, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void fwd1024p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_1024();

	forward_4i(pq, 256, Zi256, 256, zk, w, j / 256);
	forward_4(pq, 64, Zi64, w, j / 64);
	forward_4(pq, 16, Zi16, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	fwd2_write_4(pq, 256, zk, Z4, w, j);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void fwd2048p(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_2048();

	forward_4i(pq, 512, Zi512, 512, zk, w, j / 512);
	forward_4(pq, 128, Zi128, w, j / 128);
	forward_4(pq, 32, Zi32, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	write_4(512, zk, Z4);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void mul32(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_32();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k32];

	forward_4i(pq, 8, Zi8, 8, zk, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	mul_22(pq, Z4, 8, zpk, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4o(pq, 8, zk, 8, Zi8, w, ji / 8);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void mul64(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_64();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k64];

	forward_4i(pq, 16, Zi16, 16, zk, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	mul_4(pq, Z4, 16, zpk, w, j, ji);
	backward_4(pq, 4, Zi4, w, ji / 4);
	backward_4o(pq, 16, zk, 16, Zi16, w, ji / 16);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void mul128(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_128();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k128];

	forward_4i(pq, 32, Zi32, 32, zk, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	mul_22(pq, Z4, 32, zpk, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4(pq, 8, Zi8, w, ji / 8);
	backward_4o(pq, 32, zk, 32, Zi32, w, ji / 32);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void mul256(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_256();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k256];

	forward_4i(pq, 64, Zi64, 64, zk, w, j / 64);
	forward_4(pq, 16, Zi16, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	mul_4(pq, Z4, 64, zpk, w, j, ji);
	backward_4(pq, 4, Zi4, w, ji / 4);
	backward_4(pq, 16, Zi16, w, ji / 16);
	backward_4o(pq, 64, zk, 64, Zi64, w, ji / 64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((work_group_size_hint(512 / 4, 1, 1)))
#endif
void mul512(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_512();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k512];

	forward_4i(pq, 128, Zi128, 128, zk, w, j / 128);
	forward_4(pq, 32, Zi32, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	mul_22(pq, Z4, 128, zpk, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4(pq, 8, Zi8, w, ji / 8);
	backward_4(pq, 32, Zi32, w, ji / 32);
	backward_4o(pq, 128, zk, 128, Zi128, w, ji / 128);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))
#endif
void mul1024(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_1024();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k1024];

	forward_4i(pq, 256, Zi256, 256, zk, w, j / 256);
	forward_4(pq, 64, Zi64, w, j / 64);
	forward_4(pq, 16, Zi16, w, j / 16);
	forward_4(pq, 4, Zi4, w, j / 4);
	mul_4(pq, Z4, 256, zpk, w, j, ji);
	backward_4(pq, 4, Zi4, w, ji / 4);
	backward_4(pq, 16, Zi16, w, ji / 16);
	backward_4(pq, 64, Zi64, w, ji / 64);
	backward_4o(pq, 256, zk, 256, Zi256, w, ji / 256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))
#endif
void mul2048(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_2048();
	DECLARE_VARP_REG();
	__global const uint * restrict const zpk = &zp[k2048];

	forward_4i(pq, 512, Zi512, 512, zk, w, j / 512);
	forward_4(pq, 128, Zi128, w, j / 128);
	forward_4(pq, 32, Zi32, w, j / 32);
	forward_4(pq, 8, Zi8, w, j / 8);
	forward_4(pq, 2, Zi2, w, j / 2);
	mul_22(pq, Z4, 512, zpk, w, j);
	backward_4(pq, 2, Zi2, w, ji / 2);
	backward_4(pq, 8, Zi8, w, ji / 8);
	backward_4(pq, 32, Zi32, w, ji / 32);
	backward_4(pq, 128, Zi128, w, ji / 128);
	backward_4o(pq, 512, zk, 512, Zi512, w, ji / 512);
}

// -----------------

INLINE uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 * a_p)
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

	const uint32 d = mul_hi((uint32)(a >> b_s), b_inv), r = (uint32)(a) - d * b;
	const bool o = (r >= b);
	*a_p = d + (o ? 1 : 0);
	return r - (o ? b : 0);
}

INLINE int32 reduce64(int64 * f, const uint32 b, const uint32 b_inv, const int b_s)
{
	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
	const uint64 t = abs(*f);
	const uint64 t_h = t >> 29;
	const uint32 t_l = (uint32)(t) & ((1u << 29) - 1);

	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint32 d_l, r_l = barrett(((uint64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint64 d = ((uint64)(d_h) << 29) | d_l;

	const bool s = (*f < 0);
	*f = s ? -(int64)(d) : (int64)(d);
	return s ? -(int32)(r_l) : (int32)(r_l);
}

INLINE int32 reduce96(int96 * f, const uint32 b, const uint32 b_inv, const int b_s)
{
	const uint96 t = int96_abs(*f);
	const uint64 t_h = ((uint64)(t.s1) << (64 - 29)) | (t.s0 >> 29);
	const uint32 t_l = (uint32)(t.s0) & ((1u << 29) - 1);

	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint32 d_l, r_l = barrett(((uint64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint64 d = ((uint64)(d_h) << 29) | d_l;

	const bool s = int96_is_neg(*f);
	*f = int96_set_si(s ? -(int64)(d) : (int64)(d));
	return s ? -(int32)(r_l) : (int32)(r_l);
}

INLINE int64 garner2(const uint32 r1, const uint32 r2)
{
	const uint32 mfInvP2_P1 = 2130706177u;	// Montgomery form of 1 / P2 (mod P1)
	const uint64 P1P2 = P1 * (uint64)(P2);
	uint32 u12 = mulmod(submod(r1, r2, P1), mfInvP2_P1, PQ1);	// P2 < P1
	const uint64 n = r2 + u12 * (uint64)(P2);
	return (n > P1P2 / 2) ? (int64)(n - P1P2) : (int64)(n);
}

INLINE int96 garner3(const uint r1, const uint r2, const uint r3)
{
	// Montgomery form of 1 / Pi (mod Pj)
	const uint32 mfInvP3_P1 = 608773230u, mfInvP2_P1 = 2130706177u, mfInvP3_P2 = 1409286102u;
	const uint64 P2P3 = P2 * (uint64)(P3);
	const uint96 P1P2P3 = uint96_set(13049742876517335041ul, 491581440u);
	const uint96 P1P2P3_2 = uint96_set(6524871438258667520ul, 245790720u);

	const uint32 u13 = mulmod(submod(r1, r3, P1), mfInvP3_P1, PQ1);
	const uint32 u23 = mulmod(submod(r2, r3, P2), mfInvP3_P2, PQ2);
	const uint32 u123 = mulmod(submod(u13, u23, P1), mfInvP2_P1, PQ1);
	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (uint64)(P3) + r3);
	return uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);
}

__kernel
void normalize1(__global uint * restrict const z, __global long * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const unsigned int blk = abs(sblk);
	__global uint * restrict const zi = &z[blk * idx];

	prefetch(zi, (size_t)blk);

#if RNSSIZE == 2

	int64 f = 0;

	sz_t j = 0;
	do
	{
		const uint32 u1 = mulmod(zi[j + 0 * NSIZE], NORM1, PQ1);
		const uint32 u2 = mulmod(zi[j + 1 * NSIZE], NORM2, PQ2);
		int64 l = garner2(u1, u2);
		if (sblk < 0) l += l;
		f += l;
		const int32 r = reduce64(&f, b, b_inv, b_s);
		zi[j + 0 * NSIZE] = set_int(r, P1);
		zi[j + 1 * NSIZE] = set_int(r, P2);
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	c[i] = (i == 0) ? -f : f;

#else

	int96 f = int96_set_si(0);

	sz_t j = 0;
	do
	{
		const uint32 u1 = mulmod(zi[j + 0 * NSIZE], NORM1, PQ1);
		const uint32 u2 = mulmod(zi[j + 1 * NSIZE], NORM2, PQ2);
		const uint32 u3 = mulmod(zi[j + 2 * NSIZE], NORM3, PQ3);
		int96 l = garner3(u1, u2, u3);
		if (sblk < 0) l = int96_add(l, l);
		f = int96_add(f, l);
		const int32 r = reduce96(&f, b, b_inv, b_s);
		zi[j + 0 * NSIZE] = set_int(r, P1);
		zi[j + 1 * NSIZE] = set_int(r, P2);
		zi[j + 2 * NSIZE] = set_int(r, P3);
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	c[i] = (i == 0) ? -(long)f.s0 : (long)f.s0;

#endif
}

__kernel
void normalize2(__global uint * restrict const z, __global const long * restrict const c, 
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global uint * restrict const zi = &z[blk * idx];

	int64 f = c[idx];

	sz_t j = 0;
	do
	{
		f += get_int(zi[j], P1);
		const int32 r = reduce64(&f, b, b_inv, b_s);
		zi[j + 0 * NSIZE] = set_int(r, P1);
		zi[j + 1 * NSIZE] = set_int(r, P2);
#if RNSSIZE == 3
		zi[j + 2 * NSIZE] = set_int(r, P3);
#endif
		if (f == 0) return;
		++j;
	} while (j != blk - 1);

	const int32 r = (int32)(f);
	zi[blk - 1 + 0 * NSIZE] = addmod(zi[blk - 1 + 0 * NSIZE], set_int(r, P1), P1);
	zi[blk - 1 + 1 * NSIZE] = addmod(zi[blk - 1 + 1 * NSIZE], set_int(r, P2), P2);
#if RNSSIZE == 3
	zi[blk - 1 + 2 * NSIZE] = addmod(zi[blk - 1 + 2 * NSIZE], set_int(r, P3), P3);
#endif
}

__kernel
void mulscalar(__global uint * restrict const z, __global long * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global uint * restrict const zi = &z[blk * idx];

	prefetch(zi, (size_t)blk);

	int64 f = 0;

	sz_t j = 0;
	do
	{
		f += get_int(zi[j], P1) * (int64)(a);
		const int32 r = reduce64(&f, b, b_inv, b_s);
		zi[j + 0 * NSIZE] = set_int(r, P1);
		zi[j + 1 * NSIZE] = set_int(r, P2);
#if RNSSIZE == 3
		zi[j + 2 * NSIZE] = set_int(r, P3);
#endif
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	c[i] = (i == 0) ? -f : f;
}

__kernel
void set(__global uint * restrict const z, const unsigned int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[idx] = ((idx & (NSIZE - 1)) == 0) ? a : 0;
}

__kernel
void copy(__global uint * restrict const z, const unsigned int dst, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[dst + idx] = z[src + idx];
}

__kernel
void copyp(__global uint * restrict const zp, __global const uint * restrict const z, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	zp[idx] = z[src + idx];
}
