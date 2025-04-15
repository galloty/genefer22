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

#if defined(__NV_CL_C_VERSION)
	#define PTX_ASM	1
#endif

#if !defined(N_SZ)
#define N_SZ		65536u
#define LN_SZ		16
#define RNS_SZ		3
#define VSIZE		4
#define LVSIZE		2
#define NORM1		2130641409u
#define NORM2		2113864705u
#define NORM3		2013204481u
#define W_SHFT		65536u
#define WI_SHFT		32768u
// #define USE_WI		1
#define BLK32		32
#define BLK64		16
#define BLK128		8
#define BLK256		4
#define BLK512		4
#define BLK1024		2
#define CHUNK64		4
#define CHUNK256	4
#define CHUNK1024	1
// #define SHORT_VER	1
#define MAX_WG_SZ	256
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

// --- modular arithmetic

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

// 2 mul + 2 mul_hi
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

// --- v2

INLINE uint32_2 addmod2(const uint32_2 lhs, const uint32_2 rhs, const uint32 p)
{
	return (uint32_2)(addmod(lhs.s0, rhs.s0, p), addmod(lhs.s1, rhs.s1, p));
}

INLINE uint32_2 submod2(const uint32_2 lhs, const uint32_2 rhs, const uint32 p)
{
	return (uint32_2)(submod(lhs.s0, rhs.s0, p), submod(lhs.s1, rhs.s1, p));
}

INLINE uint32_2 mulmod2(const uint32_2 lhs, const uint32_2 rhs, const uint32_2 pq)
{
	return (uint32_2)(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq));
}

// --- v4

INLINE uint32_4 addmod4(const uint32_4 lhs, const uint32_4 rhs, const uint32 p)
{
	return (uint32_4)(addmod2(lhs.s01, rhs.s01, p), addmod2(lhs.s23, rhs.s23, p));
}

INLINE uint32_4 submod4(const uint32_4 lhs, const uint32_4 rhs, const uint32 p)
{
	return (uint32_4)(submod2(lhs.s01, rhs.s01, p), submod2(lhs.s23, rhs.s23, p));
}

INLINE uint32_4 mulmod4(const uint32_4 lhs, const uint32_4 rhs, const uint32_2 pq)
{
	return (uint32_4)(mulmod2(lhs.s01, rhs.s01, pq), mulmod2(lhs.s23, rhs.s23, pq));
}

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
#if defined(PTX_ASM)
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
#if defined(PTX_ASM)
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
#if defined(PTX_ASM)
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

#define FWD2(z0, z1, w) \
{ \
	const uint32 t = mulmod(z1, w, pq); \
	z1 = submod(z0, t, pq.s0); z0 = addmod(z0, t, pq.s0); \
}

#define BCK2(z0, z1, win) \
{ \
	const uint32 t = submod(z1, z0, pq.s0); z0 = addmod(z0, z1, pq.s0); \
	z1 = mulmod(t, win, pq); \
}

#define SQR2(z0, z1, w) \
{ \
	const uint32 t = mulmod(sqrmod(z1, pq), w, pq); \
	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \
	z0 = addmod(sqrmod(z0, pq), t, pq.s0); \
}

#define SQR2N(z0, z1, w) \
{ \
	const uint32 t = mulmod(sqrmod(z1, pq), w, pq); \
	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \
	z0 = submod(sqrmod(z0, pq), t, pq.s0); \
}

#define MUL2(z0, z1, zp0, zp1, w) \
{ \
	const uint32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \
	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \
	z0 = addmod(mulmod(z0, zp0, pq), t, pq.s0); \
}

#define MUL2N(z0, z1, zp0, zp1, w) \
{ \
	const uint32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \
	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \
	z0 = submod(mulmod(z0, zp0, pq), t, pq.s0); \
}

#define FWD2v2(z0, z1, w) \
{ \
	const uint32_2 t = mulmod2(z1, w, pq); \
	z1 = submod2(z0, t, pq.s0); z0 = addmod2(z0, t, pq.s0); \
}

#define BCK2v2(z0, z1, win) \
{ \
	const uint32_2 t = submod2(z1, z0, pq.s0); z0 = addmod2(z0, z1, pq.s0); \
	z1 = mulmod2(t, win, pq); \
}

#define FWD2v4(z0, z1, w) \
{ \
	const uint32_4 t = mulmod4(z1, w, pq); \
	z1 = submod4(z0, t, pq.s0); z0 = addmod4(z0, t, pq.s0); \
}

#define BCK2v4(z0, z1, win) \
{ \
	const uint32_4 t = submod4(z1, z0, pq.s0); z0 = addmod4(z0, z1, pq.s0); \
	z1 = mulmod4(t, win, pq); \
}

static void _loadg1(const sz_t n, uint32 * const zl, __global const uint * restrict const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
static void _loadl1(const sz_t n, uint32 * const zl, __local const uint * restrict const Z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
static void _storeg1(const sz_t n, __global uint * restrict const z, const size_t s, const uint32 * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }
static void _storel1(const sz_t n, __local uint * restrict const Z, const size_t s, const uint32 * const zl) { for (size_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

static void _loadg2(const sz_t n, uint32_2 * const zl, __global const uint2 * restrict const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
static void _loadl2(const sz_t n, uint32_2 * const zl, __local const uint2 * restrict const Z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
static void _storeg2(const sz_t n, __global uint2 * restrict const z, const size_t s, const uint32_2 * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }
static void _storel2(const sz_t n, __local uint2 * restrict const Z, const size_t s, const uint32_2 * const zl) { for (size_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

static void _loadg4(const sz_t n, uint32_4 * const zl, __global const uint4 * restrict const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
static void _loadl4(const sz_t n, uint32_4 * const zl, __local const uint4 * restrict const Z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
static void _storeg4(const sz_t n, __global uint4 * restrict const z, const size_t s, const uint32_4 * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }
static void _storel4(const sz_t n, __local uint4 * restrict const Z, const size_t s, const uint32_4 * const zl) { for (size_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

// ---

INLINE void _forward4x1(const uint32_2 pq, uint32 z[4], const uint32 w1, const uint32 w2[2])
{
	FWD2(z[0], z[2], w1); FWD2(z[1], z[3], w1);
	FWD2(z[0], z[1], w2[0]); FWD2(z[2], z[3], w2[1]);
}

INLINE void _backward4x1(const uint32_2 pq, uint32 z[4], const uint32 win1, const uint32 win2[2])
{
	BCK2(z[0], z[1], win2[0]); BCK2(z[2], z[3], win2[1]);
	BCK2(z[0], z[2], win1); BCK2(z[1], z[3], win1);
}

INLINE void _forward4x1_0(const uint32_2 pq, const uint32_4 f0, uint32 z[4])
{
	const uint32 rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
	z[0] = mulmod(z[0], rsq, pq); z[1] = mulmod(z[1], rsq, pq);
	FWD2(z[0], z[2], im); FWD2(z[1], z[3], im);
	FWD2(z[0], z[1], sqrti); FWD2(z[2], z[3], isqrti);
}

INLINE void _square2x2(const uint32_2 pq, uint32 z[4], const uint32 w)
{
	SQR2(z[0], z[1], w); SQR2N(z[2], z[3], w);
}

INLINE void _square4(const uint32_2 pq, uint32 z[4], const uint32 w, const uint32 win)
{
	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);
	_square2x2(pq, z, w);
	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);
}

INLINE void _fwd4(const uint32_2 pq, uint32 z[4], const uint32 w)
{
	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);
}

INLINE void _mul2x2(const uint32_2 pq, uint32 z[4], const uint32 zp[4], const uint32 w)
{
	MUL2(z[0], z[1], zp[0], zp[1], w); MUL2N(z[2], z[3], zp[2], zp[3], w);
}

INLINE void _mul4(const uint32_2 pq, uint32 z[4], const uint32 zp[4], const uint32 w, const uint32 win)
{
	_fwd4(pq, z, w);
	_mul2x2(pq, z, zp, w);
	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);
}

// --- v2

INLINE void _forward4x2(const uint32_2 pq, uint32_2 z[4], const uint32 w1, const uint32 w2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
}

INLINE void _backward4x2(const uint32_2 pq, uint32_2 z[4], const uint32 win1, const uint32 win2[2])
{
	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);
	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);
}

INLINE void _forward4x2_0(const uint32_2 pq, const uint32_4 f0, uint32_2 z[4])
{
	const uint32 rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
	z[0] = mulmod2(z[0], rsq, pq); z[1] = mulmod2(z[1], rsq, pq);
	FWD2v2(z[0], z[2], im); FWD2v2(z[1], z[3], im);
	FWD2v2(z[0], z[1], sqrti); FWD2v2(z[2], z[3], isqrti);
}

INLINE void _square4x2(const uint32_2 pq, uint32_2 z[4], const uint32 w2[2], const uint32 win2[2])
{
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
	SQR2(z[0].s0, z[0].s1, w2[0]); SQR2N(z[1].s0, z[1].s1, w2[0]);
	SQR2(z[2].s0, z[2].s1, w2[1]); SQR2N(z[3].s0, z[3].s1, w2[1]);
	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);
}

INLINE void _square8(const uint32_2 pq, uint32_2 z[4], const uint32 w1, const uint32 win1, const uint32 w2[2], const uint32 win2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	_square4x2(pq, z, w2, win2);
	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);
}

INLINE void _fwd4x2(const uint32_2 pq, uint32_2 z[4], const uint32 w2[2])
{
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
}

INLINE void _fwd8(const uint32_2 pq, uint32_2 z[4], const uint32 w1, const uint32 w2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	_fwd4x2(pq, z, w2);
}

INLINE void _mul4x2(const uint32_2 pq, uint32_2 z[4], const uint32_2 zp[4], const uint32 w2[2], const uint32 win2[2])
{
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
	MUL2(z[0].s0, z[0].s1, zp[0].s0, zp[0].s1, w2[0]); MUL2N(z[1].s0, z[1].s1, zp[1].s0, zp[1].s1, w2[0]);
	MUL2(z[2].s0, z[2].s1, zp[2].s0, zp[2].s1, w2[1]); MUL2N(z[3].s0, z[3].s1, zp[3].s0, zp[3].s1, w2[1]);
	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);
}

INLINE void _mul8(const uint32_2 pq, uint32_2 z[4], const uint32_2 zp[4], const uint32 w1, const uint32  win1, const uint32 w2[2], const uint32 win2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	_mul4x2(pq, z, zp, w2, win2);
	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);
}

// --- v4

INLINE void _forward4x4(const uint32_2 pq, uint32_4 z[4], const uint32 w1, const uint32 w2[2])
{
	FWD2v4(z[0], z[2], w1); FWD2v4(z[1], z[3], w1);
	FWD2v4(z[0], z[1], w2[0]); FWD2v4(z[2], z[3], w2[1]);
}

INLINE void _backward4x4(const uint32_2 pq, uint32_4 z[4], const uint32 win1, const uint32 win2[2])
{
	BCK2v4(z[0], z[1], win2[0]); BCK2v4(z[2], z[3], win2[1]);
	BCK2v4(z[0], z[2], win1); BCK2v4(z[1], z[3], win1);
}

INLINE void _forward4x4_0(const uint32_2 pq, const uint32_4 f0, uint32_4 z[4])
{
	const uint32 rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
	z[0] = mulmod4(z[0], rsq, pq); z[1] = mulmod4(z[1], rsq, pq);
	FWD2v4(z[0], z[2], im); FWD2v4(z[1], z[3], im);
	FWD2v4(z[0], z[1], sqrti); FWD2v4(z[2], z[3], isqrti);
}

INLINE void _square4x2v4(const uint32_2 pq, uint32_4 z[2], const uint32 w2[2], const uint32 win2[2])
{
	for (sz_t i = 0; i < 2; ++i)
	{
		FWD2v2(z[i].s01, z[i].s23, w2[i]);
		SQR2(z[i].s0, z[i].s1, w2[i]); SQR2N(z[i].s2, z[i].s3, w2[i]);
		BCK2v2(z[i].s01, z[i].s23, win2[i]);
	}
}

INLINE void _square8v4(const uint32_2 pq, uint32_4 z[2], const uint32 w1, const uint32 win1, const uint32 w2[2], const uint32 win2[2])
{
	FWD2v4(z[0], z[1], w1);
	_square4x2v4(pq, z, w2, win2);
	BCK2v4(z[0], z[1], win1);
}

INLINE void _fwd4x2v4(const uint32_2 pq, uint32_4 z[2], const uint32 w2[2])
{
	for (sz_t i = 0; i < 2; ++i) FWD2v2(z[i].s01, z[i].s23, w2[i]);
}

INLINE void _fwd8v4(const uint32_2 pq, uint32_4 z[2], const uint32 w1, const uint32 w2[2])
{
	FWD2v4(z[0], z[1], w1);
	_fwd4x2v4(pq, z, w2);
}

INLINE void _mul4x2v4(const uint32_2 pq, uint32_4 z[2], const uint32_4 zp[2], const uint32 w2[2], const uint32 win2[2])
{
	for (sz_t i = 0; i < 2; ++i)
	{
		FWD2v2(z[i].s01, z[i].s23, w2[i]);
		MUL2(z[i].s0, z[i].s1, zp[i].s0, zp[i].s1, w2[i]); MUL2N(z[i].s2, z[i].s3, zp[i].s2, zp[i].s3, w2[i]);
		BCK2v2(z[i].s01, z[i].s23, win2[i]);
	}
}

INLINE void _mul8v4(const uint32_2 pq, uint32_4 z[2], const uint32_4 zp[2], const uint32 w1, const uint32  win1, const uint32 w2[2], const uint32 win2[2])
{
	FWD2v4(z[0], z[1], w1);
	_mul4x2v4(pq, z, zp, w2, win2);
	BCK2v4(z[0], z[1], win1);
}

// --- inverse of roots is wi[s + j] or w[s + s - j - 1] ---

#define DECLARE_W1(sj)			const uint32 w1 = w[sj];
#define DECLARE_W2(sj)			uint32 w2[2]; { const uint32_2 t = ((__global const uint2 *)w)[sj]; w2[0] = t.s0; w2[1] = t.s1; }
#define DECLARE_W12(sj)			DECLARE_W1(sj); DECLARE_W2(sj);
#define DECLARE_W1_2(sj)		uint32 w1[2]; { const uint32_2 t = ((__global const uint2 *)w)[sj]; w1[0] = t.s0; w1[1] = t.s1; }
#define DECLARE_W2_4(sj)		uint32 w2[4]; { const uint32_4 t = ((__global const uint4 *)w)[sj]; w2[0] = t.s0; w2[1] = t.s1; w2[2] = t.s2; w2[3] = t.s3; }
#define DECLARE_W12_24(sj)		DECLARE_W1_2(sj); DECLARE_W2_4(sj);

#define DECLARE_WIN1(sji)		const uint32 win1 = wi[sji];
#if defined(USE_WI)
#define DECLARE_IVAR(s, j)		const sz_t sji = s + j; __global const uint * restrict const wi = &w[WI_SHFT];
#define DECLARE_WIN2(sji)		uint32 win2[2]; { const uint32_2 t = ((__global const uint2 *)wi)[sji]; win2[0] = t.s0; win2[1] = t.s1; }
#define DECLARE_WIN1_2(sji)		uint32 win1[2]; { const uint32_2 t = ((__global const uint2 *)wi)[sji]; win1[0] = t.s0; win1[1] = t.s1; }
#define DECLARE_WIN2_4(sji)		uint32 win2[4]; { const uint32_4 t = ((__global const uint4 *)wi)[sji]; win2[0] = t.s0; win2[1] = t.s1; win2[2] = t.s2; win2[3] = t.s3; }
#else
#define DECLARE_IVAR(s, j)		const sz_t sji = s + s - j - 1; __global const uint * restrict const wi = w;
#define DECLARE_WIN2(sji)		uint32 win2[2]; { const uint32_2 t = ((__global const uint2 *)wi)[sji]; win2[0] = t.s1; win2[1] = t.s0; }
#define DECLARE_WIN1_2(sji)		uint32 win1[2]; { const uint32_2 t = ((__global const uint2 *)wi)[sji]; win1[0] = t.s1; win1[1] = t.s0; }
#define DECLARE_WIN2_4(sji)		uint32 win2[4]; { const uint32_4 t = ((__global const uint4 *)wi)[sji]; win2[0] = t.s3; win2[1] = t.s2; win2[2] = t.s1; win2[3] = t.s0; }
#endif
#define DECLARE_WIN12(sj)		DECLARE_WIN1(sj); DECLARE_WIN2(sj);
#define DECLARE_WIN12_24(sj)	DECLARE_WIN1_2(sj); DECLARE_WIN2_4(sj);

// --- vector size (1, 2 or 4) ---

#if VSIZE == 4
#define VTYPE				uint32_4
#define _loadg				_loadg4
#define _loadl				_loadl4
#define _storeg				_storeg4
#define _storel				_storel4
#define _forward4			_forward4x4
#define _backward4			_backward4x4
#define _forward4_0			_forward4x4_0
#elif VSIZE == 2
#define VTYPE				uint32_2
#define _loadg				_loadg2
#define _loadl				_loadl2
#define _storeg				_storeg2
#define _storel				_storel2
#define _forward4			_forward4x2
#define _backward4			_backward4x2
#define _forward4_0			_forward4x2_0
#else
#define VTYPE				uint32
#define _loadg				_loadg1
#define _loadl				_loadl1
#define _storeg				_storeg1
#define _storel				_storel1
#define _forward4			_forward4x1
#define _backward4			_backward4x1
#define _forward4_0			_forward4x1_0
#endif

// --- transform/inline global mem ---

INLINE void forward4io(const uint32_2 pq, const sz_t m, __global VTYPE * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12(sj);
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_forward4(pq, zl, w1, w2);
	_storeg(4, z, m, zl);
}

INLINE void backward4io(const uint32_2 pq, const sz_t m, __global VTYPE * restrict const z, __global const uint * restrict const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_backward4(pq, zl, win1, win2);
	_storeg(4, z, m, zl);
}

INLINE void forward4io_0(const uint32_2 pq, const uint32_4 f0, __global VTYPE * restrict const z)
{
	const sz_t m = N_SZ / 4 / VSIZE;
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_forward4_0(pq, f0, zl);
	_storeg(4, z, m, zl);
}

// --- v1

INLINE void square2x2io(const uint32_2 pq, __global uint * restrict const z, const uint w)
{
	uint32 zl[4]; _loadg1(4, zl, z, 1);
	_square2x2(pq, zl, w);
	_storeg1(4, z, 1, zl);
}

INLINE void square4x1io(const uint32_2 pq, __global uint * restrict const z, const uint w, const uint win)
{
	uint32 zl[4]; _loadg1(4, zl, z, 1);
	_square4(pq, zl, w, win);
	_storeg1(4, z, 1, zl);
}

INLINE void fwd4x1io(const uint32_2 pq, __global uint * restrict const z, const uint w)
{
	uint32 zl[4]; _loadg1(4, zl, z, 1);
	_fwd4(pq, zl, w);
	_storeg1(4, z, 1, zl);
}

INLINE void mul2x2io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp, const uint w)
{
	uint32 zpl[4]; _loadg1(4, zpl, zp, 1);
	uint32 zl[4]; _loadg1(4, zl, z, 1);
	_mul2x2(pq, zl, zpl, w);
	_storeg1(4, z, 1, zl);
}

INLINE void mul4x1io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp, const uint w, const uint win)
{
	uint32 zpl[4]; _loadg1(4, zpl, zp, 1);
	uint32 zl[4]; _loadg1(4, zl, z, 1);
	_mul4(pq, zl, zpl, w, win);
	_storeg1(4, z, 1, zl);
}

// --- v2

INLINE void square4x2io(const uint32_2 pq, __global uint2 * restrict const z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	uint32_2 zl[4]; _loadg2(4, zl, z, 1);
	_square4x2(pq, zl, w2, win2);
	_storeg2(4, z, 1, zl);
}

INLINE void square8x1io(const uint32_2 pq, __global uint2 * restrict const z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	uint32_2 zl[4]; _loadg2(4, zl, z, 1);
	_square8(pq, zl, w1, win1, w2, win2);
	_storeg2(4, z, 1, zl);
}

INLINE void fwd4x2io(const uint32_2 pq, __global uint2 * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W2(sj);
	uint32_2 zl[4]; _loadg2(4, zl, z, 1);
	_fwd4x2(pq, zl, w2);
	_storeg2(4, z, 1, zl);
}

INLINE void fwd8x1io(const uint32_2 pq, __global uint2 * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12(sj);
	uint32_2 zl[4]; _loadg2(4, zl, z, 1);
	_fwd8(pq, zl, w1, w2);
	_storeg2(4, z, 1, zl);
}

INLINE void mul4x2io(const uint32_2 pq, __global uint2 * restrict const z, const __global uint2 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	uint32_2 zpl[4]; _loadg2(4, zpl, zp, 1);
	uint32_2 zl[4]; _loadg2(4, zl, z, 1);
	_mul4x2(pq, zl, zpl, w2, win2);
	_storeg2(4, z, 1, zl);
}

INLINE void mul8x1io(const uint32_2 pq, __global uint2 * restrict const z, const __global uint2 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	uint32_2 zpl[4]; _loadg2(4, zpl, zp, 1);
	uint32_2 zl[4]; _loadg2(4, zl, z, 1);
	_mul8(pq, zl, zpl, w1, win1, w2, win2);
	_storeg2(4, z, 1, zl);
}

// --- v4

INLINE void square4x4io(const uint32_2 pq, __global uint4 * restrict const z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	uint32_4 zl[4]; _loadg4(4, zl, z, 1);
	_square4x2v4(pq, &zl[0], &w2[0], &win2[0]);
	_square4x2v4(pq, &zl[2], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void square8x2io(const uint32_2 pq, __global uint4 * restrict const z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	uint32_4 zl[4]; _loadg4(4, zl, z, 1);
	_square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void fwd4x4io(const uint32_2 pq, __global uint4 * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W2_4(sj);
	uint32_4 zl[4]; _loadg4(4, zl, z, 1);
	_fwd4x2v4(pq, &zl[0], &w2[0]);
	_fwd4x2v4(pq, &zl[2], &w2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void fwd8x2io(const uint32_2 pq, __global uint4 * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12_24(sj);
	uint32_4 zl[4]; _loadg4(4, zl, z, 1);
	_fwd8v4(pq, &zl[0], w1[0], &w2[0]);
	_fwd8v4(pq, &zl[2], w1[1], &w2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void mul4x4io(const uint32_2 pq, __global uint4 * restrict const z, const __global uint4 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	uint32_4 zpl[4]; _loadg4(4, zpl, zp, 1);
	uint32_4 zl[4]; _loadg4(4, zl, z, 1);
	_mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);
	_mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void mul8x2io(const uint32_2 pq, __global uint4 * restrict const z, const __global uint4 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	uint32_4 zpl[4]; _loadg4(4, zpl, zp, 1);
	uint32_4 zl[4]; _loadg4(4, zl, z, 1);
	_mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

// --- v1, v2, v4

INLINE void square4io(const uint32_2 pq, __global VTYPE * restrict const z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square4x4io(pq, z, w, wi, sj, sji);
#elif VSIZE == 2
	square4x2io(pq, z, w, wi, sj, sji);
#else
	square4x1io(pq, z, w[sj], wi[sji]);
#endif
}

INLINE void fwd4io(const uint32_2 pq, __global VTYPE * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
#if VSIZE == 4
	fwd4x4io(pq, z, w, sj);
#elif VSIZE == 2
	fwd4x2io(pq, z, w, sj);
#else
	fwd4x1io(pq, z, w[sj]);
#endif
}

INLINE void mul4io(const uint32_2 pq, __global VTYPE * restrict const z, const __global VTYPE * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul4x4io(pq, z, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul4x2io(pq, z, zp, w, wi, sj, sji);
#else
	mul4x1io(pq, z, zp, w[sj], wi[sji]);
#endif
}

// --- v2, v4

INLINE void square8io(const uint32_2 pq, __global VTYPE * restrict const z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square8x2io(pq, z, w, wi, sj, sji);
#elif VSIZE == 2
	square8x1io(pq, z, w, wi, sj, sji);
#endif
}

INLINE void fwd8io(const uint32_2 pq, __global VTYPE * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
#if VSIZE == 4
	fwd8x2io(pq, z, w, sj);
#elif VSIZE == 2
	fwd8x1io(pq, z, w, sj);
#endif
}

INLINE void mul8io(const uint32_2 pq, __global VTYPE * restrict const z, const __global VTYPE * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul8x2io(pq, z, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul8x1io(pq, z, zp, w, wi, sj, sji);
#endif
}

// --- transform/inline local & global mem ---

INLINE void forward_4(const uint32_2 pq, const sz_t m, __local VTYPE * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12(sj);
	barrier(CLK_LOCAL_MEM_FENCE);
	VTYPE zl[4]; _loadl(4, zl, Z, m);
	_forward4(pq, zl, w1, w2);
	_storel(4, Z, m, zl);
}

INLINE void forward_4i(const uint32_2 pq, const sz_t ml, __local VTYPE * restrict const Z, const sz_t mg,
	__global const VTYPE * restrict const z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12(sj);
	VTYPE zl[4]; _loadg(4, zl, z, mg);
	_forward4(pq, zl, w1, w2);
	_storel(4, Z, ml, zl);
}

INLINE void forward_4i_0(const uint32_2 pq, const uint32_4 f0, const sz_t ml, __local VTYPE * restrict const Z,
	const sz_t mg, __global const VTYPE * restrict const z)
{
	VTYPE zl[4]; _loadg(4, zl, z, mg);
	_forward4_0(pq, f0, zl);
	_storel(4, Z, ml, zl);
}

INLINE void forward_4o(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z, const sz_t ml,
	__local const VTYPE * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12(sj);
	barrier(CLK_LOCAL_MEM_FENCE);
	VTYPE zl[4]; _loadl(4, zl, Z, ml);
	_forward4(pq, zl, w1, w2);
	_storeg(4, z, mg, zl);
}

INLINE void backward_4(const uint32_2 pq, const sz_t m, __local VTYPE * restrict const Z, __global const uint * restrict const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	barrier(CLK_LOCAL_MEM_FENCE);
	VTYPE zl[4]; _loadl(4, zl, Z, m);
	_backward4(pq, zl, win1, win2);
	_storel(4, Z, m, zl);
}

INLINE void backward_4i(const uint32_2 pq, const sz_t ml, __local VTYPE * restrict const Z, const sz_t mg,
	__global const VTYPE * restrict const z, __global const uint * restrict const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	VTYPE zl[4]; _loadg(4, zl, z, mg);
	_backward4(pq, zl, win1, win2);
	_storel(4, Z, ml, zl);
}

INLINE void backward_4o(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z, const sz_t ml,
	__local const VTYPE * restrict const Z, __global const uint * restrict const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	barrier(CLK_LOCAL_MEM_FENCE);
	VTYPE zl[4]; _loadl(4, zl, Z, ml);
	_backward4(pq, zl, win1, win2);
	_storeg(4, z, mg, zl);
}

// --- v1

INLINE void square_2x2(const uint32_2 pq, __local uint * restrict const Z, const uint w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32 zl[4]; _loadl1(4, zl, Z, 1);
	_square2x2(pq, zl, w);
	_storel1(4, Z, 1, zl);
}

INLINE void square_4x1(const uint32_2 pq, __local uint * restrict const Z, const uint w, const uint win)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32 zl[4]; _loadl1(4, zl, Z, 1);
	_square4(pq, zl, w, win);
	_storel1(4, Z, 1, zl);
}

INLINE void write_4(const sz_t mg, __global VTYPE * restrict const z, __local const VTYPE * restrict const Z)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	VTYPE zl[4]; _loadl(4, zl, Z, 1);
	_storeg(4, z, mg, zl);
}

INLINE void fwd4x1_write(const uint32_2 pq, const sz_t mg, __global uint * restrict const z,
	__local const uint * restrict const Z, const uint w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32 zl[4]; _loadl1(4, zl, Z, 1);
	_fwd4(pq, zl, w);
	_storeg1(4, z, mg, zl);
}

INLINE void mul_2x2(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg,
	__global const uint * restrict const zp, const uint w)
{
	uint32 zpl[4]; _loadg1(4, zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32 zl[4]; _loadl1(4, zl, Z, 1);
	_mul2x2(pq, zl, zpl, w);
	_storel1(4, Z, 1, zl);
}

INLINE void mul_4x1(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg,
	__global const uint * restrict const zp, const uint w, const uint win)
{
	uint32 zpl[4]; _loadg1(4, zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32 zl[4]; _loadl1(4, zl, Z, 1);
	_mul4(pq, zl, zpl, w, win);
	_storel1(4, Z, 1, zl);
}

// --- v2

INLINE void square_4x2(const uint32_2 pq, __local uint2 * restrict const Z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);
	_square4x2(pq, zl, w2, win2);
	_storel2(4, Z, 1, zl);
}

INLINE void square_8x1(const uint32_2 pq, __local uint2 * restrict const Z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);
	_square8(pq, zl, w1, win1, w2, win2);
	_storel2(4, Z, 1, zl);
}

INLINE void fwd4x2_write(const uint32_2 pq, const sz_t mg, __global uint2 * restrict const z,
	__local const uint2 * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W2(sj);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);
	_fwd4x2(pq, zl, w2);
	_storeg2(4, z, mg, zl);
}

INLINE void fwd8x1_write(const uint32_2 pq, const sz_t mg, __global uint2 * restrict const z,
	__local const uint2 * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12(sj);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);
	_fwd8(pq, zl, w1, w2);
	_storeg2(4, z, mg, zl);
}

INLINE void mul_4x2(const uint32_2 pq, __local uint2 * restrict const Z, const sz_t mg, const __global uint2 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	uint32_2 zpl[4]; _loadg2(4, zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);
	_mul4x2(pq, zl, zpl, w2, win2);
	_storel2(4, Z, 1, zl);
}

INLINE void mul_8x1(const uint32_2 pq, __local uint2 * restrict const Z, const sz_t mg, const __global uint2 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	uint32_2 zpl[4]; _loadg2(4, zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);
	_mul8(pq, zl, zpl, w1, win1, w2, win2);
	_storel2(4, Z, 1, zl);
}

// --- v4

INLINE void square_4x4(const uint32_2 pq, __local uint4 * restrict const Z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);
	_square4x2v4(pq, &zl[0], &w2[0], &win2[0]);
	_square4x2v4(pq, &zl[2], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

INLINE void square_8x2(const uint32_2 pq, __local uint4 * restrict const Z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);
	_square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

INLINE void fwd4x4_write(const uint32_2 pq, const sz_t mg, __global uint4 * restrict const z,
	__local const uint4 * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W2_4(sj);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);
	_fwd4x2v4(pq, &zl[0], &w2[0]);
	_fwd4x2v4(pq, &zl[2], &w2[2]);
	_storeg4(4, z, mg, zl);
}

INLINE void fwd8x2_write(const uint32_2 pq, const sz_t mg, __global uint4 * restrict const z,
	__local const uint4 * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
	DECLARE_W12_24(sj);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);
	_fwd8v4(pq, &zl[0], w1[0], &w2[0]);
	_fwd8v4(pq, &zl[2], w1[1], &w2[2]);
	_storeg4(4, z, mg, zl);
}

INLINE void mul_4x4(const uint32_2 pq, __local uint4 * restrict const Z, const sz_t mg, const __global uint4 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	uint32_4 zpl[4]; _loadg4(4, zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);
	_mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);
	_mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

INLINE void mul_8x2(const uint32_2 pq, __local uint4 * restrict const Z, const sz_t mg, const __global uint4 * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	uint32_4 zpl[4]; _loadg4(4, zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);
	_mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

// --- v1, v2, v4 -- no barrier

INLINE void square_4(const uint32_2 pq, __local VTYPE * restrict const Z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square_4x4(pq, Z, w, wi, sj, sji);
#elif VSIZE == 2
	square_4x2(pq, Z, w, wi, sj, sji);
#else
	square_4x1(pq, Z, w[sj], wi[sji]);
#endif
}

INLINE void fwd4_write(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z,
	__local const VTYPE * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
#if VSIZE == 4
	fwd4x4_write(pq, mg, z, Z, w, sj);
#elif VSIZE == 2
	fwd4x2_write(pq, mg, z, Z, w, sj);
#else
	fwd4x1_write(pq, mg, z, Z, w[sj]);
#endif
}

INLINE void mul_4(const uint32_2 pq, __local VTYPE * restrict const Z, const sz_t mg, const __global VTYPE * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul_4x4(pq, Z, mg, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul_4x2(pq, Z, mg, zp, w, wi, sj, sji);
#else
	mul_4x1(pq, Z, mg, zp, w[sj], wi[sji]);
#endif
}

// --- v2, v4 -- no barrier

INLINE void square_8(const uint32_2 pq, __local VTYPE * restrict const Z,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square_8x2(pq, Z, w, wi, sj, sji);
#elif VSIZE == 2
	square_8x1(pq, Z, w, wi, sj, sji);
#endif
}

INLINE void fwd8_write(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z,
	__local const VTYPE * restrict const Z, __global const uint * restrict const w, const sz_t sj)
{
#if VSIZE == 4
	fwd8x2_write(pq, mg, z, Z, w, sj);
#elif VSIZE == 2
	fwd8x1_write(pq, mg, z, Z, w, sj);
#endif
}

INLINE void mul_8(const uint32_2 pq, __local VTYPE * restrict const Z, const sz_t mg, const __global VTYPE * restrict const zp,
	__global const uint * restrict const w, __global const uint * restrict const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul_8x2(pq, Z, mg, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul_8x1(pq, Z, mg, zp, w, wi, sj, sji);
#endif
}

// --- transform/macro ---

#define DECLARE_VAR_REGv1() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 2), mid = gid & ~((N_SZ / 4) - 1), id = gid %  (N_SZ / 4); \
	const uint32_2 pq = g_pq[lid]; \
	__global uint * restrict const z = &zg[4 * mid]; \
	__global const uint * restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv1() \
	__global const uint * restrict const zp = &zpg[4 * mid];

#define DECLARE_VAR_REGv2() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 3), mid = gid & ~((N_SZ / 8) - 1), id = gid %  (N_SZ / 8); \
	const uint32_2 pq = g_pq[lid]; \
	__global uint2 * restrict const z = &zg[4 * mid]; \
	__global const uint * restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv2() \
	__global const uint2 * restrict const zp = &zpg[4 * mid];

#define DECLARE_VAR_REGv4() \
	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 4), mid = gid & ~((N_SZ / 16) - 1), id = gid %  (N_SZ / 16); \
	const uint32_2 pq = g_pq[lid]; \
	__global uint4 * restrict const z = &zg[4 * mid]; \
	__global const uint * restrict const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv4() \
	__global const uint4 * restrict const zp = &zpg[4 * mid];

#if VSIZE == 4
#define DECLARE_VAR_REG		DECLARE_VAR_REGv4
#define DECLARE_VARP_REG	DECLARE_VARP_REGv4
#elif VSIZE == 2
#define DECLARE_VAR_REG		DECLARE_VAR_REGv2
#define DECLARE_VARP_REG	DECLARE_VARP_REGv2
#else
#define DECLARE_VAR_REG		DECLARE_VAR_REGv1
#define DECLARE_VARP_REG	DECLARE_VARP_REGv1
#endif

// --- transform without local mem ---

__kernel
void forward4(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	DECLARE_VAR_REG();
	const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;
	forward4io(pq, m, &z[k], w, s + j);
}

__kernel
void backward4(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	DECLARE_VAR_REG();
	const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id; DECLARE_IVAR(s, j);
	backward4io(pq, m, &z[k], wi, sji);
}

__kernel
void forward4_0(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t k = id;
	forward4io_0(pq, g_f0[lid], &z[k]);
}

__kernel
void square2x2(__global uint * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REGv1();
	const sz_t j = id, k = 4 * id;
	square2x2io(pq, &z[k], w[N_SZ / 4 + j]);
}

__kernel
void square4(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	square4io(pq, &z[k], w, wi, sj, sji);
}

__kernel
void fwd4p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j;
	fwd4io(pq, &z[k], w, sj);
}

__kernel
void mul4(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	mul4io(pq, &z[k], &zp[k], w, wi, sj, sji);
}

// --- v1

__kernel
void mul2x2(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REGv1();
	DECLARE_VARP_REGv1();
	const sz_t j = id, k = 4 * id;
	mul2x2io(pq, &z[k], &zp[k], w[N_SZ / 4 + j]);
}

// --- v2, v4

__kernel
void square8(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	square8io(pq, &z[k], w, wi, sj, sji);
}

__kernel
void fwd8p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j;
	fwd8io(pq, &z[k], w, sj);
}

__kernel
void mul8(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	mul8io(pq, &z[k], &zp[k], w, wi, sj, sji);
}

// --- transform ---

#if !defined(SHORT_VER)

#define DECLARE_VAR(B_N, CHUNK_N) \
	/* threadIdx < B_N */ \
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \
	const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = group_id * CHUNK_N + chunk_idx; \
	__local VTYPE * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	const sz_t sj = s + idx_m; DECLARE_IVAR(s, idx_m);

#define DECLARE_VAR_FORWARD() \
	__global VTYPE * restrict const zi = &z[ki]; \
	__global VTYPE * restrict const zo = &z[ko];

#define DECLARE_VAR_BACKWARD() \
	__global VTYPE * restrict const zi = &z[ko]; \
	__global VTYPE * restrict const zo = &z[ki];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(pq, B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);

#define FORWARD_I_0(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_0(pq, g_f0[lid], B_N * CHUNK_N, &Z[i], B_N << lm, zi);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sji / 1);

// -----------------

#define B_64	(64 / 4)

#if MAX_WG_SZ >= B_64 * CHUNK64
#define ATTR_64() \
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#else
#define ATTR_64()
#endif

#define FORWARD_64() \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);

__kernel
ATTR_64()
void forward64(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local VTYPE Z[4 * B_64 * CHUNK64];
	FORWARD_I(B_64, CHUNK64);
	FORWARD_64();
}

__kernel
ATTR_64()
void forward64_0(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	const int lm = LN_SZ - LVSIZE - 6; const unsigned int s = 64 / 4;
	__local VTYPE Z[4 * B_64 * CHUNK64];
	FORWARD_I_0(B_64, CHUNK64);
	FORWARD_64();
}

__kernel
ATTR_64()
void backward64(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local VTYPE Z[4 * B_64 * CHUNK64];
	BACKWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], wi, sji / 4);
	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], wi, sji / B_64);
}

// -----------------

#define B_256	(256 / 4)

#if MAX_WG_SZ >= B_256 * CHUNK256
#define ATTR_256() \
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#else
#define ATTR_256()
#endif

#define FORWARD_256() \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);

__kernel
ATTR_256()
void forward256(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local VTYPE Z[4 * B_256 * CHUNK256];
	FORWARD_I(B_256, CHUNK256);
	FORWARD_256();
}

__kernel
ATTR_256()
void forward256_0(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	const int lm = LN_SZ - LVSIZE - 8; const unsigned int s = 256 / 4;
	__local VTYPE Z[4 * B_256 * CHUNK256];
	FORWARD_I_0(B_256, CHUNK256);
	FORWARD_256();
}

__kernel
ATTR_256()
void backward256(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local VTYPE Z[4 * B_256 * CHUNK256];
	BACKWARD_I(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sji / 16);
	backward_4o(pq, B_256 << lm, zo, B_256 * CHUNK256, &Z[i], wi, sji / B_256);
}

// -----------------

#define B_1024	(1024 / 4)

#if MAX_WG_SZ >= B_1024 * CHUNK1024
#define ATTR_1024() \
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
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

__kernel
ATTR_1024()
void forward1024(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local VTYPE Z[4 * B_1024 * CHUNK1024];
	FORWARD_I(B_1024, CHUNK1024);
	FORWARD_1024();
}

__kernel
ATTR_1024()
void forward1024_0(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	const int lm = LN_SZ - LVSIZE - 10; const unsigned int s = 1024 / 4;
	__local VTYPE Z[4 * B_1024 * CHUNK1024];
	FORWARD_I_0(B_1024, CHUNK1024);
	FORWARD_1024();
}

__kernel
ATTR_1024()
void backward1024(__global VTYPE * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)
{
	__local VTYPE Z[4 * B_1024 * CHUNK1024];
	BACKWARD_I(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], wi, sji / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], wi, sji / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], wi, sji / 64);
	backward_4o(pq, B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], wi, sji / B_1024);
}

// -----------------

#define L32S	(32 / VSIZE)

#define DECLARE_VAR_32() \
	__local VTYPE Z[L32S * BLK32]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L32S / 4 * BLK32), group_id = id / (L32S / 4 * BLK32); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i32 = (local_id & ~(L32S / 4 - 1)) * 4, i8 = local_id % (L32S / 4); \
	const sz_t k32 = group_id * L32S * BLK32 + i32 + i8; \
	\
	__global VTYPE * restrict const zk = &z[k32]; \
	__local VTYPE * const Z32 = &Z[i32]; \
	__local VTYPE * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & ~(4 * 2 - 1)) + (i8 % 2); \
	__local VTYPE * const Zi2 = &Z32[i2]; \
	__local VTYPE * const Z4 = &Z32[4 * i8];

__kernel
#if MAX_WG_SZ >= L32S / 4 * BLK32
	__attribute__((reqd_work_group_size(L32S / 4 * BLK32, 1, 1)))
#endif
void square32(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_32();

	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4o(pq, L32S / 4, zk, L32S / 4, Zi8, wi, sji / (L32S / 4));
}

#define L64S	(64 / VSIZE)

#define DECLARE_VAR_64() \
	__local VTYPE Z[L64S * BLK64]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L64S / 4 * BLK64), group_id = id / (L64S / 4 * BLK64); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i64 = (local_id & ~(L64S / 4 - 1)) * 4, i16 = local_id % (L64S / 4); \
	const sz_t k64 = group_id * L64S * BLK64 + i64 + i16; \
	\
	__global VTYPE * restrict const zk = &z[k64]; \
	__local VTYPE * const Z64 = &Z[i64]; \
	__local VTYPE * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & ~(4 * (L64S / 16) - 1)) + (i16 % (L64S / 16)); \
	__local VTYPE * const Zi4 = &Z64[i4]; \
	__local VTYPE * const Z4 = &Z64[4 * i16];

__kernel
#if MAX_WG_SZ >= L64S / 4 * BLK64
	__attribute__((reqd_work_group_size(L64S / 4 * BLK64, 1, 1)))
#endif
void square64(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_64();

	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));
	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L64S / 16, Zi4, wi, sji / (L64S / 16));
	backward_4o(pq, L64S / 4, zk, L64S / 4, Zi16, wi, sji / (L64S / 4));
}

#define L128S	(128 / VSIZE)

#define DECLARE_VAR_128() \
	__local VTYPE Z[L128S * BLK128]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L128S / 4 * BLK128), group_id = id / (L128S / 4 * BLK128); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i128 = (local_id & ~(L128S / 4 - 1)) * 4, i32 = local_id % (L128S / 4); \
	const sz_t k128 = group_id * L128S * BLK128 + i128 + i32; \
	\
	__global VTYPE * restrict const zk = &z[k128]; \
	__local VTYPE * const Z128 = &Z[i128]; \
	__local VTYPE * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & ~(4 * (L128S / 16) - 1)) + (i32 % (L128S / 16)); \
	__local VTYPE * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & ~(4 * 2 - 1)) + (i32 % 2); \
	__local VTYPE * const Zi2 = &Z128[i2]; \
	__local VTYPE * const Z4 = &Z128[4 * i32];

__kernel
#if MAX_WG_SZ >= L128S / 4 * BLK128
	__attribute__((reqd_work_group_size(L128S / 4 * BLK128, 1, 1)))
#endif
void square128(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_128();

	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));
	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4(pq, L128S / 16, Zi8, wi, sji / (L128S / 16));
	backward_4o(pq, L128S / 4, zk, L128S / 4, Zi32, wi, sji / (L128S / 4));
}

#define L256S	(256 / VSIZE)

#define DECLARE_VAR_256() \
	__local VTYPE Z[L256S * BLK256]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L256S / 4 * BLK256), group_id = id / (L256S / 4 * BLK256); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i256 = (local_id & ~(L256S / 4 - 1)) * 4, i64 = local_id % (L256S / 4); \
	const sz_t k256 = group_id * L256S * BLK256 + i256 + i64; \
	\
	__global VTYPE * restrict const zk = &z[k256]; \
	__local VTYPE * const Z256 = &Z[i256]; \
	__local VTYPE * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & ~(4 * (L256S / 16) - 1)) + (i64 % (L256S / 16)); \
	__local VTYPE * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & ~(4 * (L256S / 64) - 1)) + (i64 % (L256S / 64)); \
	__local VTYPE * const Zi4 = &Z256[i4]; \
	__local VTYPE * const Z4 = &Z256[4 * i64];

__kernel
#if MAX_WG_SZ >= L256S / 4 * BLK256
	__attribute__((reqd_work_group_size(L256S / 4 * BLK256, 1, 1)))
#endif
void square256(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_256();

	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));
	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));
	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L256S / 64, Zi4, wi, sji / (L256S / 64));
	backward_4(pq, L256S / 16, Zi16, wi, sji / (L256S / 16));
	backward_4o(pq, L256S / 4, zk, L256S / 4, Zi64, wi, sji / (L256S / 4));
}

#define L512S	(512 / VSIZE)

#define DECLARE_VAR_512() \
	__local VTYPE Z[L512S * BLK512]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L512S / 4 * BLK512), group_id = id / (L512S / 4 * BLK512); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i512 = (local_id & ~(L512S / 4 - 1)) * 4, i128 = local_id % (L512S / 4); \
	const sz_t k512 = group_id * L512S * BLK512 + i512 + i128; \
	\
	__global VTYPE * restrict const zk = &z[k512]; \
	__local VTYPE * const Z512 = &Z[i512]; \
	__local VTYPE * const Zi128 = &Z512[i128]; \
	const sz_t i32 = ((4 * i128) & ~(4 * (L512S / 16) - 1)) + (i128 % (L512S / 16)); \
	__local VTYPE * const Zi32 = &Z512[i32]; \
	const sz_t i8 = ((4 * i128) & ~(4 * (L512S / 64) - 1)) + (i128 % (L512S / 64)); \
	__local VTYPE * const Zi8 = &Z512[i8]; \
	const sz_t i2 = ((4 * i128) & ~(4 * 2 - 1)) + (i128 % 2); \
	__local VTYPE * const Zi2 = &Z512[i2]; \
	__local VTYPE * const Z4 = &Z512[4 * i128];

__kernel
#if MAX_WG_SZ >= L512S / 4 * BLK512
	__attribute__((reqd_work_group_size(L512S / 4 * BLK512, 1, 1)))
#endif
void square512(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_512();

	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));
	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));
	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4(pq, L512S / 64, Zi8, wi, sji / (L512S / 64));
	backward_4(pq, L512S / 16, Zi32, wi, sji / (L512S / 16));
	backward_4o(pq, L512S / 4, zk, L512S / 4, Zi128, wi, sji / (L512S / 4));
}

#define L1024S	(1024 / VSIZE)

// if BLK1024 != 1 then const sz_t i1024 = (local_id & ~(L1024S / 4 - 1)) * 4, i256 = local_id % (L1024S / 4);
// if BLK1024 = 1 then const sz_t i1024 = 0, i256 = local_id;
#define DECLARE_VAR_1024() \
	__local VTYPE Z[L1024S * BLK1024]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L1024S / 4 * BLK1024), group_id = id / (L1024S / 4 * BLK1024); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i1024 = (local_id & ~(L1024S / 4 - 1)) * 4, i256 = local_id % (L1024S / 4); \
	const sz_t k1024 = group_id * L1024S * BLK1024 + i1024 + i256; \
	\
	__global VTYPE * restrict const zk = &z[k1024]; \
	__local VTYPE * const Z1024 = &Z[i1024]; \
	__local VTYPE * const Zi256 = &Z1024[i256]; \
	const sz_t i64 = ((4 * i256) & ~(4 * (L1024S / 16) - 1)) + (i256 % (L1024S / 16)); \
	__local VTYPE * const Zi64 = &Z1024[i64]; \
	const sz_t i16 = ((4 * i256) & ~(4 * (L1024S / 64) - 1)) + (i256 % (L1024S / 64)); \
	__local VTYPE * const Zi16 = &Z1024[i16]; \
	const sz_t i4 = ((4 * i256) & ~(4 * (L1024S / 256) - 1)) + (i256 % (L1024S / 256)); \
	__local VTYPE * const Zi4 = &Z1024[i4]; \
	__local VTYPE * const Z4 = &Z1024[4 * i256];

__kernel
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
	__attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))
#endif
void square1024(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_1024();

	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));
	backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));
	backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));
	backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));
}

#define L2048S	(2048 / VSIZE)

#define DECLARE_VAR_2048() \
	__local VTYPE Z[L2048S]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L2048S / 4), group_id = id / (L2048S / 4); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i512 = local_id, k2048 = group_id * L2048S + i512; \
	\
	__global VTYPE * restrict const zk = &z[k2048]; \
	__local VTYPE * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & ~(4 * (L2048S / 16) - 1)) + (i512 % (L2048S / 16)); \
	__local VTYPE * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & ~(4 * (L2048S / 64) - 1)) + (i512 % (L2048S / 64)); \
	__local VTYPE * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & ~(4 * (L2048S / 256) - 1)) + (i512 % (L2048S / 256)); \
	__local VTYPE * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & ~(4 * 2 - 1)) + (i512 % 2); \
	__local VTYPE * const Zi2 = &Z[i2]; \
	__local VTYPE * const Z4 = &Z[4 * i512];

__kernel
#if MAX_WG_SZ >= L2048S / 4
	__attribute__((reqd_work_group_size(L2048S / 4, 1, 1)))
#endif
void square2048(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_2048();

	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));
	backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));
	backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));
	backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));
}

#define L4096S	(4096 / VSIZE)

#define DECLARE_VAR_4096() \
	__local VTYPE Z[L4096S]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L4096S / 4), group_id = id / (L4096S / 4); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i1024 = local_id, k4096 = group_id * L4096S + i1024; \
	\
	__global VTYPE * restrict const zk = &z[k4096]; \
	__local VTYPE * const Zi1024 = &Z[i1024]; \
	const sz_t i256 = ((4 * i1024) & ~(4 * (L4096S / 16) - 1)) + (i1024 % (L4096S / 16)); \
	__local VTYPE * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i1024) & ~(4 * (L4096S / 64) - 1)) + (i1024 % (L4096S / 64)); \
	__local VTYPE * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i1024) & ~(4 * (L4096S / 256) - 1)) + (i1024 % (L4096S / 256)); \
	__local VTYPE * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i1024) & ~(4 * (L4096S / 1024) - 1)) + (i1024 % (L4096S / 1024)); \
	__local VTYPE * const Zi4 = &Z[i4]; \
	__local VTYPE * const Z4 = &Z[4 * i1024];

__kernel
#if MAX_WG_SZ >= L4096S / 4
	__attribute__((reqd_work_group_size(L4096S / 4, 1, 1)))
#endif
void square4096(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_4096();

	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));
	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));
	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));
	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));
	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L4096S / 1024, Zi4, wi, sji / (L4096S / 1024));
	backward_4(pq, L4096S / 256, Zi16, wi, sji / (L4096S / 256));
	backward_4(pq, L4096S / 64, Zi64, wi, sji / (L4096S / 64));
	backward_4(pq, L4096S / 16, Zi256, wi, sji / (L4096S / 16));
	backward_4o(pq, L4096S / 4, zk, L4096S / 4, Zi1024, wi, sji / (L4096S / 4));
}

// -----------------

__kernel
#if MAX_WG_SZ >= L32S / 4 * BLK32
	__attribute__((reqd_work_group_size(L32S / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_32();

	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(8, zk, Z4);
#else
	fwd8_write(pq, L32S / 4, zk, Z4, w, sj);
#endif
}

__kernel
#if MAX_WG_SZ >= L64S / 4 * BLK64
	__attribute__((reqd_work_group_size(L64S / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_64();

	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));
	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));
	fwd4_write(pq, L64S / 4, zk, Z4, w, sj);
}

__kernel
#if MAX_WG_SZ >= L128S / 4 * BLK128
	__attribute__((reqd_work_group_size(L128S / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_128();

	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));
	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(32, zk, Z4);
#else
	fwd8_write(pq, L128S / 4, zk, Z4, w, sj);
#endif
}

__kernel
#if MAX_WG_SZ >= L256S / 4 * BLK256
	__attribute__((reqd_work_group_size(L256S / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_256();

	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));
	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));
	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));
	fwd4_write(pq, L256S / 4, zk, Z4, w, sj);
}

__kernel
#if MAX_WG_SZ >= L512S / 4 * BLK512
	__attribute__((reqd_work_group_size(L512S / 4 * BLK512, 1, 1)))
#endif
void fwd512p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_512();

	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));
	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));
	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(128, zk, Z4);
#else
	fwd8_write(pq, L512S / 4, zk, Z4, w, sj);
#endif
}

__kernel
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
	__attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))
#endif
void fwd1024p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_1024();

	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
	fwd4_write(pq, L1024S / 4, zk, Z4, w, sj);
}

__kernel
#if MAX_WG_SZ >= L2048S / 4
	__attribute__((reqd_work_group_size(L2048S / 4, 1, 1)))
#endif
void fwd2048p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_2048();

	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(512, zk, Z4);
#else
	fwd8_write(pq, L2048S / 4, zk, Z4, w, sj);
#endif
}

__kernel
#if MAX_WG_SZ >= L4096S / 4
	__attribute__((reqd_work_group_size(L4096S / 4, 1, 1)))
#endif
void fwd4096p(__global VTYPE * restrict const zg, __global const uint * restrict const wg)
{
	DECLARE_VAR_4096();

	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));
	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));
	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));
	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));
	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));
	fwd4_write(pq, L4096S / 4, zk, Z4, w, sj);
}

// -----------------

__kernel
#if MAX_WG_SZ >= L32S / 4 * BLK32
	__attribute__((reqd_work_group_size(L32S / 4 * BLK32, 1, 1)))
#endif
void mul32(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_32();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k32];

	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 8, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L32S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4o(pq, L32S / 4, zk, L32S / 4, Zi8, wi, sji / (L32S / 4));
}

__kernel
#if MAX_WG_SZ >= L64S / 4 * BLK64
	__attribute__((reqd_work_group_size(L64S / 4 * BLK64, 1, 1)))
#endif
void mul64(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_64();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k64];

	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));
	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));
	mul_4(pq, Z4, L64S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L64S / 16, Zi4, wi, sji / (L64S / 16));
	backward_4o(pq, L64S / 4, zk, L64S / 4, Zi16, wi, sji / (L64S / 4));
}

__kernel
#if MAX_WG_SZ >= L128S / 4 * BLK128
	__attribute__((reqd_work_group_size(L128S / 4 * BLK128, 1, 1)))
#endif
void mul128(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_128();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k128];

	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));
	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 32, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L128S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4(pq, L128S / 16, Zi8, wi, sji / (L128S / 16));
	backward_4o(pq, L128S / 4, zk, L128S / 4, Zi32, wi, sji / (L128S / 4));
}

__kernel
#if MAX_WG_SZ >= L256S / 4 * BLK256
	__attribute__((reqd_work_group_size(L256S / 4 * BLK256, 1, 1)))
#endif
void mul256(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_256();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k256];

	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));
	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));
	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));
	mul_4(pq, Z4, L256S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L256S / 64, Zi4, wi, sji / (L256S / 64));
	backward_4(pq, L256S / 16, Zi16, wi, sji / (L256S / 16));
	backward_4o(pq, L256S / 4, zk, L256S / 4, Zi64, wi, sji / (L256S / 4));
}

__kernel
#if MAX_WG_SZ >= L512S / 4 * BLK512
	__attribute__((reqd_work_group_size(L512S / 4 * BLK512, 1, 1)))
#endif
void mul512(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_512();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k512];

	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));
	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));
	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 128, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L512S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4(pq, L512S / 64, Zi8, wi, sji / (L512S / 64));
	backward_4(pq, L512S / 16, Zi32, wi, sji / (L512S / 16));
	backward_4o(pq, L512S / 4, zk, L512S / 4, Zi128, wi, sji / (L512S / 4));
}

__kernel
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
	__attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))
#endif
void mul1024(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_1024();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k1024];

	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
	mul_4(pq, Z4, L1024S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));
	backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));
	backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));
	backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));
}

__kernel
#if MAX_WG_SZ >= L2048S / 4
	__attribute__((reqd_work_group_size(L2048S / 4, 1, 1)))
#endif
void mul2048(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_2048();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k2048];

	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 512, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L2048S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));
	backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));
	backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));
	backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));
}

__kernel
#if MAX_WG_SZ >= L4096S / 4
	__attribute__((reqd_work_group_size(L4096S / 4, 1, 1)))
#endif
void mul4096(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint * restrict const wg)
{
	DECLARE_VAR_4096();
	DECLARE_VARP_REG();
	__global const VTYPE * restrict const zpk = &zp[k4096];

	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));
	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));
	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));
	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));
	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));
	mul_4(pq, Z4, L4096S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L4096S / 1024, Zi4, wi, sji / (L4096S / 1024));
	backward_4(pq, L4096S / 256, Zi16, wi, sji / (L4096S / 256));
	backward_4(pq, L4096S / 64, Zi64, wi, sji / (L4096S / 64));
	backward_4(pq, L4096S / 16, Zi256, wi, sji / (L4096S / 16));
	backward_4o(pq, L4096S / 4, zk, L4096S / 4, Zi1024, wi, sji / (L4096S / 4));
}

#endif	// SHORT_VER

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
	// 2- t < 2^23 b^2 => t_h < b^2 / 2^6. If 2 <= b < 32 then t_h < 32^2 / 2^6 = 16 < 2^29 b
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

#if RNS_SZ == 2

	int64 f = 0;

	sz_t j = 0;
	do
	{
		const uint32 u1 = mulmod(zi[j + 0 * N_SZ], NORM1, PQ1);
		const uint32 u2 = mulmod(zi[j + 1 * N_SZ], NORM2, PQ2);
		int64 l = garner2(u1, u2);
		if (sblk < 0) l += l;
		f += l;
		const int32 r = reduce64(&f, b, b_inv, b_s);
		zi[j + 0 * N_SZ] = set_int(r, P1);
		zi[j + 1 * N_SZ] = set_int(r, P2);
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	c[i] = (i == 0) ? -f : f;

#else

	int96 f = int96_set_si(0);

	sz_t j = 0;
	do
	{
		const uint32 u1 = mulmod(zi[j + 0 * N_SZ], NORM1, PQ1);
		const uint32 u2 = mulmod(zi[j + 1 * N_SZ], NORM2, PQ2);
		const uint32 u3 = mulmod(zi[j + 2 * N_SZ], NORM3, PQ3);
		int96 l = garner3(u1, u2, u3);
		if (sblk < 0) l = int96_add(l, l);
		f = int96_add(f, l);
		const int32 r = reduce96(&f, b, b_inv, b_s);
		zi[j + 0 * N_SZ] = set_int(r, P1);
		zi[j + 1 * N_SZ] = set_int(r, P2);
		zi[j + 2 * N_SZ] = set_int(r, P3);
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
		zi[j + 0 * N_SZ] = set_int(r, P1);
		zi[j + 1 * N_SZ] = set_int(r, P2);
#if RNS_SZ == 3
		zi[j + 2 * N_SZ] = set_int(r, P3);
#endif
		if (f == 0) return;
		++j;
	} while (j != blk - 1);

	const int32 r = (int32)(f);
	zi[blk - 1 + 0 * N_SZ] = addmod(zi[blk - 1 + 0 * N_SZ], set_int(r, P1), P1);
	zi[blk - 1 + 1 * N_SZ] = addmod(zi[blk - 1 + 1 * N_SZ], set_int(r, P2), P2);
#if RNS_SZ == 3
	zi[blk - 1 + 2 * N_SZ] = addmod(zi[blk - 1 + 2 * N_SZ], set_int(r, P3), P3);
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
		zi[j + 0 * N_SZ] = set_int(r, P1);
		zi[j + 1 * N_SZ] = set_int(r, P2);
#if RNS_SZ == 3
		zi[j + 2 * N_SZ] = set_int(r, P3);
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
	z[idx] = ((idx & (N_SZ - 1)) == 0) ? a : 0;
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
