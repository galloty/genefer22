/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2022, Yves Gallot\n" \
"\n" \
"genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#if __OPENCL_VERSION__ >= 120\n" \
"	#define INLINE	static inline\n" \
"#else\n" \
"	#define INLINE\n" \
"#endif\n" \
"\n" \
"#if defined(__NV_CL_C_VERSION)\n" \
"	#define PTX_ASM	1\n" \
"#endif\n" \
"\n" \
"#if !defined(N_SZ)\n" \
"#define N_SZ		65536u\n" \
"#define LN_SZ		16\n" \
"#define RNS_SZ		3\n" \
"#define VSIZE		4\n" \
"#define LVSIZE		2\n" \
"// #define IS32		1\n" \
"#define P1			2130706433u\n" \
"#define Q1			2164260865u\n" \
"#define RSQ1		402124772u\n" \
"#define IM1			1930170389u\n" \
"#define SQRTI1		1626730317u\n" \
"#define ISQRTI1		856006302u\n" \
"#define P2			2113929217u\n" \
"#define Q2			2181038081u\n" \
"#define RSQ2		2111798781u\n" \
"#define IM2			1036950657u\n" \
"#define SQRTI2		338852760u\n" \
"#define ISQRTI2		1090446030u\n" \
"#define P3			2013265921u\n" \
"#define Q3			2281701377u\n" \
"#define RSQ3		1172168163u\n" \
"#define IM3			734725699u\n" \
"#define SQRTI3		1032137103u\n" \
"#define ISQRTI3		1964242958u\n" \
"#define INVP2_P1	2130706177u\n" \
"#define INVP3_P1	608773230u\n" \
"#define INVP3_P2	1409286102u\n" \
"#define P1P2P3L		13049742876517335041ul\n" \
"#define P1P2P3H		491581440u\n" \
"#define P1P2P3_2L	6524871438258667520ul\n" \
"#define P1P2P3_2H	245790720u\n" \
"#define NORM1		2130641409u\n" \
"#define NORM2		2113864705u\n" \
"#define NORM3		2013204481u\n" \
"#define W_SHFT		65536u\n" \
"#define WI_SHFT		32768u\n" \
"// #define USE_WI		1\n" \
"#define BLK32		32\n" \
"#define BLK64		16\n" \
"#define BLK128		8\n" \
"#define BLK256		4\n" \
"#define BLK512		2\n" \
"#define BLK1024		1\n" \
"#define CHUNK64		4\n" \
"#define CHUNK256	4\n" \
"#define CHUNK1024	1\n" \
"// #define SHORT_VER	1\n" \
"#define NORM_WG_SZ	32\n" \
"#define MAX_WG_SZ	256\n" \
"#endif\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uint	uint32;\n" \
"typedef int		int32;\n" \
"typedef ulong	uint64;\n" \
"typedef long	int64;\n" \
"typedef uint2	uint32_2;\n" \
"typedef uint4	uint32_4;\n" \
"typedef int4	int32_4;\n" \
"typedef long4	int64_4;\n" \
"\n" \
"// --- modular arithmetic\n" \
"\n" \
"#define	PQ1		(uint32_2)(P1, Q1)\n" \
"#define	PQ2		(uint32_2)(P2, Q2)\n" \
"#define	PQ3		(uint32_2)(P3, Q3)\n" \
"\n" \
"__constant uint32_2 g_pq[3] = { PQ1, PQ2, PQ3 };\n" \
"__constant uint32_4 g_f0[3] = { (uint32_4)(RSQ1, IM1, SQRTI1, ISQRTI1), (uint32_4)(RSQ2, IM2, SQRTI2, ISQRTI2), (uint32_4)(RSQ3, IM3, SQRTI3, ISQRTI3) };\n" \
"\n" \
"INLINE uint32 addmod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"#if defined(IS32)\n" \
"	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);\n" \
"#else\n" \
"	const uint32 t = lhs + rhs;\n" \
"	return t - ((t >= p) ? p : 0);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE uint32 submod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"#if defined(IS32)\n" \
"	return lhs - rhs + ((lhs < rhs) ? p : 0);\n" \
"#else\n" \
"	const uint32 t = lhs - rhs;\n" \
"	return t + (((int32)(t) < 0) ? p : 0);\n" \
"#endif\n" \
"}\n" \
"\n" \
"// 2 mul + 2 mul_hi\n" \
"INLINE uint32 mulmod(const uint32 lhs, const uint32 rhs, const uint32_2 pq)\n" \
"{\n" \
"	const uint64 t = lhs * (uint64)(rhs);\n" \
"	const uint32 lo = (uint32)(t), hi = (uint32)(t >> 32);\n" \
"	const uint32 mp = mul_hi(lo * pq.s1, pq.s0);\n" \
"	return submod(hi, mp, pq.s0);\n" \
"}\n" \
"\n" \
"INLINE uint32 sqrmod(const uint32 lhs, const uint32_2 pq) { return mulmod(lhs, lhs, pq); }\n" \
"\n" \
"INLINE int32 get_int(const uint32 n, const uint32 p) { return (n >= p / 2) ? (int32)(n - p) : (int32)(n); }	// ? 2n >= p ?\n" \
"INLINE uint32 set_int(const int32 i, const uint32 p) { return (i < 0) ? ((uint32)(i) + p) : (uint32)(i); }\n" \
"\n" \
"// --- v2\n" \
"\n" \
"INLINE uint32_2 addmod2(const uint32_2 lhs, const uint32_2 rhs, const uint32 p)\n" \
"{\n" \
"	return (uint32_2)(addmod(lhs.s0, rhs.s0, p), addmod(lhs.s1, rhs.s1, p));\n" \
"}\n" \
"\n" \
"INLINE uint32_2 submod2(const uint32_2 lhs, const uint32_2 rhs, const uint32 p)\n" \
"{\n" \
"	return (uint32_2)(submod(lhs.s0, rhs.s0, p), submod(lhs.s1, rhs.s1, p));\n" \
"}\n" \
"\n" \
"INLINE uint32_2 mulmod2(const uint32_2 lhs, const uint32_2 rhs, const uint32_2 pq)\n" \
"{\n" \
"	return (uint32_2)(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq));\n" \
"}\n" \
"\n" \
"// --- v4\n" \
"\n" \
"INLINE uint32_4 addmod4(const uint32_4 lhs, const uint32_4 rhs, const uint32 p)\n" \
"{\n" \
"	return (uint32_4)(addmod2(lhs.s01, rhs.s01, p), addmod2(lhs.s23, rhs.s23, p));\n" \
"}\n" \
"\n" \
"INLINE uint32_4 submod4(const uint32_4 lhs, const uint32_4 rhs, const uint32 p)\n" \
"{\n" \
"	return (uint32_4)(submod2(lhs.s01, rhs.s01, p), submod2(lhs.s23, rhs.s23, p));\n" \
"}\n" \
"\n" \
"INLINE uint32_4 mulmod4(const uint32_4 lhs, const uint32_4 rhs, const uint32_2 pq)\n" \
"{\n" \
"	return (uint32_4)(mulmod2(lhs.s01, rhs.s01, pq), mulmod2(lhs.s23, rhs.s23, pq));\n" \
"}\n" \
"\n" \
"// --- uint96/int96 ---\n" \
"\n" \
"typedef struct { uint64 s0; uint32 s1; } uint96;\n" \
"typedef struct { uint64 s0; int32 s1; } int96;\n" \
"\n" \
"INLINE int96 int96_zero() { int96 r; r.s0 = 0; r.s1 = 0; return r; }\n" \
"INLINE int96 int96_set_si(const int64 n) { int96 r; r.s0 = (uint64)(n); r.s1 = (n < 0) ? -1 : 0; return r; }\n" \
"INLINE uint96 uint96_set(const uint64 s0, const uint32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }\n" \
"\n" \
"INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int32)(x.s1); return r; }\n" \
"INLINE uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint32)(x.s1); return r; }\n" \
"\n" \
"INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }\n" \
"\n" \
"INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }\n" \
"\n" \
"INLINE uint96 uint96_add_64(const uint96 x, const uint64 y)\n" \
"{\n" \
"	uint96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y));\n" \
"	asm volatile (\"addc.u32 %0, %1, 0;\" : \"=r\" (r.s1) : \"r\" (x.s1));\n" \
"#else\n" \
"	const uint64 s0 = x.s0 + y;\n" \
"	r.s0 = s0; r.s1 = x.s1 + ((s0 < y) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int96 int96_add(const int96 x, const int96 y)\n" \
"{\n" \
"	int96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y.s0));\n" \
"	asm volatile (\"addc.s32 %0, %1, %2;\" : \"=r\" (r.s1) : \"r\" (x.s1), \"r\" (y.s1));\n" \
"#else\n" \
"	const uint64 s0 = x.s0 + y.s0;\n" \
"	r.s0 = s0; r.s1 = x.s1 + y.s1 + ((s0 < y.s0) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 uint96_sub(const uint96 x, const uint96 y)\n" \
"{\n" \
"	uint96 r;\n" \
"#if defined(PTX_ASM)\n" \
"	asm volatile (\"sub.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y.s0));\n" \
"	asm volatile (\"subc.u32 %0, %1, %2;\" : \"=r\" (r.s1) : \"r\" (x.s1), \"r\" (y.s1));\n" \
"#else\n" \
"	r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - ((x.s0 < y.s0) ? 1 : 0));\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 int96_abs(const int96 x)\n" \
"{\n" \
"	const bool is_neg = int96_is_neg(x);\n" \
"	const uint96 mask = uint96_set(is_neg ? ~0ul : 0ul, is_neg ? ~0u : 0u);\n" \
"	uint96 t = uint96_set(x.s0 ^ mask.s0, (uint32)(x.s1) ^ mask.s1);\n" \
"	return uint96_sub(t, mask);\n" \
"}\n" \
"\n" \
"INLINE uint96 uint96_mul_64_32(const uint64 x, const uint32 y)\n" \
"{\n" \
"	const uint64 l = (uint32)(x) * (uint64)(y), h = (x >> 32) * y + (l >> 32);\n" \
"	uint96 r; r.s0 = (h << 32) | (uint32)(l); r.s1 = (uint32)(h >> 32);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define FWD2(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint32 t = mulmod(z1, w, pq); \\\n" \
"	z1 = submod(z0, t, pq.s0); z0 = addmod(z0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define BCK2(z0, z1, win) \\\n" \
"{ \\\n" \
"	const uint32 t = submod(z1, z0, pq.s0); z0 = addmod(z0, z1, pq.s0); \\\n" \
"	z1 = mulmod(t, win, pq); \\\n" \
"}\n" \
"\n" \
"#define SQR2(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint32 t = mulmod(sqrmod(z1, pq), w, pq); \\\n" \
"	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \\\n" \
"	z0 = addmod(sqrmod(z0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define SQR2N(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint32 t = mulmod(sqrmod(z1, pq), w, pq); \\\n" \
"	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \\\n" \
"	z0 = submod(sqrmod(z0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2(z0, z1, zp0, zp1, w) \\\n" \
"{ \\\n" \
"	const uint32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \\\n" \
"	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \\\n" \
"	z0 = addmod(mulmod(z0, zp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define MUL2N(z0, z1, zp0, zp1, w) \\\n" \
"{ \\\n" \
"	const uint32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \\\n" \
"	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \\\n" \
"	z0 = submod(mulmod(z0, zp0, pq), t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define FWD2v2(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint32_2 t = mulmod2(z1, w, pq); \\\n" \
"	z1 = submod2(z0, t, pq.s0); z0 = addmod2(z0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define BCK2v2(z0, z1, win) \\\n" \
"{ \\\n" \
"	const uint32_2 t = submod2(z1, z0, pq.s0); z0 = addmod2(z0, z1, pq.s0); \\\n" \
"	z1 = mulmod2(t, win, pq); \\\n" \
"}\n" \
"\n" \
"#define FWD2v4(z0, z1, w) \\\n" \
"{ \\\n" \
"	const uint32_4 t = mulmod4(z1, w, pq); \\\n" \
"	z1 = submod4(z0, t, pq.s0); z0 = addmod4(z0, t, pq.s0); \\\n" \
"}\n" \
"\n" \
"#define BCK2v4(z0, z1, win) \\\n" \
"{ \\\n" \
"	const uint32_4 t = submod4(z1, z0, pq.s0); z0 = addmod4(z0, z1, pq.s0); \\\n" \
"	z1 = mulmod4(t, win, pq); \\\n" \
"}\n" \
"\n" \
"INLINE void _loadg1(const sz_t n, uint32 * const zl, __global const uint32 * restrict const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }\n" \
"INLINE void _loadl1(const sz_t n, uint32 * const zl, __local const uint32 * restrict const Z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }\n" \
"INLINE void _storeg1(const sz_t n, __global uint32 * restrict const z, const size_t s, const uint32 * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }\n" \
"INLINE void _storel1(const sz_t n, __local uint32 * restrict const Z, const size_t s, const uint32 * const zl) { for (size_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }\n" \
"\n" \
"INLINE void _loadg2(const sz_t n, uint32_2 * const zl, __global const uint32_2 * restrict const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }\n" \
"INLINE void _loadl2(const sz_t n, uint32_2 * const zl, __local const uint32_2 * restrict const Z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }\n" \
"INLINE void _storeg2(const sz_t n, __global uint32_2 * restrict const z, const size_t s, const uint32_2 * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }\n" \
"INLINE void _storel2(const sz_t n, __local uint32_2 * restrict const Z, const size_t s, const uint32_2 * const zl) { for (size_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }\n" \
"\n" \
"INLINE void _loadg4(const sz_t n, uint32_4 * const zl, __global const uint32_4 * restrict const z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = z[l * s]; }\n" \
"INLINE void _loadl4(const sz_t n, uint32_4 * const zl, __local const uint32_4 * restrict const Z, const size_t s) { for (size_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }\n" \
"INLINE void _storeg4(const sz_t n, __global uint32_4 * restrict const z, const size_t s, const uint32_4 * const zl) { for (size_t l = 0; l < n; ++l) z[l * s] = zl[l]; }\n" \
"INLINE void _storel4(const sz_t n, __local uint32_4 * restrict const Z, const size_t s, const uint32_4 * const zl) { for (size_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }\n" \
"\n" \
"// ---\n" \
"\n" \
"INLINE void _forward4x1(const uint32_2 pq, uint32 z[4], const uint32 w1, const uint32 w2[2])\n" \
"{\n" \
"	FWD2(z[0], z[2], w1); FWD2(z[1], z[3], w1);\n" \
"	FWD2(z[0], z[1], w2[0]); FWD2(z[2], z[3], w2[1]);\n" \
"}\n" \
"\n" \
"INLINE void _backward4x1(const uint32_2 pq, uint32 z[4], const uint32 win1, const uint32 win2[2])\n" \
"{\n" \
"	BCK2(z[0], z[1], win2[0]); BCK2(z[2], z[3], win2[1]);\n" \
"	BCK2(z[0], z[2], win1); BCK2(z[1], z[3], win1);\n" \
"}\n" \
"\n" \
"INLINE void _forward4x1_0(const uint32_2 pq, const uint32_4 f0, uint32 z[4])\n" \
"{\n" \
"	const uint32 rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3;\n" \
"	z[0] = mulmod(z[0], rsq, pq); z[1] = mulmod(z[1], rsq, pq);\n" \
"	FWD2(z[0], z[2], im); FWD2(z[1], z[3], im);\n" \
"	FWD2(z[0], z[1], sqrti); FWD2(z[2], z[3], isqrti);\n" \
"}\n" \
"\n" \
"INLINE void _square2x2(const uint32_2 pq, uint32 z[4], const uint32 w)\n" \
"{\n" \
"	SQR2(z[0], z[1], w); SQR2N(z[2], z[3], w);\n" \
"}\n" \
"\n" \
"INLINE void _square4(const uint32_2 pq, uint32 z[4], const uint32 w, const uint32 win)\n" \
"{\n" \
"	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);\n" \
"	_square2x2(pq, z, w);\n" \
"	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4(const uint32_2 pq, uint32 z[4], const uint32 w)\n" \
"{\n" \
"	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);\n" \
"}\n" \
"\n" \
"INLINE void _mul2x2(const uint32_2 pq, uint32 z[4], const uint32 zp[4], const uint32 w)\n" \
"{\n" \
"	MUL2(z[0], z[1], zp[0], zp[1], w); MUL2N(z[2], z[3], zp[2], zp[3], w);\n" \
"}\n" \
"\n" \
"INLINE void _mul4(const uint32_2 pq, uint32 z[4], const uint32 zp[4], const uint32 w, const uint32 win)\n" \
"{\n" \
"	_fwd4(pq, z, w);\n" \
"	_mul2x2(pq, z, zp, w);\n" \
"	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);\n" \
"}\n" \
"\n" \
"// --- v2\n" \
"\n" \
"INLINE void _forward4x2(const uint32_2 pq, uint32_2 z[4], const uint32 w1, const uint32 w2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);\n" \
"	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);\n" \
"}\n" \
"\n" \
"INLINE void _backward4x2(const uint32_2 pq, uint32_2 z[4], const uint32 win1, const uint32 win2[2])\n" \
"{\n" \
"	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);\n" \
"	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);\n" \
"}\n" \
"\n" \
"INLINE void _forward4x2_0(const uint32_2 pq, const uint32_4 f0, uint32_2 z[4])\n" \
"{\n" \
"	const uint32 rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3;\n" \
"	z[0] = mulmod2(z[0], rsq, pq); z[1] = mulmod2(z[1], rsq, pq);\n" \
"	FWD2v2(z[0], z[2], im); FWD2v2(z[1], z[3], im);\n" \
"	FWD2v2(z[0], z[1], sqrti); FWD2v2(z[2], z[3], isqrti);\n" \
"}\n" \
"\n" \
"INLINE void _square4x2(const uint32_2 pq, uint32_2 z[4], const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);\n" \
"	SQR2(z[0].s0, z[0].s1, w2[0]); SQR2N(z[1].s0, z[1].s1, w2[0]);\n" \
"	SQR2(z[2].s0, z[2].s1, w2[1]); SQR2N(z[3].s0, z[3].s1, w2[1]);\n" \
"	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);\n" \
"}\n" \
"\n" \
"INLINE void _square8(const uint32_2 pq, uint32_2 z[4], const uint32 w1, const uint32 win1, const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);\n" \
"	_square4x2(pq, z, w2, win2);\n" \
"	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4x2(const uint32_2 pq, uint32_2 z[4], const uint32 w2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);\n" \
"}\n" \
"\n" \
"INLINE void _fwd8(const uint32_2 pq, uint32_2 z[4], const uint32 w1, const uint32 w2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);\n" \
"	_fwd4x2(pq, z, w2);\n" \
"}\n" \
"\n" \
"INLINE void _mul4x2(const uint32_2 pq, uint32_2 z[4], const uint32_2 zp[4], const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);\n" \
"	MUL2(z[0].s0, z[0].s1, zp[0].s0, zp[0].s1, w2[0]); MUL2N(z[1].s0, z[1].s1, zp[1].s0, zp[1].s1, w2[0]);\n" \
"	MUL2(z[2].s0, z[2].s1, zp[2].s0, zp[2].s1, w2[1]); MUL2N(z[3].s0, z[3].s1, zp[3].s0, zp[3].s1, w2[1]);\n" \
"	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);\n" \
"}\n" \
"\n" \
"INLINE void _mul8(const uint32_2 pq, uint32_2 z[4], const uint32_2 zp[4], const uint32 w1, const uint32  win1, const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);\n" \
"	_mul4x2(pq, z, zp, w2, win2);\n" \
"	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);\n" \
"}\n" \
"\n" \
"// --- v4\n" \
"\n" \
"INLINE void _forward4x4(const uint32_2 pq, uint32_4 z[4], const uint32 w1, const uint32 w2[2])\n" \
"{\n" \
"	FWD2v4(z[0], z[2], w1); FWD2v4(z[1], z[3], w1);\n" \
"	FWD2v4(z[0], z[1], w2[0]); FWD2v4(z[2], z[3], w2[1]);\n" \
"}\n" \
"\n" \
"INLINE void _backward4x4(const uint32_2 pq, uint32_4 z[4], const uint32 win1, const uint32 win2[2])\n" \
"{\n" \
"	BCK2v4(z[0], z[1], win2[0]); BCK2v4(z[2], z[3], win2[1]);\n" \
"	BCK2v4(z[0], z[2], win1); BCK2v4(z[1], z[3], win1);\n" \
"}\n" \
"\n" \
"INLINE void _forward4x4_0(const uint32_2 pq, const uint32_4 f0, uint32_4 z[4])\n" \
"{\n" \
"	const uint32 rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3;\n" \
"	z[0] = mulmod4(z[0], rsq, pq); z[1] = mulmod4(z[1], rsq, pq);\n" \
"	FWD2v4(z[0], z[2], im); FWD2v4(z[1], z[3], im);\n" \
"	FWD2v4(z[0], z[1], sqrti); FWD2v4(z[2], z[3], isqrti);\n" \
"}\n" \
"\n" \
"INLINE void _square4x2v4(const uint32_2 pq, uint32_4 z[2], const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		FWD2v2(z[i].s01, z[i].s23, w2[i]);\n" \
"		SQR2(z[i].s0, z[i].s1, w2[i]); SQR2N(z[i].s2, z[i].s3, w2[i]);\n" \
"		BCK2v2(z[i].s01, z[i].s23, win2[i]);\n" \
"	}\n" \
"}\n" \
"\n" \
"INLINE void _square8v4(const uint32_2 pq, uint32_4 z[2], const uint32 w1, const uint32 win1, const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	FWD2v4(z[0], z[1], w1);\n" \
"	_square4x2v4(pq, z, w2, win2);\n" \
"	BCK2v4(z[0], z[1], win1);\n" \
"}\n" \
"\n" \
"INLINE void _fwd4x2v4(const uint32_2 pq, uint32_4 z[2], const uint32 w2[2])\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i) FWD2v2(z[i].s01, z[i].s23, w2[i]);\n" \
"}\n" \
"\n" \
"INLINE void _fwd8v4(const uint32_2 pq, uint32_4 z[2], const uint32 w1, const uint32 w2[2])\n" \
"{\n" \
"	FWD2v4(z[0], z[1], w1);\n" \
"	_fwd4x2v4(pq, z, w2);\n" \
"}\n" \
"\n" \
"INLINE void _mul4x2v4(const uint32_2 pq, uint32_4 z[2], const uint32_4 zp[2], const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		FWD2v2(z[i].s01, z[i].s23, w2[i]);\n" \
"		MUL2(z[i].s0, z[i].s1, zp[i].s0, zp[i].s1, w2[i]); MUL2N(z[i].s2, z[i].s3, zp[i].s2, zp[i].s3, w2[i]);\n" \
"		BCK2v2(z[i].s01, z[i].s23, win2[i]);\n" \
"	}\n" \
"}\n" \
"\n" \
"INLINE void _mul8v4(const uint32_2 pq, uint32_4 z[2], const uint32_4 zp[2], const uint32 w1, const uint32  win1, const uint32 w2[2], const uint32 win2[2])\n" \
"{\n" \
"	FWD2v4(z[0], z[1], w1);\n" \
"	_mul4x2v4(pq, z, zp, w2, win2);\n" \
"	BCK2v4(z[0], z[1], win1);\n" \
"}\n" \
"\n" \
"// --- inverse of roots is wi[s + j] or w[s + s - j - 1] ---\n" \
"\n" \
"#define DECLARE_W1(sj)			const uint32 w1 = w[sj];\n" \
"#define DECLARE_W2(sj)			uint32 w2[2]; { const uint32_2 t = ((__global const uint32_2 *)w)[sj]; w2[0] = t.s0; w2[1] = t.s1; }\n" \
"#define DECLARE_W12(sj)			DECLARE_W1(sj); DECLARE_W2(sj);\n" \
"#define DECLARE_W1_2(sj)		uint32 w1[2]; { const uint32_2 t = ((__global const uint32_2 *)w)[sj]; w1[0] = t.s0; w1[1] = t.s1; }\n" \
"#define DECLARE_W2_4(sj)		uint32 w2[4]; { const uint32_4 t = ((__global const uint32_4 *)w)[sj]; w2[0] = t.s0; w2[1] = t.s1; w2[2] = t.s2; w2[3] = t.s3; }\n" \
"#define DECLARE_W12_24(sj)		DECLARE_W1_2(sj); DECLARE_W2_4(sj);\n" \
"\n" \
"#define DECLARE_WIN1(sji)		const uint32 win1 = wi[sji];\n" \
"#if defined(USE_WI)\n" \
"#define DECLARE_IVAR(s, j)		const sz_t sji = s + j; __global const uint32 * restrict const wi = &w[WI_SHFT];\n" \
"#define DECLARE_WIN2(sji)		uint32 win2[2]; { const uint32_2 t = ((__global const uint32_2 *)wi)[sji]; win2[0] = t.s0; win2[1] = t.s1; }\n" \
"#define DECLARE_WIN1_2(sji)		uint32 win1[2]; { const uint32_2 t = ((__global const uint32_2 *)wi)[sji]; win1[0] = t.s0; win1[1] = t.s1; }\n" \
"#define DECLARE_WIN2_4(sji)		uint32 win2[4]; { const uint32_4 t = ((__global const uint32_4 *)wi)[sji]; win2[0] = t.s0; win2[1] = t.s1; win2[2] = t.s2; win2[3] = t.s3; }\n" \
"#else\n" \
"#define DECLARE_IVAR(s, j)		const sz_t sji = s + s - j - 1; __global const uint32 * restrict const wi = w;\n" \
"#define DECLARE_WIN2(sji)		uint32 win2[2]; { const uint32_2 t = ((__global const uint32_2 *)wi)[sji]; win2[0] = t.s1; win2[1] = t.s0; }\n" \
"#define DECLARE_WIN1_2(sji)		uint32 win1[2]; { const uint32_2 t = ((__global const uint32_2 *)wi)[sji]; win1[0] = t.s1; win1[1] = t.s0; }\n" \
"#define DECLARE_WIN2_4(sji)		uint32 win2[4]; { const uint32_4 t = ((__global const uint32_4 *)wi)[sji]; win2[0] = t.s3; win2[1] = t.s2; win2[2] = t.s1; win2[3] = t.s0; }\n" \
"#endif\n" \
"#define DECLARE_WIN12(sj)		DECLARE_WIN1(sj); DECLARE_WIN2(sj);\n" \
"#define DECLARE_WIN12_24(sj)	DECLARE_WIN1_2(sj); DECLARE_WIN2_4(sj);\n" \
"\n" \
"// --- vector size (1, 2 or 4) ---\n" \
"\n" \
"#if VSIZE == 4\n" \
"#define VTYPE				uint32_4\n" \
"#define _loadg				_loadg4\n" \
"#define _loadl				_loadl4\n" \
"#define _storeg				_storeg4\n" \
"#define _storel				_storel4\n" \
"#define _forward4			_forward4x4\n" \
"#define _backward4			_backward4x4\n" \
"#define _forward4_0			_forward4x4_0\n" \
"#elif VSIZE == 2\n" \
"#define VTYPE				uint32_2\n" \
"#define _loadg				_loadg2\n" \
"#define _loadl				_loadl2\n" \
"#define _storeg				_storeg2\n" \
"#define _storel				_storel2\n" \
"#define _forward4			_forward4x2\n" \
"#define _backward4			_backward4x2\n" \
"#define _forward4_0			_forward4x2_0\n" \
"#else\n" \
"#define VTYPE				uint32\n" \
"#define _loadg				_loadg1\n" \
"#define _loadl				_loadl1\n" \
"#define _storeg				_storeg1\n" \
"#define _storel				_storel1\n" \
"#define _forward4			_forward4x1\n" \
"#define _backward4			_backward4x1\n" \
"#define _forward4_0			_forward4x1_0\n" \
"#endif\n" \
"\n" \
"// --- transform/inline global mem ---\n" \
"\n" \
"INLINE void forward4io(const uint32_2 pq, const sz_t m, __global VTYPE * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	VTYPE zl[4]; _loadg(4, zl, z, m);\n" \
"	_forward4(pq, zl, w1, w2);\n" \
"	_storeg(4, z, m, zl);\n" \
"}\n" \
"\n" \
"INLINE void backward4io(const uint32_2 pq, const sz_t m, __global VTYPE * restrict const z, __global const uint32 * restrict const wi, const sz_t sji)\n" \
"{\n" \
"	DECLARE_WIN12(sji);\n" \
"	VTYPE zl[4]; _loadg(4, zl, z, m);\n" \
"	_backward4(pq, zl, win1, win2);\n" \
"	_storeg(4, z, m, zl);\n" \
"}\n" \
"\n" \
"INLINE void forward4io_0(const uint32_2 pq, const uint32_4 f0, __global VTYPE * restrict const z)\n" \
"{\n" \
"	const sz_t m = N_SZ / 4 / VSIZE;\n" \
"	VTYPE zl[4]; _loadg(4, zl, z, m);\n" \
"	_forward4_0(pq, f0, zl);\n" \
"	_storeg(4, z, m, zl);\n" \
"}\n" \
"\n" \
"// --- v1\n" \
"\n" \
"INLINE void square2x2io(const uint32_2 pq, __global uint32 * restrict const z, const uint32 w)\n" \
"{\n" \
"	uint32 zl[4]; _loadg1(4, zl, z, 1);\n" \
"	_square2x2(pq, zl, w);\n" \
"	_storeg1(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void square4x1io(const uint32_2 pq, __global uint32 * restrict const z, const uint32 w, const uint32 win)\n" \
"{\n" \
"	uint32 zl[4]; _loadg1(4, zl, z, 1);\n" \
"	_square4(pq, zl, w, win);\n" \
"	_storeg1(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x1io(const uint32_2 pq, __global uint32 * restrict const z, const uint32 w)\n" \
"{\n" \
"	uint32 zl[4]; _loadg1(4, zl, z, 1);\n" \
"	_fwd4(pq, zl, w);\n" \
"	_storeg1(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul2x2io(const uint32_2 pq, __global uint32 * restrict const z, const __global uint32 * restrict const zp, const uint32 w)\n" \
"{\n" \
"	uint32 zpl[4]; _loadg1(4, zpl, zp, 1);\n" \
"	uint32 zl[4]; _loadg1(4, zl, z, 1);\n" \
"	_mul2x2(pq, zl, zpl, w);\n" \
"	_storeg1(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul4x1io(const uint32_2 pq, __global uint32 * restrict const z, const __global uint32 * restrict const zp, const uint32 w, const uint32 win)\n" \
"{\n" \
"	uint32 zpl[4]; _loadg1(4, zpl, zp, 1);\n" \
"	uint32 zl[4]; _loadg1(4, zl, z, 1);\n" \
"	_mul4(pq, zl, zpl, w, win);\n" \
"	_storeg1(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"// --- v2\n" \
"\n" \
"INLINE void square4x2io(const uint32_2 pq, __global uint32_2 * restrict const z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2(sj);\n" \
"	DECLARE_WIN2(sji);\n" \
"	uint32_2 zl[4]; _loadg2(4, zl, z, 1);\n" \
"	_square4x2(pq, zl, w2, win2);\n" \
"	_storeg2(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void square8x1io(const uint32_2 pq, __global uint32_2 * restrict const z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	DECLARE_WIN12(sji);\n" \
"	uint32_2 zl[4]; _loadg2(4, zl, z, 1);\n" \
"	_square8(pq, zl, w1, win1, w2, win2);\n" \
"	_storeg2(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x2io(const uint32_2 pq, __global uint32_2 * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2(sj);\n" \
"	uint32_2 zl[4]; _loadg2(4, zl, z, 1);\n" \
"	_fwd4x2(pq, zl, w2);\n" \
"	_storeg2(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd8x1io(const uint32_2 pq, __global uint32_2 * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	uint32_2 zl[4]; _loadg2(4, zl, z, 1);\n" \
"	_fwd8(pq, zl, w1, w2);\n" \
"	_storeg2(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul4x2io(const uint32_2 pq, __global uint32_2 * restrict const z, const __global uint32_2 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2(sj);\n" \
"	DECLARE_WIN2(sji);\n" \
"	uint32_2 zpl[4]; _loadg2(4, zpl, zp, 1);\n" \
"	uint32_2 zl[4]; _loadg2(4, zl, z, 1);\n" \
"	_mul4x2(pq, zl, zpl, w2, win2);\n" \
"	_storeg2(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul8x1io(const uint32_2 pq, __global uint32_2 * restrict const z, const __global uint32_2 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	DECLARE_WIN12(sji);\n" \
"	uint32_2 zpl[4]; _loadg2(4, zpl, zp, 1);\n" \
"	uint32_2 zl[4]; _loadg2(4, zl, z, 1);\n" \
"	_mul8(pq, zl, zpl, w1, win1, w2, win2);\n" \
"	_storeg2(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"// --- v4\n" \
"\n" \
"INLINE void square4x4io(const uint32_2 pq, __global uint32_4 * restrict const z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2_4(sj);\n" \
"	DECLARE_WIN2_4(sji);\n" \
"	uint32_4 zl[4]; _loadg4(4, zl, z, 1);\n" \
"	_square4x2v4(pq, &zl[0], &w2[0], &win2[0]);\n" \
"	_square4x2v4(pq, &zl[2], &w2[2], &win2[2]);\n" \
"	_storeg4(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void square8x2io(const uint32_2 pq, __global uint32_4 * restrict const z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12_24(sj);\n" \
"	DECLARE_WIN12_24(sji);\n" \
"	uint32_4 zl[4]; _loadg4(4, zl, z, 1);\n" \
"	_square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);\n" \
"	_square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);\n" \
"	_storeg4(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x4io(const uint32_2 pq, __global uint32_4 * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2_4(sj);\n" \
"	uint32_4 zl[4]; _loadg4(4, zl, z, 1);\n" \
"	_fwd4x2v4(pq, &zl[0], &w2[0]);\n" \
"	_fwd4x2v4(pq, &zl[2], &w2[2]);\n" \
"	_storeg4(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd8x2io(const uint32_2 pq, __global uint32_4 * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12_24(sj);\n" \
"	uint32_4 zl[4]; _loadg4(4, zl, z, 1);\n" \
"	_fwd8v4(pq, &zl[0], w1[0], &w2[0]);\n" \
"	_fwd8v4(pq, &zl[2], w1[1], &w2[2]);\n" \
"	_storeg4(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul4x4io(const uint32_2 pq, __global uint32_4 * restrict const z, const __global uint32_4 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2_4(sj);\n" \
"	DECLARE_WIN2_4(sji);\n" \
"	uint32_4 zpl[4]; _loadg4(4, zpl, zp, 1);\n" \
"	uint32_4 zl[4]; _loadg4(4, zl, z, 1);\n" \
"	_mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);\n" \
"	_mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);\n" \
"	_storeg4(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul8x2io(const uint32_2 pq, __global uint32_4 * restrict const z, const __global uint32_4 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12_24(sj);\n" \
"	DECLARE_WIN12_24(sji);\n" \
"	uint32_4 zpl[4]; _loadg4(4, zpl, zp, 1);\n" \
"	uint32_4 zl[4]; _loadg4(4, zl, z, 1);\n" \
"	_mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);\n" \
"	_mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);\n" \
"	_storeg4(4, z, 1, zl);\n" \
"}\n" \
"\n" \
"// --- v1, v2, v4\n" \
"\n" \
"INLINE void square4io(const uint32_2 pq, __global VTYPE * restrict const z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	square4x4io(pq, z, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	square4x2io(pq, z, w, wi, sj, sji);\n" \
"#else\n" \
"	square4x1io(pq, z, w[sj], wi[sji]);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void fwd4io(const uint32_2 pq, __global VTYPE * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	fwd4x4io(pq, z, w, sj);\n" \
"#elif VSIZE == 2\n" \
"	fwd4x2io(pq, z, w, sj);\n" \
"#else\n" \
"	fwd4x1io(pq, z, w[sj]);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void mul4io(const uint32_2 pq, __global VTYPE * restrict const z, const __global VTYPE * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	mul4x4io(pq, z, zp, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	mul4x2io(pq, z, zp, w, wi, sj, sji);\n" \
"#else\n" \
"	mul4x1io(pq, z, zp, w[sj], wi[sji]);\n" \
"#endif\n" \
"}\n" \
"\n" \
"// --- v2, v4\n" \
"\n" \
"INLINE void square8io(const uint32_2 pq, __global VTYPE * restrict const z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	square8x2io(pq, z, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	square8x1io(pq, z, w, wi, sj, sji);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void fwd8io(const uint32_2 pq, __global VTYPE * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	fwd8x2io(pq, z, w, sj);\n" \
"#elif VSIZE == 2\n" \
"	fwd8x1io(pq, z, w, sj);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void mul8io(const uint32_2 pq, __global VTYPE * restrict const z, const __global VTYPE * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	mul8x2io(pq, z, zp, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	mul8x1io(pq, z, zp, w, wi, sj, sji);\n" \
"#endif\n" \
"}\n" \
"\n" \
"// --- transform/inline local & global mem ---\n" \
"\n" \
"INLINE void forward_4(const uint32_2 pq, const sz_t m, __local VTYPE * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	VTYPE zl[4]; _loadl(4, zl, Z, m);\n" \
"	_forward4(pq, zl, w1, w2);\n" \
"	_storel(4, Z, m, zl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i(const uint32_2 pq, const sz_t ml, __local VTYPE * restrict const Z, const sz_t mg,\n" \
"	__global const VTYPE * restrict const z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	VTYPE zl[4]; _loadg(4, zl, z, mg);\n" \
"	_forward4(pq, zl, w1, w2);\n" \
"	_storel(4, Z, ml, zl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i_0(const uint32_2 pq, const uint32_4 f0, const sz_t ml, __local VTYPE * restrict const Z,\n" \
"	const sz_t mg, __global const VTYPE * restrict const z)\n" \
"{\n" \
"	VTYPE zl[4]; _loadg(4, zl, z, mg);\n" \
"	_forward4_0(pq, f0, zl);\n" \
"	_storel(4, Z, ml, zl);\n" \
"}\n" \
"\n" \
"INLINE void forward_4o(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z, const sz_t ml,\n" \
"	__local const VTYPE * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	VTYPE zl[4]; _loadl(4, zl, Z, ml);\n" \
"	_forward4(pq, zl, w1, w2);\n" \
"	_storeg(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4(const uint32_2 pq, const sz_t m, __local VTYPE * restrict const Z, __global const uint32 * restrict const wi, const sz_t sji)\n" \
"{\n" \
"	DECLARE_WIN12(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	VTYPE zl[4]; _loadl(4, zl, Z, m);\n" \
"	_backward4(pq, zl, win1, win2);\n" \
"	_storel(4, Z, m, zl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const uint32_2 pq, const sz_t ml, __local VTYPE * restrict const Z, const sz_t mg,\n" \
"	__global const VTYPE * restrict const z, __global const uint32 * restrict const wi, const sz_t sji)\n" \
"{\n" \
"	DECLARE_WIN12(sji);\n" \
"	VTYPE zl[4]; _loadg(4, zl, z, mg);\n" \
"	_backward4(pq, zl, win1, win2);\n" \
"	_storel(4, Z, ml, zl);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z, const sz_t ml,\n" \
"	__local const VTYPE * restrict const Z, __global const uint32 * restrict const wi, const sz_t sji)\n" \
"{\n" \
"	DECLARE_WIN12(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	VTYPE zl[4]; _loadl(4, zl, Z, ml);\n" \
"	_backward4(pq, zl, win1, win2);\n" \
"	_storeg(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"// --- v1\n" \
"\n" \
"INLINE void square_2x2(const uint32_2 pq, __local uint32 * restrict const Z, const uint32 w)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32 zl[4]; _loadl1(4, zl, Z, 1);\n" \
"	_square2x2(pq, zl, w);\n" \
"	_storel1(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void square_4x1(const uint32_2 pq, __local uint32 * restrict const Z, const uint32 w, const uint32 win)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32 zl[4]; _loadl1(4, zl, Z, 1);\n" \
"	_square4(pq, zl, w, win);\n" \
"	_storel1(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void write_4(const sz_t mg, __global VTYPE * restrict const z, __local const VTYPE * restrict const Z)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	VTYPE zl[4]; _loadl(4, zl, Z, 1);\n" \
"	_storeg(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x1_write(const uint32_2 pq, const sz_t mg, __global uint32 * restrict const z,\n" \
"	__local const uint32 * restrict const Z, const uint32 w)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32 zl[4]; _loadl1(4, zl, Z, 1);\n" \
"	_fwd4(pq, zl, w);\n" \
"	_storeg1(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul_2x2(const uint32_2 pq, __local uint32 * restrict const Z, const sz_t mg,\n" \
"	__global const uint32 * restrict const zp, const uint32 w)\n" \
"{\n" \
"	uint32 zpl[4]; _loadg1(4, zpl, zp, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32 zl[4]; _loadl1(4, zl, Z, 1);\n" \
"	_mul2x2(pq, zl, zpl, w);\n" \
"	_storel1(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul_4x1(const uint32_2 pq, __local uint32 * restrict const Z, const sz_t mg,\n" \
"	__global const uint32 * restrict const zp, const uint32 w, const uint32 win)\n" \
"{\n" \
"	uint32 zpl[4]; _loadg1(4, zpl, zp, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32 zl[4]; _loadl1(4, zl, Z, 1);\n" \
"	_mul4(pq, zl, zpl, w, win);\n" \
"	_storel1(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"// --- v2\n" \
"\n" \
"INLINE void square_4x2(const uint32_2 pq, __local uint32_2 * restrict const Z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2(sj);\n" \
"	DECLARE_WIN2(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);\n" \
"	_square4x2(pq, zl, w2, win2);\n" \
"	_storel2(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void square_8x1(const uint32_2 pq, __local uint32_2 * restrict const Z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	DECLARE_WIN12(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);\n" \
"	_square8(pq, zl, w1, win1, w2, win2);\n" \
"	_storel2(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x2_write(const uint32_2 pq, const sz_t mg, __global uint32_2 * restrict const z,\n" \
"	__local const uint32_2 * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2(sj);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);\n" \
"	_fwd4x2(pq, zl, w2);\n" \
"	_storeg2(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd8x1_write(const uint32_2 pq, const sz_t mg, __global uint32_2 * restrict const z,\n" \
"	__local const uint32_2 * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);\n" \
"	_fwd8(pq, zl, w1, w2);\n" \
"	_storeg2(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul_4x2(const uint32_2 pq, __local uint32_2 * restrict const Z, const sz_t mg, const __global uint32_2 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2(sj);\n" \
"	DECLARE_WIN2(sji);\n" \
"	uint32_2 zpl[4]; _loadg2(4, zpl, zp, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);\n" \
"	_mul4x2(pq, zl, zpl, w2, win2);\n" \
"	_storel2(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul_8x1(const uint32_2 pq, __local uint32_2 * restrict const Z, const sz_t mg, const __global uint32_2 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12(sj);\n" \
"	DECLARE_WIN12(sji);\n" \
"	uint32_2 zpl[4]; _loadg2(4, zpl, zp, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_2 zl[4]; _loadl2(4, zl, Z, 1);\n" \
"	_mul8(pq, zl, zpl, w1, win1, w2, win2);\n" \
"	_storel2(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"// --- v4\n" \
"\n" \
"INLINE void square_4x4(const uint32_2 pq, __local uint32_4 * restrict const Z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2_4(sj);\n" \
"	DECLARE_WIN2_4(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);\n" \
"	_square4x2v4(pq, &zl[0], &w2[0], &win2[0]);\n" \
"	_square4x2v4(pq, &zl[2], &w2[2], &win2[2]);\n" \
"	_storel4(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void square_8x2(const uint32_2 pq, __local uint32_4 * restrict const Z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12_24(sj);\n" \
"	DECLARE_WIN12_24(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);\n" \
"	_square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);\n" \
"	_square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);\n" \
"	_storel4(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd4x4_write(const uint32_2 pq, const sz_t mg, __global uint32_4 * restrict const z,\n" \
"	__local const uint32_4 * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W2_4(sj);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);\n" \
"	_fwd4x2v4(pq, &zl[0], &w2[0]);\n" \
"	_fwd4x2v4(pq, &zl[2], &w2[2]);\n" \
"	_storeg4(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void fwd8x2_write(const uint32_2 pq, const sz_t mg, __global uint32_4 * restrict const z,\n" \
"	__local const uint32_4 * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"	DECLARE_W12_24(sj);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);\n" \
"	_fwd8v4(pq, &zl[0], w1[0], &w2[0]);\n" \
"	_fwd8v4(pq, &zl[2], w1[1], &w2[2]);\n" \
"	_storeg4(4, z, mg, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul_4x4(const uint32_2 pq, __local uint32_4 * restrict const Z, const sz_t mg, const __global uint32_4 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W2_4(sj);\n" \
"	DECLARE_WIN2_4(sji);\n" \
"	uint32_4 zpl[4]; _loadg4(4, zpl, zp, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);\n" \
"	_mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);\n" \
"	_mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);\n" \
"	_storel4(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"INLINE void mul_8x2(const uint32_2 pq, __local uint32_4 * restrict const Z, const sz_t mg, const __global uint32_4 * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"	DECLARE_W12_24(sj);\n" \
"	DECLARE_WIN12_24(sji);\n" \
"	uint32_4 zpl[4]; _loadg4(4, zpl, zp, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	uint32_4 zl[4]; _loadl4(4, zl, Z, 1);\n" \
"	_mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);\n" \
"	_mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);\n" \
"	_storel4(4, Z, 1, zl);\n" \
"}\n" \
"\n" \
"// --- v1, v2, v4 -- no barrier\n" \
"\n" \
"INLINE void square_4(const uint32_2 pq, __local VTYPE * restrict const Z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	square_4x4(pq, Z, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	square_4x2(pq, Z, w, wi, sj, sji);\n" \
"#else\n" \
"	square_4x1(pq, Z, w[sj], wi[sji]);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void fwd4_write(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z,\n" \
"	__local const VTYPE * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	fwd4x4_write(pq, mg, z, Z, w, sj);\n" \
"#elif VSIZE == 2\n" \
"	fwd4x2_write(pq, mg, z, Z, w, sj);\n" \
"#else\n" \
"	fwd4x1_write(pq, mg, z, Z, w[sj]);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void mul_4(const uint32_2 pq, __local VTYPE * restrict const Z, const sz_t mg, const __global VTYPE * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	mul_4x4(pq, Z, mg, zp, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	mul_4x2(pq, Z, mg, zp, w, wi, sj, sji);\n" \
"#else\n" \
"	mul_4x1(pq, Z, mg, zp, w[sj], wi[sji]);\n" \
"#endif\n" \
"}\n" \
"\n" \
"// --- v2, v4 -- no barrier\n" \
"\n" \
"INLINE void square_8(const uint32_2 pq, __local VTYPE * restrict const Z,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	square_8x2(pq, Z, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	square_8x1(pq, Z, w, wi, sj, sji);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void fwd8_write(const uint32_2 pq, const sz_t mg, __global VTYPE * restrict const z,\n" \
"	__local const VTYPE * restrict const Z, __global const uint32 * restrict const w, const sz_t sj)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	fwd8x2_write(pq, mg, z, Z, w, sj);\n" \
"#elif VSIZE == 2\n" \
"	fwd8x1_write(pq, mg, z, Z, w, sj);\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE void mul_8(const uint32_2 pq, __local VTYPE * restrict const Z, const sz_t mg, const __global VTYPE * restrict const zp,\n" \
"	__global const uint32 * restrict const w, __global const uint32 * restrict const wi, const sz_t sj, const sz_t sji)\n" \
"{\n" \
"#if VSIZE == 4\n" \
"	mul_8x2(pq, Z, mg, zp, w, wi, sj, sji);\n" \
"#elif VSIZE == 2\n" \
"	mul_8x1(pq, Z, mg, zp, w, wi, sj, sji);\n" \
"#endif\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define DECLARE_VAR_REGv1() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 2), mid = gid & ~((N_SZ / 4) - 1), id = gid %  (N_SZ / 4); \\\n" \
"	const uint32_2 pq = g_pq[lid]; \\\n" \
"	__global uint32 * restrict const z = &zg[4 * mid]; \\\n" \
"	__global const uint32 * restrict const w = &wg[lid * W_SHFT];\n" \
"\n" \
"#define DECLARE_VARP_REGv1() \\\n" \
"	__global const uint32 * restrict const zp = &zpg[4 * mid];\n" \
"\n" \
"#define DECLARE_VAR_REGv2() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 3), mid = gid & ~((N_SZ / 8) - 1), id = gid %  (N_SZ / 8); \\\n" \
"	const uint32_2 pq = g_pq[lid]; \\\n" \
"	__global uint32_2 * restrict const z = &zg[4 * mid]; \\\n" \
"	__global const uint32 * restrict const w = &wg[lid * W_SHFT];\n" \
"\n" \
"#define DECLARE_VARP_REGv2() \\\n" \
"	__global const uint32_2 * restrict const zp = &zpg[4 * mid];\n" \
"\n" \
"#define DECLARE_VAR_REGv4() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 4), mid = gid & ~((N_SZ / 16) - 1), id = gid %  (N_SZ / 16); \\\n" \
"	const uint32_2 pq = g_pq[lid]; \\\n" \
"	__global uint32_4 * restrict const z = &zg[4 * mid]; \\\n" \
"	__global const uint32 * restrict const w = &wg[lid * W_SHFT];\n" \
"\n" \
"#define DECLARE_VARP_REGv4() \\\n" \
"	__global const uint32_4 * restrict const zp = &zpg[4 * mid];\n" \
"\n" \
"#if VSIZE == 4\n" \
"#define DECLARE_VAR_REG		DECLARE_VAR_REGv4\n" \
"#define DECLARE_VARP_REG	DECLARE_VARP_REGv4\n" \
"#elif VSIZE == 2\n" \
"#define DECLARE_VAR_REG		DECLARE_VAR_REGv2\n" \
"#define DECLARE_VARP_REG	DECLARE_VARP_REGv2\n" \
"#else\n" \
"#define DECLARE_VAR_REG		DECLARE_VAR_REGv1\n" \
"#define DECLARE_VARP_REG	DECLARE_VARP_REGv1\n" \
"#endif\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"__kernel\n" \
"void forward4(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;\n" \
"	forward4io(pq, m, &z[k], w, s + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward4(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id; DECLARE_IVAR(s, j);\n" \
"	backward4io(pq, m, &z[k], wi, sji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward4_0(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t k = id;\n" \
"	forward4io_0(pq, g_f0[lid], &z[k]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2x2(__global uint32 * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REGv1();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	square2x2io(pq, &z[k], w[N_SZ / 4 + j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);\n" \
"	square4io(pq, &z[k], w, wi, sj, sji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd4p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j;\n" \
"	fwd4io(pq, &z[k], w, sj);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);\n" \
"	mul4io(pq, &z[k], &zp[k], w, wi, sj, sji);\n" \
"}\n" \
"\n" \
"// --- v1\n" \
"\n" \
"__kernel\n" \
"void mul2x2(__global uint32 * restrict const zg, __global const uint32 * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REGv1();\n" \
"	DECLARE_VARP_REGv1();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	mul2x2io(pq, &z[k], &zp[k], w[N_SZ / 4 + j]);\n" \
"}\n" \
"\n" \
"// --- v2, v4\n" \
"\n" \
"__kernel\n" \
"void square8(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);\n" \
"	square8io(pq, &z[k], w, wi, sj, sji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd8p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j;\n" \
"	fwd8io(pq, &z[k], w, sj);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul8(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);\n" \
"	mul8io(pq, &z[k], &zp[k], w, wi, sj, sji);\n" \
"}\n" \
"\n" \
"// --- transform ---\n" \
"\n" \
"#if !defined(SHORT_VER)\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	/* threadIdx < B_N */ \\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \\\n" \
"	const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = group_id * CHUNK_N + chunk_idx; \\\n" \
"	__local VTYPE * const Zi = &Z[chunk_idx]; \\\n" \
"	\\\n" \
"	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \\\n" \
"	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \\\n" \
"	\\\n" \
"	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \\\n" \
"	\\\n" \
"	const sz_t sj = s + idx_m; DECLARE_IVAR(s, idx_m);\n" \
"\n" \
"#define DECLARE_VAR_FORWARD() \\\n" \
"	__global VTYPE * restrict const zi = &z[ki]; \\\n" \
"	__global VTYPE * restrict const zo = &z[ko];\n" \
"\n" \
"#define DECLARE_VAR_BACKWARD() \\\n" \
"	__global VTYPE * restrict const zi = &z[ko]; \\\n" \
"	__global VTYPE * restrict const zo = &z[ki];\n" \
"\n" \
"#define FORWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	forward_4i(pq, B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);\n" \
"\n" \
"#define FORWARD_I_0(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	forward_4i_0(pq, g_f0[lid], B_N * CHUNK_N, &Z[i], B_N << lm, zi);\n" \
"\n" \
"#define BACKWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_BACKWARD(); \\\n" \
"	\\\n" \
"	backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sji / 1);\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_64	(64 / 4)\n" \
"\n" \
"#if MAX_WG_SZ >= B_64 * CHUNK64\n" \
"#define ATTR_64() \\\n" \
"	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"#else\n" \
"#define ATTR_64()\n" \
"#endif\n" \
"\n" \
"#define FORWARD_64() \\\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \\\n" \
"	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4); \\\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void forward64(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local VTYPE Z[4 * B_64 * CHUNK64];\n" \
"	FORWARD_I(B_64, CHUNK64);\n" \
"	FORWARD_64();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void forward64_0(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	const int lm = LN_SZ - LVSIZE - 6; const unsigned int s = 64 / 4;\n" \
"	__local VTYPE Z[4 * B_64 * CHUNK64];\n" \
"	FORWARD_I_0(B_64, CHUNK64);\n" \
"	FORWARD_64();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void backward64(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local VTYPE Z[4 * B_64 * CHUNK64];\n" \
"	BACKWARD_I(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], wi, sji / 4);\n" \
"	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], wi, sji / B_64);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_256	(256 / 4)\n" \
"\n" \
"#if MAX_WG_SZ >= B_256 * CHUNK256\n" \
"#define ATTR_256() \\\n" \
"	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"#else\n" \
"#define ATTR_256()\n" \
"#endif\n" \
"\n" \
"#define FORWARD_256() \\\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \\\n" \
"	forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16); \\\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \\\n" \
"	forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4); \\\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void forward256(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local VTYPE Z[4 * B_256 * CHUNK256];\n" \
"	FORWARD_I(B_256, CHUNK256);\n" \
"	FORWARD_256();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void forward256_0(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	const int lm = LN_SZ - LVSIZE - 8; const unsigned int s = 256 / 4;\n" \
"	__local VTYPE Z[4 * B_256 * CHUNK256];\n" \
"	FORWARD_I_0(B_256, CHUNK256);\n" \
"	FORWARD_256();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void backward256(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local VTYPE Z[4 * B_256 * CHUNK256];\n" \
"	BACKWARD_I(B_256, CHUNK256);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sji / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sji / 16);\n" \
"	backward_4o(pq, B_256 << lm, zo, B_256 * CHUNK256, &Z[i], wi, sji / B_256);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_1024	(1024 / 4)\n" \
"\n" \
"#if MAX_WG_SZ >= B_1024 * CHUNK1024\n" \
"#define ATTR_1024() \\\n" \
"	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"#else\n" \
"#define ATTR_1024()\n" \
"#endif\n" \
"\n" \
"#define FORWARD_1024() \\\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 ); \\\n" \
"	forward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64); \\\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \\\n" \
"	forward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16); \\\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \\\n" \
"	forward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4); \\\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void forward1024(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local VTYPE Z[4 * B_1024 * CHUNK1024];\n" \
"	FORWARD_I(B_1024, CHUNK1024);\n" \
"	FORWARD_1024();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void forward1024_0(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	const int lm = LN_SZ - LVSIZE - 10; const unsigned int s = 1024 / 4;\n" \
"	__local VTYPE Z[4 * B_1024 * CHUNK1024];\n" \
"	FORWARD_I_0(B_1024, CHUNK1024);\n" \
"	FORWARD_1024();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void backward1024(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local VTYPE Z[4 * B_1024 * CHUNK1024];\n" \
"	BACKWARD_I(B_1024, CHUNK1024);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], wi, sji / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], wi, sji / 16);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);\n" \
"	backward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], wi, sji / 64);\n" \
"	backward_4o(pq, B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], wi, sji / B_1024);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define L32S	(32 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local VTYPE Z[L32S * BLK32]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L32S / 4 * BLK32), group_id = id / (L32S / 4 * BLK32); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i32 = (local_id & ~(L32S / 4 - 1)) * 4, i8 = local_id % (L32S / 4); \\\n" \
"	const sz_t k32 = group_id * L32S * BLK32 + i32 + i8; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k32]; \\\n" \
"	__local VTYPE * const Z32 = &Z[i32]; \\\n" \
"	__local VTYPE * const Zi8 = &Z32[i8]; \\\n" \
"	const sz_t i2 = ((4 * i8) & ~(4 * 2 - 1)) + (i8 % 2); \\\n" \
"	__local VTYPE * const Zi2 = &Z32[i2]; \\\n" \
"	__local VTYPE * const Z4 = &Z32[4 * i8];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L32S / 4 * BLK32\n" \
"	__attribute__((reqd_work_group_size(L32S / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void square32(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_2x2(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	square_8(pq, Z4, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4o(pq, L32S / 4, zk, L32S / 4, Zi8, wi, sji / (L32S / 4));\n" \
"}\n" \
"\n" \
"#define L64S	(64 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local VTYPE Z[L64S * BLK64]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L64S / 4 * BLK64), group_id = id / (L64S / 4 * BLK64); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i64 = (local_id & ~(L64S / 4 - 1)) * 4, i16 = local_id % (L64S / 4); \\\n" \
"	const sz_t k64 = group_id * L64S * BLK64 + i64 + i16; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k64]; \\\n" \
"	__local VTYPE * const Z64 = &Z[i64]; \\\n" \
"	__local VTYPE * const Zi16 = &Z64[i16]; \\\n" \
"	const sz_t i4 = ((4 * i16) & ~(4 * (L64S / 16) - 1)) + (i16 % (L64S / 16)); \\\n" \
"	__local VTYPE * const Zi4 = &Z64[i4]; \\\n" \
"	__local VTYPE * const Z4 = &Z64[4 * i16];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L64S / 4 * BLK64\n" \
"	__attribute__((reqd_work_group_size(L64S / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void square64(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));\n" \
"	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));\n" \
"	square_4(pq, Z4, w, wi, sj, sji);\n" \
"	backward_4(pq, L64S / 16, Zi4, wi, sji / (L64S / 16));\n" \
"	backward_4o(pq, L64S / 4, zk, L64S / 4, Zi16, wi, sji / (L64S / 4));\n" \
"}\n" \
"\n" \
"#define L128S	(128 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_128() \\\n" \
"	__local VTYPE Z[L128S * BLK128]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L128S / 4 * BLK128), group_id = id / (L128S / 4 * BLK128); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i128 = (local_id & ~(L128S / 4 - 1)) * 4, i32 = local_id % (L128S / 4); \\\n" \
"	const sz_t k128 = group_id * L128S * BLK128 + i128 + i32; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k128]; \\\n" \
"	__local VTYPE * const Z128 = &Z[i128]; \\\n" \
"	__local VTYPE * const Zi32 = &Z128[i32]; \\\n" \
"	const sz_t i8 = ((4 * i32) & ~(4 * (L128S / 16) - 1)) + (i32 % (L128S / 16)); \\\n" \
"	__local VTYPE * const Zi8 = &Z128[i8]; \\\n" \
"	const sz_t i2 = ((4 * i32) & ~(4 * 2 - 1)) + (i32 % 2); \\\n" \
"	__local VTYPE * const Zi2 = &Z128[i2]; \\\n" \
"	__local VTYPE * const Z4 = &Z128[4 * i32];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L128S / 4 * BLK128\n" \
"	__attribute__((reqd_work_group_size(L128S / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void square128(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));\n" \
"	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_2x2(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	square_8(pq, Z4, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4(pq, L128S / 16, Zi8, wi, sji / (L128S / 16));\n" \
"	backward_4o(pq, L128S / 4, zk, L128S / 4, Zi32, wi, sji / (L128S / 4));\n" \
"}\n" \
"\n" \
"#define L256S	(256 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_256() \\\n" \
"	__local VTYPE Z[L256S * BLK256]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L256S / 4 * BLK256), group_id = id / (L256S / 4 * BLK256); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i256 = (local_id & ~(L256S / 4 - 1)) * 4, i64 = local_id % (L256S / 4); \\\n" \
"	const sz_t k256 = group_id * L256S * BLK256 + i256 + i64; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k256]; \\\n" \
"	__local VTYPE * const Z256 = &Z[i256]; \\\n" \
"	__local VTYPE * const Zi64 = &Z256[i64]; \\\n" \
"	const sz_t i16 = ((4 * i64) & ~(4 * (L256S / 16) - 1)) + (i64 % (L256S / 16)); \\\n" \
"	__local VTYPE * const Zi16 = &Z256[i16]; \\\n" \
"	const sz_t i4 = ((4 * i64) & ~(4 * (L256S / 64) - 1)) + (i64 % (L256S / 64)); \\\n" \
"	__local VTYPE * const Zi4 = &Z256[i4]; \\\n" \
"	__local VTYPE * const Z4 = &Z256[4 * i64];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L256S / 4 * BLK256\n" \
"	__attribute__((reqd_work_group_size(L256S / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void square256(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));\n" \
"	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));\n" \
"	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));\n" \
"	square_4(pq, Z4, w, wi, sj, sji);\n" \
"	backward_4(pq, L256S / 64, Zi4, wi, sji / (L256S / 64));\n" \
"	backward_4(pq, L256S / 16, Zi16, wi, sji / (L256S / 16));\n" \
"	backward_4o(pq, L256S / 4, zk, L256S / 4, Zi64, wi, sji / (L256S / 4));\n" \
"}\n" \
"\n" \
"#define L512S	(512 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_512() \\\n" \
"	__local VTYPE Z[L512S * BLK512]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L512S / 4 * BLK512), group_id = id / (L512S / 4 * BLK512); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i512 = (local_id & ~(L512S / 4 - 1)) * 4, i128 = local_id % (L512S / 4); \\\n" \
"	const sz_t k512 = group_id * L512S * BLK512 + i512 + i128; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k512]; \\\n" \
"	__local VTYPE * const Z512 = &Z[i512]; \\\n" \
"	__local VTYPE * const Zi128 = &Z512[i128]; \\\n" \
"	const sz_t i32 = ((4 * i128) & ~(4 * (L512S / 16) - 1)) + (i128 % (L512S / 16)); \\\n" \
"	__local VTYPE * const Zi32 = &Z512[i32]; \\\n" \
"	const sz_t i8 = ((4 * i128) & ~(4 * (L512S / 64) - 1)) + (i128 % (L512S / 64)); \\\n" \
"	__local VTYPE * const Zi8 = &Z512[i8]; \\\n" \
"	const sz_t i2 = ((4 * i128) & ~(4 * 2 - 1)) + (i128 % 2); \\\n" \
"	__local VTYPE * const Zi2 = &Z512[i2]; \\\n" \
"	__local VTYPE * const Z4 = &Z512[4 * i128];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L512S / 4 * BLK512\n" \
"	__attribute__((reqd_work_group_size(L512S / 4 * BLK512, 1, 1)))\n" \
"#endif\n" \
"void square512(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));\n" \
"	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));\n" \
"	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_2x2(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	square_8(pq, Z4, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4(pq, L512S / 64, Zi8, wi, sji / (L512S / 64));\n" \
"	backward_4(pq, L512S / 16, Zi32, wi, sji / (L512S / 16));\n" \
"	backward_4o(pq, L512S / 4, zk, L512S / 4, Zi128, wi, sji / (L512S / 4));\n" \
"}\n" \
"\n" \
"#define L1024S	(1024 / VSIZE)\n" \
"\n" \
"// if BLK1024 != 1 then const sz_t i1024 = (local_id & ~(L1024S / 4 - 1)) * 4, i256 = local_id % (L1024S / 4);\n" \
"// if BLK1024 = 1 then const sz_t i1024 = 0, i256 = local_id;\n" \
"#define DECLARE_VAR_1024() \\\n" \
"	__local VTYPE Z[L1024S * BLK1024]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L1024S / 4 * BLK1024), group_id = id / (L1024S / 4 * BLK1024); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i1024 = 0, i256 = local_id; \\\n" \
"	const sz_t k1024 = group_id * L1024S * BLK1024 + i1024 + i256; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k1024]; \\\n" \
"	__local VTYPE * const Z1024 = &Z[i1024]; \\\n" \
"	__local VTYPE * const Zi256 = &Z1024[i256]; \\\n" \
"	const sz_t i64 = ((4 * i256) & ~(4 * (L1024S / 16) - 1)) + (i256 % (L1024S / 16)); \\\n" \
"	__local VTYPE * const Zi64 = &Z1024[i64]; \\\n" \
"	const sz_t i16 = ((4 * i256) & ~(4 * (L1024S / 64) - 1)) + (i256 % (L1024S / 64)); \\\n" \
"	__local VTYPE * const Zi16 = &Z1024[i16]; \\\n" \
"	const sz_t i4 = ((4 * i256) & ~(4 * (L1024S / 256) - 1)) + (i256 % (L1024S / 256)); \\\n" \
"	__local VTYPE * const Zi4 = &Z1024[i4]; \\\n" \
"	__local VTYPE * const Z4 = &Z1024[4 * i256];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L1024S / 4 * BLK1024\n" \
"	__attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))\n" \
"#endif\n" \
"void square1024(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));\n" \
"	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));\n" \
"	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));\n" \
"	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));\n" \
"	square_4(pq, Z4, w, wi, sj, sji);\n" \
"	backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));\n" \
"	backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));\n" \
"	backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));\n" \
"	backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));\n" \
"}\n" \
"\n" \
"#define L2048S	(2048 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_2048() \\\n" \
"	__local VTYPE Z[L2048S]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L2048S / 4), group_id = id / (L2048S / 4); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i512 = local_id, k2048 = group_id * L2048S + i512; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k2048]; \\\n" \
"	__local VTYPE * const Zi512 = &Z[i512]; \\\n" \
"	const sz_t i128 = ((4 * i512) & ~(4 * (L2048S / 16) - 1)) + (i512 % (L2048S / 16)); \\\n" \
"	__local VTYPE * const Zi128 = &Z[i128]; \\\n" \
"	const sz_t i32 = ((4 * i512) & ~(4 * (L2048S / 64) - 1)) + (i512 % (L2048S / 64)); \\\n" \
"	__local VTYPE * const Zi32 = &Z[i32]; \\\n" \
"	const sz_t i8 = ((4 * i512) & ~(4 * (L2048S / 256) - 1)) + (i512 % (L2048S / 256)); \\\n" \
"	__local VTYPE * const Zi8 = &Z[i8]; \\\n" \
"	const sz_t i2 = ((4 * i512) & ~(4 * 2 - 1)) + (i512 % 2); \\\n" \
"	__local VTYPE * const Zi2 = &Z[i2]; \\\n" \
"	__local VTYPE * const Z4 = &Z[4 * i512];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L2048S / 4\n" \
"	__attribute__((reqd_work_group_size(L2048S / 4, 1, 1)))\n" \
"#endif\n" \
"void square2048(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));\n" \
"	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));\n" \
"	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));\n" \
"	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_2x2(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	square_8(pq, Z4, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));\n" \
"	backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));\n" \
"	backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));\n" \
"	backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));\n" \
"}\n" \
"\n" \
"#define L4096S	(4096 / VSIZE)\n" \
"\n" \
"#define DECLARE_VAR_4096() \\\n" \
"	__local VTYPE Z[L4096S]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (L4096S / 4), group_id = id / (L4096S / 4); \\\n" \
"	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \\\n" \
"	\\\n" \
"	const sz_t i1024 = local_id, k4096 = group_id * L4096S + i1024; \\\n" \
"	\\\n" \
"	__global VTYPE * restrict const zk = &z[k4096]; \\\n" \
"	__local VTYPE * const Zi1024 = &Z[i1024]; \\\n" \
"	const sz_t i256 = ((4 * i1024) & ~(4 * (L4096S / 16) - 1)) + (i1024 % (L4096S / 16)); \\\n" \
"	__local VTYPE * const Zi256 = &Z[i256]; \\\n" \
"	const sz_t i64 = ((4 * i1024) & ~(4 * (L4096S / 64) - 1)) + (i1024 % (L4096S / 64)); \\\n" \
"	__local VTYPE * const Zi64 = &Z[i64]; \\\n" \
"	const sz_t i16 = ((4 * i1024) & ~(4 * (L4096S / 256) - 1)) + (i1024 % (L4096S / 256)); \\\n" \
"	__local VTYPE * const Zi16 = &Z[i16]; \\\n" \
"	const sz_t i4 = ((4 * i1024) & ~(4 * (L4096S / 1024) - 1)) + (i1024 % (L4096S / 1024)); \\\n" \
"	__local VTYPE * const Zi4 = &Z[i4]; \\\n" \
"	__local VTYPE * const Z4 = &Z[4 * i1024];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L4096S / 4\n" \
"	__attribute__((reqd_work_group_size(L4096S / 4, 1, 1)))\n" \
"#endif\n" \
"void square4096(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_4096();\n" \
"\n" \
"	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));\n" \
"	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));\n" \
"	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));\n" \
"	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));\n" \
"	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));\n" \
"	square_4(pq, Z4, w, wi, sj, sji);\n" \
"	backward_4(pq, L4096S / 1024, Zi4, wi, sji / (L4096S / 1024));\n" \
"	backward_4(pq, L4096S / 256, Zi16, wi, sji / (L4096S / 256));\n" \
"	backward_4(pq, L4096S / 64, Zi64, wi, sji / (L4096S / 64));\n" \
"	backward_4(pq, L4096S / 16, Zi256, wi, sji / (L4096S / 16));\n" \
"	backward_4o(pq, L4096S / 4, zk, L4096S / 4, Zi1024, wi, sji / (L4096S / 4));\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L32S / 4 * BLK32\n" \
"	__attribute__((reqd_work_group_size(L32S / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void fwd32p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(8, zk, Z4);\n" \
"#else\n" \
"	fwd8_write(pq, L32S / 4, zk, Z4, w, sj);\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L64S / 4 * BLK64\n" \
"	__attribute__((reqd_work_group_size(L64S / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void fwd64p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));\n" \
"	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));\n" \
"	fwd4_write(pq, L64S / 4, zk, Z4, w, sj);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L128S / 4 * BLK128\n" \
"	__attribute__((reqd_work_group_size(L128S / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void fwd128p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));\n" \
"	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(32, zk, Z4);\n" \
"#else\n" \
"	fwd8_write(pq, L128S / 4, zk, Z4, w, sj);\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L256S / 4 * BLK256\n" \
"	__attribute__((reqd_work_group_size(L256S / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void fwd256p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));\n" \
"	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));\n" \
"	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));\n" \
"	fwd4_write(pq, L256S / 4, zk, Z4, w, sj);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L512S / 4 * BLK512\n" \
"	__attribute__((reqd_work_group_size(L512S / 4 * BLK512, 1, 1)))\n" \
"#endif\n" \
"void fwd512p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));\n" \
"	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));\n" \
"	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(128, zk, Z4);\n" \
"#else\n" \
"	fwd8_write(pq, L512S / 4, zk, Z4, w, sj);\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L1024S / 4 * BLK1024\n" \
"	__attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))\n" \
"#endif\n" \
"void fwd1024p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));\n" \
"	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));\n" \
"	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));\n" \
"	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));\n" \
"	fwd4_write(pq, L1024S / 4, zk, Z4, w, sj);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L2048S / 4\n" \
"	__attribute__((reqd_work_group_size(L2048S / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd2048p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));\n" \
"	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));\n" \
"	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));\n" \
"	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(512, zk, Z4);\n" \
"#else\n" \
"	fwd8_write(pq, L2048S / 4, zk, Z4, w, sj);\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L4096S / 4\n" \
"	__attribute__((reqd_work_group_size(L4096S / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd4096p(__global VTYPE * restrict const zg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_4096();\n" \
"\n" \
"	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));\n" \
"	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));\n" \
"	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));\n" \
"	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));\n" \
"	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));\n" \
"	fwd4_write(pq, L4096S / 4, zk, Z4, w, sj);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L32S / 4 * BLK32\n" \
"	__attribute__((reqd_work_group_size(L32S / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void mul32(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k32];\n" \
"\n" \
"	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_2x2(pq, Z4, 8, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	mul_8(pq, Z4, L32S / 4, zpk, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4o(pq, L32S / 4, zk, L32S / 4, Zi8, wi, sji / (L32S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L64S / 4 * BLK64\n" \
"	__attribute__((reqd_work_group_size(L64S / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void mul64(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k64];\n" \
"\n" \
"	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));\n" \
"	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));\n" \
"	mul_4(pq, Z4, L64S / 4, zpk, w, wi, sj, sji);\n" \
"	backward_4(pq, L64S / 16, Zi4, wi, sji / (L64S / 16));\n" \
"	backward_4o(pq, L64S / 4, zk, L64S / 4, Zi16, wi, sji / (L64S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L128S / 4 * BLK128\n" \
"	__attribute__((reqd_work_group_size(L128S / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void mul128(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k128];\n" \
"\n" \
"	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));\n" \
"	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_2x2(pq, Z4, 32, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	mul_8(pq, Z4, L128S / 4, zpk, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4(pq, L128S / 16, Zi8, wi, sji / (L128S / 16));\n" \
"	backward_4o(pq, L128S / 4, zk, L128S / 4, Zi32, wi, sji / (L128S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L256S / 4 * BLK256\n" \
"	__attribute__((reqd_work_group_size(L256S / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void mul256(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k256];\n" \
"\n" \
"	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));\n" \
"	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));\n" \
"	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));\n" \
"	mul_4(pq, Z4, L256S / 4, zpk, w, wi, sj, sji);\n" \
"	backward_4(pq, L256S / 64, Zi4, wi, sji / (L256S / 64));\n" \
"	backward_4(pq, L256S / 16, Zi16, wi, sji / (L256S / 16));\n" \
"	backward_4o(pq, L256S / 4, zk, L256S / 4, Zi64, wi, sji / (L256S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L512S / 4 * BLK512\n" \
"	__attribute__((reqd_work_group_size(L512S / 4 * BLK512, 1, 1)))\n" \
"#endif\n" \
"void mul512(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k512];\n" \
"\n" \
"	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));\n" \
"	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));\n" \
"	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_2x2(pq, Z4, 128, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	mul_8(pq, Z4, L512S / 4, zpk, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4(pq, L512S / 64, Zi8, wi, sji / (L512S / 64));\n" \
"	backward_4(pq, L512S / 16, Zi32, wi, sji / (L512S / 16));\n" \
"	backward_4o(pq, L512S / 4, zk, L512S / 4, Zi128, wi, sji / (L512S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L1024S / 4 * BLK1024\n" \
"	__attribute__((reqd_work_group_size(L1024S / 4 * BLK1024, 1, 1)))\n" \
"#endif\n" \
"void mul1024(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k1024];\n" \
"\n" \
"	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));\n" \
"	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));\n" \
"	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));\n" \
"	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));\n" \
"	mul_4(pq, Z4, L1024S / 4, zpk, w, wi, sj, sji);\n" \
"	backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));\n" \
"	backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));\n" \
"	backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));\n" \
"	backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L2048S / 4\n" \
"	__attribute__((reqd_work_group_size(L2048S / 4, 1, 1)))\n" \
"#endif\n" \
"void mul2048(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k2048];\n" \
"\n" \
"	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));\n" \
"	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));\n" \
"	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));\n" \
"	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));\n" \
"#if VSIZE == 1\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_2x2(pq, Z4, 512, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"#else\n" \
"	mul_8(pq, Z4, L2048S / 4, zpk, w, wi, sj, sji);\n" \
"#endif\n" \
"	backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));\n" \
"	backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));\n" \
"	backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));\n" \
"	backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= L4096S / 4\n" \
"	__attribute__((reqd_work_group_size(L4096S / 4, 1, 1)))\n" \
"#endif\n" \
"void mul4096(__global VTYPE * restrict const zg, __global const VTYPE * restrict const zpg, __global const uint32 * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_4096();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const VTYPE * restrict const zpk = &zp[k4096];\n" \
"\n" \
"	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));\n" \
"	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));\n" \
"	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));\n" \
"	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));\n" \
"	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));\n" \
"	mul_4(pq, Z4, L4096S / 4, zpk, w, wi, sj, sji);\n" \
"	backward_4(pq, L4096S / 1024, Zi4, wi, sji / (L4096S / 1024));\n" \
"	backward_4(pq, L4096S / 256, Zi16, wi, sji / (L4096S / 256));\n" \
"	backward_4(pq, L4096S / 64, Zi64, wi, sji / (L4096S / 64));\n" \
"	backward_4(pq, L4096S / 16, Zi256, wi, sji / (L4096S / 16));\n" \
"	backward_4o(pq, L4096S / 4, zk, L4096S / 4, Zi1024, wi, sji / (L4096S / 4));\n" \
"}\n" \
"\n" \
"#endif	// SHORT_VER\n" \
"\n" \
"// -----------------\n" \
"\n" \
"INLINE uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 * a_p)\n" \
"{\n" \
"	// Using notations of Modular SIMD arithmetic in Mathemagix, Joris van der Hoeven, Grégoire Lecerf, Guillaume Quintin, 2014, HAL.\n" \
"	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.\n" \
"	// b < 2^31, alpha = 2^29 => a < 2^29 b\n" \
"	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31\n" \
"	// b_inv = [2^{s + 32} / b]\n" \
"	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31\n" \
"	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].\n" \
"	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])\n" \
"	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,\n" \
"	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.\n" \
"\n" \
"	const uint32 d = mul_hi((uint32)(a >> b_s), b_inv), r = (uint32)(a) - d * b;\n" \
"	const bool o = (r >= b);\n" \
"	*a_p = d + (o ? 1 : 0);\n" \
"	return r - (o ? b : 0);\n" \
"}\n" \
"\n" \
"INLINE int32 reduce64(int64 * f, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32\n" \
"	// 2- t < 2^23 b^2 => t_h < b^2 / 2^6. If 2 <= b < 32 then t_h < 32^2 / 2^6 = 16 < 2^29 b\n" \
"	const uint64 t = abs(*f);\n" \
"	const uint64 t_h = t >> 29;\n" \
"	const uint32 t_l = (uint32)(t) % (1u << 29);\n" \
"\n" \
"	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint32 d_l, r_l = barrett(((uint64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint64 d = ((uint64)(d_h) << 29) | d_l;\n" \
"\n" \
"	const bool s = (*f < 0);\n" \
"	*f = s ? -(int64)(d) : (int64)(d);\n" \
"	return s ? -(int32)(r_l) : (int32)(r_l);\n" \
"}\n" \
"\n" \
"INLINE int32 reduce96(int96 * f, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	const uint96 t = int96_abs(*f);\n" \
"	const uint64 t_h = ((uint64)(t.s1) << (64 - 29)) | (t.s0 >> 29);\n" \
"	const uint32 t_l = (uint32)(t.s0) % (1u << 29);\n" \
"\n" \
"	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint32 d_l, r_l = barrett(((uint64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint64 d = ((uint64)(d_h) << 29) | d_l;\n" \
"\n" \
"	const bool s = int96_is_neg(*f);\n" \
"	*f = int96_set_si(s ? -(int64)(d) : (int64)(d));\n" \
"	return s ? -(int32)(r_l) : (int32)(r_l);\n" \
"}\n" \
"\n" \
"INLINE int64 garner2(const uint32 r1, const uint32 r2)\n" \
"{\n" \
"	const uint64 P1P2 = P1 * (uint64)(P2);\n" \
"	uint32 u12 = mulmod(submod(r1, r2, P1), INVP2_P1, PQ1);	// P2 < P1\n" \
"	const uint64 n = r2 + u12 * (uint64)(P2);\n" \
"	const bool b = (n > P1P2 / 2);\n" \
"	return (int64)(n - (b ? P1P2 : 0));\n" \
"}\n" \
"\n" \
"INLINE int96 garner3(const uint32 r1, const uint32 r2, const uint32 r3)\n" \
"{\n" \
"	const uint32 u13 = mulmod(submod(r1, r3, P1), INVP3_P1, PQ1);\n" \
"	const uint32 u23 = mulmod(submod(r2, r3, P2), INVP3_P2, PQ2);\n" \
"	const uint32 u123 = mulmod(submod(u13, u23, P1), INVP2_P1, PQ1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2 * (uint64)(P3), u123), u23 * (uint64)(P3) + r3);\n" \
"	const bool b = uint96_is_greater(n, uint96_set(P1P2P3_2L, P1P2P3_2H));\n" \
"	return uint96_i(uint96_sub(n, uint96_set(b ? P1P2P3L : 0ul, b ? P1P2P3H : 0u)));\n" \
"}\n" \
"\n" \
"INLINE void write_rns(__global uint32_4 * restrict const zi, const int32_4 r)\n" \
"{\n" \
"	uint32_4 zo1, zo2;\n" \
"#if RNS_SZ == 3\n" \
"	uint32_4 zo3;\n" \
"#endif\n" \
"	zo1.s0 = set_int(r.s0, P1); zo2.s0 = set_int(r.s0, P2);\n" \
"#if RNS_SZ == 3\n" \
"	zo3.s0 = set_int(r.s0, P3);\n" \
"#endif\n" \
"	zo1.s1 = set_int(r.s1, P1); zo2.s1 = set_int(r.s1, P2);\n" \
"#if RNS_SZ == 3\n" \
"	zo3.s1 = set_int(r.s1, P3);\n" \
"#endif\n" \
"	zo1.s2 = set_int(r.s2, P1); zo2.s2 = set_int(r.s2, P2);\n" \
"#if RNS_SZ == 3\n" \
"	zo3.s2 = set_int(r.s2, P3);\n" \
"#endif\n" \
"	zo1.s3 = set_int(r.s3, P1); zo2.s3 = set_int(r.s3, P2);\n" \
"#if RNS_SZ == 3\n" \
"	zo3.s3 = set_int(r.s3, P3);\n" \
"#endif\n" \
"\n" \
"	zi[0 * N_SZ / 4] = zo1; zi[1 * N_SZ / 4] = zo2;\n" \
"#if RNS_SZ == 3\n" \
"	zi[2 * N_SZ / 4] = zo3;\n" \
"#endif\n" \
"}\n" \
"\n" \
"INLINE int32_4 normalize_1(__global uint32_4 * restrict const zi, __global int64 * restrict const c,\n" \
"	__local int64 * const cl, const sz_t gid, const sz_t lid,\n" \
"	const uint32 b, const uint32 b_inv, const int b_s, const bool dup)\n" \
"{\n" \
"	const uint32_4 u1 = mulmod4(zi[0 * N_SZ / 4], NORM1, PQ1), u2 = mulmod4(zi[1 * N_SZ / 4], NORM2, PQ2);\n" \
"	int32_4 r;\n" \
"\n" \
"#if RNS_SZ == 2\n" \
"\n" \
"	int64_4 l = (int64_4)(garner2(u1.s0, u2.s0), garner2(u1.s1, u2.s1), garner2(u1.s2, u2.s2), garner2(u1.s3, u2.s3));\n" \
"	l += dup ? l : (int64_4)(0, 0, 0, 0);\n" \
"\n" \
"	int64 f = l.s0; r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += l.s1; r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += l.s2; r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += l.s3; r.s3 = reduce64(&f, b, b_inv, b_s);\n" \
"\n" \
"#else\n" \
"\n" \
"	const uint32_4 u3 = mulmod4(zi[2 * N_SZ / 4], NORM3, PQ3);\n" \
"\n" \
"	int96 l0 = garner3(u1.s0, u2.s0, u3.s0), l1 = garner3(u1.s1, u2.s1, u3.s1);\n" \
"	int96 l2 = garner3(u1.s2, u2.s2, u3.s2), l3 = garner3(u1.s3, u2.s3, u3.s3);\n" \
"\n" \
"	l0 = int96_add(l0, dup ? l0 : int96_zero());\n" \
"	l1 = int96_add(l1, dup ? l1 : int96_zero());\n" \
"	l2 = int96_add(l2, dup ? l2 : int96_zero());\n" \
"	l3 = int96_add(l3, dup ? l3 : int96_zero());\n" \
"\n" \
"	int96 f96 = l0; r.s0 = reduce96(&f96, b, b_inv, b_s);\n" \
"	f96 = int96_add(f96, l1); r.s1 = reduce96(&f96, b, b_inv, b_s);\n" \
"	f96 = int96_add(f96, l2); r.s2 = reduce96(&f96, b, b_inv, b_s);\n" \
"	f96 = int96_add(f96, l3); r.s3 = reduce96(&f96, b, b_inv, b_s);\n" \
"	int64 f = (int64)(f96.s0);\n" \
"\n" \
"#endif\n" \
"\n" \
"	cl[lid] = f;\n" \
"\n" \
"	if (lid == NORM_WG_SZ - 1)\n" \
"	{\n" \
"		const sz_t i = (gid / NORM_WG_SZ + 1) % (N_SZ / 4 / NORM_WG_SZ);\n" \
"		c[i] = (i == 0) ? -f : f;\n" \
"	}\n" \
"\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE void normalize_2(__global uint32_4 * restrict const zi, __local int64 * const cl, const sz_t lid,\n" \
"	const int32_4 r, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	int64 f = (lid == 0) ? 0 : cl[lid - 1];\n" \
"	int32_4 ro;\n" \
"	f += r.s0; ro.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s1; ro.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s2; ro.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s3; ro.s3 = (sz_t)(f);\n" \
"\n" \
"	write_rns(zi, ro);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))\n" \
"void normalize1(__global uint32_4 * restrict const z, __global int64 * restrict const c,\n" \
"	const uint32 b, const uint32 b_inv, const int b_s, const int32 dup)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;\n" \
"	__global uint32_4 * restrict const zi = &z[gid];\n" \
"	__local int64 cl[NORM_WG_SZ];\n" \
"\n" \
"	const int32_4 r = normalize_1(zi, c, cl, gid, lid, b, b_inv, b_s, dup != 0);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	normalize_2(zi, cl, lid, r, b, b_inv, b_s);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2(__global uint32_4 * restrict const z, __global const int64 * restrict const c, \n" \
"	const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	__global uint32_4 * restrict const zi = &z[NORM_WG_SZ * gid];\n" \
"\n" \
"	const uint32_4 u1 = zi[0 * N_SZ / 4];\n" \
"	int32_4 r;\n" \
"\n" \
"	int64 f = c[gid] + get_int(u1.s0, P1);\n" \
"	r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += get_int(u1.s1, P1);\n" \
"	r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += get_int(u1.s2, P1);\n" \
"	r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += get_int(u1.s3, P1);\n" \
"	r.s3 = (int32)(f);\n" \
"\n" \
"	write_rns(zi, r);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))\n" \
"void mulscalar(__global uint32_4 * restrict const z, __global int64 * restrict const c,\n" \
"	const uint32 b, const uint32 b_inv, const int b_s, const int32 a)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;\n" \
"	__global uint32_4 * restrict const zi = &z[gid];\n" \
"	__local int64 cl[NORM_WG_SZ];\n" \
"\n" \
"	const uint32_4 u1 = zi[0 * N_SZ / 4];\n" \
"	int32_4 r;\n" \
"\n" \
"	int64 f = get_int(u1.s0, P1) * (int64)(a);\n" \
"	r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += get_int(u1.s1, P1) * (int64)(a);\n" \
"	r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += get_int(u1.s2, P1) * (int64)(a);\n" \
"	r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += get_int(u1.s3, P1) * (int64)(a);\n" \
"	r.s3 = reduce64(&f, b, b_inv, b_s);\n" \
"\n" \
"	cl[lid] = f;\n" \
"\n" \
"	if (lid == NORM_WG_SZ - 1)\n" \
"	{\n" \
"		const sz_t i = (gid / NORM_WG_SZ + 1) % (N_SZ / 4 / NORM_WG_SZ);\n" \
"		c[i] = (i == 0) ? -f : f;\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	normalize_2(zi, cl, lid, r, b, b_inv, b_s);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set(__global uint32_4 * restrict const z, const uint32 a)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	z[gid] = (gid % (N_SZ / 4) == 0) ? (uint32_4)(a, 0, 0, 0) : (uint32_4)(0, 0, 0, 0);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global uint32_4 * restrict const z, const sz_t dst, const sz_t src)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	z[dst + gid] = z[src + gid];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copyp(__global uint32_4 * restrict const zp, __global const uint32_4 * restrict const z, const sz_t src)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	zp[gid] = z[src + gid];\n" \
"}\n" \
"";
