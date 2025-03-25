/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernels = \
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
"#ifdef __NV_CL_C_VERSION\n" \
"	#define PTX_ASM	1\n" \
"#endif\n" \
"\n" \
"#ifndef N_SZ\n" \
"#define N_SZ		65536u\n" \
"#define LN_SZ		16\n" \
"#define RNS_SZ		3\n" \
"#define NORM1		2130641409u\n" \
"#define NORM2		2113864705u\n" \
"#define NORM3		2013204481u\n" \
"#define W_SHFT		65536u\n" \
"#define WI_SHFT		32768u\n" \
"#define USE_WI		1\n" \
"#define BLK32		32\n" \
"#define BLK64		16\n" \
"#define BLK128		8\n" \
"#define BLK256		4\n" \
"#define BLK512		2\n" \
"#define CHUNK64		16\n" \
"#define CHUNK256	4\n" \
"#define CHUNK1024	1\n" \
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
"\n" \
"// --- Z/(127*2^24 + 1)Z ---\n" \
"\n" \
"#define	P1		2130706433u\n" \
"#define	Q1		2164260865u		// p * q = 1 (mod 2^32)\n" \
"// #define	R1		33554430u		// 2^32 mod p\n" \
"#define	RSQ1	402124772u		// (2^32)^2 mod p\n" \
"// #define	H1		100663290u		// Montgomery form of the primitive root 3\n" \
"#define	IM1		1930170389u		// MF of MF of I = 3^{(p - 1)/4} to convert input into MF\n" \
"#define	SQRTI1	1626730317u		// MF of 3^{(p - 1)/8}\n" \
"#define	ISQRTI1	856006302u		// MF of i * sqrt(i)\n" \
"\n" \
"// --- Z/(63*2^25 + 1)Z ---\n" \
"\n" \
"#define	P2		2113929217u\n" \
"#define	Q2		2181038081u\n" \
"// #define	R2		67108862u\n" \
"#define	RSQ2	2111798781u\n" \
"// #define	H2		335544310u		// MF of the primitive root 5\n" \
"#define	IM2		1036950657u\n" \
"#define	SQRTI2	338852760u\n" \
"#define	ISQRTI2	1090446030u\n" \
"\n" \
"// --- Z/(15*2^27 + 1)Z ---\n" \
"\n" \
"#define	P3		2013265921u\n" \
"#define	Q3		2281701377u\n" \
"// #define	R3		268435454u\n" \
"#define	RSQ3	1172168163u\n" \
"// #define	H3		268435390u		// MF of the primitive root 31\n" \
"#define	IM3		734725699u\n" \
"#define	SQRTI3	1032137103u\n" \
"#define	ISQRTI3	1964242958u\n" \
"\n" \
"// ---\n" \
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
"	const uint32 t = lhs + rhs;\n" \
"	return t - ((t >= p) ? p : 0);\n" \
"}\n" \
"\n" \
"INLINE uint32 submod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 t = lhs - rhs;\n" \
"	return t + (((int32)(t) < 0) ? p : 0);\n" \
"}\n" \
"\n" \
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
"// --- uint96/int96 ---\n" \
"\n" \
"typedef struct { uint64 s0; uint32 s1; } uint96;\n" \
"typedef struct { uint64 s0; int32 s1; } int96;\n" \
"\n" \
"INLINE int96 int96_set_si(const int64 n) { int96 r; r.s0 = (uint64)(n); r.s1 = (n < 0) ? -1 : 0; return r; }\n" \
"INLINE uint96 uint96_set(const uint64 s0, const int32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }\n" \
"\n" \
"INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int32)(x.s1); return r; }\n" \
"INLINE uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint32)(x.s1); return r; }\n" \
"\n" \
"INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }\n" \
"\n" \
"INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }\n" \
"\n" \
"INLINE int96 int96_neg(const int96 x)\n" \
"{\n" \
"	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - ((x.s0 != 0) ? 1 : 0);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 int96_abs(const int96 x)\n" \
"{\n" \
"	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;\n" \
"	return int96_u(t);\n" \
"}\n" \
"\n" \
"INLINE int96 int96_add(const int96 x, const int96 y)\n" \
"{\n" \
"	int96 r;\n" \
"#ifdef PTX_ASM\n" \
"	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y.s0));\n" \
"	asm volatile (\"addc.s32 %0, %1, %2;\" : \"=r\" (r.s1) : \"r\" (x.s1), \"r\" (y.s1));\n" \
"#else\n" \
"	const uint64 s0 = x.s0 + y.s0;\n" \
"	r.s0 = s0; r.s1 = x.s1 + y.s1 + ((s0 < y.s0) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 uint96_add_64(const uint96 x, const ulong y)\n" \
"{\n" \
"	uint96 r;\n" \
"#ifdef PTX_ASM\n" \
"	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y));\n" \
"	asm volatile (\"addc.u32 %0, %1, 0;\" : \"=r\" (r.s1) : \"r\" (x.s1));\n" \
"#else\n" \
"	const uint64 s0 = x.s0 + y;\n" \
"	r.s0 = s0; r.s1 = x.s1 + ((s0 < y) ? 1 : 0);\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE int96 uint96_subi(const uint96 x, const uint96 y)\n" \
"{\n" \
"	int96 r;\n" \
"#ifdef PTX_ASM\n" \
"	asm volatile (\"sub.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y.s0));\n" \
"	asm volatile (\"subc.s32 %0, %1, %2;\" : \"=r\" (r.s1) : \"r\" (x.s1), \"r\" (y.s1));\n" \
"#else\n" \
"	r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - ((x.s0 < y.s0) ? 1 : 0));\n" \
"#endif\n" \
"	return r;\n" \
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
"// 16 mul + 16 mul_hi\n" \
"#define FORWARD_4(pq, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w1, w20, w21) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0 = zi0, u2 = mulmod(zi2, w1, pq), u1 = zi1, u3 = mulmod(zi3, w1, pq); \\\n" \
"	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p); \\\n" \
"	const uint32 v1 = mulmod(addmod(u1, u3, p), w20, pq), v3 = mulmod(submod(u1, u3, p), w21, pq); \\\n" \
"	zo0 = addmod(v0, v1, p); zo1 = submod(v0, v1, p); zo2 = addmod(v2, v3, p); zo3 = submod(v2, v3, p); \\\n" \
"}\n" \
"\n" \
"#define BACKWARD_4(pq, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, win1, win20, win21) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0 = zi0, u1 = zi1, u2 = zi2, u3 = zi3; \\\n" \
"	const uint32 v0 = addmod(u0, u1, p), v1 = mulmod(submod(u1, u0, p), win20, pq); \\\n" \
"	const uint32 v2 = addmod(u2, u3, p), v3 = mulmod(submod(u3, u2, p), win21, pq); \\\n" \
"	zo0 = addmod(v0, v2, p); zo2 = mulmod(submod(v2, v0, p), win1, pq); \\\n" \
"	zo1 = addmod(v1, v3, p); zo3 = mulmod(submod(v3, v1, p), win1, pq); \\\n" \
"}\n" \
"\n" \
"#define FORWARD_4_0(pq, f0, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0, rsq = f0.s0, im = f0.s1, sqrti = f0.s2, isqrti = f0.s3; \\\n" \
"	const uint32 u0 = mulmod(zi0, rsq, pq), u2 = mulmod(zi2, im, pq); \\\n" \
"	const uint32 u1 = mulmod(zi1, rsq, pq), u3 = mulmod(zi3, im, pq); \\\n" \
"	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p); \\\n" \
"	const uint32 v1 = mulmod(addmod(u1, u3, p), sqrti, pq), v3 = mulmod(submod(u1, u3, p), isqrti, pq); \\\n" \
"	zo0 = addmod(v0, v1, p); zo1 = submod(v0, v1, p); zo2 = addmod(v2, v3, p); zo3 = submod(v2, v3, p); \\\n" \
"}\n" \
"\n" \
"#define SQUARE_22(pq, z0, z1, z2, z3, w) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \\\n" \
"	z0 = addmod(sqrmod(u0, pq), mulmod(sqrmod(u1, pq), w, pq), p); z1 = mulmod(addmod(u0, u0, p), u1, pq); \\\n" \
"	z2 = submod(sqrmod(u2, pq), mulmod(sqrmod(u3, pq), w, pq), p); z3 = mulmod(addmod(u2, u2, p), u3, pq); \\\n" \
"}\n" \
"\n" \
"#define SQUARE_4(pq, z0, z1, z2, z3, w, win) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0 = z0, u2 = mulmod(z2, w, pq), u1 = z1, u3 = mulmod(z3, w, pq); \\\n" \
"	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p), v1 = addmod(u1, u3, p), v3 = submod(u1, u3, p); \\\n" \
"	const uint32 s0 = addmod(sqrmod(v0, pq), mulmod(sqrmod(v1, pq), w, pq), p); \\\n" \
"	const uint32 s1 = mulmod(addmod(v0, v0, p), v1, pq); \\\n" \
"	const uint32 s2 = submod(sqrmod(v2, pq), mulmod(sqrmod(v3, pq), w, pq), p); \\\n" \
"	const uint32 s3 = mulmod(addmod(v2, v2, p), v3, pq); \\\n" \
"	z0 = addmod(s0, s2, p); z2 = mulmod(submod(s2, s0, p), win, pq); \\\n" \
"	z1 = addmod(s1, s3, p); z3 = mulmod(submod(s3, s1, p), win, pq); \\\n" \
"}\n" \
"\n" \
"#define FWD_2(pq, zi0, zi1, zi2, zi3, zo0, zo1, zo2, zo3, w) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0 = zi0, u2 = mulmod(zi2, w, pq), u1 = zi1, u3 = mulmod(zi3, w, pq); \\\n" \
"	zo0 = addmod(u0, u2, p); zo2 = submod(u0, u2, p); zo1 = addmod(u1, u3, p); zo3 = submod(u1, u3, p); \\\n" \
"}\n" \
"\n" \
"#define MUL_22(pq, z0, z1, z2, z3, z0p, z1p, z2p, z3p, w) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0p = z0p, u1p = z1p, u2p = z2p, u3p = z3p; \\\n" \
"	const uint32 u0 = z0, u1 = z1, u2 = z2, u3 = z3; \\\n" \
"	z0 = addmod(mulmod(u0, u0p, pq), mulmod(mulmod(u1, u1p, pq), w, pq), p); \\\n" \
"	z1 = addmod(mulmod(u0, u1p, pq), mulmod(u0p, u1, pq), p); \\\n" \
"	z2 = submod(mulmod(u2, u2p, pq), mulmod(mulmod(u3, u3p, pq), w, pq), p); \\\n" \
"	z3 = addmod(mulmod(u2, u3p, pq), mulmod(u2p, u3, pq), p); \\\n" \
"}\n" \
"\n" \
"#define MUL_4(pq, z0, z1, z2, z3, z0p, z1p, z2p, z3p, w, win) \\\n" \
"{ \\\n" \
"	const uint32 p = pq.s0; \\\n" \
"	const uint32 u0 = z0, u2 = mulmod(z2, w, pq), u1 = z1, u3 = mulmod(z3, w, pq); \\\n" \
"	const uint32 v0 = addmod(u0, u2, p), v2 = submod(u0, u2, p), v1 = addmod(u1, u3, p), v3 = submod(u1, u3, p); \\\n" \
"	const uint32 v0p = z0p, v1p = z1p, v2p = z2p, v3p = z3p; \\\n" \
"	const uint32 s0 = addmod(mulmod(v0, v0p, pq), mulmod(mulmod(v1, v1p, pq), w, pq), p); \\\n" \
"	const uint32 s1 = addmod(mulmod(v0, v1p, pq), mulmod(v0p, v1, pq), p); \\\n" \
"	const uint32 s2 = submod(mulmod(v2, v2p, pq), mulmod(mulmod(v3, v3p, pq), w, pq), p); \\\n" \
"	const uint32 s3 = addmod(mulmod(v2, v3p, pq), mulmod(v2p, v3, pq), p); \\\n" \
"	z0 = addmod(s0, s2, p); z2 = mulmod(submod(s2, s0, p), win, pq); \\\n" \
"	z1 = addmod(s1, s3, p); z3 = mulmod(submod(s3, s1, p), win, pq); \\\n" \
"}\n" \
"\n" \
"// --- inverse of roots is wi[s + j] or w[s + s - j - 1] ---\n" \
"\n" \
"#ifdef USE_WI\n" \
"#define DECLARE_IVAR(s, j)	const sz_t sji = s + j; __global const uint * restrict const wi = &w[WI_SHFT];\n" \
"#define DECLARE_WIN(sji)	const uint32 win1 = w[sji], win20 = w[2 * sji + 0], win21 = w[2 * sji + 1];\n" \
"#else\n" \
"#define DECLARE_IVAR(s, j)	const sz_t sji = s + s - j - 1; __global const uint * restrict const wi = w;\n" \
"#define DECLARE_WIN(sji)	const uint32 win1 = w[sji], win20 = w[2 * sji + 1], win21 = w[2 * sji + 0];\n" \
"#endif\n" \
"\n" \
"// --- transform/inline global mem ---\n" \
"\n" \
"INLINE void forward_4io(const uint32_2 pq, const sz_t m, __global uint * restrict const z, __global const uint * restrict const w, const sz_t sj)\n" \
"{\n" \
"	const uint32 w1 = w[sj], w20 = w[2 * sj + 0], w21 = w[2 * sj + 1];\n" \
"	FORWARD_4(pq, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], w1, w20, w21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4io(const uint32_2 pq, const sz_t m, __global uint * restrict const z, __global const uint * restrict const w, const sz_t sji)\n" \
"{\n" \
"	DECLARE_WIN(sji);\n" \
"	BACKWARD_4(pq, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void forward_4io_0(const uint32_2 pq, const uint32_4 f0,	__global uint * restrict const z)\n" \
"{\n" \
"	const sz_t m = N_SZ / 4;\n" \
"	FORWARD_4_0(pq, f0, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m]);\n" \
"}\n" \
"\n" \
"INLINE void square_22io(const uint32_2 pq, __global uint * restrict const z, const uint w)\n" \
"{\n" \
"	SQUARE_22(pq, z[0], z[1], z[2], z[3], w);\n" \
"}\n" \
"\n" \
"INLINE void square_4io(const uint32_2 pq, __global uint * restrict const z, const uint w, const uint win)\n" \
"{\n" \
"	SQUARE_4(pq, z[0], z[1], z[2], z[3], w, win);\n" \
"}\n" \
"\n" \
"INLINE void fwd_2io(const uint32_2 pq, __global uint * restrict const z, const uint w)\n" \
"{\n" \
"	FWD_2(pq, z[0], z[1], z[2], z[3], z[0], z[1], z[2], z[3], w);\n" \
"}\n" \
"\n" \
"INLINE void mul_22io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp, const uint w)\n" \
"{\n" \
"	MUL_22(pq, z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w);\n" \
"}\n" \
"\n" \
"INLINE void mul_4io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp, const uint w, const uint win)\n" \
"{\n" \
"	MUL_4(pq, z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w, win);\n" \
"}\n" \
"\n" \
"// --- transform/inline local & global mem ---\n" \
"\n" \
"INLINE void forward_4(const uint32_2 pq, const sz_t m, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FORWARD_4(pq, Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], w1, w20, w21);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i(const uint32_2 pq, const sz_t ml, __local uint * restrict const Z, const sz_t mg,\n" \
"	__global const uint * restrict const z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];\n" \
"	FORWARD_4(pq, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], w1, w20, w21);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i_0(const uint32_2 pq, const uint32_4 f0, const sz_t ml, __local uint * restrict const Z, const sz_t mg,\n" \
"	__global const uint * restrict const z, __global const uint * restrict const w)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	FORWARD_4_0(pq, f0, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml]);\n" \
"}\n" \
"\n" \
"INLINE void forward_4o(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, const sz_t ml,\n" \
"	__local const uint * restrict const Z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FORWARD_4(pq, Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], w1, w20, w21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4(const uint32_2 pq, const sz_t m, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t sji)\n" \
"{\n" \
"	DECLARE_WIN(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4(pq, Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const uint32_2 pq, const sz_t ml, __local uint * restrict const Z, const sz_t mg,\n" \
"	__global const uint * restrict const z, __global const uint * restrict const w, const sz_t sji)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	DECLARE_WIN(sji);\n" \
"	BACKWARD_4(pq, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, const sz_t ml,\n" \
"	__local const uint * restrict const Z, __global const uint * restrict const w, const sz_t sji)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	DECLARE_WIN(sji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4(pq, Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void square_22(const uint32_2 pq, __local uint * restrict const Z, const uint w)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	SQUARE_22(pq, Z[0], Z[1], Z[2], Z[3], w);\n" \
"}\n" \
"\n" \
"INLINE void square_4(const uint32_2 pq, __local uint * restrict const Z, const uint w, const uint win)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	SQUARE_4(pq, Z[0], Z[1], Z[2], Z[3], w, win);\n" \
"}\n" \
"\n" \
"INLINE void write_4(const sz_t mg, __global uint * restrict const z, __local const uint * restrict const Z)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];\n" \
"}\n" \
"\n" \
"INLINE void fwd2_write_4(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, __local const uint * restrict const Z, const uint w)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FWD_2(pq, Z[0], Z[1], Z[2], Z[3], z[0], z[mg], z2mg[0], z2mg[mg], w);\n" \
"}\n" \
"\n" \
"INLINE void mul_22(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg, __global const uint * restrict const z, const uint w)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	const uint z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	MUL_22(pq, Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w);\n" \
"}\n" \
"\n" \
"INLINE void mul_4(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg, __global const uint * restrict const z, const uint w, const uint win)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	const uint z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	MUL_4(pq, Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w, win);\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LN_SZ - 2), mid = gid & ~((N_SZ / 4) - 1), id = gid %  (N_SZ / 4); \\\n" \
"	const uint32_2 pq = g_pq[lid]; \\\n" \
"	__global uint * restrict const z = &zg[4 * mid]; \\\n" \
"	__global const uint * restrict const w = &wg[lid * W_SHFT];\n" \
"\n" \
"#define DECLARE_VARP_REG() \\\n" \
"	__global const uint * restrict const zp = &zpg[4 * mid];\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"__kernel\n" \
"void forward4(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id >> lm, k = 3 * (j << lm) + id;\n" \
"	forward_4io(pq, (sz_t)(1) << lm, &z[k], w, s + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void backward4(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id >> lm, k = 3 * (j << lm) + id; DECLARE_IVAR(s, j);\n" \
"	backward_4io(pq, (sz_t)(1) << lm, &z[k], wi, sji);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void forward4_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t k = id;\n" \
"	forward_4io_0(pq, g_f0[lid], &z[k]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square22(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	square_22io(pq, &z[k], w[N_SZ / 4 + j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 + j; DECLARE_IVAR(N_SZ / 4, j);\n" \
"	square_4io(pq, &z[k], w[sj], wi[sji]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd4p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	fwd_2io(pq, &z[k], w[N_SZ / 4 + j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul22(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	mul_22io(pq, &z[k], &zp[k], w[N_SZ / 4 + j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 + j; DECLARE_IVAR(N_SZ / 4, j);\n" \
"	mul_4io(pq, &z[k], &zp[k], w[sj], wi[sji]);\n" \
"}\n" \
"\n" \
"// --- transform ---\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	/* threadIdx < B_N */ \\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \\\n" \
"	const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = group_id * CHUNK_N + chunk_idx; \\\n" \
"	__local uint * const Zi = &Z[chunk_idx]; \\\n" \
"	\\\n" \
"	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \\\n" \
"	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \\\n" \
"	\\\n" \
"	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \\\n" \
"	\\\n" \
"	const sz_t sj = s + idx_m; DECLARE_IVAR(s, idx_m);\n" \
"\n" \
"#define DECLARE_VAR_FORWARD() \\\n" \
"	__global uint * restrict const zi = &z[ki]; \\\n" \
"	__global uint * restrict const zo = &z[ko];\n" \
"\n" \
"#define DECLARE_VAR_BACKWARD() \\\n" \
"	__global uint * restrict const zi = &z[ko]; \\\n" \
"	__global uint * restrict const zo = &z[ki];\n" \
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
"	forward_4i_0(pq, g_f0[lid], B_N * CHUNK_N, &Z[i], B_N << lm, zi, w);\n" \
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
"	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))\n" \
"#else\n" \
"#define ATTR_64()\n" \
"#endif\n" \
"\n" \
"#define FORWARD_64() \\\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \\\n" \
"	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4); \\\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);\n" \
"\n" \
"INLINE void _forward64(__global uint * restrict const zg, __global const uint * restrict const wg,\n" \
"	__local uint * const Z, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_64, CHUNK64);\n" \
"	FORWARD_64();\n" \
"}\n" \
"\n" \
"INLINE void _backward64(__global uint * restrict const zg, __global const uint * restrict const wg,\n" \
"	__local uint * const Z, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], wi, sji / 4);\n" \
"	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], wi, sji / B_64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void forward64(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local uint Z[4 * B_64 * CHUNK64];\n" \
"	_forward64(zg, wg, Z, lm, s);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void forward64_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	const int lm = LN_SZ - 6; const unsigned int s = 64 / 4;\n" \
"	__local uint Z[4 * B_64 * CHUNK64];\n" \
"	FORWARD_I_0(B_64, CHUNK64);\n" \
"	FORWARD_64();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_64()\n" \
"void backward64(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local uint Z[4 * B_64 * CHUNK64];\n" \
"	_backward64(zg, wg, Z, lm, s);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_256	(256 / 4)\n" \
"\n" \
"#if MAX_WG_SZ >= B_256 * CHUNK256\n" \
"#define ATTR_256() \\\n" \
"	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))\n" \
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
"INLINE void _forward256(__global uint * restrict const zg, __global const uint * restrict const wg,\n" \
"	__local uint * const Z, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_256, CHUNK256);\n" \
"	FORWARD_256();\n" \
"}\n" \
"\n" \
"INLINE void _backward256(__global uint * restrict const zg, __global const uint * restrict const wg,\n" \
"	__local uint * const Z, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_256, CHUNK256);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sji / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sji / 16);\n" \
"	backward_4o(pq, B_256 << lm, zo, B_256 * CHUNK256, &Z[i], wi, sji / B_256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void forward256(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local uint Z[4 * B_256 * CHUNK256];\n" \
"	_forward256(zg, wg, Z, lm, s);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void forward256_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	const int lm = LN_SZ - 8; const unsigned int s = 256 / 4;\n" \
"	__local uint Z[4 * B_256 * CHUNK256];\n" \
"	FORWARD_I_0(B_256, CHUNK256);\n" \
"	FORWARD_256();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_256()\n" \
"void backward256(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local uint Z[4 * B_256 * CHUNK256];\n" \
"	_backward256(zg, wg, Z, lm, s);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_1024	(1024 / 4)\n" \
"\n" \
"#if MAX_WG_SZ >= B_1024 * CHUNK1024\n" \
"#define ATTR_1024() \\\n" \
"	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))\n" \
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
"INLINE void _forward1024(__global uint * restrict const zg, __global const uint * restrict const wg,\n" \
"	__local uint * const Z, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_1024, CHUNK1024);\n" \
"	FORWARD_1024();\n" \
"}\n" \
"\n" \
"INLINE void _backward1024(__global uint * restrict const zg, __global const uint * restrict const wg,\n" \
"	__local uint * const Z, const int lm, const unsigned int s)\n" \
"{\n" \
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
"__kernel\n" \
"ATTR_1024()\n" \
"void forward1024(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local uint Z[4 * B_1024 * CHUNK1024];\n" \
"	_forward1024(zg, wg, Z, lm, s);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void forward1024_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	const int lm = LN_SZ - 10; const unsigned int s = 1024 / 4;\n" \
"	__local uint Z[4 * B_1024 * CHUNK1024];\n" \
"	FORWARD_I_0(B_1024, CHUNK1024);\n" \
"	FORWARD_1024();\n" \
"}\n" \
"\n" \
"__kernel\n" \
"ATTR_1024()\n" \
"void backward1024(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	__local uint Z[4 * B_1024 * CHUNK1024];\n" \
"	_backward1024(zg, wg, Z, lm, s);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define DEFINE_KERNEL_FORWARD(m, n) \\\n" \
"	__kernel \\\n" \
"	ATTR_##m() \\\n" \
"	void forward##m##_##n(__global uint * restrict const zg, __global const uint * restrict const wg) \\\n" \
"	{ \\\n" \
"		__local uint Z[4 * B_##m * CHUNK##m]; \\\n" \
"		_forward##m(zg, wg, Z, n, (N_SZ / 4) >> n); \\\n" \
"	}\n" \
"\n" \
"#define DEFINE_KERNEL_BACKWARD(m, n) \\\n" \
"	__kernel \\\n" \
"	ATTR_##m() \\\n" \
"	void backward##m##_##n(__global uint * restrict const zg, __global const uint * restrict const wg) \\\n" \
"	{ \\\n" \
"		__local uint Z[4 * B_##m * CHUNK##m]; \\\n" \
"		_backward##m(zg, wg, Z, n, (N_SZ / 4) >> n); \\\n" \
"	}\n" \
"\n" \
"#if LN_SZ % 2 != 0\n" \
"\n" \
"DEFINE_KERNEL_FORWARD(64, 5);\n" \
"DEFINE_KERNEL_BACKWARD(64, 5);\n" \
"\n" \
"#if LN_SZ >= 19\n" \
"\n" \
"DEFINE_KERNEL_FORWARD(64, 7);\n" \
"DEFINE_KERNEL_BACKWARD(64, 7);\n" \
"DEFINE_KERNEL_FORWARD(256, 5);\n" \
"DEFINE_KERNEL_BACKWARD(256, 5);\n" \
"\n" \
"#endif\n" \
"\n" \
"#else // LN_SZ % 2 == 0\n" \
"\n" \
"DEFINE_KERNEL_FORWARD(64, 6);\n" \
"DEFINE_KERNEL_BACKWARD(64, 6);\n" \
"\n" \
"#if LN_SZ >= 20\n" \
"\n" \
"DEFINE_KERNEL_FORWARD(64, 8);\n" \
"DEFINE_KERNEL_BACKWARD(64, 8);\n" \
"DEFINE_KERNEL_FORWARD(256, 6);\n" \
"DEFINE_KERNEL_BACKWARD(256, 6);\n" \
"\n" \
"#endif\n" \
"\n" \
"#endif\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local uint Z[32 * BLK32]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (32 / 4 * BLK32), group_id = id / (32 / 4 * BLK32); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i32 = (local_id & ~(32 / 4 - 1)) * 4, i8 = local_id % (32 / 4); \\\n" \
"	const sz_t k32 = group_id * 32 * BLK32 + i32 + i8; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k32]; \\\n" \
"	__local uint * const Z32 = &Z[i32]; \\\n" \
"	__local uint * const Zi8 = &Z32[i8]; \\\n" \
"	const sz_t i2 = ((4 * i8) & ~(4 * 2 - 1)) + (i8 % 2); \\\n" \
"	__local uint * const Zi2 = &Z32[i2]; \\\n" \
"	__local uint * const Z4 = &Z32[4 * i8];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void square32(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(pq, 8, Zi8, 8, zk, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_22(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4o(pq, 8, zk, 8, Zi8, wi, sji / 8);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local uint Z[64 * BLK64]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (64 / 4 * BLK64), group_id = id / (64 / 4 * BLK64); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i64 = (local_id & ~(64 / 4 - 1)) * 4, i16 = local_id % (64 / 4); \\\n" \
"	const sz_t k64 = group_id * 64 * BLK64 + i64 + i16; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k64]; \\\n" \
"	__local uint * const Z64 = &Z[i64]; \\\n" \
"	__local uint * const Zi16 = &Z64[i16]; \\\n" \
"	const sz_t i4 = ((4 * i16) & ~(4 * 4 - 1)) + (i16 % 4); \\\n" \
"	__local uint * const Zi4 = &Z64[i4]; \\\n" \
"	__local uint * const Z4 = &Z64[4 * i16];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void square64(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(pq, 16, Zi16, 16, zk, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	square_4(pq, Z4, w[sj], wi[sji]);\n" \
"	backward_4(pq, 4, Zi4, wi, sji / 4);\n" \
"	backward_4o(pq, 16, zk, 16, Zi16, wi, sji / 16);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_128() \\\n" \
"	__local uint Z[128 * BLK128]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (128 / 4 * BLK128), group_id = id / (128 / 4 * BLK128); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i128 = (local_id & ~(128 / 4 - 1)) * 4, i32 = local_id % (128 / 4); \\\n" \
"	const sz_t k128 = group_id * 128 * BLK128 + i128 + i32; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k128]; \\\n" \
"	__local uint * const Z128 = &Z[i128]; \\\n" \
"	__local uint * const Zi32 = &Z128[i32]; \\\n" \
"	const sz_t i8 = ((4 * i32) & ~(4 * 8 - 1)) + (i32 % 8); \\\n" \
"	__local uint * const Zi8 = &Z128[i8]; \\\n" \
"	const sz_t i2 = ((4 * i32) & ~(4 * 2 - 1)) + (i32 % 2); \\\n" \
"	__local uint * const Zi2 = &Z128[i2]; \\\n" \
"	__local uint * const Z4 = &Z128[4 * i32];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 128 / 4 * BLK128\n" \
"	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void square128(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(pq, 32, Zi32, 32, zk, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_22(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4(pq, 8, Zi8, wi, sji / 8);\n" \
"	backward_4o(pq, 32, zk, 32, Zi32, wi, sji / 32);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_256() \\\n" \
"	__local uint Z[256 * BLK256]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (256 / 4 * BLK256), group_id = id / (256 / 4 * BLK256); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i256 = (local_id & ~(256 / 4 - 1)) * 4, i64 = local_id % (256 / 4); \\\n" \
"	const sz_t k256 = group_id * 256 * BLK256 + i256 + i64; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k256]; \\\n" \
"	__local uint * const Z256 = &Z[i256]; \\\n" \
"	__local uint * const Zi64 = &Z256[i64]; \\\n" \
"	const sz_t i16 = ((4 * i64) & ~(4 * 16 - 1)) + (i64 % 16); \\\n" \
"	__local uint * const Zi16 = &Z256[i16]; \\\n" \
"	const sz_t i4 = ((4 * i64) & ~(4 * 4 - 1)) + (i64 % 4); \\\n" \
"	__local uint * const Zi4 = &Z256[i4]; \\\n" \
"	__local uint * const Z4 = &Z256[4 * i64];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 256 / 4 * BLK256\n" \
"	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void square256(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(pq, 64, Zi64, 64, zk, w, sj / 64);\n" \
"	forward_4(pq, 16, Zi16, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	square_4(pq, Z4, w[sj], wi[sji]);\n" \
"	backward_4(pq, 4, Zi4, wi, sji / 4);\n" \
"	backward_4(pq, 16, Zi16, wi, sji / 16);\n" \
"	backward_4o(pq, 64, zk, 64, Zi64, wi, sji / 64);\n" \
"}\n" \
"\n" \
"// if BLK512 != 1 then const sz_t i512 = (i & ~(512 / 4 - 1)) * 4, i128 = i % (512 / 4);\n" \
"// if BLK512 = 1 then const sz_t i512 = 0, i128 = i;\n" \
"#define DECLARE_VAR_512() \\\n" \
"	__local uint Z[512 * BLK512]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (512 / 4 * BLK512), group_id = id / (512 / 4 * BLK512); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i512 = (local_id & ~(512 / 4 - 1)) * 4, i128 = local_id % (512 / 4); \\\n" \
"	const sz_t k512 = group_id * 512 * BLK512 + i512 + i128; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k512]; \\\n" \
"	__local uint * const Z512 = &Z[i512]; \\\n" \
"	__local uint * const Zi128 = &Z512[i128]; \\\n" \
"	const sz_t i32 = ((4 * i128) & ~(4 * 32 - 1)) + (i128 % 32); \\\n" \
"	__local uint * const Zi32 = &Z512[i32]; \\\n" \
"	const sz_t i8 = ((4 * i128) & ~(4 * 8 - 1)) + (i128 % 8); \\\n" \
"	__local uint * const Zi8 = &Z512[i8]; \\\n" \
"	const sz_t i2 = ((4 * i128) & ~(4 * 2 - 1)) + (i128 % 2); \\\n" \
"	__local uint * const Zi2 = &Z512[i2]; \\\n" \
"	__local uint * const Z4 = &Z512[4 * i128];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 512 / 4\n" \
"	__attribute__((work_group_size_hint(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void square512(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(pq, 128, Zi128, 128, zk, w, sj / 128);\n" \
"	forward_4(pq, 32, Zi32, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_22(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4(pq, 8, Zi8, wi, sji / 8);\n" \
"	backward_4(pq, 32, Zi32, wi, sji / 32);\n" \
"	backward_4o(pq, 128, zk, 128, Zi128, wi, sji / 128);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_1024() \\\n" \
"	__local uint Z[1024]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (1024 / 4), group_id = id / (1024 / 4); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i256 = local_id, k1024 = group_id * 1024 + i256; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k1024]; \\\n" \
"	__local uint * const Zi256 = &Z[i256]; \\\n" \
"	const sz_t i64 = ((4 * i256) & ~(4 * 64 - 1)) + (i256 % 64); \\\n" \
"	__local uint * const Zi64 = &Z[i64]; \\\n" \
"	const sz_t i16 = ((4 * i256) & ~(4 * 16 - 1)) + (i256 % 16); \\\n" \
"	__local uint * const Zi16 = &Z[i16]; \\\n" \
"	const sz_t i4 = ((4 * i256) & ~(4 * 4 - 1)) + (i256 % 4); \\\n" \
"	__local uint * const Zi4 = &Z[i4]; \\\n" \
"	__local uint * const Z4 = &Z[4 * i256];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 1024 / 4\n" \
"	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void square1024(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(pq, 256, Zi256, 256, zk, w, sj / 256);\n" \
"	forward_4(pq, 64, Zi64, w, sj / 64);\n" \
"	forward_4(pq, 16, Zi16, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	square_4(pq, Z4, w[sj], wi[sji]);\n" \
"	backward_4(pq, 4, Zi4, wi, sji / 4);\n" \
"	backward_4(pq, 16, Zi16, wi, sji / 16);\n" \
"	backward_4(pq, 64, Zi64, wi, sji / 64);\n" \
"	backward_4o(pq, 256, zk, 256, Zi256, wi, sji / 256);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_2048() \\\n" \
"	__local uint Z[2048]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (2048 / 4), group_id = id / (2048 / 4); \\\n" \
"	const sz_t sj = N_SZ / 4 + id; DECLARE_IVAR(N_SZ / 4, id); \\\n" \
"	\\\n" \
"	const sz_t i512 = local_id, k2048 = group_id * 2048 + i512; \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k2048]; \\\n" \
"	__local uint * const Zi512 = &Z[i512]; \\\n" \
"	const sz_t i128 = ((4 * i512) & ~(4 * 128 - 1)) + (i512 % 128); \\\n" \
"	__local uint * const Zi128 = &Z[i128]; \\\n" \
"	const sz_t i32 = ((4 * i512) & ~(4 * 32 - 1)) + (i512 % 32); \\\n" \
"	__local uint * const Zi32 = &Z[i32]; \\\n" \
"	const sz_t i8 = ((4 * i512) & ~(4 * 8 - 1)) + (i512 % 8); \\\n" \
"	__local uint * const Zi8 = &Z[i8]; \\\n" \
"	const sz_t i2 = ((4 * i512) & ~(4 * 2 - 1)) + (i512 % 2); \\\n" \
"	__local uint * const Zi2 = &Z[i2]; \\\n" \
"	__local uint * const Z4 = &Z[4 * i512];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 2048 / 4\n" \
"	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void square2048(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(pq, 512, Zi512, 512, zk, w, sj / 512);\n" \
"	forward_4(pq, 128, Zi128, w, sj / 128);\n" \
"	forward_4(pq, 32, Zi32, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	square_22(pq, Z4, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4(pq, 8, Zi8, wi, sji / 8);\n" \
"	backward_4(pq, 32, Zi32, wi, sji / 32);\n" \
"	backward_4(pq, 128, Zi128, wi, sji / 128);\n" \
"	backward_4o(pq, 512, zk, 512, Zi512, wi, sji / 512);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void fwd32p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(pq, 8, Zi8, 8, zk, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(8, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void fwd64p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(pq, 16, Zi16, 16, zk, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	fwd2_write_4(pq, 16, zk, Z4, w[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 128 / 4 * BLK128\n" \
"	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void fwd128p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(pq, 32, Zi32, 32, zk, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(32, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 256 / 4 * BLK256\n" \
"	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void fwd256p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(pq, 64, Zi64, 64, zk, w, sj / 64);\n" \
"	forward_4(pq, 16, Zi16, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	fwd2_write_4(pq, 64, zk, Z4, w[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 512 / 4\n" \
"	__attribute__((work_group_size_hint(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd512p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(pq, 128, Zi128, 128, zk, w, sj / 128);\n" \
"	forward_4(pq, 32, Zi32, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(128, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 1024 / 4\n" \
"	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd1024p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(pq, 256, Zi256, 256, zk, w, sj / 256);\n" \
"	forward_4(pq, 64, Zi64, w, sj / 64);\n" \
"	forward_4(pq, 16, Zi16, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	fwd2_write_4(pq, 256, zk, Z4, w[sj]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 2048 / 4\n" \
"	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd2048p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(pq, 512, Zi512, 512, zk, w, sj / 512);\n" \
"	forward_4(pq, 128, Zi128, w, sj / 128);\n" \
"	forward_4(pq, 32, Zi32, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	write_4(512, zk, Z4);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void mul32(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k32];\n" \
"\n" \
"	forward_4i(pq, 8, Zi8, 8, zk, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_22(pq, Z4, 8, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4o(pq, 8, zk, 8, Zi8, wi, sji / 8);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void mul64(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k64];\n" \
"\n" \
"	forward_4i(pq, 16, Zi16, 16, zk, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	mul_4(pq, Z4, 16, zpk, w[sj], wi[sji]);\n" \
"	backward_4(pq, 4, Zi4, wi, sji / 4);\n" \
"	backward_4o(pq, 16, zk, 16, Zi16, wi, sji / 16);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 128 / 4 * BLK128\n" \
"	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void mul128(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k128];\n" \
"\n" \
"	forward_4i(pq, 32, Zi32, 32, zk, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_22(pq, Z4, 32, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4(pq, 8, Zi8, wi, sji / 8);\n" \
"	backward_4o(pq, 32, zk, 32, Zi32, wi, sji / 32);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 256 / 4 * BLK256\n" \
"	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void mul256(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k256];\n" \
"\n" \
"	forward_4i(pq, 64, Zi64, 64, zk, w, sj / 64);\n" \
"	forward_4(pq, 16, Zi16, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	mul_4(pq, Z4, 64, zpk, w[sj], wi[sji]);\n" \
"	backward_4(pq, 4, Zi4, wi, sji / 4);\n" \
"	backward_4(pq, 16, Zi16, wi, sji / 16);\n" \
"	backward_4o(pq, 64, zk, 64, Zi64, wi, sji / 64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 512 / 4\n" \
"	__attribute__((work_group_size_hint(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul512(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k512];\n" \
"\n" \
"	forward_4i(pq, 128, Zi128, 128, zk, w, sj / 128);\n" \
"	forward_4(pq, 32, Zi32, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_22(pq, Z4, 128, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4(pq, 8, Zi8, wi, sji / 8);\n" \
"	backward_4(pq, 32, Zi32, wi, sji / 32);\n" \
"	backward_4o(pq, 128, zk, 128, Zi128, wi, sji / 128);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 1024 / 4\n" \
"	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul1024(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k1024];\n" \
"\n" \
"	forward_4i(pq, 256, Zi256, 256, zk, w, sj / 256);\n" \
"	forward_4(pq, 64, Zi64, w, sj / 64);\n" \
"	forward_4(pq, 16, Zi16, w, sj / 16);\n" \
"	forward_4(pq, 4, Zi4, w, sj / 4);\n" \
"	mul_4(pq, Z4, 256, zpk, w[sj], wi[sji]);\n" \
"	backward_4(pq, 4, Zi4, wi, sji / 4);\n" \
"	backward_4(pq, 16, Zi16, wi, sji / 16);\n" \
"	backward_4(pq, 64, Zi64, wi, sji / 64);\n" \
"	backward_4o(pq, 256, zk, 256, Zi256, wi, sji / 256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WG_SZ >= 2048 / 4\n" \
"	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul2048(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k2048];\n" \
"\n" \
"	forward_4i(pq, 512, Zi512, 512, zk, w, sj / 512);\n" \
"	forward_4(pq, 128, Zi128, w, sj / 128);\n" \
"	forward_4(pq, 32, Zi32, w, sj / 32);\n" \
"	forward_4(pq, 8, Zi8, w, sj / 8);\n" \
"	forward_4(pq, 2, Zi2, w, sj / 2);\n" \
"	mul_22(pq, Z4, 512, zpk, w[sj]);\n" \
"	backward_4(pq, 2, Zi2, wi, sji / 2);\n" \
"	backward_4(pq, 8, Zi8, wi, sji / 8);\n" \
"	backward_4(pq, 32, Zi32, wi, sji / 32);\n" \
"	backward_4(pq, 128, Zi128, wi, sji / 128);\n" \
"	backward_4o(pq, 512, zk, 512, Zi512, wi, sji / 512);\n" \
"}\n" \
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
"	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b\n" \
"	const uint64 t = abs(*f);\n" \
"	const uint64 t_h = t >> 29;\n" \
"	const uint32 t_l = (uint32)(t) & ((1u << 29) - 1);\n" \
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
"	const uint32 t_l = (uint32)(t.s0) & ((1u << 29) - 1);\n" \
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
"	const uint32 mfInvP2_P1 = 2130706177u;	// Montgomery form of 1 / P2 (mod P1)\n" \
"	const uint64 P1P2 = P1 * (uint64)(P2);\n" \
"	uint32 u12 = mulmod(submod(r1, r2, P1), mfInvP2_P1, PQ1);	// P2 < P1\n" \
"	const uint64 n = r2 + u12 * (uint64)(P2);\n" \
"	return (n > P1P2 / 2) ? (int64)(n - P1P2) : (int64)(n);\n" \
"}\n" \
"\n" \
"INLINE int96 garner3(const uint r1, const uint r2, const uint r3)\n" \
"{\n" \
"	// Montgomery form of 1 / Pi (mod Pj)\n" \
"	const uint32 mfInvP3_P1 = 608773230u, mfInvP2_P1 = 2130706177u, mfInvP3_P2 = 1409286102u;\n" \
"	const uint64 P2P3 = P2 * (uint64)(P3);\n" \
"	const uint96 P1P2P3 = uint96_set(13049742876517335041ul, 491581440u);\n" \
"	const uint96 P1P2P3_2 = uint96_set(6524871438258667520ul, 245790720u);\n" \
"\n" \
"	const uint32 u13 = mulmod(submod(r1, r3, P1), mfInvP3_P1, PQ1);\n" \
"	const uint32 u23 = mulmod(submod(r2, r3, P2), mfInvP3_P2, PQ2);\n" \
"	const uint32 u123 = mulmod(submod(u13, u23, P1), mfInvP2_P1, PQ1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (uint64)(P3) + r3);\n" \
"	return uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize1(__global uint * restrict const z, __global long * restrict const c,\n" \
"	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	const unsigned int blk = abs(sblk);\n" \
"	__global uint * restrict const zi = &z[blk * idx];\n" \
"\n" \
"	prefetch(zi, (size_t)blk);\n" \
"\n" \
"#if RNS_SZ == 2\n" \
"\n" \
"	int64 f = 0;\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		const uint32 u1 = mulmod(zi[j + 0 * N_SZ], NORM1, PQ1);\n" \
"		const uint32 u2 = mulmod(zi[j + 1 * N_SZ], NORM2, PQ2);\n" \
"		int64 l = garner2(u1, u2);\n" \
"		if (sblk < 0) l += l;\n" \
"		f += l;\n" \
"		const int32 r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j + 0 * N_SZ] = set_int(r, P1);\n" \
"		zi[j + 1 * N_SZ] = set_int(r, P2);\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	c[i] = (i == 0) ? -f : f;\n" \
"\n" \
"#else\n" \
"\n" \
"	int96 f = int96_set_si(0);\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		const uint32 u1 = mulmod(zi[j + 0 * N_SZ], NORM1, PQ1);\n" \
"		const uint32 u2 = mulmod(zi[j + 1 * N_SZ], NORM2, PQ2);\n" \
"		const uint32 u3 = mulmod(zi[j + 2 * N_SZ], NORM3, PQ3);\n" \
"		int96 l = garner3(u1, u2, u3);\n" \
"		if (sblk < 0) l = int96_add(l, l);\n" \
"		f = int96_add(f, l);\n" \
"		const int32 r = reduce96(&f, b, b_inv, b_s);\n" \
"		zi[j + 0 * N_SZ] = set_int(r, P1);\n" \
"		zi[j + 1 * N_SZ] = set_int(r, P2);\n" \
"		zi[j + 2 * N_SZ] = set_int(r, P3);\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	c[i] = (i == 0) ? -(long)f.s0 : (long)f.s0;\n" \
"\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2(__global uint * restrict const z, __global const long * restrict const c, \n" \
"	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	__global uint * restrict const zi = &z[blk * idx];\n" \
"\n" \
"	int64 f = c[idx];\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		f += get_int(zi[j], P1);\n" \
"		const int32 r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j + 0 * N_SZ] = set_int(r, P1);\n" \
"		zi[j + 1 * N_SZ] = set_int(r, P2);\n" \
"#if RNS_SZ == 3\n" \
"		zi[j + 2 * N_SZ] = set_int(r, P3);\n" \
"#endif\n" \
"		if (f == 0) return;\n" \
"		++j;\n" \
"	} while (j != blk - 1);\n" \
"\n" \
"	const int32 r = (int32)(f);\n" \
"	zi[blk - 1 + 0 * N_SZ] = addmod(zi[blk - 1 + 0 * N_SZ], set_int(r, P1), P1);\n" \
"	zi[blk - 1 + 1 * N_SZ] = addmod(zi[blk - 1 + 1 * N_SZ], set_int(r, P2), P2);\n" \
"#if RNS_SZ == 3\n" \
"	zi[blk - 1 + 2 * N_SZ] = addmod(zi[blk - 1 + 2 * N_SZ], set_int(r, P3), P3);\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mulscalar(__global uint * restrict const z, __global long * restrict const c,\n" \
"	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	__global uint * restrict const zi = &z[blk * idx];\n" \
"\n" \
"	prefetch(zi, (size_t)blk);\n" \
"\n" \
"	int64 f = 0;\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		f += get_int(zi[j], P1) * (int64)(a);\n" \
"		const int32 r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j + 0 * N_SZ] = set_int(r, P1);\n" \
"		zi[j + 1 * N_SZ] = set_int(r, P2);\n" \
"#if RNS_SZ == 3\n" \
"		zi[j + 2 * N_SZ] = set_int(r, P3);\n" \
"#endif\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	c[i] = (i == 0) ? -f : f;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set(__global uint * restrict const z, const unsigned int a)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	z[idx] = ((idx & (N_SZ - 1)) == 0) ? a : 0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global uint * restrict const z, const unsigned int dst, const unsigned int src)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	z[dst + idx] = z[src + idx];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copyp(__global uint * restrict const zp, __global const uint * restrict const z, const unsigned int src)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	zp[idx] = z[src + idx];\n" \
"}\n" \
"";
