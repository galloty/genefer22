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
"#ifndef NSIZE\n" \
"#define NSIZE		4096u\n" \
"#define	LNSZ		12\n" \
"#define	NORM1		2129666049u\n" \
"#define	NORM2		2112897025u\n" \
"#define	WOFFSET		2048u\n" \
"#define BLK32		8\n" \
"#define BLK64		4\n" \
"#define BLK128		2\n" \
"#define BLK256		1\n" \
"#define CHUNK64		4\n" \
"#define CHUNK256	2\n" \
"#define CHUNK1024	1\n" \
"#define MAX_WORK_GROUP_SIZE	256\n" \
"#endif\n" \
"\n" \
"typedef uint	sz_t;\n" \
"typedef uint	uint32;\n" \
"typedef int		int32;\n" \
"typedef ulong	uint64;\n" \
"typedef long	int64;\n" \
"typedef uint2	uint32_2;\n" \
"typedef uint4	uint32_4;\n" \
"// typedef int2	int32_2;\n" \
"// typedef ulong2	uint64_2;\n" \
"// typedef long2	int64_2;\n" \
"\n" \
"// --- Z/(127*2^24 + 1)Z ---\n" \
"\n" \
"#define	P1		2130706433u\n" \
"#define	Q1		2164260865u		// p * q = 1 (mod 2^32)\n" \
"// #define	R1		33554430u		// 2^32 mod p\n" \
"#define	RSQ1	402124772u		// (2^32)^2 mod p\n" \
"// #define	H1		167772150u		// Montgomery form of the primitive root 5\n" \
"#define	IM1		200536044u		// MF of MF of I = 5^{(p - 1)/4} to convert input into MF\n" \
"#define	SQRTI1	856006302u		// MF of 5^{(p - 1)/8}\n" \
"#define	ISQRTI1	1626730317u		// MF of i * sqrt(i)\n" \
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
"__constant uint32_2 g_pq[2] = { (uint32_2)(P1, Q1), (uint32_2)(P2, Q2) };\n" \
"__constant uint32_4 g_f0[2] = { (uint32_4)(RSQ1, IM1, SQRTI1, ISQRTI1), (uint32_4)(RSQ2, IM2, SQRTI2, ISQRTI2) };\n" \
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
"// --- transform/inline global mem ---\n" \
"\n" \
"INLINE void forward_4io(const uint32_2 pq, const sz_t m, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	const uint32 w1 = w[j], w20 = w[2 * j + 0], w21 = w[2 * j + 1];\n" \
"	FORWARD_4(pq, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], w1, w20, w21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4io(const uint32_2 pq, const sz_t m, __global uint * restrict const z, __global const uint * restrict const w, const sz_t ji)\n" \
"{\n" \
"	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];\n" \
"	BACKWARD_4(pq, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void forward_4io_0(const uint32_2 pq, const uint32_4 f0,	__global uint * restrict const z)\n" \
"{\n" \
"	const sz_t m = NSIZE / 4;\n" \
"	FORWARD_4_0(pq, f0, z[0 * m], z[1 * m], z[2 * m], z[3 * m], z[0 * m], z[1 * m], z[2 * m], z[3 * m]);\n" \
"}\n" \
"\n" \
"INLINE void square_22io(const uint32_2 pq, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	SQUARE_22(pq, z[0], z[1], z[2], z[3], w[j]);\n" \
"}\n" \
"\n" \
"INLINE void square_4io(const uint32_2 pq, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	SQUARE_4(pq, z[0], z[1], z[2], z[3], w[j], w[ji]);\n" \
"}\n" \
"\n" \
"INLINE void fwd_2io(const uint32_2 pq, __global uint * restrict const z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	FWD_2(pq, z[0], z[1], z[2], z[3], z[0], z[1], z[2], z[3], w[j]);\n" \
"}\n" \
"\n" \
"INLINE void mul_22io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp,\n" \
"	__global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	MUL_22(pq, z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w[j]);\n" \
"}\n" \
"\n" \
"INLINE void mul_4io(const uint32_2 pq, __global uint * restrict const z, const __global uint * restrict const zp,\n" \
"	__global const uint * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	MUL_4(pq, z[0], z[1], z[2], z[3], zp[0], zp[1], zp[2], zp[3], w[j], w[ji]);\n" \
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
"INLINE void backward_4(const uint32_2 pq, const sz_t m, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t ji)\n" \
"{\n" \
"	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4(pq, Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], Z[0 * m], Z[1 * m], Z[2 * m], Z[3 * m], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const uint32_2 pq, const sz_t ml, __local uint * restrict const Z, const sz_t mg,\n" \
"	__global const uint * restrict const z, __global const uint * restrict const w, const sz_t ji)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];\n" \
"	BACKWARD_4(pq, z[0], z[mg], z2mg[0], z2mg[mg], Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, const sz_t ml,\n" \
"	__local const uint * restrict const Z, __global const uint * restrict const w, const sz_t ji)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	const uint32 win1 = w[ji], win20 = w[2 * ji + 1], win21 = w[2 * ji + 0];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4(pq, Z[0 * ml], Z[1 * ml], Z[2 * ml], Z[3 * ml], z[0], z[mg], z2mg[0], z2mg[mg], win1, win20, win21);\n" \
"}\n" \
"\n" \
"INLINE void square_22(const uint32_2 pq, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	SQUARE_22(pq, Z[0], Z[1], Z[2], Z[3], w[j]);\n" \
"}\n" \
"\n" \
"INLINE void square_4(const uint32_2 pq, __local uint * restrict const Z, __global const uint * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	SQUARE_4(pq, Z[0], Z[1], Z[2], Z[3], w[j], w[ji]);\n" \
"}\n" \
"\n" \
"INLINE void write_4(const sz_t mg, __global uint * restrict const z, __local const uint * restrict const Z)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];\n" \
"}\n" \
"\n" \
"INLINE void fwd2_write_4(const uint32_2 pq, const sz_t mg, __global uint * restrict const z, __local const uint * restrict const Z,\n" \
"	__global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global uint * const z2mg = &z[2 * mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FWD_2(pq, Z[0], Z[1], Z[2], Z[3], z[0], z[mg], z2mg[0], z2mg[mg], w[j]);\n" \
"}\n" \
"\n" \
"INLINE void mul_22(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg, __global const uint * restrict const z,\n" \
"	__global const uint * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	const uint z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	MUL_22(pq, Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w[j]);\n" \
"}\n" \
"\n" \
"INLINE void mul_4(const uint32_2 pq, __local uint * restrict const Z, const sz_t mg, __global const uint * restrict const z,\n" \
"	__global const uint * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	__global const uint * const z2mg = &z[2 * mg];\n" \
"	const uint z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	MUL_4(pq, Z[0], Z[1], Z[2], Z[3], z0p, z1p, z2p, z3p, w[j], w[ji]);\n" \
"}\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define DECLARE_VAR_REG() \\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid >> (LNSZ - 2), mid = gid & ~((NSIZE / 4) - 1), id = gid & ((NSIZE / 4) - 1); \\\n" \
"	const uint32_2 pq = g_pq[lid]; \\\n" \
"	__global uint * restrict const z = &zg[4 * mid]; \\\n" \
"	__global const uint * restrict const w = &wg[lid * WOFFSET];\n" \
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
"	const sz_t j = id >> lm, k = 3 * (j << lm) + id;\n" \
"	backward_4io(pq, (sz_t)(1) << lm, &z[k], w, s + s - j - 1);\n" \
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
"	square_22io(pq, &z[k], w, NSIZE / 4 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	square_4io(pq, &z[k], w, NSIZE / 4 +  j, NSIZE / 4 + NSIZE / 4 - j - 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void fwd4p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	fwd_2io(pq, &z[k], w, NSIZE / 4 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul22(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	mul_22io(pq, &z[k], &zp[k], w, NSIZE / 4 + j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_REG();\n" \
"	DECLARE_VARP_REG();\n" \
"	const sz_t j = id, k = 4 * id;\n" \
"	mul_4io(pq, &z[k], &zp[k], w, NSIZE / 4 + j, NSIZE / 4 + NSIZE / 4 - j - 1);\n" \
"}\n" \
"\n" \
"// --- transform ---\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	__local uint Z[4 * B_N * CHUNK_N]; \\\n" \
"	\\\n" \
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
"	const sz_t sj = s + idx_m, sji = s + s - idx_m - 1;\n" \
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
"	backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, w, sji / 1);\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_64	(64 / 4)\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64\n" \
"	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))\n" \
"#endif\n" \
"void forward64(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64\n" \
"	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))\n" \
"#endif\n" \
"void forward64_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	const int lm = LNSZ - 6; const unsigned int s = 64 / 4;\n" \
"\n" \
"	FORWARD_I_0(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64\n" \
"	__attribute__((work_group_size_hint(B_64 * CHUNK64, 1, 1)))\n" \
"#endif\n" \
"void backward64(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sji / 4);\n" \
"	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], w, sji / B_64);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_256	(256 / 4)\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256\n" \
"	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))\n" \
"#endif\n" \
"void forward256(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_256, CHUNK256);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256\n" \
"	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))\n" \
"#endif\n" \
"void forward256_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	const int lm = LNSZ - 8; const unsigned int s = 256 / 4;\n" \
"\n" \
"	FORWARD_I_0(B_256, CHUNK256);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256\n" \
"	__attribute__((work_group_size_hint(B_256 * CHUNK256, 1, 1)))\n" \
"#endif\n" \
"void backward256(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_256, CHUNK256);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sji / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sji / 16);\n" \
"	backward_4o(pq, B_256 << lm, zo, B_256 * CHUNK256, &Z[i], w, sji / B_256);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_1024	(1024 / 4)\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024\n" \
"	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))\n" \
"#endif\n" \
"void forward1024(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_1024, CHUNK1024);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	forward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024\n" \
"	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))\n" \
"#endif\n" \
"void forward1024_0(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	const int lm = LNSZ - 10; const unsigned int s = 1024 / 4;\n" \
"\n" \
"	FORWARD_I_0(B_1024, CHUNK1024);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	forward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);\n" \
"	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024\n" \
"	__attribute__((work_group_size_hint(B_1024 * CHUNK1024, 1, 1)))\n" \
"#endif\n" \
"void backward1024(__global uint * restrict const zg, __global const uint * restrict const wg, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_1024, CHUNK1024);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sji / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sji / 16);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);\n" \
"	backward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sji / 64);\n" \
"	backward_4o(pq, B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], w, sji / B_1024);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local uint Z[32 * BLK32]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (32 / 4 * BLK32), group_id = id / (32 / 4 * BLK32); \\\n" \
"	const sz_t idx = id, j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k32 = group_id * 32 * BLK32, i = local_id; \\\n" \
"	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k32 + i32 + i8]; \\\n" \
"	__local uint * const Z32 = &Z[i32]; \\\n" \
"	__local uint * const Zi8 = &Z32[i8]; \\\n" \
"	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \\\n" \
"	__local uint * const Zi2 = &Z32[i2]; \\\n" \
"	__local uint * const Z4 = &Z32[4 * i8];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void square32(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(pq, 8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(pq, 2, Zi2, w, j / 2);\n" \
"	square_22(pq, Z4, w, j);\n" \
"	backward_4(pq, 2, Zi2, w, ji / 2);\n" \
"	backward_4o(pq, 8, zk, 8, Zi8, w, ji / 8);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local uint Z[64 * BLK64]; \\\n" \
"	\\\n" \
"	DECLARE_VAR_REG(); \\\n" \
"	const sz_t local_id = id % (64 / 4 * BLK64), group_id = id / (64 / 4 * BLK64); \\\n" \
"	const sz_t idx = id, j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k64 = group_id * 64 * BLK64, i = local_id; \\\n" \
"	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \\\n" \
"	\\\n" \
"	__global uint * restrict const zk = &z[k64 + i64 + i16]; \\\n" \
"	__local uint * const Z64 = &Z[i64]; \\\n" \
"	__local uint * const Zi16 = &Z64[i16]; \\\n" \
"	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \\\n" \
"	__local uint * const Zi4 = &Z64[i4]; \\\n" \
"	__local uint * const Z4 = &Z64[4 * i16];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void square64(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(pq, 16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(pq, 4, Zi4, w, j / 4);\n" \
"	square_4(pq, Z4, w, j, ji);\n" \
"	backward_4(pq, 4, Zi4, w, ji / 4);\n" \
"	backward_4o(pq, 16, zk, 16, Zi16, w, ji / 16);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void fwd32p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(pq, 8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(pq, 2, Zi2, w, j / 2);\n" \
"	write_4(8, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void fwd64p(__global uint * restrict const zg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(pq, 16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(pq, 4, Zi4, w, j / 4);\n" \
"	fwd2_write_4(pq, 16, zk, Z4, w, j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void mul32(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k32 + i32 + i8];\n" \
"\n" \
"	forward_4i(pq, 8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(pq, 2, Zi2, w, j / 2);\n" \
"	mul_22(pq, Z4, 8, zpk, w, j);\n" \
"	backward_4(pq, 2, Zi2, w, ji / 2);\n" \
"	backward_4o(pq, 8, zk, 8, Zi8, w, ji / 8);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void mul64(__global uint * restrict const zg, __global const uint * restrict const zpg, __global const uint * restrict const wg)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	DECLARE_VARP_REG();\n" \
"	__global const uint * restrict const zpk = &zp[k64 + i64 + i16];\n" \
"\n" \
"	forward_4i(pq, 16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(pq, 4, Zi4, w, j / 4);\n" \
"	mul_4(pq, Z4, 16, zpk, w, j, ji);\n" \
"	backward_4(pq, 4, Zi4, w, ji / 4);\n" \
"	backward_4o(pq, 16, zk, 16, Zi16, w, ji / 16);\n" \
"}\n" \
"\n" \
"/*\n" \
"// --- set pair P1/P2 ---\n" \
"\n" \
"#ifdef W123\n" \
"#define DECLARE_VAR_INDEXW(j) \\\n" \
"	const sz_t j0 = 3 * j + 0, j1 = 3 * j + 1, j2 = 3 * j + 2;\n" \
"#define DECLARE_VAR_INDEXWIN(ji) \\\n" \
"	const sz_t ji0 = 3 * ji + 0, ji1 = 3 * ji + 1, ji2 = 3 * ji + 2;\n" \
"#define DECLARE_VAR_INDEXWS() \\\n" \
"	const sz_t ws_offset = NSIZE / 2;\n" \
"#else\n" \
"#define DECLARE_VAR_INDEXW(j) \\\n" \
"	const sz_t j0 = j, j1 = NSIZE / 2 + j, j2 = NSIZE + j;\n" \
"#define DECLARE_VAR_INDEXWIN(ji) \\\n" \
"	const sz_t ji0 = ji, ji1 = NSIZE / 2 + ji, ji2 = NSIZE + ji;\n" \
"#define DECLARE_VAR_INDEXWS() \\\n" \
"	const sz_t ws_offset = 0;\n" \
"#endif\n" \
"\n" \
"#define DECLARE_VAR_W(j) \\\n" \
"	DECLARE_VAR_INDEXW(j); \\\n" \
"	const Zp1 w1_0 = w[j0], w2_0 = w[j1], w3_0 = w[j2]; \\\n" \
"	const Zp2 w1_1 = w[WOFFSET_1 + j0], w2_1 = w[WOFFSET_1 + j1], w3_1 = w[WOFFSET_1 + j2];\n" \
"\n" \
"#define DECLARE_VAR_WIN(j, ji) \\\n" \
"	DECLARE_VAR_INDEXWIN(ji); \\\n" \
"	const Zp1 wi1_0 = swap(w[ji0]), wi3_0 = swap(w[ji1]), wi2_0 = swap(w[ji2]); \\\n" \
"	const Zp2 wi1_1 = swap(w[WOFFSET_1 + ji0]), wi3_1 = swap(w[WOFFSET_1 + ji1]), wi2_1 = swap(w[WOFFSET_1 + ji2]);\n" \
"\n" \
"#define DECLARE_VAR_WS(j) \\\n" \
"	DECLARE_VAR_INDEXWS(); \\\n" \
"	const Zp1 w_0 = w[ws_offset + j]; \\\n" \
"	const Zp2 w_1 = w[WOFFSET_1 + ws_offset + j];\n" \
"\n" \
"#define DECLARE_VAR_WINS(ji) \\\n" \
"	const Zp1 wi_0 = swap(w[ws_offset + ji]); \\\n" \
"	const Zp2 wi_1 = swap(w[WOFFSET_1 + ws_offset + ji]);\n" \
"\n" \
"// --- transform/inline global mem ---\n" \
"\n" \
"\n" \
"}\n" \
"\n" \
"// --- transform/inline local & global mem ---\n" \
"\n" \
"INLINE void forward_4(const sz_t m, __local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j)\n" \
"{\n" \
"	DECLARE_VAR_W(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FORWARD_4_a(Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, w1_0, w2_0, w3_0);\n" \
"	FORWARD_4_b(Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, w1_1, w2_1, w3_1);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i(const sz_t ml, __local uint4 * restrict const Z, const sz_t mg,\n" \
"	__global const uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_W(j);\n" \
"	FORWARD_4_a(z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, w1_0, w2_0, w3_0);\n" \
"	FORWARD_4_b(z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, w1_1, w2_1, w3_1);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i_0(const sz_t ml, __local uint4 * restrict const Z, const sz_t mg,\n" \
"	__global const uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	__global const uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_W(1);\n" \
"	FORWARD_4_0_a(z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, w1_0, w2_0, w3_0);\n" \
"	FORWARD_4_0_b(z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, w1_1, w2_1, w3_1);\n" \
"}\n" \
"\n" \
"INLINE void forward_4o(const sz_t mg, __global uint4 * restrict const z, const sz_t ml,\n" \
"	__local const uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_W(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FORWARD_4_a(Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, w1_0, w2_0, w3_0);\n" \
"	FORWARD_4_b(Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, w1_1, w2_1, w3_1);\n" \
"}\n" \
"\n" \
"INLINE void backward_4(const sz_t m, __local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	DECLARE_VAR_WIN(j, ji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4_a(Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, Z[0 * m].s01, Z[1 * m].s01, Z[2 * m].s01, Z[3 * m].s01, wi1_0, wi2_0, wi3_0);\n" \
"	BACKWARD_4_b(Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, Z[0 * m].s23, Z[1 * m].s23, Z[2 * m].s23, Z[3 * m].s23, wi1_1, wi2_1, wi3_1);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const sz_t ml, __local uint4 * restrict const Z, const sz_t mg,\n" \
"	__global const uint4 * restrict const z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	__global const uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_WIN(j, ji);\n" \
"	BACKWARD_4_a(z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, wi1_0, wi2_0, wi3_0);\n" \
"	BACKWARD_4_b(z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, wi1_1, wi2_1, wi3_1);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const sz_t mg, __global uint4 * restrict const z, const sz_t ml,\n" \
"	__local const uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	__global uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_WIN(j, ji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4_a(Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, wi1_0, wi2_0, wi3_0);\n" \
"	BACKWARD_4_b(Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, wi1_1, wi2_1, wi3_1);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o_0(const sz_t mg, __global uint4 * restrict const z, const sz_t ml,\n" \
"	__local const uint4 * restrict const Z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	__global uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_WIN(1, 1);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	BACKWARD_4_0_a(Z[0 * ml].s01, Z[1 * ml].s01, Z[2 * ml].s01, Z[3 * ml].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, wi1_0, wi2_0, wi3_0);\n" \
"	BACKWARD_4_0_b(Z[0 * ml].s23, Z[1 * ml].s23, Z[2 * ml].s23, Z[3 * ml].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, wi1_1, wi2_1, wi3_1);\n" \
"}\n" \
"\n" \
"INLINE void square_22(__local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j)\n" \
"{\n" \
"	DECLARE_VAR_WS(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	SQUARE_22_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, w_0);\n" \
"	SQUARE_22_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, w_1);\n" \
"}\n" \
"\n" \
"INLINE void square_4(__local uint4 * restrict const Z, __global const uint2 * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	DECLARE_VAR_WS(j);\n" \
"	DECLARE_VAR_WINS(ji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	SQUARE_4_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, w_0, wi_0);\n" \
"	SQUARE_4_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, w_1, wi_1);\n" \
"}\n" \
"\n" \
"INLINE void write_4(const sz_t mg, __global uint4 * restrict const z, __local const uint4 * restrict const Z)\n" \
"{\n" \
"	__global uint4 * const z2mg = &z[2 * mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];\n" \
"}\n" \
"\n" \
"INLINE void fwd2_write_4(const sz_t mg, __global uint4 * restrict const z, __local const uint4 * restrict const Z,\n" \
"	__global const uint2 * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global uint4 * const z2mg = &z[2 * mg];\n" \
"	DECLARE_VAR_WS(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	FWD_2_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, z[0].s01, z[mg].s01, z2mg[0].s01, z2mg[mg].s01, w_0);\n" \
"	FWD_2_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, z[0].s23, z[mg].s23, z2mg[0].s23, z2mg[mg].s23, w_1);\n" \
"}\n" \
"\n" \
"INLINE void mul_22(__local uint4 * restrict const Z, const sz_t mg, __global const uint4 * restrict const z,\n" \
"	__global const uint2 * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const uint4 * const z2mg = &z[2 * mg];\n" \
"	const uint4 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];\n" \
"	DECLARE_VAR_WS(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	MUL_22_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, z0p.s01, z1p.s01, z2p.s01, z3p.s01, w_0);\n" \
"	MUL_22_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, z0p.s23, z1p.s23, z2p.s23, z3p.s23, w_1);\n" \
"}\n" \
"\n" \
"INLINE void mul_4(__local uint4 * restrict const Z, const sz_t mg, __global const uint4 * restrict const z,\n" \
"	__global const uint2 * restrict const w, const sz_t j, const sz_t ji)\n" \
"{\n" \
"	__global const uint4 * const z2mg = &z[2 * mg];\n" \
"	const uint4 z0p = z[0], z1p = z[mg], z2p = z2mg[0], z3p = z2mg[mg];\n" \
"	DECLARE_VAR_WS(j);\n" \
"	DECLARE_VAR_WINS(ji);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	MUL_4_a(Z[0].s01, Z[1].s01, Z[2].s01, Z[3].s01, z0p.s01, z1p.s01, z2p.s01, z3p.s01, w_0, wi_0);\n" \
"	MUL_4_b(Z[0].s23, Z[1].s23, Z[2].s23, Z[3].s23, z0p.s23, z1p.s23, z2p.s23, z3p.s23, w_1, wi_1);\n" \
"}\n" \
"\n" \
"// --- transform without local mem ---\n" \
"\n" \
"\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local uint4 Z[32 * BLK32]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k32 = (sz_t)get_group_id(0) * 32 * BLK32, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k32 + i32 + i8]; \\\n" \
"	__local uint4 * const Z32 = &Z[i32]; \\\n" \
"	__local uint4 * const Zi8 = &Z32[i8]; \\\n" \
"	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \\\n" \
"	__local uint4 * const Zi2 = &Z32[i2]; \\\n" \
"	__local uint4 * const Z4 = &Z32[4 * i8];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void square32(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4o(8, zk, 8, Zi8, w, j / 8, ji / 8);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local uint4 Z[64 * BLK64]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k64 = (sz_t)get_group_id(0) * 64 * BLK64, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k64 + i64 + i16]; \\\n" \
"	__local uint4 * const Z64 = &Z[i64]; \\\n" \
"	__local uint4 * const Zi16 = &Z64[i16]; \\\n" \
"	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \\\n" \
"	__local uint4 * const Zi4 = &Z64[i4]; \\\n" \
"	__local uint4 * const Z4 = &Z64[4 * i16];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void square64(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	square_4(Z4, w, j, ji);\n" \
"	backward_4(4, Zi4, w, j / 4, ji / 4);\n" \
"	backward_4o(16, zk, 16, Zi16, w, j / 16, ji / 16);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_128() \\\n" \
"	__local uint4 Z[128 * BLK128]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k128 = (sz_t)get_group_id(0) * 128 * BLK128, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k128 + i128 + i32]; \\\n" \
"	__local uint4 * const Z128 = &Z[i128]; \\\n" \
"	__local uint4 * const Zi32 = &Z128[i32]; \\\n" \
"	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \\\n" \
"	__local uint4 * const Zi8 = &Z128[i8]; \\\n" \
"	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \\\n" \
"	__local uint4 * const Zi2 = &Z128[i2]; \\\n" \
"	__local uint4 * const Z4 = &Z128[4 * i32];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128\n" \
"	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void square128(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(32, Zi32, 32, zk, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4(8, Zi8, w, j / 8, ji / 8);\n" \
"	backward_4o(32, zk, 32, Zi32, w, j / 32, ji / 32);\n" \
"}\n" \
"\n" \
"// if BLK256 != 1 then const sz_t i256 = (i & (sz_t)~(256 / 4 - 1)) * 4, i64 = i % (256 / 4);\n" \
"// if BLK256 = 1 then const sz_t i256 = 0, i64 = i;\n" \
"#define DECLARE_VAR_256() \\\n" \
"	__local uint4 Z[256 * BLK256]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k256 = (sz_t)get_group_id(0) * 256 * BLK256, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i256 = 0, i64 = i; \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k256 + i256 + i64]; \\\n" \
"	__local uint4 * const Z256 = &Z[i256]; \\\n" \
"	__local uint4 * const Zi64 = &Z256[i64]; \\\n" \
"	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \\\n" \
"	__local uint4 * const Zi16 = &Z256[i16]; \\\n" \
"	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \\\n" \
"	__local uint4 * const Zi4 = &Z256[i4]; \\\n" \
"	__local uint4 * const Z4 = &Z256[4 * i64];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256\n" \
"	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void square256(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(64, Zi64, 64, zk, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	square_4(Z4, w, j, ji);\n" \
"	backward_4(4, Zi4, w, j / 4, ji / 4);\n" \
"	backward_4(16, Zi16, w, j / 16, ji / 16);\n" \
"	backward_4o(64, zk, 64, Zi64, w, j / 64, ji / 64);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_512() \\\n" \
"	__local uint4 Z[512]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k512 = (sz_t)get_group_id(0) * 512, i128 = (sz_t)get_local_id(0); \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k512 + i128]; \\\n" \
"	__local uint4 * const Zi128 = &Z[i128]; \\\n" \
"	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \\\n" \
"	__local uint4 * const Zi32 = &Z[i32]; \\\n" \
"	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \\\n" \
"	__local uint4 * const Zi8 = &Z[i8]; \\\n" \
"	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \\\n" \
"	__local uint4 * const Zi2 = &Z[i2]; \\\n" \
"	__local uint4 * const Z4 = &Z[4 * i128];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 512 / 4\n" \
"	__attribute__((work_group_size_hint(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void square512(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(128, Zi128, 128, zk, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4(8, Zi8, w, j / 8, ji / 8);\n" \
"	backward_4(32, Zi32, w, j / 32, ji / 32);\n" \
"	backward_4o(128, zk, 128, Zi128, w, j / 128, ji / 128);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_1024() \\\n" \
"	__local uint4 Z[1024]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k1024 = (sz_t)get_group_id(0) * 1024, i256 = (sz_t)get_local_id(0); \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k1024 + i256]; \\\n" \
"	__local uint4 * const Zi256 = &Z[i256]; \\\n" \
"	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \\\n" \
"	__local uint4 * const Zi64 = &Z[i64]; \\\n" \
"	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \\\n" \
"	__local uint4 * const Zi16 = &Z[i16]; \\\n" \
"	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \\\n" \
"	__local uint4 * const Zi4 = &Z[i4]; \\\n" \
"	__local uint4 * const Z4 = &Z[4 * i256];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 1024 / 4\n" \
"	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void square1024(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(256, Zi256, 256, zk, w, j / 256);\n" \
"	forward_4(64, Zi64, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	square_4(Z4, w, j, ji);\n" \
"	backward_4(4, Zi4, w, j / 4, ji / 4);\n" \
"	backward_4(16, Zi16, w, j / 16, ji / 16);\n" \
"	backward_4(64, Zi64, w, j / 64, ji / 64);\n" \
"	backward_4o(256, zk, 256, Zi256, w, j / 256, ji / 256);\n" \
"}\n" \
"\n" \
"/*#define DECLARE_VAR_2048() \\\n" \
"	__local uint4 Z[2048]; \\\n" \
"	\\\n" \
"	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE / 4 + idx, ji = NSIZE / 4 + NSIZE / 4 - idx - 1; \\\n" \
"	\\\n" \
"	const sz_t k2048 = (sz_t)get_group_id(0) * 2048, i512 = (sz_t)get_local_id(0); \\\n" \
"	\\\n" \
"	__global uint4 * restrict const zk = &z[k2048 + i512]; \\\n" \
"	__local uint4 * const Zi512 = &Z[i512]; \\\n" \
"	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \\\n" \
"	__local uint4 * const Zi128 = &Z[i128]; \\\n" \
"	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \\\n" \
"	__local uint4 * const Zi32 = &Z[i32]; \\\n" \
"	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \\\n" \
"	__local uint4 * const Zi8 = &Z[i8]; \\\n" \
"	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \\\n" \
"	__local uint4 * const Zi2 = &Z[i2]; \\\n" \
"	__local uint4 * const Z4 = &Z[4 * i512];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 2048 / 4\n" \
"	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void square2048(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(512, Zi512, 512, zk, w, j / 512);\n" \
"	forward_4(128, Zi128, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4(8, Zi8, w, j / 8, ji / 8);\n" \
"	backward_4(32, Zi32, w, j / 32, ji / 32);\n" \
"	backward_4(128, Zi128, w, j / 128, ji / 128);\n" \
"	backward_4o(512, zk, 512, Zi512, w, j / 512, ji / 51);\n" \
"}*/\n" \
"\n" \
"// -----------------\n" \
"/*\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void fwd32p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(8, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void fwd64p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	fwd2_write_4(16, zk, Z4, w, j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128\n" \
"	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void fwd128p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(32, Zi32, 32, zk, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(32, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256\n" \
"	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void fwd256p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(64, Zi64, 64, zk, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	fwd2_write_4(64, zk, Z4, w, j);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 512 / 4\n" \
"	__attribute__((work_group_size_hint(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd512p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(128, Zi128, 128, zk, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(128, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 1024 / 4\n" \
"	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd1024p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(256, Zi256, 256, zk, w, j / 256);\n" \
"	forward_4(64, Zi64, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	fwd2_write_4(256, zk, Z4, w, j);\n" \
"}\n" \
"\n" \
"/*__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 2048 / 4\n" \
"	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd2048p(__global uint4 * restrict const z, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(512, Zi512, 512, zk, w, j / 512);\n" \
"	forward_4(128, Zi128, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(512, zk, Z4);\n" \
"}*/\n" \
"\n" \
"// -----------------\n" \
"/*\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void mul32(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	__global const uint4 * restrict const zpk = &zp[k32 + i32 + i8];\n" \
"\n" \
"	forward_4i(8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	mul_22(Z4, 8, zpk, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4o(8, zk, 8, Zi8, w, j / 8, ji / 8);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void mul64(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	__global const uint4 * restrict const zpk = &zp[k64 + i64 + i16];\n" \
"\n" \
"	forward_4i(16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	mul_4(Z4, 16, zpk, w, j, ji);\n" \
"	backward_4(4, Zi4, w, j / 4, ji / 4);\n" \
"	backward_4o(16, zk, 16, Zi16, w, j / 16, ji / 16);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128\n" \
"	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void mul128(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	__global const uint4 * restrict const zpk = &zp[k128 + i128 + i32];\n" \
"\n" \
"	forward_4i(32, Zi32, 32, zk, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	mul_22(Z4, 32, zpk, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4(8, Zi8, w, j / 8, ji / 8);\n" \
"	backward_4o(32, zk, 32, Zi32, w, j / 32, ji / 32);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256\n" \
"	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void mul256(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	__global const uint4 * restrict const zpk = &zp[k256 + i256 + i64];\n" \
"\n" \
"	forward_4i(64, Zi64, 64, zk, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	mul_4(Z4, 64, zpk, w, j, ji);\n" \
"	backward_4(4, Zi4, w, j / 4, ji / 4);\n" \
"	backward_4(16, Zi16, w, j / 16, ji / 16);\n" \
"	backward_4o(64, zk, 64, Zi64, w, j / 64, ji / 64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 512 / 4\n" \
"	__attribute__((work_group_size_hint(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul512(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	__global const uint4 * restrict const zpk = &zp[k512 + i128];\n" \
"\n" \
"	forward_4i(128, Zi128, 128, zk, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	mul_22(Z4, 128, zpk, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4(8, Zi8, w, j / 8, ji / 8);\n" \
"	backward_4(32, Zi32, w, j / 32, ji / 32);\n" \
"	backward_4o(128, zk, 128, Zi128, w, j / 128, ji / 128);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 1024 / 4\n" \
"	__attribute__((work_group_size_hint(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul1024(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	__global const uint4 * restrict const zpk = &zp[k1024 + i256];\n" \
"\n" \
"	forward_4i(256, Zi256, 256, zk, w, j / 256);\n" \
"	forward_4(64, Zi64, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	mul_4(Z4, 256, zpk, w, j, ji);\n" \
"	backward_4(4, Zi4, w, j / 4, ji / 4);\n" \
"	backward_4(16, Zi16, w, j / 16, ji / 16);\n" \
"	backward_4(64, Zi64, w, j / 64, ji / 64);\n" \
"	backward_4o(256, zk, 256, Zi256, w, j / 256, ji / 256);\n" \
"}\n" \
"\n" \
"/*__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 2048 / 4\n" \
"	__attribute__((work_group_size_hint(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul2048(__global uint4 * restrict const z, __global const uint4 * restrict const zp, __global const uint2 * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	__global const uint4 * restrict const zpk = &zp[k2048 + i512];\n" \
"\n" \
"	forward_4i(512, Zi512, 512, zk, w, j / 512);\n" \
"	forward_4(128, Zi128, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	mul_22(Z4, 512, zpk, w, j);\n" \
"	backward_4(2, Zi2, w, j / 2, ji / 2);\n" \
"	backward_4(8, Zi8, w, j / 8, ji / 8);\n" \
"	backward_4(32, Zi32, w, j / 32, ji / 32);\n" \
"	backward_4(128, Zi128, w, j / 128, ji / 128);\n" \
"	backward_4o(512, zk, 512, Zi512, w, j / 512, ji / 512);\n" \
"}*/\n" \
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
"INLINE int64 garner2(const uint32 r1, const uint32 r2)\n" \
"{\n" \
"	const uint32 mfInvP2_P1 = 2130706177u;	// Montgomery form of 1 / P2 (mod P1)\n" \
"	const uint64 P1P2 = P1 * (uint64)(P2);\n" \
"	uint32 u12 = mulmod(submod(r1, r2, P1), mfInvP2_P1, (uint2)(P1, Q1));	// P2 < P1\n" \
"	const uint64 n = r2 + u12 * (uint64)(P2);\n" \
"	return (n > P1P2 / 2) ? (int64)(n - P1P2) : (int64)(n);\n" \
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
"	int64 f = 0;\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		const uint32 u1 = mulmod(zi[j], NORM1, (uint2)(P1, Q1));\n" \
"		const uint32 u2 = mulmod(zi[j + NSIZE], NORM2, (uint2)(P2, Q2));\n" \
"		int64 l = garner2(u1, u2);\n" \
"		if (sblk < 0) l += l;\n" \
"		f += l;\n" \
"		const int32 r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j] = set_int(r, P1); zi[j + NSIZE] = set_int(r, P2);\n" \
"\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	if (i == 0) f = -f;		// a_n = -a_0\n" \
"	c[i] = f;\n" \
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
"		zi[j] = set_int(r, P1); zi[j + NSIZE] = set_int(r, P2);\n" \
"\n" \
"		if (f == 0) return;\n" \
"		++j;\n" \
"	} while (j != blk - 1);\n" \
"\n" \
"	const int32 r = (int32)(f);\n" \
"	zi[blk - 1] = addmod(zi[blk - 1], set_int(r, P1), P1);\n" \
"	zi[blk - 1 + NSIZE] = addmod(zi[blk - 1 + NSIZE], set_int(r, P2), P2);\n" \
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
"		f += garner2(zi[j], zi[j + NSIZE]) * a;\n" \
"		const int32 r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j] = set_int(r, P1); zi[j + NSIZE] = set_int(r, P2);\n" \
"\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	if (i == 0) f = -f;		// a_n = -a_0\n" \
"	c[i] = f;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set(__global uint * restrict const z, const unsigned int a)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	z[idx] = ((idx & (NSIZE - 1)) == 0) ? a : 0;\n" \
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
