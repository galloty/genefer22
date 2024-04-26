/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel2 = \
"/*\n" \
"Copyright 2022, Yves Gallot\n" \
"\n" \
"genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"typedef uint	sz_t;\n" \
"\n" \
"#define P1P2	(P1 * (ulong)P2)\n" \
"\n" \
"// --- mod arith ---\n" \
"\n" \
"inline uint _addMod(const uint lhs, const uint rhs, const uint p)\n" \
"{\n" \
"	const uint c = (lhs >= p - rhs) ? p : 0;\n" \
"	return lhs + rhs - c;\n" \
"}\n" \
"\n" \
"inline uint _subMod(const uint lhs, const uint rhs, const uint p)\n" \
"{\n" \
"	const uint c = (lhs < rhs) ? p : 0;\n" \
"	return lhs - rhs + c;\n" \
"}\n" \
"\n" \
"// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.\n" \
"\n" \
"// Montgomery form (lhs, rhs and output): if 0 <= r < p then f is r * 2^32 mod p\n" \
"inline uint _mulMonty(const uint lhs, const uint rhs, const uint p, const uint q)\n" \
"{\n" \
"	const uint t_lo = lhs * rhs, t_hi = mul_hi(lhs, rhs);\n" \
"	const uint mp = mul_hi(t_lo * q, p);\n" \
"	return _subMod(t_hi, mp, p);\n" \
"}\n" \
"\n" \
"// Conversion into Montgomery form\n" \
"inline uint _toMonty(const uint n, const uint r2, const uint p, const uint q)\n" \
"{\n" \
"	// n * (2^32)^2 = (n * 2^32) * (1 * 2^32)\n" \
"	return _mulMonty(n, r2, p, q);\n" \
"}\n" \
"\n" \
"// Conversion out of Montgomery form\n" \
"// inline uint _fromMonty(const uint n, const uint p, const uint q)\n" \
"// {\n" \
"// 	// REDC(n * 2^32, 1)\n" \
"// 	const uint mp = mul_hi(n * q, p);\n" \
"// 	return (mp != 0) ? p - mp : 0;\n" \
"// }\n" \
"\n" \
"inline uint add_P1(const uint lhs, const uint rhs) { return _addMod(lhs, rhs, P1); }\n" \
"inline uint add_P2(const uint lhs, const uint rhs) { return _addMod(lhs, rhs, P2); }\n" \
"\n" \
"inline uint sub_P1(const uint lhs, const uint rhs) { return _subMod(lhs, rhs, P1); }\n" \
"inline uint sub_P2(const uint lhs, const uint rhs) { return _subMod(lhs, rhs, P2); }\n" \
"\n" \
"// Montgomery form\n" \
"inline uint mul_P1(const uint lhs, const uint rhs) { return _mulMonty(lhs, rhs, P1, Q1); }\n" \
"inline uint mul_P2(const uint lhs, const uint rhs) { return _mulMonty(lhs, rhs, P2, Q2); }\n" \
"\n" \
"inline uint toMonty_P1(const uint lhs) { return _toMonty(lhs, R1, P1, Q1); }\n" \
"inline uint toMonty_P2(const uint lhs) { return _toMonty(lhs, R2, P2, Q2); }\n" \
"\n" \
"// inline uint fromMonty_P1(const uint lhs) { return _fromMonty(lhs, P1, Q1); }\n" \
"// inline uint fromMonty_P2(const uint lhs) { return _fromMonty(lhs, P2, Q2); }\n" \
"\n" \
"inline int geti_P1(const uint r) { return (r > P1 / 2) ? (int)(r - P1) : (int)r; }\n" \
"\n" \
"inline long garner2(const uint r1, const uint r2)\n" \
"{\n" \
"	const uint u12 = mul_P1(sub_P1(r1, r2), InvP2_P1);\n" \
"	const ulong n = r2 + u12 * (ulong)P2;\n" \
"	return (n > P1P2 / 2) ? (long)(n - P1P2) : (long)n;\n" \
"}\n" \
"\n" \
"// --- RNS ---\n" \
"\n" \
"typedef uint2	RNS;\n" \
"typedef RNS		RNS_W;\n" \
"\n" \
"inline RNS toRNS(const int i) { return ((RNS)(i, i) + ((i < 0) ? (RNS)(P1, P2) : (RNS)(0, 0))); }\n" \
"\n" \
"inline RNS add(const RNS lhs, const RNS rhs) { return (RNS)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }\n" \
"inline RNS sub(const RNS lhs, const RNS rhs) { return (RNS)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }\n" \
"inline RNS mul(const RNS lhs, const RNS rhs) { return (RNS)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }\n" \
"\n" \
"inline RNS sqr(const RNS lhs) { return mul(lhs, lhs); }\n" \
"\n" \
"inline RNS mulW(const RNS lhs, const RNS_W w) { return mul(lhs, w); }\n" \
"\n" \
"inline RNS toMonty(const RNS lhs) { return (RNS)(toMonty_P1(lhs.s0), toMonty_P2(lhs.s1)); }\n" \
"\n" \
"// --- transform/inline ---\n" \
"\n" \
"inline void forward_4(const sz_t m, __local RNS * restrict const Z, __global const RNS_W * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const w_j = &w[j];\n" \
"	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * m], u2 = mulW(Z[2 * m], w1), u1 = Z[1 * m], u3 = mulW(Z[3 * m], w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);\n" \
"	Z[0 * m] = add(v0, v1); Z[1 * m] = sub(v0, v1); Z[2 * m] = add(v2, v3); Z[3 * m] = sub(v2, v3);\n" \
"}\n" \
"\n" \
"inline void forward_4i(const sz_t ml, __local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z, __global const RNS_W * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS z0 = z[0], z2 = z2mg[0], z1 = z[mg], z3 = z2mg[mg];\n" \
"	__global const RNS_W * restrict const w_j = &w[j];\n" \
"	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];\n" \
"	const RNS u0 = z0, u2 = mulW(z2, w1), u1 = z1, u3 = mulW(z3, w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);\n" \
"	Z[0 * ml] = add(v0, v1); Z[1 * ml] = sub(v0, v1); Z[2 * ml] = add(v2, v3); Z[3 * ml] = sub(v2, v3);\n" \
"}\n" \
"\n" \
"inline void forward_4i_0(const sz_t ml, __local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS z0 = z[0], z2 = z2mg[0], z1 = z[mg], z3 = z2mg[mg];\n" \
"	const RNS_W w1 = w[1], w2 = w[2], w3 = w[3];\n" \
"	const RNS u0 = toMonty(z0), u2 = mulW(z2, w1), u1 = toMonty(z1), u3 = mulW(z3, w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);\n" \
"	Z[0 * ml] = add(v0, v1); Z[1 * ml] = sub(v0, v1); Z[2 * ml] = add(v2, v3); Z[3 * ml] = sub(v2, v3);\n" \
"}\n" \
"\n" \
"inline void forward_4o(const sz_t mg, __global RNS * restrict const z, const sz_t ml, __local const RNS * restrict const Z, __global const RNS_W * restrict const w, const sz_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const w_j = &w[j];\n" \
"	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * ml], u2 = mulW(Z[2 * ml], w1), u1 = Z[1 * ml], u3 = mulW(Z[3 * ml], w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);\n" \
"	__global RNS * const z2mg = &z[2 * mg];\n" \
"	z[0] = add(v0, v1); z[mg] = sub(v0, v1); z2mg[0] = add(v2, v3); z2mg[mg] = sub(v2, v3);\n" \
"}\n" \
"\n" \
"inline void backward_4(const sz_t m, __local RNS * restrict const Z, __global const RNS_W * restrict const wi, const sz_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const wi_j = &wi[j];\n" \
"	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * m], u1 = Z[1 * m], u2 = Z[2 * m], u3 = Z[3 * m];\n" \
"	const RNS v0 = add(u0, u1), v1 = mulW(sub(u0, u1), wi2), v2 = add(u2, u3), v3 = mulW(sub(u2, u3), wi3);\n" \
"	Z[0 * m] = add(v0, v2); Z[2 * m] = mulW(sub(v0, v2), wi1); Z[1 * m] = add(v1, v3); Z[3 * m] = mulW(sub(v1, v3), wi1);\n" \
"}\n" \
"\n" \
"inline void backward_4i(const sz_t ml, __local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z, __global const RNS_W * restrict const wi,const sz_t j)\n" \
"{\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS u0 = z[0], u1 = z[mg], u2 = z2mg[0], u3 = z2mg[mg];\n" \
"	__global const RNS_W * restrict const wi_j = &wi[j];\n" \
"	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];\n" \
"	const RNS v0 = add(u0, u1), v1 = mulW(sub(u0, u1), wi2), v2 = add(u2, u3), v3 = mulW(sub(u2, u3), wi3);\n" \
"	Z[0 * ml] = add(v0, v2); Z[2 * ml] = mulW(sub(v0, v2), wi1); Z[1 * ml] = add(v1, v3); Z[3 * ml] = mulW(sub(v1, v3), wi1);\n" \
"}\n" \
"\n" \
"inline void backward_4o(const sz_t mg, __global RNS * restrict const z, const sz_t ml, __local const RNS * restrict const Z, __global const RNS_W * restrict const wi, const sz_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const wi_j = &wi[j];\n" \
"	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * ml], u1 = Z[1 * ml], u2 = Z[2 * ml], u3 = Z[3 * ml];\n" \
"	const RNS v0 = add(u0, u1), v1 = mulW(sub(u0, u1), wi2), v2 = add(u2, u3), v3 = mulW(sub(u2, u3), wi3);\n" \
"	__global RNS * const z2mg = &z[2 * mg];\n" \
"	z[0] = add(v0, v2); z2mg[0] = mulW(sub(v0, v2), wi1); z[mg] = add(v1, v3); z2mg[mg] = mulW(sub(v1, v3), wi1);\n" \
"}\n" \
"\n" \
"inline void write_4(const sz_t mg, __global RNS * restrict const z, __local const RNS * restrict const Z)\n" \
"{\n" \
"	__global RNS * const z2mg = &z[2 * mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];\n" \
"}\n" \
"\n" \
"inline void fwd2write_4(const sz_t mg, __global RNS * restrict const z, __local const RNS * restrict const Z, const RNS_W w1)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u2 = mulW(Z[2], w1), u1 = Z[1], u3 = mulW(Z[3], w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = add(u1, u3), v3 = sub(u1, u3);\n" \
"	__global RNS * const z2mg = &z[2 * mg];\n" \
"	z[0] = v0; z2mg[0] = v2; z[mg] = v1; z2mg[mg] = v3;\n" \
"}\n" \
"\n" \
"inline void square_22(__local RNS * restrict const Z, const RNS_W w0)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];\n" \
"	Z[0] = add(sqr(u0), sqr(mulW(u1, w0))); Z[1] = mul(add(u0, u0), u1);\n" \
"	Z[2] = sub(sqr(u2), sqr(mulW(u3, w0))); Z[3] = mul(add(u2, u2), u3);\n" \
"}\n" \
"\n" \
"inline void square_4(__local RNS * restrict const Z, const RNS_W w1, const RNS_W w1i, const RNS_W w0)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u2 = mulW(Z[2], w1), u1 = Z[1], u3 = mulW(Z[3], w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = add(u1, u3), v3 = sub(u1, u3);\n" \
"	const RNS s0 = add(sqr(v0), sqr(mulW(v1, w0))), s1 = mul(add(v0, v0), v1);\n" \
"	const RNS s2 = sub(sqr(v2), sqr(mulW(v3, w0))), s3 = mul(add(v2, v2), v3);\n" \
"	Z[0] = add(s0, s2); Z[2] = mulW(sub(s0, s2), w1i); Z[1] = add(s1, s3); Z[3] = mulW(sub(s1, s3), w1i);\n" \
"}\n" \
"\n" \
"inline void mul_22(__local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z, const RNS_W w0)\n" \
"{\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS u0p = z[0], u1p = z[mg], u2p = z2mg[0], u3p = z2mg[mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];\n" \
"	Z[0] = add(mul(u0, u0p), mul(mulW(u1, w0), mulW(u1p, w0)));\n" \
"	Z[1] = add(mul(u0, u1p), mul(u0p, u1));\n" \
"	Z[2] = sub(mul(u2, u2p), mul(mulW(u3, w0), mulW(u3p, w0)));\n" \
"	Z[3] = add(mul(u2, u3p), mul(u2p, u3));\n" \
"}\n" \
"\n" \
"inline void mul_4(__local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z, const RNS_W w1, const RNS_W w1i, const RNS_W w0)\n" \
"{\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS v0p = z[0], v1p = z[mg], v2p = z2mg[0], v3p = z2mg[mg];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u2 = mulW(Z[2], w1), u1 = Z[1], u3 = mulW(Z[3], w1);\n" \
"	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = add(u1, u3), v3 = sub(u1, u3);\n" \
"	const RNS s0 = add(mul(v0, v0p), mul(mulW(v1, w0), mulW(v1p, w0)));\n" \
"	const RNS s1 = add(mul(v0, v1p), mul(v0p, v1));\n" \
"	const RNS s2 = sub(mul(v2, v2p), mul(mulW(v3, w0), mulW(v3p, w0)));\n" \
"	const RNS s3 = add(mul(v2, v3p), mul(v2p, v3));\n" \
"	Z[0] = add(s0, s2); Z[2] = mulW(sub(s0, s2), w1i); Z[1] = add(s1, s3); Z[3] = mulW(sub(s1, s3), w1i);\n" \
"}\n" \
"\n" \
"// --- transform ---\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	__local RNS Z[4 * B_N * CHUNK_N]; \\\n" \
"	\\\n" \
"	/* threadIdx < B_N */ \\\n" \
"	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \\\n" \
"	__local RNS * const Zi = &Z[chunk_idx]; \\\n" \
"	\\\n" \
"	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \\\n" \
"	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \\\n" \
"	\\\n" \
"	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \\\n" \
"	\\\n" \
"	sz_t sj = s + idx_m;\n" \
"\n" \
"#define DECLARE_VAR_FORWARD() \\\n" \
"	__global RNS * __restrict__ const zi = &z[ki]; \\\n" \
"	__global RNS * __restrict__ const zo = &z[ko];\n" \
"\n" \
"#define DECLARE_VAR_BACKWARD() \\\n" \
"	__global RNS * __restrict__ const zi = &z[ko]; \\\n" \
"	__global RNS * __restrict__ const zo = &z[ki]; \\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0); \\\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"\n" \
"#define FORWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	forward_4i(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);\n" \
"\n" \
"#define FORWARD_I_0(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	forward_4i_0(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w);\n" \
"\n" \
"#define FORWARD_O(CHUNK_N) \\\n" \
"	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], w, sj / 1);\n" \
"\n" \
"#define BACKWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_BACKWARD(); \\\n" \
"	\\\n" \
"	backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sj / 1);\n" \
"\n" \
"#define BACKWARD_O(B_N, CHUNK_N) \\\n" \
"	backward_4o(B_N << lm, zo, B_N * CHUNK_N, &Z[i], wi, sj / B_N);\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_64	(64 / 4)\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"void forward64(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_64, CHUNK64);\n" \
"\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"void backward64(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_64, CHUNK64);\n" \
"\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], wi, sj / 4);\n" \
"\n" \
"	BACKWARD_O(B_64, CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"void forward64_0(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	const int lm = LNSIZE - 6; const unsigned int s = 64 / 4;\n" \
"\n" \
"	FORWARD_I_0(B_64, CHUNK64);\n" \
"\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK64);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_256	(256 / 4)\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"void forward256(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_256, CHUNK256);\n" \
"\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"void backward256(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_256, CHUNK256);\n" \
"\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sj / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sj / 16);\n" \
"\n" \
"	BACKWARD_O(B_256, CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"void forward256_0(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	const int lm = LNSIZE - 8; const unsigned int s = 256 / 4;\n" \
"\n" \
"	FORWARD_I_0(B_256, CHUNK256);\n" \
"\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK256);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_1024	(1024 / 4)\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"void forward1024(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_1024, CHUNK1024);\n" \
"\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK1024);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"void backward1024(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_1024, CHUNK1024);\n" \
"\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], wi, sj / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], wi, sj / 16);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);\n" \
"	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], wi, sj / 64);\n" \
"\n" \
"	BACKWARD_O(B_1024, CHUNK1024);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"void forward1024_0(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	const int lm = LNSIZE - 10; const unsigned int s = 1024 / 4;\n" \
"\n" \
"	FORWARD_I_0(B_1024, CHUNK1024);\n" \
"\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK1024);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local RNS Z[32 * BLK32]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k32 = (sz_t)get_group_id(0) * 32 * BLK32, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k32 + i32 + i8]; \\\n" \
"	__local RNS * const Z32 = &Z[i32]; \\\n" \
"	__local RNS * const Zi8 = &Z32[i8]; \\\n" \
"	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z32[i2]; \\\n" \
"	__local RNS * const Z4 = &Z32[4 * i8];\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"void square32(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4o(8, zk, 8, Zi8, wi, j / 8);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local RNS Z[64 * BLK64]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k64 = (sz_t)get_group_id(0) * 64 * BLK64, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k64 + i64 + i16]; \\\n" \
"	__local RNS * const Z64 = &Z[i64]; \\\n" \
"	__local RNS * const Zi16 = &Z64[i16]; \\\n" \
"	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \\\n" \
"	__local RNS * const Zi4 = &Z64[i4]; \\\n" \
"	__local RNS * const Z4 = &Z64[4 * i16];\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"void square64(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	__global const RNS_W * const wi = &w[4 * n_4];\n" \
"	square_4(Z4, w[j], wi[j], w[n_4 + j]);\n" \
"	backward_4(4, Zi4, wi, j / 4);\n" \
"	backward_4o(16, zk, 16, Zi16, wi, j / 16);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_128() \\\n" \
"	__local RNS Z[128 * BLK128]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k128 = (sz_t)get_group_id(0) * 128 * BLK128, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k128 + i128 + i32]; \\\n" \
"	__local RNS * const Z128 = &Z[i128]; \\\n" \
"	__local RNS * const Zi32 = &Z128[i32]; \\\n" \
"	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \\\n" \
"	__local RNS * const Zi8 = &Z128[i8]; \\\n" \
"	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z128[i2]; \\\n" \
"	__local RNS * const Z4 = &Z128[4 * i32];\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"void square128(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(32, Zi32, 32, zk, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4(8, Zi8, wi, j / 8);\n" \
"	backward_4o(32, zk, 32, Zi32, wi, j / 32);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_256() \\\n" \
"	__local RNS Z[256 * BLK256]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k256 = (sz_t)get_group_id(0) * 256 * BLK256, i = (sz_t)get_local_id(0); \\\n" \
"	const sz_t i256 = 0, i64 = i; \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k256 + i256 + i64]; \\\n" \
"	__local RNS * const Z256 = &Z[i256]; \\\n" \
"	__local RNS * const Zi64 = &Z256[i64]; \\\n" \
"	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \\\n" \
"	__local RNS * const Zi16 = &Z256[i16]; \\\n" \
"	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \\\n" \
"	__local RNS * const Zi4 = &Z256[i4]; \\\n" \
"	__local RNS * const Z4 = &Z256[4 * i64];\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"void square256(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(64, Zi64, 64, zk, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	square_4(Z4, w[j], wi[j], w[n_4 + j]);\n" \
"	backward_4(4, Zi4, wi, j / 4);\n" \
"	backward_4(16, Zi16, wi, j / 16);\n" \
"	backward_4o(64, zk, 64, Zi64, wi, j / 64);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_512() \\\n" \
"	__local RNS Z[512]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k512 = (sz_t)get_group_id(0) * 512, i128 = (sz_t)get_local_id(0); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k512 + i128]; \\\n" \
"	__local RNS * const Zi128 = &Z[i128]; \\\n" \
"	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \\\n" \
"	__local RNS * const Zi32 = &Z[i32]; \\\n" \
"	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \\\n" \
"	__local RNS * const Zi8 = &Z[i8]; \\\n" \
"	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z[i2]; \\\n" \
"	__local RNS * const Z4 = &Z[4 * i128];\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"void square512(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(128, Zi128, 128, zk, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4(8, Zi8, wi, j / 8);\n" \
"	backward_4(32, Zi32, wi, j / 32);\n" \
"	backward_4o(128, zk, 128, Zi128, wi, j / 128);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_1024() \\\n" \
"	__local RNS Z[1024]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k1024 = (sz_t)get_group_id(0) * 1024, i256 = (sz_t)get_local_id(0); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k1024 + i256]; \\\n" \
"	__local RNS * const Zi256 = &Z[i256]; \\\n" \
"	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \\\n" \
"	__local RNS * const Zi64 = &Z[i64]; \\\n" \
"	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \\\n" \
"	__local RNS * const Zi16 = &Z[i16]; \\\n" \
"	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \\\n" \
"	__local RNS * const Zi4 = &Z[i4]; \\\n" \
"	__local RNS * const Z4 = &Z[4 * i256];\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"void square1024(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(256, Zi256, 256, zk, w, j / 256);\n" \
"	forward_4(64, Zi64, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	square_4(Z4, w[j], wi[j], w[n_4 + j]);\n" \
"	backward_4(4, Zi4, wi, j / 4);\n" \
"	backward_4(16, Zi16, wi, j / 16);\n" \
"	backward_4(64, Zi64, wi, j / 64);\n" \
"	backward_4o(256, zk, 256, Zi256, wi, j / 256);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_2048() \\\n" \
"	__local RNS Z[2048]; \\\n" \
"	\\\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \\\n" \
"	\\\n" \
"	const sz_t k2048 = (sz_t)get_group_id(0) * 2048, i512 = (sz_t)get_local_id(0); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k2048 + i512]; \\\n" \
"	__local RNS * const Zi512 = &Z[i512]; \\\n" \
"	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \\\n" \
"	__local RNS * const Zi128 = &Z[i128]; \\\n" \
"	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \\\n" \
"	__local RNS * const Zi32 = &Z[i32]; \\\n" \
"	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \\\n" \
"	__local RNS * const Zi8 = &Z[i8]; \\\n" \
"	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z[i2]; \\\n" \
"	__local RNS * const Z4 = &Z[4 * i512];\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"void square2048(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(512, Zi512, 512, zk, w, j / 512);\n" \
"	forward_4(128, Zi128, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	square_22(Z4, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4(8, Zi8, wi, j / 8);\n" \
"	backward_4(32, Zi32, wi, j / 32);\n" \
"	backward_4(128, Zi128, wi, j / 128);\n" \
"	backward_4o(512, zk, 512, Zi512, wi, j / 512);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"void fwd32p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(8, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"void fwd64p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	fwd2write_4(16, zk, Z4, w[j]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"void fwd128p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(32, Zi32, 32, zk, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(32, zk, Z4);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"void fwd256p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(64, Zi64, 64, zk, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	fwd2write_4(64, zk, Z4, w[j]);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"void fwd512p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
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
"__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"void fwd1024p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(256, Zi256, 256, zk, w, j / 256);\n" \
"	forward_4(64, Zi64, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	fwd2write_4(256, zk, Z4, w[j]);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"void fwd2048p(__global RNS * restrict const z, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(512, Zi512, 512, zk, w, j / 512);\n" \
"	forward_4(128, Zi128, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	write_4(512, zk, Z4);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"void mul32(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(8, Zi8, 8, zk, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	__global const RNS * restrict const zpk = &zp[k32 + i32 + i8];\n" \
"	mul_22(Z4, 8, zpk, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4o(8, zk, 8, Zi8, wi, j / 8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"void mul64(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(16, Zi16, 16, zk, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	__global const RNS_W * const wi = &w[4 * n_4];\n" \
"	__global const RNS * restrict const zpk = &zp[k64 + i64 + i16];\n" \
"	mul_4(Z4, 16, zpk, w[j], wi[j], w[n_4 + j]);\n" \
"	backward_4(4, Zi4, wi, j / 4);\n" \
"	backward_4o(16, zk, 16, Zi16, wi, j / 16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"void mul128(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(32, Zi32, 32, zk, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	__global const RNS * restrict const zpk = &zp[k128 + i128 + i32];\n" \
"	mul_22(Z4, 32, zpk, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4(8, Zi8, wi, j / 8);\n" \
"	backward_4o(32, zk, 32, Zi32, wi, j / 32);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"void mul256(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(64, Zi64, 64, zk, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	__global const RNS * restrict const zpk = &zp[k256 + i256 + i64];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	mul_4(Z4, 64, zpk, w[j], wi[j], w[n_4 + j]);\n" \
"	backward_4(4, Zi4, wi, j / 4);\n" \
"	backward_4(16, Zi16, wi, j / 16);\n" \
"	backward_4o(64, zk, 64, Zi64, wi, j / 64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"void mul512(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(128, Zi128, 128, zk, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	__global const RNS * restrict const zpk = &zp[k512 + i128];\n" \
"	mul_22(Z4, 128, zpk, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4(8, Zi8, wi, j / 8);\n" \
"	backward_4(32, Zi32, wi, j / 32);\n" \
"	backward_4o(128, zk, 128, Zi128, wi, j / 128);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"void mul1024(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(256, Zi256, 256, zk, w, j / 256);\n" \
"	forward_4(64, Zi64, w, j / 64);\n" \
"	forward_4(16, Zi16, w, j / 16);\n" \
"	forward_4(4, Zi4, w, j / 4);\n" \
"	__global const RNS * restrict const zpk = &zp[k1024 + i256];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	mul_4(Z4, 256, zpk, w[j], wi[j], w[n_4 + j]);\n" \
"	backward_4(4, Zi4, wi, j / 4);\n" \
"	backward_4(16, Zi16, wi, j / 16);\n" \
"	backward_4(64, Zi64, wi, j / 64);\n" \
"	backward_4o(256, zk, 256, Zi256, wi, j / 256);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"void mul2048(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(512, Zi512, 512, zk, w, j / 512);\n" \
"	forward_4(128, Zi128, w, j / 128);\n" \
"	forward_4(32, Zi32, w, j / 32);\n" \
"	forward_4(8, Zi8, w, j / 8);\n" \
"	forward_4(2, Zi2, w, j / 2);\n" \
"	__global const RNS * restrict const zpk = &zp[k2048 + i512];\n" \
"	mul_22(Z4, 512, zpk, w[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	backward_4(2, Zi2, wi, j / 2);\n" \
"	backward_4(8, Zi8, wi, j / 8);\n" \
"	backward_4(32, Zi32, wi, j / 32);\n" \
"	backward_4(128, Zi128, wi, j / 128);\n" \
"	backward_4o(512, zk, 512, Zi512, wi, j / 512);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"inline uint barrett(const ulong a, const uint b, const uint b_inv, const int b_s, uint * a_p)\n" \
"{\n" \
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
"	const uint d = mul_hi((uint)(a >> b_s), b_inv), r = (uint)a - d * b;\n" \
"	const bool o = (r >= b);\n" \
"	*a_p = o ? d + 1 : d;\n" \
"	return o ? r - b : r;\n" \
"}\n" \
"\n" \
"inline int reduce64(long * f, const uint b, const uint b_inv, const int b_s)\n" \
"{\n" \
"	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32\n" \
"	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b\n" \
"	const ulong t = abs(*f);\n" \
"	const ulong t_h = t >> 29;\n" \
"	const uint t_l = (uint)t & ((1u << 29) - 1);\n" \
"\n" \
"	uint d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint d_l, r_l = barrett(((ulong)r_h << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const ulong d = ((ulong)d_h << 29) | d_l;\n" \
"\n" \
"	const bool s = (*f < 0);\n" \
"	*f = s ? -(long)d : (long)d;\n" \
"	return s ? -(int)r_l : (int)r_l;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize1(__global RNS * restrict const z, __global long * restrict const c,\n" \
"	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	const unsigned int blk = abs(sblk);\n" \
"	__global RNS * restrict const zi = &z[blk * idx];\n" \
"\n" \
"	prefetch(zi, (size_t)blk);\n" \
"\n" \
"	// Not converted into Montgomery form such that output is converted out of Montgomery form\n" \
"	const RNS norm = (RNS)(NORM1, NORM2);\n" \
"\n" \
"	long f = 0;\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		const RNS zj = mul(zi[j], norm);\n" \
"		long l = garner2(zj.s0, zj.s1);\n" \
"		if (sblk < 0) l += l;\n" \
"		f += l;\n" \
"\n" \
"		const int r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j] = toRNS(r);\n" \
"\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	c[i] = (i == 0) ? -f : f;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul1(__global RNS * restrict const z, __global long * restrict const c,\n" \
"	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const zi = &z[blk * idx];\n" \
"\n" \
"	prefetch(zi, (size_t)blk);\n" \
"\n" \
"	long f = 0;\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		f += geti_P1(zi[j].s0) * (long)a;\n" \
"		const int r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j] = toRNS(r);\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);\n" \
"	c[i] = (i == 0) ? -f : f;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2(__global RNS * restrict const z, __global const long * restrict const c, \n" \
"	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const zi = &z[blk * idx];\n" \
"\n" \
"	long f = c[idx];\n" \
"\n" \
"	sz_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		f += geti_P1(zi[j].s0);\n" \
"		const int r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j] = toRNS(r);\n" \
"		if (f == 0) return;\n" \
"		++j;\n" \
"	} while (j != blk - 1);\n" \
"\n" \
"	const int r = (int)f;\n" \
"	zi[blk - 1] = add(zi[blk - 1], toRNS(r));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set(__global RNS * restrict const z, const int a)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	z[idx] = (idx == 0) ? toRNS(a) : (RNS)(0, 0);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global RNS * restrict const z, const unsigned int dst, const unsigned int src)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	z[dst + idx] = z[src + idx];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copyp(__global RNS * restrict const zp, __global const RNS * restrict const z, const unsigned int src)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	zp[idx] = z[src + idx];\n" \
"}\n" \
"";
