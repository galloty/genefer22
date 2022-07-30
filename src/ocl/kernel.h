/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2022, Yves Gallot\n" \
"\n" \
"genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" \
"\n" \
"typedef uint2	RNS;\n" \
"typedef uint4	RNS_W;\n" \
"typedef uint	RNSe;\n" \
"typedef uint2	RNS_We;\n" \
"\n" \
"typedef struct\n" \
"{\n" \
"	ulong lo;\n" \
"	uint hi;\n" \
"} uint96;\n" \
"\n" \
"typedef struct\n" \
"{\n" \
"	ulong lo;\n" \
"	int hi;\n" \
"} int96;\n" \
"\n" \
"int96 MAdd96_64_32_64(const ulong a, const uint b, const ulong c)\n" \
"{\n" \
"	int96 r; r.lo = a * b + c; r.hi = (int)mul_hi(a, (ulong)b) + (r.lo < c);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"int96 Add96(const int96 lhs, const int96 rhs)\n" \
"{\n" \
"	int96 r; r.lo = lhs.lo + rhs.lo; r.hi = lhs.hi + rhs.hi;\n" \
"	r.hi += (r.lo < rhs.lo);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"int96 Sub96(const int96 lhs, const int96 rhs)\n" \
"{\n" \
"	int96 r; r.lo = lhs.lo - rhs.lo; r.hi = lhs.hi - rhs.hi;\n" \
"	r.hi -= (lhs.lo < rhs.lo);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"int96 Mul96_96_u32(const int96 lhs, const int rhs)\n" \
"{\n" \
"	int96 r; r.lo = lhs.lo * (ulong)rhs; r.hi = (int)mul_hi(lhs.lo, (ulong)rhs);\n" \
"	r.hi += lhs.hi * rhs;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"int96 Mul96_64_u32(const long lhs, const int rhs)\n" \
"{\n" \
"	int96 r; r.lo = lhs * (long)rhs; r.hi = (int)mul_hi(lhs, (long)rhs);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"long MulHi64_96_u64(const int96 lhs, const long rhs)\n" \
"{\n" \
"	return (long)mul_hi(lhs.lo, (ulong)rhs) + lhs.hi * rhs;\n" \
"}\n" \
"\n" \
"#define	P1			2130706433u		// 127 * 2^24 + 1\n" \
"#define	P2			2013265921u		// 15 * 2^27 + 1\n" \
"#define	P3			1711276033u		// 51 * 2^25 + 1\n" \
"#define	R1			2164392967u		// 2^62 / P1\n" \
"#define	R2			2290649223u		// 2^62 / P2\n" \
"#define	R3			2694881439u		// 2^62 / P3\n" \
"#define	InvP2_P1	 913159918u		// 1 / P2 mod P1\n" \
"#define	InvP2_P1p	1840700306u		// (InvP2_P1 * 2^32) / P1\n" \
"#define	InvP3_P2	 671088647u		// 1 / P3 mod P2\n" \
"#define	InvP3_P2p	1431655779u		// (InvP3_P2 * 2^32) / P2\n" \
"#define	InvP2P3_P1	1059265576u		// 1 / (P2*P3) mod P1\n" \
"#define	InvP2P3_P1p	2135212498u		// (InvP2P3_P1 * 2^32) / P1\n" \
"\n" \
"#define	P1P2	(P1 * (ulong)P2)\n" \
"#define	P2P3	(P2 * (ulong)P3)\n" \
"\n" \
"#define	P1P2P3hi	397946880				// P1 * P2 * P3 / 2^64\n" \
"#define	P1P2P3lo	11381159214173913089ul	// P1 * P2 * P3 mod 2^64\n" \
"\n" \
"/*\n" \
"Barrett's product/reduction, where P is such that h (the number of iterations in the 'while loop') is 0 or 1.\n" \
"\n" \
"Let m < P^2 < 2^62, R = [2^62/P] and h = [m/P] - [[m/2^30] R / 2^32].\n" \
"\n" \
"We have h = ([m/P] - m/P) + m/2^62 (2^62/P - R) + R/2^32 (m/2^30 - [m/2^30]) + ([m/2^30] R / 2^32 - [[m/2^30] * R / 2^32]).\n" \
"Then -1 + 0 + 0 + 0 < h < 0 + (2^62/P - R) + R/2^32 + 1,\n" \
"0 <= h < 1 + (2^62/P - R) + R/2^32.\n" \
"\n" \
"P = 127 * 2^24 + 1 = 2130706433 => R = 2164392967, h < 1.56\n" \
"P =  63 * 2^25 + 1 = 2113929217 => R = 2181570688, h < 2.51 NOK\n" \
"P =  15 * 2^27 + 1 = 2013265921 => R = 2290649223, h < 1.93\n" \
"P =  27 * 2^26 + 1 = 1811939329 => R = 2545165803, h < 2.23 NOK\n" \
"P =  51 * 2^25 + 1 = 1711276033 => R = 2694881439, h < 1.69\n" \
"*/\n" \
"\n" \
"inline uint MulMod_P1(const uint lhs, const uint rhs)\n" \
"{\n" \
"	const ulong m = lhs * (ulong)rhs;\n" \
"	const uint q = mul_hi((uint)(m >> 30), R1);\n" \
"	uint r = (uint)m - q * P1;\n" \
"	if (r >= P1) r -= P1;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint MulMod_P2(const uint lhs, const uint rhs)\n" \
"{\n" \
"	const ulong m = lhs * (ulong)rhs;\n" \
"	const uint q = mul_hi((uint)(m >> 30), R2);\n" \
"	uint r = (uint)m - q * P2;\n" \
"	if (r >= P2) r -= P2;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint MulMod_P3(const uint lhs, const uint rhs)\n" \
"{\n" \
"	const ulong m = lhs * (ulong)rhs;\n" \
"	const uint q = mul_hi((uint)(m >> 30), R3);\n" \
"	uint r = (uint)m - q * P3;\n" \
"	if (r >= P3) r -= P3;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// Shoup's modular multiplication: Faster arithmetic for number-theoretic transforms, David Harvey, J.Symb.Comp. 60 (2014) 113-119\n" \
"\n" \
"inline uint MulConstMod_P1(const uint lhs, const uint c, const uint cp)\n" \
"{\n" \
"	uint r = lhs * c - mul_hi(lhs, cp) * P1;\n" \
"	if (r >= P1) r -= P1;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint MulConstMod_P2(const uint lhs, const uint c, const uint cp)\n" \
"{\n" \
"	uint r = lhs * c - mul_hi(lhs, cp) * P2;\n" \
"	if (r >= P2) r -= P2;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint MulConstMod_P3(const uint lhs, const uint c, const uint cp)\n" \
"{\n" \
"	uint r = lhs * c - mul_hi(lhs, cp) * P3;\n" \
"	if (r >= P3) r -= P3;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// ---------------------------------------------\n" \
"// Garner Algorithm\n" \
"\n" \
"inline int96 GetLong_e(const RNS lhs, const RNSe lhse)\n" \
"{\n" \
"	uint d2 = lhs.s1 - lhse; if (lhs.s1 < lhse) d2 += P2;	// mod P2\n" \
"	const uint u2 = MulConstMod_P2(d2, InvP3_P2, InvP3_P2p);\n" \
"	const ulong n2 = lhse + u2 * (ulong)P3;\n" \
"\n" \
"	const uint n1 = (uint)(n2 % P1);\n" \
"	uint d1 = lhs.s0 - n1; if (lhs.s0 < n1) d1 += P1;		// mod P1\n" \
"	const uint u1 = MulConstMod_P1(d1, InvP2P3_P1, InvP2P3_P1p);\n" \
"\n" \
"	int96 r = MAdd96_64_32_64(P2P3, u1, n2);\n" \
"\n" \
"	if (r.hi > P1P2P3hi / 2)\n" \
"	{\n" \
"		int96 P1P2P3; P1P2P3.lo = P1P2P3lo; P1P2P3.hi = P1P2P3hi;\n" \
"		r = Sub96(r, P1P2P3);\n" \
"	}\n" \
"\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// ---------------------------------------------\n" \
"\n" \
"inline RNS ToRNS(const int i) { return (RNS)(i, i) + ((i < 0) ? (RNS)(P1, P2) : (RNS)(0, 0)); }\n" \
"\n" \
"inline int GetInt(const RNS lhs) { return (lhs.s0 >= P1 / 2) ? lhs.s0 - P1 : lhs.s0; }\n" \
"\n" \
"inline RNS Add(const RNS lhs, const RNS rhs)\n" \
"{\n" \
"	RNS r = lhs + rhs;\n" \
"	if (r.s0 >= P1) r.s0 -= P1;\n" \
"	if (r.s1 >= P2) r.s1 -= P2;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline RNS Sub(const RNS lhs, const RNS rhs)\n" \
"{\n" \
"	RNS r = lhs - rhs;\n" \
"	if (lhs.s0 < rhs.s0) r.s0 += P1;\n" \
"	if (lhs.s1 < rhs.s1) r.s1 += P2;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline RNS Mul(const RNS lhs, const RNS rhs)\n" \
"{\n" \
"	return (RNS)(MulMod_P1(lhs.s0, rhs.s0), MulMod_P2(lhs.s1, rhs.s1));\n" \
"}\n" \
"\n" \
"inline RNS Sqr(const RNS lhs) { return Mul(lhs, lhs); }\n" \
"\n" \
"inline RNS MulW(const RNS lhs, const RNS_W w)\n" \
"{\n" \
"	return (RNS)(MulConstMod_P1(lhs.s0, w.s0, w.s2), MulConstMod_P2(lhs.s1, w.s1, w.s3));\n" \
"}\n" \
"\n" \
"// ---------------------------------------------\n" \
"\n" \
"inline RNSe ToRNSe(const int i) { return (RNSe)(i) + ((i < 0) ? (RNSe)(P3) : (RNSe)(0)); }\n" \
"\n" \
"inline RNSe Adde(const RNSe lhs, const RNSe rhs)\n" \
"{\n" \
"	RNSe r = lhs + rhs;\n" \
"	if (r >= P3) r -= P3;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline RNSe Sube(const RNSe lhs, const RNSe rhs)\n" \
"{\n" \
"	RNSe r = lhs - rhs;\n" \
"	if (lhs < rhs) r += P3;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline RNSe Mule(const RNSe lhs, const RNSe rhs)\n" \
"{\n" \
"	return (RNSe)(MulMod_P3(lhs, rhs));\n" \
"}\n" \
"\n" \
"inline RNSe Sqre(const RNSe lhs) { return Mule(lhs, lhs); }\n" \
"\n" \
"inline RNSe MulWe(const RNSe lhs, const RNS_We w)\n" \
"{\n" \
"	return (RNSe)(MulConstMod_P3(lhs, w.s0, w.s1));\n" \
"}\n" \
"\n" \
"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline void Forward_4(const size_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const size_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const w_j = &w[j];\n" \
"	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];\n" \
"	__global const RNS_We * restrict const we_j = &we[j];\n" \
"	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * m], u2 = MulW(Z[2 * m], w1), u1 = Z[1 * m], u3 = MulW(Z[3 * m], w1);\n" \
"	const RNSe u0e = Ze[0 * m], u2e = MulWe(Ze[2 * m], w1e), u1e = Ze[1 * m], u3e = MulWe(Ze[3 * m], w1e);\n" \
"	const RNS v0 = Add(u0, u2), v2 = Sub(u0, u2), v1 = MulW(Add(u1, u3), w2), v3 = MulW(Sub(u1, u3), w3);\n" \
"	const RNSe v0e = Adde(u0e, u2e), v2e = Sube(u0e, u2e), v1e = MulWe(Adde(u1e, u3e), w2e), v3e = MulWe(Sube(u1e, u3e), w3e);\n" \
"	Z[0 * m] = Add(v0, v1); Z[1 * m] = Sub(v0, v1); Z[2 * m] = Add(v2, v3); Z[3 * m] = Sub(v2, v3);\n" \
"	Ze[0 * m] = Adde(v0e, v1e); Ze[1 * m] = Sube(v0e, v1e); Ze[2 * m] = Adde(v2e, v3e); Ze[3 * m] = Sube(v2e, v3e);\n" \
"}\n" \
"\n" \
"inline void Forward_4i(const size_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const size_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const size_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const w_j = &w[j];\n" \
"	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];\n" \
"	__global const RNS_We * restrict const we_j = &we[j];\n" \
"	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS u0 = z[0], u2 = MulW(z2mg[0], w1), u1 = z[mg], u3 = MulW(z2mg[mg], w1);\n" \
"	__global const RNSe * const z2mge = &ze[2 * mg];\n" \
"	const RNSe u0e = ze[0], u2e = MulWe(z2mge[0], w1e), u1e = ze[mg], u3e = MulWe(z2mge[mg], w1e);\n" \
"	const RNS v0 = Add(u0, u2), v2 = Sub(u0, u2), v1 = MulW(Add(u1, u3), w2), v3 = MulW(Sub(u1, u3), w3);\n" \
"	const RNSe v0e = Adde(u0e, u2e), v2e = Sube(u0e, u2e), v1e = MulWe(Adde(u1e, u3e), w2e), v3e = MulWe(Sube(u1e, u3e), w3e);\n" \
"	Z[0 * ml] = Add(v0, v1); Z[1 * ml] = Sub(v0, v1); Z[2 * ml] = Add(v2, v3); Z[3 * ml] = Sub(v2, v3);\n" \
"	Ze[0 * ml] = Adde(v0e, v1e); Ze[1 * ml] = Sube(v0e, v1e); Ze[2 * ml] = Adde(v2e, v3e); Ze[3 * ml] = Sube(v2e, v3e);\n" \
"}\n" \
"\n" \
"inline void Forward_4o(const size_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	const size_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const size_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const w_j = &w[j];\n" \
"	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];\n" \
"	__global const RNS_We * restrict const we_j = &we[j];\n" \
"	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * ml], u2 = MulW(Z[2 * ml], w1), u1 = Z[1 * ml], u3 = MulW(Z[3 * ml], w1);\n" \
"	const RNSe u0e = Ze[0 * ml], u2e = MulWe(Ze[2 * ml], w1e), u1e = Ze[1 * ml], u3e = MulWe(Ze[3 * ml], w1e);\n" \
"	const RNS v0 = Add(u0, u2), v2 = Sub(u0, u2), v1 = MulW(Add(u1, u3), w2), v3 = MulW(Sub(u1, u3), w3);\n" \
"	const RNSe v0e = Adde(u0e, u2e), v2e = Sube(u0e, u2e), v1e = MulWe(Adde(u1e, u3e), w2e), v3e = MulWe(Sube(u1e, u3e), w3e);\n" \
"	__global RNS * const z2mg = &z[2 * mg];\n" \
"	z[0] = Add(v0, v1); z[mg] = Sub(v0, v1); z2mg[0] = Add(v2, v3); z2mg[mg] = Sub(v2, v3);\n" \
"	__global RNSe * const z2mge = &ze[2 * mg];\n" \
"	ze[0] = Adde(v0e, v1e); ze[mg] = Sube(v0e, v1e); z2mge[0] = Adde(v2e, v3e); z2mge[mg] = Sube(v2e, v3e);\n" \
"}\n" \
"\n" \
"inline void Backward_4(const size_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const size_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const wi_j = &wi[j];\n" \
"	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];\n" \
"	__global const RNS_We * restrict const wie_j = &wie[j];\n" \
"	const RNS_We wi1e = wie_j[0], wi2e = wie_j[j], wi3e = wie_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * m], u1 = Z[1 * m], u2 = Z[2 * m], u3 = Z[3 * m];\n" \
"	const RNSe u0e = Ze[0 * m], u1e = Ze[1 * m], u2e = Ze[2 * m], u3e = Ze[3 * m];\n" \
"	const RNS v0 = Add(u0, u1), v1 = MulW(Sub(u0, u1), wi2), v2 = Add(u2, u3), v3 = MulW(Sub(u2, u3), wi3);\n" \
"	const RNSe v0e = Adde(u0e, u1e), v1e = MulWe(Sube(u0e, u1e), wi2e), v2e = Adde(u2e, u3e), v3e = MulWe(Sube(u2e, u3e), wi3e);\n" \
"	Z[0 * m] = Add(v0, v2); Z[2 * m] = MulW(Sub(v0, v2), wi1); Z[1 * m] = Add(v1, v3); Z[3 * m] = MulW(Sub(v1, v3), wi1);\n" \
"	Ze[0 * m] = Adde(v0e, v2e); Ze[2 * m] = MulWe(Sube(v0e, v2e), wi1e); Ze[1 * m] = Adde(v1e, v3e); Ze[3 * m] = MulWe(Sube(v1e, v3e), wi1e);\n" \
"}\n" \
"\n" \
"inline void Backward_4i(const size_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const size_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const size_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const wi_j = &wi[j];\n" \
"	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];\n" \
"	__global const RNS_We * restrict const wie_j = &wie[j];\n" \
"	const RNS_We wi1e = wie_j[0], wi2e = wie_j[j], wi3e = wie_j[j + 1];\n" \
"	__global const RNS * const z2mg = &z[2 * mg];\n" \
"	const RNS u0 = z[0], u1 = z[mg], u2 = z2mg[0], u3 = z2mg[mg];\n" \
"	__global const RNSe * const z2mge = &ze[2 * mg];\n" \
"	const RNSe u0e = ze[0], u1e = ze[mg], u2e = z2mge[0], u3e = z2mge[mg];\n" \
"	const RNS v0 = Add(u0, u1), v1 = MulW(Sub(u0, u1), wi2), v2 = Add(u2, u3), v3 = MulW(Sub(u2, u3), wi3);\n" \
"	const RNSe v0e = Adde(u0e, u1e), v1e = MulWe(Sube(u0e, u1e), wi2e), v2e = Adde(u2e, u3e), v3e = MulWe(Sube(u2e, u3e), wi3e);\n" \
"	Z[0 * ml] = Add(v0, v2); Z[2 * ml] = MulW(Sub(v0, v2), wi1); Z[1 * ml] = Add(v1, v3); Z[3 * ml] = MulW(Sub(v1, v3), wi1);\n" \
"	Ze[0 * ml] = Adde(v0e, v2e); Ze[2 * ml] = MulWe(Sube(v0e, v2e), wi1e); Ze[1 * ml] = Adde(v1e, v3e); Ze[3 * ml] = MulWe(Sube(v1e, v3e), wi1e);\n" \
"}\n" \
"\n" \
"inline void Backward_4o(const size_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	const size_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const size_t j)\n" \
"{\n" \
"	__global const RNS_W * restrict const wi_j = &wi[j];\n" \
"	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];\n" \
"	__global const RNS_We * restrict const wie_j = &wie[j];\n" \
"	const RNS_We wi1e = wie_j[0], wi2e = wie_j[j], wi3e = wie_j[j + 1];\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0 * ml], u1 = Z[1 * ml], u2 = Z[2 * ml], u3 = Z[3 * ml];\n" \
"	const RNSe u0e = Ze[0 * ml], u1e = Ze[1 * ml], u2e = Ze[2 * ml], u3e = Ze[3 * ml];\n" \
"	const RNS v0 = Add(u0, u1), v1 = MulW(Sub(u0, u1), wi2), v2 = Add(u2, u3), v3 = MulW(Sub(u2, u3), wi3);\n" \
"	const RNSe v0e = Adde(u0e, u1e), v1e = MulWe(Sube(u0e, u1e), wi2e), v2e = Adde(u2e, u3e), v3e = MulWe(Sube(u2e, u3e), wi3e);\n" \
"	__global RNS * const z2mg = &z[2 * mg];\n" \
"	z[0] = Add(v0, v2); z2mg[0] = MulW(Sub(v0, v2), wi1); z[mg] = Add(v1, v3); z2mg[mg] = MulW(Sub(v1, v3), wi1);\n" \
"	__global RNSe * const z2mge = &ze[2 * mg];\n" \
"	ze[0] = Adde(v0e, v2e); z2mge[0] = MulWe(Sube(v0e, v2e), wi1e); ze[mg] = Adde(v1e, v3e); z2mge[mg] = MulWe(Sube(v1e, v3e), wi1e);\n" \
"}\n" \
"\n" \
"inline void Square_22_e(__local RNS * restrict const Z, __local RNSe * restrict const Ze, const RNS_W w0, const RNS_We w0e)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];\n" \
"	const RNSe u0e = Ze[0], u1e = Ze[1], u2e = Ze[2], u3e = Ze[3];\n" \
"	Z[0] = Add(Sqr(u0), Sqr(MulW(u1, w0))); Ze[0] = Adde(Sqre(u0e), Sqre(MulWe(u1e, w0e)));\n" \
"	Z[1] = Mul(Add(u0, u0), u1); Ze[1] = Mule(Adde(u0e, u0e), u1e);\n" \
"	Z[2] = Sub(Sqr(u2), Sqr(MulW(u3, w0))); Ze[2] = Sube(Sqre(u2e), Sqre(MulWe(u3e, w0e)));\n" \
"	Z[3] = Mul(Add(u2, u2), u3); Ze[3] = Mule(Adde(u2e, u2e), u3e);\n" \
"}\n" \
"\n" \
"inline void Square_4_e(__local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const RNS_W w1, const RNS_W w1i, const RNS_W w0,  const RNS_We w1e, const RNS_We w1ie, const RNS_We w0e)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	const RNS u0 = Z[0], u2 = MulW(Z[2], w1), u1 = Z[1], u3 = MulW(Z[3], w1);\n" \
"	const RNSe u0e = Ze[0], u2e = MulWe(Ze[2], w1e), u1e = Ze[1], u3e = MulWe(Ze[3], w1e);\n" \
"	const RNS v0 = Add(u0, u2), v2 = Sub(u0, u2), v1 = Add(u1, u3), v3 = Sub(u1, u3);\n" \
"	const RNSe v0e = Adde(u0e, u2e), v2e = Sube(u0e, u2e), v1e = Adde(u1e, u3e), v3e = Sube(u1e, u3e);\n" \
"	const RNS s0 = Add(Sqr(v0), Sqr(MulW(v1, w0))), s1 = Mul(Add(v0, v0), v1);\n" \
"	const RNSe s0e = Adde(Sqre(v0e), Sqre(MulWe(v1e, w0e))), s1e = Mule(Adde(v0e, v0e), v1e);\n" \
"	const RNS s2 = Sub(Sqr(v2), Sqr(MulW(v3, w0))), s3 = Mul(Add(v2, v2), v3);\n" \
"	const RNSe s2e = Sube(Sqre(v2e), Sqre(MulWe(v3e, w0e))), s3e = Mule(Adde(v2e, v2e), v3e);\n" \
"	Z[0] = Add(s0, s2); Z[2] = MulW(Sub(s0, s2), w1i); Z[1] = Add(s1, s3); Z[3] = MulW(Sub(s1, s3), w1i);\n" \
"	Ze[0] = Adde(s0e, s2e); Ze[2] = MulWe(Sube(s0e, s2e), w1ie); Ze[1] = Adde(s1e, s3e); Ze[3] = MulWe(Sube(s1e, s3e), w1ie);\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" \
"\n" \
"#define BLK32	8\n" \
"#define BLK64	4\n" \
"#define BLK128	2\n" \
"#define BLK256	1\n" \
"\n" \
"#define CHUNK64		8\n" \
"#define CHUNK256	4\n" \
"#define CHUNK1024	2\n" \
"\n" \
"// ---------------------------------------------\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	__local RNS Z[4 * B_N * CHUNK_N]; \\\n" \
"	__local RNSe Ze[4 * B_N * CHUNK_N]; \\\n" \
"	\\\n" \
"	/* threadIdx < B_N */ \\\n" \
"	const size_t i = get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = get_group_id(0) * CHUNK_N + chunk_idx; \\\n" \
"	__local RNS * const Zi = &Z[chunk_idx]; \\\n" \
"	__local RNSe * const Zie = &Ze[chunk_idx]; \\\n" \
"	\\\n" \
"	const size_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \\\n" \
"	const size_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \\\n" \
"	\\\n" \
"	const size_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \\\n" \
"	\\\n" \
"	size_t sj = s + idx_m;\n" \
"\n" \
"#define DECLARE_VAR_FORWARD() \\\n" \
"	__global RNS * __restrict__ const zi = &z[ki]; \\\n" \
"	__global RNSe * __restrict__ const zie = &ze[ki]; \\\n" \
"	__global RNS * __restrict__ const zo = &z[ko]; \\\n" \
"	__global RNSe * __restrict__ const zoe = &ze[ko];\n" \
"\n" \
"#define DECLARE_VAR_BACKWARD() \\\n" \
"	__global RNS * __restrict__ const zi = &z[ko]; \\\n" \
"	__global RNSe * __restrict__ const zie = &ze[ko]; \\\n" \
"	__global RNS * __restrict__ const zo = &z[ki]; \\\n" \
"	__global RNSe * __restrict__ const zoe = &ze[ki]; \\\n" \
"	const size_t n_4 = get_global_size(0); \\\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4]; \\\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"\n" \
"#define FORWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	Forward_4i(B_N * CHUNK_N, &Z[i], &Ze[i], B_N << lm, zi, zie, w, we, sj / B_N);\n" \
"\n" \
"#define FORWARD_O(CHUNK_N) \\\n" \
"	Forward_4o((size_t)1 << lm, zo, zoe, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &Zie[CHUNK_N * 4 * threadIdx], w, we, sj / 1);\n" \
"\n" \
"#define BACKWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_BACKWARD(); \\\n" \
"	\\\n" \
"	Backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &Zie[CHUNK_N * 4 * threadIdx], (size_t)1 << lm, zi, zie, wi, wie, sj / 1); \\\n" \
"\n" \
"#define BACKWARD_O(B_N, CHUNK_N) \\\n" \
"	Backward_4o(B_N << lm, zo, zoe, B_N * CHUNK_N, &Z[i], &Ze[i], wi, wie, sj / B_N);\n" \
"\n" \
"// ---------------------------------------------\n" \
"\n" \
"#define B_64	(64 / 4)\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"void Forward64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const unsigned int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_64, CHUNK64);\n" \
"\n" \
"	const size_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	Forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], w, we, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"void Backward64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const unsigned int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_64, CHUNK64);\n" \
"\n" \
"	const size_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	Backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], wi, wie, sj / 4);\n" \
"\n" \
"	BACKWARD_O(B_64, CHUNK64);\n" \
"}\n" \
"\n" \
"// ---------------------------------------------\n" \
"\n" \
"#define B_256	(256 / 4)\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"void Forward256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const unsigned int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_256, CHUNK256);\n" \
"\n" \
"	const size_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	Forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], w, we, sj / 16);\n" \
"	const size_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	Forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], w, we, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"void Backward256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const unsigned int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_256, CHUNK256);\n" \
"\n" \
"	const size_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	Backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], wi, wie, sj / 4);\n" \
"	const size_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	Backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], wi, wie, sj / 16);\n" \
"\n" \
"	BACKWARD_O(B_256, CHUNK256);\n" \
"}\n" \
"\n" \
"// ---------------------------------------------\n" \
"\n" \
"#define B_1024	(1024 / 4)\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"void Forward1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const unsigned int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_1024, CHUNK1024);\n" \
"\n" \
"	const size_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	Forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], w, we, sj / 64);\n" \
"	const size_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	Forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], w, we, sj / 16);\n" \
"	const size_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	Forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], w, we, sj / 4);\n" \
"\n" \
"	FORWARD_O(CHUNK1024);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"void Backward1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const unsigned int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_1024, CHUNK1024);\n" \
"\n" \
"	const size_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	Backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], wi, wie, sj / 4);\n" \
"	const size_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	Backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], wi, wie, sj / 16);\n" \
"	const size_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);\n" \
"	Backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], wi, wie, sj / 64);\n" \
"\n" \
"	BACKWARD_O(B_1024, CHUNK1024);\n" \
"}\n" \
"\n" \
"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))\n" \
"void Square32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[32 * BLK32];\n" \
"	__local RNSe Ze[32 * BLK32];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k32 = get_group_id(0) * 32 * BLK32, i = get_local_id(0);\n" \
"#if BLK32 > 1\n" \
"	const size_t i32 = (i & (size_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4);\n" \
"#else\n" \
"	const size_t i32 = 0, i8 = i;\n" \
"#endif\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k32 + i32 + i8];\n" \
"	__global RNSe * restrict const zke = &ze[k32 + i32 + i8];\n" \
"	__local RNS * const Z32 = &Z[i32];\n" \
"	__local RNSe * const Z32e = &Ze[i32];\n" \
"	__local RNS * const Zi8 = &Z32[i8];\n" \
"	__local RNSe * const Zi8e = &Z32e[i8];\n" \
"	const size_t i2 = ((4 * i8) & (size_t)~(4 * 2 - 1)) + (i8 % 2);\n" \
"	__local RNS * const Zi2 = &Z32[i2];\n" \
"	__local RNSe * const Zi2e = &Z32e[i2];\n" \
"	__local RNS * const Z4 = &Z32[4 * i8];\n" \
"	__local RNSe * const Z4e = &Z32e[4 * i8];\n" \
"\n" \
"	Forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);\n" \
"	Forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	Square_22_e(Z4, Z4e, w[n_4 + j], we[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"	Backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	Backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))\n" \
"void Square64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[64 * BLK64];\n" \
"	__local RNSe Ze[64 * BLK64];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k64 = get_group_id(0) * 64 * BLK64, i = get_local_id(0);\n" \
"#if BLK64 > 1\n" \
"	const size_t i64 = (i & (size_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4);\n" \
"#else\n" \
"	const size_t i64 = 0, i16 = i;\n" \
"#endif\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k64 + i64 + i16];\n" \
"	__global RNSe * restrict const zke = &ze[k64 + i64 + i16];\n" \
"	__local RNS * const Z64 = &Z[i64];\n" \
"	__local RNSe * const Z64e = &Ze[i64];\n" \
"	__local RNS * const Zi16 = &Z64[i16];\n" \
"	__local RNSe * const Zi16e = &Z64e[i16];\n" \
"	const size_t i4 = ((4 * i16) & (size_t)~(4 * 4 - 1)) + (i16 % 4);\n" \
"	__local RNS * const Zi4 = &Z64[i4];\n" \
"	__local RNSe * const Zi4e = &Z64e[i4];\n" \
"	__local RNS * const Z4 = &Z64[4 * i16];\n" \
"	__local RNSe * const Z4e = &Z64e[4 * i16];\n" \
"\n" \
"	Forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);\n" \
"	Forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	__global const RNS_W * const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * const wie = &we[4 * n_4];\n" \
"	Square_4_e(Z4, Z4e, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);\n" \
"	Backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	Backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))\n" \
"void Square128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[128 * BLK128];\n" \
"	__local RNSe Ze[128 * BLK128];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k128 = get_group_id(0) * 128 * BLK128, i = get_local_id(0);\n" \
"#if BLK128 > 1\n" \
"	const size_t i128 = (i & (size_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4);\n" \
"#else\n" \
"	const size_t i128 = 0, i32 = i;\n" \
"#endif\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k128 + i128 + i32];\n" \
"	__global RNSe * restrict const zke = &ze[k128 + i128 + i32];\n" \
"	__local RNS * const Z128 = &Z[i128];\n" \
"	__local RNSe * const Z128e = &Ze[i128];\n" \
"	__local RNS * const Zi32 = &Z128[i32];\n" \
"	__local RNSe * const Zi32e = &Z128e[i32];\n" \
"	const size_t i8 = ((4 * i32) & (size_t)~(4 * 8 - 1)) + (i32 % 8);\n" \
"	__local RNS * const Zi8 = &Z128[i8];\n" \
"	__local RNSe * const Zi8e = &Z128e[i8];\n" \
"	const size_t i2 = ((4 * i32) & (size_t)~(4 * 2 - 1)) + (i32 % 2);\n" \
"	__local RNS * const Zi2 = &Z128[i2];\n" \
"	__local RNSe * const Zi2e = &Z128e[i2];\n" \
"	__local RNS * const Z4 = &Z128[4 * i32];\n" \
"	__local RNSe * const Z4e = &Z128e[4 * i32];\n" \
"\n" \
"	Forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);\n" \
"	Forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	Forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	Square_22_e(Z4, Z4e, w[n_4 + j], we[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"	Backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	Backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	Backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))\n" \
"void Square256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[256 * BLK256];\n" \
"	__local RNSe Ze[256 * BLK256];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k256 = get_group_id(0) * 256 * BLK256, i = get_local_id(0);\n" \
"#if BLK256 > 1\n" \
"	const size_t i256 = (i & (size_t)~(256 / 4 - 1)) * 4, i64 = i % (256 / 4);\n" \
"#else\n" \
"	const size_t i256 = 0, i64 = i;\n" \
"#endif\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k256 + i256 + i64];\n" \
"	__global RNSe * restrict const zke = &ze[k256 + i256 + i64];\n" \
"	__local RNS * const Z256 = &Z[i256];\n" \
"	__local RNSe * const Z256e = &Ze[i256];\n" \
"	__local RNS * const Zi64 = &Z256[i64];\n" \
"	__local RNSe * const Zi64e = &Z256e[i64];\n" \
"	const size_t i16 = ((4 * i64) & (size_t)~(4 * 16 - 1)) + (i64 % 16);\n" \
"	__local RNS * const Zi16 = &Z256[i16];\n" \
"	__local RNSe * const Zi16e = &Z256e[i16];\n" \
"	const size_t i4 = ((4 * i64) & (size_t)~(4 * 4 - 1)) + (i64 % 4);\n" \
"	__local RNS * const Zi4 = &Z256[i4];\n" \
"	__local RNSe * const Zi4e = &Z256e[i4];\n" \
"	__local RNS * const Z4 = &Z256[4 * i64];\n" \
"	__local RNSe * const Z4e = &Z256e[4 * i64];\n" \
"\n" \
"	Forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);\n" \
"	Forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	Forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"	Square_4_e(Z4, Z4e, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);\n" \
"	Backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	Backward_4(16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"	Backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"void Square512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[512];\n" \
"	__local RNSe Ze[512];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k512 = get_group_id(0) * 512, i128 = get_local_id(0);\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k512 + i128];\n" \
"	__global RNSe * restrict const zke = &ze[k512 + i128];\n" \
"	__local RNS * const Zi128 = &Z[i128];\n" \
"	__local RNSe * const Zi128e = &Ze[i128];\n" \
"	const size_t i32 = ((4 * i128) & (size_t)~(4 * 32 - 1)) + (i128 % 32);\n" \
"	__local RNS * const Zi32 = &Z[i32];\n" \
"	__local RNSe * const Zi32e = &Ze[i32];\n" \
"	const size_t i8 = ((4 * i128) & (size_t)~(4 * 8 - 1)) + (i128 % 8);\n" \
"	__local RNS * const Zi8 = &Z[i8];\n" \
"	__local RNSe * const Zi8e = &Ze[i8];\n" \
"	const size_t i2 = ((4 * i128) & (size_t)~(4 * 2 - 1)) + (i128 % 2);\n" \
"	__local RNS * const Zi2 = &Z[i2];\n" \
"	__local RNSe * const Zi2e = &Ze[i2];\n" \
"	__local RNS * const Z4 = &Z[4 * i128];\n" \
"	__local RNSe * const Z4e = &Ze[4 * i128];\n" \
"\n" \
"	Forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);\n" \
"	Forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	Forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	Forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	Square_22_e(Z4, Z4e, w[n_4 + j], we[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"	Backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	Backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	Backward_4(32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"	Backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"void Square1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[1024];\n" \
"	__local RNSe Ze[1024];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k1024 = get_group_id(0) * 1024, i256 = get_local_id(0);\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k1024 + i256];\n" \
"	__global RNSe * restrict const zke = &ze[k1024 + i256];\n" \
"	__local RNS * const Zi256 = &Z[i256];\n" \
"	__local RNSe * const Zi256e = &Ze[i256];\n" \
"	const size_t i64 = ((4 * i256) & (size_t)~(4 * 64 - 1)) + (i256 % 64);\n" \
"	__local RNS * const Zi64 = &Z[i64];\n" \
"	__local RNSe * const Zi64e = &Ze[i64];\n" \
"	const size_t i16 = ((4 * i256) & (size_t)~(4 * 16 - 1)) + (i256 % 16);\n" \
"	__local RNS * const Zi16 = &Z[i16];\n" \
"	__local RNSe * const Zi16e = &Ze[i16];\n" \
"	const size_t i4 = ((4 * i256) & (size_t)~(4 * 4 - 1)) + (i256 % 4);\n" \
"	__local RNS * const Zi4 = &Z[i4];\n" \
"	__local RNSe * const Zi4e = &Ze[i4];\n" \
"	__local RNS * const Z4 = &Z[4 * i256];\n" \
"	__local RNSe * const Z4e = &Ze[4 * i256];\n" \
"\n" \
"	Forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);\n" \
"	Forward_4(64, Zi64, Zi64e, w, we, j / 64);\n" \
"	Forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	Forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"	Square_4_e(Z4, Z4e, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);\n" \
"	Backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	Backward_4(16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"	Backward_4(64, Zi64, Zi64e, wi, wie, j / 64);\n" \
"	Backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"void Square2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	__local RNS Z[2048];\n" \
"	__local RNSe Ze[2048];\n" \
"\n" \
"	const size_t n_4 = get_global_size(0), idx = get_global_id(0), j = n_4 + idx;\n" \
"\n" \
"	const size_t k2048 = get_group_id(0) * 2048, i512 = get_local_id(0);\n" \
"\n" \
"	__global RNS * restrict const zk = &z[k2048 + i512];\n" \
"	__global RNSe * restrict const zke = &ze[k2048 + i512];\n" \
"	__local RNS * const Zi512 = &Z[i512];\n" \
"	__local RNSe * const Zi512e = &Ze[i512];\n" \
"	const size_t i128 = ((4 * i512) & (size_t)~(4 * 128 - 1)) + (i512 % 128);\n" \
"	__local RNS * const Zi128 = &Z[i128];\n" \
"	__local RNSe * const Zi128e = &Ze[i128];\n" \
"	const size_t i32 = ((4 * i512) & (size_t)~(4 * 32 - 1)) + (i512 % 32);\n" \
"	__local RNS * const Zi32 = &Z[i32];\n" \
"	__local RNSe * const Zi32e = &Ze[i32];\n" \
"	const size_t i8 = ((4 * i512) & (size_t)~(4 * 8 - 1)) + (i512 % 8);\n" \
"	__local RNS * const Zi8 = &Z[i8];\n" \
"	__local RNSe * const Zi8e = &Ze[i8];\n" \
"	const size_t i2 = ((4 * i512) & (size_t)~(4 * 2 - 1)) + (i512 % 2);\n" \
"	__local RNS * const Zi2 = &Z[i2];\n" \
"	__local RNSe * const Zi2e = &Ze[i2];\n" \
"	__local RNS * const Z4 = &Z[4 * i512];\n" \
"	__local RNSe * const Z4e = &Ze[4 * i512];\n" \
"\n" \
"	Forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);\n" \
"	Forward_4(128, Zi128, Zi128e, w, we, j / 128);\n" \
"	Forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	Forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	Forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	Square_22_e(Z4, Z4e, w[n_4 + j], we[n_4 + j]);\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"	Backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	Backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	Backward_4(32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"	Backward_4(128, Zi128, Zi128e, wi, wie, j / 128);\n" \
"	Backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);\n" \
"}\n" \
"\n" \
"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n" \
"\n" \
"/*\n" \
"Barrett's reduction.\n" \
"\n" \
"Let h be the number of iterations in the 'while loop': h = [x/b] - [x [2^64/b] / 2^64].\n" \
"\n" \
"We have h = ([x/b] - x/b) + x/2^64 (2^64/b - [2^64/b]) + (x [2^64/b] / 2^64 - [x [2^64/b] / 2^64]).\n" \
"Then -1 + 0 + 0 < h < 0 + x/2^64 + 1,\n" \
"0 <= h < 1 + x/2^64.\n" \
"\n" \
"*/\n" \
"\n" \
"__kernel\n" \
"void BaseMod0(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global long * restrict const c, __global unsigned int * restrict const bErr,\n" \
"	const int base, const long recBase, const unsigned int ln, const unsigned int blk, const int g)\n" \
"{\n" \
"	const size_t idx = get_global_id(0);\n" \
"	__global RNS * restrict const zi = &z[blk * idx];\n" \
"	__global RNSe * restrict const zie = &ze[blk * idx];\n" \
"\n" \
"	prefetch(zi, (size_t)blk);\n" \
"	prefetch(zie, (size_t)blk);\n" \
"\n" \
"	const RNS norm = (RNS)(P1 - ((P1 - 1) >> (ln - 1)), P2 - ((P2 - 1) >> (ln - 1)));\n" \
"	const RNSe norme = (RNSe)(P3 - ((P3 - 1) >> (ln - 1)));\n" \
"	const long maxProd_32 = (long)max(mul_hi(base, base + 1), 1) << ln;\n" \
"\n" \
"	int96 f; f.lo = 0; f.hi = 0;\n" \
"	bool err = false;\n" \
"\n" \
"	size_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		const int96 l = GetLong_e(Mul(zi[j], norm), Mule(zie[j], norme));\n" \
"		f = Add96(f, Mul96_96_u32(l, g));\n" \
"\n" \
"		const long d1 = MulHi64_96_u64(f, recBase);\n" \
"		const int96 r = Sub96(f, Mul96_64_u32(d1, base));\n" \
"		const long r64 = (long)r.lo;\n" \
"\n" \
"		const long d2 = mul_hi(r64, recBase);\n" \
"		const long r2 = r64 - d2 * base;\n" \
"		const int ri = (int)r2;\n" \
"		zi[j] = ToRNS(ri);\n" \
"		zie[j] = ToRNSe(ri);\n" \
"\n" \
"		const long l_32 = ((long)l.hi << 32) | (l.lo >> 32);\n" \
"		err |= (l_32 > maxProd_32) | (l_32 < -maxProd_32) | ((r.hi != 0) && (r.hi != -1)) | ((r.hi == 0) && (r64 < 0)) | ((r.hi == -1) && (r64 >= 0)) | (ri != r2);\n" \
"\n" \
"		const long d = d1 + d2;\n" \
"		f.lo = d; f.hi = (d < 0) ? -1 : 0;\n" \
"		++j;\n" \
"	} while (j != blk);\n" \
"\n" \
"	const size_t i = (idx + 1) & (get_global_size(0) - 1);\n" \
"	if (err) bErr[idx] = 1;\n" \
"	c[i] = (i == 0) ? -(long)f.lo : (long)f.lo;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void BaseMod1(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const long * restrict const c, \n" \
"	const int base, const long recBase, const unsigned int blk)\n" \
"{\n" \
"	const size_t idx = get_global_id(0);\n" \
"	__global RNS * restrict const zi = &z[blk * idx];\n" \
"	__global RNSe * restrict const zie = &ze[blk * idx];\n" \
"\n" \
"	long f = c[idx];\n" \
"\n" \
"	size_t j = 0;\n" \
"	do\n" \
"	{\n" \
"		f += GetInt(zi[j]);\n" \
"\n" \
"		const long d = mul_hi(f, recBase);\n" \
"		const long r = f - d * base;\n" \
"		zi[j] = ToRNS((int)r);\n" \
"		zie[j] = ToRNSe((int)r);\n" \
"		if (d == 0) return;\n" \
"		f = d;\n" \
"		++j;\n" \
"	} while (j != blk - 1);\n" \
"\n" \
"	zi[blk - 1] = Add(zi[blk - 1], ToRNS((int)f));\n" \
"	zie[blk - 1] = Adde(zie[blk - 1], ToRNSe((int)f));\n" \
"}\n" \
"";
