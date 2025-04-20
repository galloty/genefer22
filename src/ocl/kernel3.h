/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel3 = \
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
"typedef uint	sz_t;\n" \
"typedef uint	uint32;\n" \
"typedef int		int32;\n" \
"typedef ulong	uint64;\n" \
"typedef long	int64;\n" \
"typedef uint2	uint32_2;\n" \
"typedef uint4	uint32_4;\n" \
"typedef int4	int32_4;\n" \
"\n" \
"#if !defined(LNSIZE)\n" \
"#define LNSIZE		16\n" \
"#define NSIZE_4		16384u\n" \
"#define P1			4194304001u\n" \
"#define P2			4076863489u\n" \
"#define P3			3942645761u\n" \
"#define Q1			100663297u\n" \
"#define Q2			218103809u\n" \
"#define Q3			352321537u\n" \
"#define R1			232465106u\n" \
"#define R2			3444438393u\n" \
"#define R3			3810498414u\n" \
"#define NORM1		4193792001u\n" \
"#define NORM2		4076365825u\n" \
"#define NORM3		3941683201u\n" \
"#define InvP2_P1	1797558821u\n" \
"#define InvP3_P1	3075822917u\n" \
"#define InvP3_P2	4076863457u\n" \
"#define P1P2P3l		12816400126780112897ul\n" \
"#define P1P2P3h		3654720002u\n" \
"#define P1P2P3_2l	6408200063390056448ul\n" \
"#define P1P2P3_2h	1827360001u\n" \
"#define BLK32		8\n" \
"#define BLK64		4\n" \
"#define BLK128		2\n" \
"#define BLK256		1\n" \
"#define CHUNK64		8\n" \
"#define CHUNK256	4\n" \
"#define CHUNK1024	2\n" \
"#define NORM_WG_SZ	64\n" \
"#define MAX_WORK_GROUP_SIZE	256\n" \
"#endif\n" \
"\n" \
"#define P1P2	(P1 * (uint64)(P2))\n" \
"#define P2P3	(P2 * (uint64)(P3))\n" \
"\n" \
"// --- uint96/int96 ---\n" \
"\n" \
"typedef struct { uint64 s0; uint32 s1; } uint96;\n" \
"typedef struct { uint64 s0; int32 s1; } int96;\n" \
"\n" \
"INLINE int96 int96_set_si(const int64 n) { int96 r; r.s0 = (ulong)n; r.s1 = (n < 0) ? -1 : 0; return r; }\n" \
"INLINE uint96 uint96_set(const uint64 s0, const uint32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }\n" \
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
"	const int32 c = (x.s0 != 0) ? 1 : 0;\n" \
"	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;\n" \
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
"	const int32 c = (s0 < y.s0) ? 1 : 0;\n" \
"	r.s0 = s0; r.s1 = x.s1 + y.s1 + c;\n" \
"#endif\n" \
"	return r;\n" \
"}\n" \
"\n" \
"INLINE uint96 uint96_add_64(const uint96 x, const uint64 y)\n" \
"{\n" \
"	uint96 r;\n" \
"#ifdef PTX_ASM\n" \
"	asm volatile (\"add.cc.u64 %0, %1, %2;\" : \"=l\" (r.s0) : \"l\" (x.s0), \"l\" (y));\n" \
"	asm volatile (\"addc.u32 %0, %1, 0;\" : \"=r\" (r.s1) : \"r\" (x.s1));\n" \
"#else\n" \
"	const uint64 s0 = x.s0 + y;\n" \
"	const uint32 c = (s0 < y) ? 1 : 0;\n" \
"	r.s0 = s0; r.s1 = x.s1 + c;\n" \
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
"	const uint32 c = (x.s0 < y.s0) ? 1 : 0;\n" \
"	r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - c);\n" \
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
"// --- mod arith ---\n" \
"\n" \
"INLINE uint32 _addMod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 c = (lhs >= p - rhs) ? p : 0;\n" \
"	return lhs + rhs - c;\n" \
"}\n" \
"\n" \
"INLINE uint32 _subMod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 c = (lhs < rhs) ? p : 0;\n" \
"	return lhs - rhs + c;\n" \
"}\n" \
"\n" \
"// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.\n" \
"\n" \
"// Montgomery form of n is n * 2^32 mod p. q * p = 1 mod 2^32.\n" \
"\n" \
"// r = lhs * rhs * 2^-32 mod p\n" \
"// If lhs = x * 2^32 and rhs = y * 2^32 then r = (x * y) * 2^32 mod p.\n" \
"// If lhs = x and rhs = y * 2^32 then r = x * y mod p.\n" \
"INLINE uint32 _mulMonty(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 q)\n" \
"{\n" \
"	const uint32 t_lo = lhs * rhs, t_hi = mul_hi(lhs, rhs);\n" \
"	const uint32 mp = mul_hi(t_lo * q, p);\n" \
"	return _subMod(t_hi, mp, p);\n" \
"}\n" \
"\n" \
"// Conversion into Montgomery form\n" \
"INLINE uint32 _toMonty(const uint32 n, const uint32 r2, const uint32 p, const uint32 q)\n" \
"{\n" \
"	// n * (2^32)^2 = (n * 2^32) * (1 * 2^32)\n" \
"	return _mulMonty(n, r2, p, q);\n" \
"}\n" \
"\n" \
"// Conversion out of Montgomery form\n" \
"// INLINE uint32 _fromMonty(const uint32 n, const uint32 p, const uint32 q)\n" \
"// {\n" \
"// 	// If n = x * 2^32 mod p then _mulMonty(n, 1, p, q) = x.\n" \
"// 	const uint32 mp = mul_hi(n * q, p);\n" \
"// 	return (mp != 0) ? p - mp : 0;\n" \
"// }\n" \
"\n" \
"INLINE uint32 add_P1(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P1); }\n" \
"INLINE uint32 add_P2(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P2); }\n" \
"INLINE uint32 add_P3(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P3); }\n" \
"\n" \
"INLINE uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }\n" \
"INLINE uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }\n" \
"INLINE uint32 sub_P3(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P3); }\n" \
"\n" \
"// Montgomery form\n" \
"INLINE uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P1, Q1); }\n" \
"INLINE uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P2, Q2); }\n" \
"INLINE uint32 mul_P3(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P3, Q3); }\n" \
"\n" \
"INLINE uint32 toMonty_P1(const uint32 lhs) { return _toMonty(lhs, R1, P1, Q1); }\n" \
"INLINE uint32 toMonty_P2(const uint32 lhs) { return _toMonty(lhs, R2, P2, Q2); }\n" \
"INLINE uint32 toMonty_P3(const uint32 lhs) { return _toMonty(lhs, R3, P3, Q3); }\n" \
"\n" \
"// INLINE uint32 fromMonty_P1(const uint32 lhs) { return _fromMonty(lhs, P1, Q1); }\n" \
"// INLINE uint32 fromMonty_P2(const uint32 lhs) { return _fromMonty(lhs, P2, Q2); }\n" \
"// INLINE uint32 fromMonty_P3(const uint32 lhs) { return _fromMonty(lhs, P3, Q3); }\n" \
"\n" \
"INLINE int32 geti_P3(const uint32 n) { return (n > P3 / 2) ? (int32)(n - P3) : (int32)(n); }\n" \
"\n" \
"INLINE int96 garner3(const uint32 r1, const uint32 r2, const uint32 r3)\n" \
"{\n" \
"	const uint32 u13 = mul_P1(sub_P1(r1, r3), InvP3_P1);\n" \
"	const uint32 u23 = mul_P2(sub_P2(r2, r3), InvP3_P2);\n" \
"	const uint32 u123 = mul_P1(sub_P1(u13, u23), InvP2_P1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (ulong)P3 + r3);\n" \
"	const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);\n" \
"	const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// --- RNS/RNSe ---\n" \
"\n" \
"\n" \
"typedef uint32_2	RNS;\n" \
"typedef uint32_4	RNS2;\n" \
"typedef uint32_2	RNS_W;\n" \
"typedef uint32_4	RNS_W2;\n" \
"typedef uint32		RNSe;\n" \
"typedef uint32_2	RNS2e;\n" \
"typedef uint32		RNS_We;\n" \
"typedef uint32_2	RNS_W2e;\n" \
"\n" \
"INLINE RNS toRNS(const int32 i) { return ((RNS)(i, i) + ((i < 0) ? (RNS)(P1, P2) : (RNS)(0, 0))); }\n" \
"\n" \
"INLINE RNS add(const RNS lhs, const RNS rhs) { return (RNS)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }\n" \
"INLINE RNS sub(const RNS lhs, const RNS rhs) { return (RNS)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }\n" \
"INLINE RNS mul(const RNS lhs, const RNS rhs) { return (RNS)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }\n" \
"\n" \
"INLINE RNS sqr(const RNS lhs) { return mul(lhs, lhs); }\n" \
"\n" \
"INLINE RNS mulW(const RNS lhs, const RNS_W w) { return mul(lhs, w); }\n" \
"\n" \
"INLINE RNS toMonty(const RNS lhs) { return (RNS)(toMonty_P1(lhs.s0), toMonty_P2(lhs.s1)); }\n" \
"\n" \
"INLINE RNS2 mul2(const RNS2 lhs, const RNS rhs) { return (RNS2)(mul(lhs.s01, rhs), mul(lhs.s23, rhs)); }\n" \
"\n" \
"INLINE RNSe toRNSe(const int i) { return ((RNSe)(i) + ((i < 0) ? (RNSe)(P3) : (RNSe)(0))); }\n" \
"\n" \
"INLINE RNSe adde(const RNSe lhs, const RNSe rhs) { return (RNSe)(add_P3(lhs, rhs)); }\n" \
"INLINE RNSe sube(const RNSe lhs, const RNSe rhs) { return (RNSe)(sub_P3(lhs, rhs)); }\n" \
"INLINE RNSe mule(const RNSe lhs, const RNSe rhs) { return (RNSe)(mul_P3(lhs, rhs)); }\n" \
"\n" \
"INLINE RNSe sqre(const RNSe lhs) { return mule(lhs, lhs); }\n" \
"\n" \
"INLINE RNSe mulWe(const RNSe lhs, const RNS_We w) { return mule(lhs, w); }\n" \
"\n" \
"INLINE RNSe toMontye(const RNSe lhs) { return (RNSe)toMonty_P3(lhs); }\n" \
"\n" \
"INLINE RNS2e mul2e(const RNS2e lhs, const RNSe rhs) { return (RNS2e)(mule(lhs.s0, rhs), mule(lhs.s1, rhs)); }\n" \
"\n" \
"// --- transform/macro ---\n" \
"\n" \
"#define FWD2(z0, z1, w) { const RNS t = mulW(z1, w); z1 = sub(z0, t); z0 = add(z0, t); }\n" \
"#define FWD2e(z0e, z1e, we) { const RNSe t = mulWe(z1e, we); z1e = sube(z0e, t); z0e = adde(z0e, t); }\n" \
"\n" \
"#define BCK2(z0, z1, wi) { const RNS t = sub(z0, z1); z0 = add(z0, z1); z1 = mulW(t, wi); }\n" \
"#define BCK2e(z0e, z1e, wie) { const RNSe t = sube(z0e, z1e); z0e = adde(z0e, z1e); z1e = mulWe(t, wie); }\n" \
"\n" \
"#define SQR2(z0, z1, w) { const RNS t = sqr(mulW(z1, w)); z1 = mul(add(z0, z0), z1); z0 = add(sqr(z0), t); }\n" \
"#define SQR2e(z0e, z1e, we) { const RNSe t = sqre(mulWe(z1e, we)); z1e = mule(adde(z0e, z0e), z1e); z0e = adde(sqre(z0e), t); }\n" \
"#define SQR2N(z0, z1, w) { const RNS t = sqr(mulW(z1, w)); z1 = mul(add(z0, z0), z1); z0 = sub(sqr(z0), t); }\n" \
"#define SQR2Ne(z0e, z1e, we) { const RNSe t = sqre(mulWe(z1e, we)); z1e = mule(adde(z0e, z0e), z1e); z0e = sube(sqre(z0e), t); }\n" \
"\n" \
"#define MUL2(z0, z1, zp0, zp1, w) { const RNS t = mul(mulW(z1, w), mulW(zp1, w)); z1 = add(mul(z0, zp1), mul(zp0, z1)); z0 = add(mul(z0, zp0), t); }\n" \
"#define MUL2e(z0e, z1e, zp0e, zp1e, we) { const RNSe t = mule(mulWe(z1e, we), mulWe(zp1e, we)); z1e = adde(mule(z0e, zp1e), mule(zp0e, z1e)); z0e = adde(mule(z0e, zp0e), t); }\n" \
"#define MUL2N(z0, z1, zp0, zp1, w) { const RNS t = mul(mulW(z1, w), mulW(zp1, w)); z1 = add(mul(z0, zp1), mul(zp0, z1)); z0 = sub(mul(z0, zp0), t); }\n" \
"#define MUL2Ne(z0e, z1e, zp0e, zp1e, we) { const RNSe t = mule(mulWe(z1e, we), mulWe(zp1e, we)); z1e = adde(mule(z0e, zp1e), mule(zp0e, z1e)); z0e = sube(mule(z0e, zp0e), t); }\n" \
"\n" \
"#define DECLARE_W(j) \\\n" \
"	const RNS_W w1 = w[j]; const RNS_W2 w2 = ((__global const RNS_W2 *)w)[j]; \\\n" \
"	const RNS_We w1e = we[j]; const RNS_W2e w2e = ((__global const RNS_W2e *)we)[j];\n" \
"\n" \
"#define DECLARE_WI(j) \\\n" \
"	const RNS_W wi1 = wi[j]; const RNS_W2 wi2 = ((__global const RNS_W2 *)wi)[j]; \\\n" \
"	const RNS_We wi1e = wie[j]; const RNS_W2e wi2e = ((__global const RNS_W2e *)wie)[j];\n" \
"\n" \
"#define FORWARD4() \\\n" \
"	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e); \\\n" \
"	FWD2(zl[0], zl[1], w2.s01); FWD2(zl[2], zl[3], w2.s23); FWD2e(zle[0], zle[1], w2e.s0); FWD2e(zle[2], zle[3], w2e.s1);\n" \
"\n" \
"#define BACKWARD4() \\\n" \
"	BCK2(zl[0], zl[1], wi2.s01); BCK2(zl[2], zl[3], wi2.s23); BCK2e(zle[0], zle[1], wi2e.s0); BCK2e(zle[2], zle[3], wi2e.s1); \\\n" \
"	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);\n" \
"\n" \
"#define FORWARD22() \\\n" \
"	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);\n" \
"\n" \
"#define BACKWARD22() \\\n" \
"	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);\n" \
"\n" \
"#define SQUARE22() \\\n" \
"	SQR2(zl[0], zl[1], w0); SQR2N(zl[2], zl[3], w0); SQR2e(zle[0], zle[1], w0e); SQR2Ne(zle[2], zle[3], w0e);\n" \
"\n" \
"#define MUL22() \\\n" \
"	MUL2(zl[0], zl[1], zpl[0], zpl[1], w0); MUL2N(zl[2], zl[3], zpl[2], zpl[3], w0); \\\n" \
"	MUL2e(zle[0], zle[1], zple[0], zple[1], w0e); MUL2Ne(zle[2], zle[3], zple[2], zple[3], w0e);\n" \
"\n" \
"// --- transform/inline ---\n" \
"\n" \
"INLINE void _loadg(RNS zl[4], __global const RNS * restrict const z, const size_t s) { for (size_t l = 0; l < 4; ++l) zl[l] = z[l * s]; }\n" \
"INLINE void _loadge(RNSe zle[4], __global const RNSe * restrict const ze, const size_t s) { for (size_t l = 0; l < 4; ++l) zle[l] = ze[l * s]; }\n" \
"INLINE void _loadl(RNS zl[4], __local const RNS * restrict const Z, const size_t s) { for (size_t l = 0; l < 4; ++l) zl[l] = Z[l * s]; }\n" \
"INLINE void _loadle(RNSe zle[4], __local const RNSe * restrict const Ze, const size_t s) { for (size_t l = 0; l < 4; ++l) zle[l] = Ze[l * s]; }\n" \
"INLINE void _storeg(__global RNS * restrict const z, const size_t s, const RNS zl[4]) { for (size_t l = 0; l < 4; ++l) z[l * s] = zl[l]; }\n" \
"INLINE void _storege(__global RNSe * restrict const ze, const size_t s, const RNSe zle[4]) { for (size_t l = 0; l < 4; ++l) ze[l * s] = zle[l]; }\n" \
"INLINE void _storel(__local RNS * restrict const Z, const size_t s, const RNS zl[4]) { for (size_t l = 0; l < 4; ++l) Z[l * s] = zl[l]; }\n" \
"INLINE void _storele(__local RNSe * restrict const Ze, const size_t s, const RNSe zle[4]) { for (size_t l = 0; l < 4; ++l) Ze[l * s] = zle[l]; }\n" \
"\n" \
"INLINE void forward_4(const sz_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)\n" \
"{\n" \
"	DECLARE_W(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, m); _loadle(zle, Ze, m);\n" \
"	FORWARD4();\n" \
"	_storel(Z, m, zl); _storele(Ze, m, zle);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)\n" \
"{\n" \
"	DECLARE_W(j);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadg(zl, z, mg); _loadge(zle, ze, mg);\n" \
"	FORWARD4();\n" \
"	_storel(Z, ml, zl); _storele(Ze, ml, zle);\n" \
"}\n" \
"\n" \
"INLINE void forward_4i_0(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_W(1);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadg(zl, z, mg); _loadge(zle, ze, mg);\n" \
"	zl[0] = toMonty(zl[0]); zl[1] = toMonty(zl[1]); zle[0] = toMontye(zle[0]); zle[1] = toMontye(zle[1]);\n" \
"	FORWARD4();\n" \
"	_storel(Z, ml, zl); _storele(Ze, ml, zle);\n" \
"}\n" \
"\n" \
"INLINE void forward_4o(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	const sz_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)\n" \
"{\n" \
"	DECLARE_W(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, ml); _loadle(zle, Ze, ml);\n" \
"	FORWARD4();\n" \
"	_storeg(z, mg, zl); _storege(ze, mg, zle);\n" \
"}\n" \
"\n" \
"INLINE void backward_4(const sz_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)\n" \
"{\n" \
"	DECLARE_WI(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, m); _loadle(zle, Ze, m);\n" \
"	BACKWARD4();\n" \
"	_storel(Z, m, zl); _storele(Ze, m, zle);\n" \
"}\n" \
"\n" \
"INLINE void backward_4i(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)\n" \
"{\n" \
"	DECLARE_WI(j);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadg(zl, z, mg); _loadge(zle, ze, mg);\n" \
"	BACKWARD4();\n" \
"	_storel(Z, ml, zl); _storele(Ze, ml, zle);\n" \
"}\n" \
"\n" \
"INLINE void backward_4o(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	const sz_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,\n" \
"	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)\n" \
"{\n" \
"	DECLARE_WI(j);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, ml); _loadle(zle, Ze, ml);\n" \
"	BACKWARD4();\n" \
"	_storeg(z, mg, zl); _storege(ze, mg, zle);\n" \
"}\n" \
"\n" \
"INLINE void write_4(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__local const RNS * restrict const Z, __local const RNSe * restrict const Ze)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	z[0 * mg] = Z[0]; z[1 * mg] = Z[1]; z[2 * mg] = Z[2]; z[3 * mg] = Z[3];\n" \
"	ze[0 * mg] = Ze[0]; ze[1 * mg] = Ze[1]; ze[2 * mg] = Ze[2]; ze[3 * mg] = Ze[3];\n" \
"}\n" \
"\n" \
"INLINE void fwd2write_4(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__local const RNS * restrict const Z, __local const RNSe * restrict const Ze, const RNS_W w1, const RNS_We w1e)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);\n" \
"	FORWARD22();\n" \
"	_storeg(z, mg, zl); _storege(ze, mg, zle);\n" \
"}\n" \
"\n" \
"INLINE void square_22(__local RNS * restrict const Z, __local RNSe * restrict const Ze, const RNS_W w0, const RNS_We w0e)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);\n" \
"	SQUARE22();\n" \
"	_storel(Z, 1, zl); _storele(Ze, 1, zle);\n" \
"}\n" \
"\n" \
"INLINE void square_4(__local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const RNS_W w1, const RNS_W wi1, const RNS_W w0, const RNS_We w1e, const RNS_We wi1e, const RNS_We w0e)\n" \
"{\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);\n" \
"	FORWARD22();\n" \
"	SQUARE22();\n" \
"	BACKWARD22();\n" \
"	_storel(Z, 1, zl); _storele(Ze, 1, zle);\n" \
"}\n" \
"\n" \
"INLINE void mul_22(__local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const sz_t mg, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe, const RNS_W w0, const RNS_We w0e)\n" \
"{\n" \
"	RNS zpl[4]; RNSe zple[4]; _loadg(zpl, zp, mg); _loadge(zple, zpe, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);\n" \
"	MUL22();\n" \
"	_storel(Z, 1, zl); _storele(Ze, 1, zle);\n" \
"}\n" \
"\n" \
"INLINE void mul_4(__local RNS * restrict const Z, __local RNSe * restrict const Ze,\n" \
"	const sz_t mg, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe, \n" \
"	const RNS_W w1, const RNS_W wi1, const RNS_W w0, const RNS_We w1e, const RNS_We wi1e, const RNS_We w0e)\n" \
"{\n" \
"	RNS zpl[4]; RNSe zple[4]; _loadg(zpl, zp, mg); _loadge(zple, zpe, mg);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);\n" \
"	FORWARD22();\n" \
"	MUL22();\n" \
"	BACKWARD22();\n" \
"	_storel(Z, 1, zl); _storele(Ze, 1, zle);\n" \
"}\n" \
"\n" \
"// --- transform ---\n" \
"\n" \
"#define DECLARE_VAR(B_N, CHUNK_N) \\\n" \
"	__local RNS Z[4 * B_N * CHUNK_N]; \\\n" \
"	__local RNSe Ze[4 * B_N * CHUNK_N]; \\\n" \
"	\\\n" \
"	/* threadIdx < B_N */ \\\n" \
"	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \\\n" \
"	__local RNS * const Zi = &Z[chunk_idx]; \\\n" \
"	__local RNSe * const Zie = &Ze[chunk_idx]; \\\n" \
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
"	__global RNSe * __restrict__ const zie = &ze[ki]; \\\n" \
"	__global RNS * __restrict__ const zo = &z[ko]; \\\n" \
"	__global RNSe * __restrict__ const zoe = &ze[ko];\n" \
"\n" \
"#define DECLARE_VAR_BACKWARD() \\\n" \
"	__global RNS * __restrict__ const zi = &z[ko]; \\\n" \
"	__global RNSe * __restrict__ const zie = &ze[ko]; \\\n" \
"	__global RNS * __restrict__ const zo = &z[ki]; \\\n" \
"	__global RNSe * __restrict__ const zoe = &ze[ki]; \\\n" \
"	const sz_t n_4 = NSIZE_4; \\\n" \
"	__global const RNS_W * restrict const wi = &w[4 * n_4]; \\\n" \
"	__global const RNS_We * restrict const wie = &we[4 * n_4];\n" \
"\n" \
"#define FORWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	forward_4i(B_N * CHUNK_N, &Z[i], &Ze[i], B_N << lm, zi, zie, w, we, sj / B_N);\n" \
"\n" \
"#define FORWARD_I_0(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_FORWARD(); \\\n" \
"	\\\n" \
"	forward_4i_0(B_N * CHUNK_N, &Z[i], &Ze[i], B_N << lm, zi, zie, w, we);\n" \
"\n" \
"#define FORWARD_O(CHUNK_N) \\\n" \
"	forward_4o((sz_t)1 << lm, zo, zoe, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &Zie[CHUNK_N * 4 * threadIdx], w, we, sj / 1);\n" \
"\n" \
"#define BACKWARD_I(B_N, CHUNK_N) \\\n" \
"	DECLARE_VAR(B_N, CHUNK_N); \\\n" \
"	DECLARE_VAR_BACKWARD(); \\\n" \
"	\\\n" \
"	backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &Zie[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, zie, wi, wie, sj / 1);\n" \
"\n" \
"#define BACKWARD_O(B_N, CHUNK_N) \\\n" \
"	backward_4o(B_N << lm, zo, zoe, B_N * CHUNK_N, &Z[i], &Ze[i], wi, wie, sj / B_N);\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_64	(64 / 4)\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64\n" \
"	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"#endif\n" \
"void forward64(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	 __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], w, we, sj / 4);\n" \
"	FORWARD_O(CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64\n" \
"	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"#endif\n" \
"void backward64(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], wi, wie, sj / 4);\n" \
"	BACKWARD_O(B_64, CHUNK64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64\n" \
"	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))\n" \
"#endif\n" \
"void forward64_0(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	 __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	const int lm = LNSIZE - 6; const unsigned int s = 64 / 4;\n" \
"	FORWARD_I_0(B_64, CHUNK64);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], w, we, sj / 4);\n" \
"	FORWARD_O(CHUNK64);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_256	(256 / 4)\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256\n" \
"	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"#endif\n" \
"void forward256(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_256, CHUNK256);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], w, we, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], w, we, sj / 4);\n" \
"	FORWARD_O(CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256\n" \
"	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"#endif\n" \
"void backward256(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_256, CHUNK256);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], wi, wie, sj / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], wi, wie, sj / 16);\n" \
"	BACKWARD_O(B_256, CHUNK256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256\n" \
"	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))\n" \
"#endif\n" \
"void forward256_0(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	const int lm = LNSIZE - 8; const unsigned int s = 256 / 4;\n" \
"	FORWARD_I_0(B_256, CHUNK256);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], w, we, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], w, we, sj / 4);\n" \
"	FORWARD_O(CHUNK256);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define B_1024	(1024 / 4)\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024\n" \
"	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"#endif\n" \
"void forward1024(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const int lm, const unsigned int s)\n" \
"{\n" \
"	FORWARD_I(B_1024, CHUNK1024);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], w, we, sj / 64);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], w, we, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], w, we, sj / 4);\n" \
"	FORWARD_O(CHUNK1024);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024\n" \
"	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"#endif\n" \
"void backward1024(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,\n" \
"	const int lm, const unsigned int s)\n" \
"{\n" \
"	BACKWARD_I(B_1024, CHUNK1024);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], wi, wie, sj / 4);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], wi, wie, sj / 16);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);\n" \
"	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], wi, wie, sj / 64);\n" \
"	BACKWARD_O(B_1024, CHUNK1024);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024\n" \
"	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))\n" \
"#endif\n" \
"void forward1024_0(__global RNS * restrict const z, __global RNSe * restrict const ze,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	const int lm = LNSIZE - 10; const unsigned int s = 1024 / 4;\n" \
"	FORWARD_I_0(B_1024, CHUNK1024);\n" \
"	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );\n" \
"	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], w, we, sj / 64);\n" \
"	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);\n" \
"	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], w, we, sj / 16);\n" \
"	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);\n" \
"	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], w, we, sj / 4);\n" \
"	FORWARD_O(CHUNK1024);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"#define DECLARE_VAR_32() \\\n" \
"	__local RNS Z[32 * BLK32]; \\\n" \
"	__local RNSe Ze[32 * BLK32]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (32 / 4 * BLK32), group_id = gid / (32 / 4 * BLK32); \\\n" \
"	const sz_t k32 = group_id * 32 * BLK32, i = local_id; \\\n" \
"	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k32 + i32 + i8]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k32 + i32 + i8]; \\\n" \
"	__local RNS * const Z32 = &Z[i32]; \\\n" \
"	__local RNSe * const Z32e = &Ze[i32]; \\\n" \
"	__local RNS * const Zi8 = &Z32[i8]; \\\n" \
"	__local RNSe * const Zi8e = &Z32e[i8]; \\\n" \
"	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z32[i2]; \\\n" \
"	__local RNSe * const Zi2e = &Z32e[i2]; \\\n" \
"	__local RNS * const Z4 = &Z32[4 * i8]; \\\n" \
"	__local RNSe * const Z4e = &Z32e[4 * i8];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void square32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_64() \\\n" \
"	__local RNS Z[64 * BLK64]; \\\n" \
"	__local RNSe Ze[64 * BLK64]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (64 / 4 * BLK64), group_id = gid / (64 / 4 * BLK64); \\\n" \
"	const sz_t k64 = group_id * 64 * BLK64, i = local_id; \\\n" \
"	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k64 + i64 + i16]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k64 + i64 + i16]; \\\n" \
"	__local RNS * const Z64 = &Z[i64]; \\\n" \
"	__local RNSe * const Z64e = &Ze[i64]; \\\n" \
"	__local RNS * const Zi16 = &Z64[i16]; \\\n" \
"	__local RNSe * const Zi16e = &Z64e[i16]; \\\n" \
"	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \\\n" \
"	__local RNS * const Zi4 = &Z64[i4]; \\\n" \
"	__local RNSe * const Zi4e = &Z64e[i4]; \\\n" \
"	__local RNS * const Z4 = &Z64[4 * i16]; \\\n" \
"	__local RNSe * const Z4e = &Z64e[4 * i16];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void square64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	__global const RNS_W * const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	square_4(Z4, Z4e, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);\n" \
"	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_128() \\\n" \
"	__local RNS Z[128 * BLK128]; \\\n" \
"	__local RNSe Ze[128 * BLK128]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (128 / 4 * BLK128), group_id = gid / (128 / 4 * BLK128); \\\n" \
"	const sz_t k128 = group_id * 128 * BLK128, i = local_id; \\\n" \
"	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k128 + i128 + i32]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k128 + i128 + i32]; \\\n" \
"	__local RNS * const Z128 = &Z[i128]; \\\n" \
"	__local RNSe * const Z128e = &Ze[i128]; \\\n" \
"	__local RNS * const Zi32 = &Z128[i32]; \\\n" \
"	__local RNSe * const Zi32e = &Z128e[i32]; \\\n" \
"	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \\\n" \
"	__local RNS * const Zi8 = &Z128[i8]; \\\n" \
"	__local RNSe * const Zi8e = &Z128e[i8]; \\\n" \
"	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z128[i2]; \\\n" \
"	__local RNSe * const Zi2e = &Z128e[i2]; \\\n" \
"	__local RNS * const Z4 = &Z128[4 * i32]; \\\n" \
"	__local RNSe * const Z4e = &Z128e[4 * i32];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128\n" \
"	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void square128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_256() \\\n" \
"	__local RNS Z[256 * BLK256]; \\\n" \
"	__local RNSe Ze[256 * BLK256]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (256 / 4 * BLK256), group_id = gid / (256 / 4 * BLK256); \\\n" \
"	const sz_t k256 = group_id * 256 * BLK256, i = local_id; \\\n" \
"	const sz_t i256 = 0, i64 = i; \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k256 + i256 + i64]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k256 + i256 + i64]; \\\n" \
"	__local RNS * const Z256 = &Z[i256]; \\\n" \
"	__local RNSe * const Z256e = &Ze[i256]; \\\n" \
"	__local RNS * const Zi64 = &Z256[i64]; \\\n" \
"	__local RNSe * const Zi64e = &Z256e[i64]; \\\n" \
"	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \\\n" \
"	__local RNS * const Zi16 = &Z256[i16]; \\\n" \
"	__local RNSe * const Zi16e = &Z256e[i16]; \\\n" \
"	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \\\n" \
"	__local RNS * const Zi4 = &Z256[i4]; \\\n" \
"	__local RNSe * const Zi4e = &Z256e[i4]; \\\n" \
"	__local RNS * const Z4 = &Z256[4 * i64]; \\\n" \
"	__local RNSe * const Z4e = &Z256e[4 * i64];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256\n" \
"	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void square256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);\n" \
"	forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	square_4(Z4, Z4e, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);\n" \
"	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"	backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_512() \\\n" \
"	__local RNS Z[512]; \\\n" \
"	__local RNSe Ze[512]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (512 / 4), group_id = gid / (512 / 4); \\\n" \
"	const sz_t k512 = group_id * 512, i128 = local_id; \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k512 + i128]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k512 + i128]; \\\n" \
"	__local RNS * const Zi128 = &Z[i128]; \\\n" \
"	__local RNSe * const Zi128e = &Ze[i128]; \\\n" \
"	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \\\n" \
"	__local RNS * const Zi32 = &Z[i32]; \\\n" \
"	__local RNSe * const Zi32e = &Ze[i32]; \\\n" \
"	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \\\n" \
"	__local RNS * const Zi8 = &Z[i8]; \\\n" \
"	__local RNSe * const Zi8e = &Ze[i8]; \\\n" \
"	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z[i2]; \\\n" \
"	__local RNSe * const Zi2e = &Ze[i2]; \\\n" \
"	__local RNS * const Z4 = &Z[4 * i128]; \\\n" \
"	__local RNSe * const Z4e = &Ze[4 * i128];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 512 / 4\n" \
"	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void square512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);\n" \
"	forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"	backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_1024() \\\n" \
"	__local RNS Z[1024]; \\\n" \
"	__local RNSe Ze[1024]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (1024 / 4), group_id = gid / (1024 / 4); \\\n" \
"	const sz_t k1024 = group_id * 1024, i256 = local_id; \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k1024 + i256]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k1024 + i256]; \\\n" \
"	__local RNS * const Zi256 = &Z[i256]; \\\n" \
"	__local RNSe * const Zi256e = &Ze[i256]; \\\n" \
"	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \\\n" \
"	__local RNS * const Zi64 = &Z[i64]; \\\n" \
"	__local RNSe * const Zi64e = &Ze[i64]; \\\n" \
"	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \\\n" \
"	__local RNS * const Zi16 = &Z[i16]; \\\n" \
"	__local RNSe * const Zi16e = &Ze[i16]; \\\n" \
"	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \\\n" \
"	__local RNS * const Zi4 = &Z[i4]; \\\n" \
"	__local RNSe * const Zi4e = &Ze[i4]; \\\n" \
"	__local RNS * const Z4 = &Z[4 * i256]; \\\n" \
"	__local RNSe * const Z4e = &Ze[4 * i256];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 1024 / 4\n" \
"	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void square1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);\n" \
"	forward_4(64, Zi64, Zi64e, w, we, j / 64);\n" \
"	forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	square_4(Z4, Z4e, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);\n" \
"	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"	backward_4(64, Zi64, Zi64e, wi, wie, j / 64);\n" \
"	backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);\n" \
"}\n" \
"\n" \
"#define DECLARE_VAR_2048() \\\n" \
"	__local RNS Z[2048]; \\\n" \
"	__local RNSe Ze[2048]; \\\n" \
"	\\\n" \
"	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \\\n" \
"	const sz_t local_id = gid % (2048 / 4), group_id = gid / (2048 / 4); \\\n" \
"	const sz_t k2048 = group_id * 2048, i512 = local_id; \\\n" \
"	\\\n" \
"	__global RNS * restrict const zk = &z[k2048 + i512]; \\\n" \
"	__global RNSe * restrict const zke = &ze[k2048 + i512]; \\\n" \
"	__local RNS * const Zi512 = &Z[i512]; \\\n" \
"	__local RNSe * const Zi512e = &Ze[i512]; \\\n" \
"	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \\\n" \
"	__local RNS * const Zi128 = &Z[i128]; \\\n" \
"	__local RNSe * const Zi128e = &Ze[i128]; \\\n" \
"	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \\\n" \
"	__local RNS * const Zi32 = &Z[i32]; \\\n" \
"	__local RNSe * const Zi32e = &Ze[i32]; \\\n" \
"	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \\\n" \
"	__local RNS * const Zi8 = &Z[i8]; \\\n" \
"	__local RNSe * const Zi8e = &Ze[i8]; \\\n" \
"	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \\\n" \
"	__local RNS * const Zi2 = &Z[i2]; \\\n" \
"	__local RNSe * const Zi2e = &Ze[i2]; \\\n" \
"	__local RNS * const Z4 = &Z[4 * i512]; \\\n" \
"	__local RNSe * const Z4e = &Ze[4 * i512];\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 2048 / 4\n" \
"	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void square2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);\n" \
"	forward_4(128, Zi128, Zi128e, w, we, j / 128);\n" \
"	forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"	backward_4(128, Zi128, Zi128e, wi, wie, j / 128);\n" \
"	backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void fwd32p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"\n" \
"	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	write_4(8, zk, zke, Z4, Z4e);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void fwd64p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"\n" \
"	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	fwd2write_4(16, zk, zke, Z4, Z4e, w[j], we[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128\n" \
"	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void fwd128p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"\n" \
"	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	write_4(32, zk, zke, Z4, Z4e);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256\n" \
"	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void fwd256p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"\n" \
"	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);\n" \
"	forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	fwd2write_4(64, zk, zke, Z4, Z4e, w[j], we[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 512 / 4\n" \
"	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd512p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"\n" \
"	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);\n" \
"	forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	write_4(128, zk, zke, Z4, Z4e);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 1024 / 4\n" \
"	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd1024p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"\n" \
"	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);\n" \
"	forward_4(64, Zi64, Zi64e, w, we, j / 64);\n" \
"	forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	fwd2write_4(256, zk, zke, Z4, Z4e, w[j], we[j]);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 2048 / 4\n" \
"	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void fwd2048p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"\n" \
"	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);\n" \
"	forward_4(128, Zi128, Zi128e, w, we, j / 128);\n" \
"	forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	write_4(512, zk, zke, Z4, Z4e);\n" \
"}\n" \
"\n" \
"// -----------------\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32\n" \
"	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))\n" \
"#endif\n" \
"void mul32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_32();\n" \
"	__global const RNS * restrict const zpk = &zp[k32 + i32 + i8];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k32 + i32 + i8];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	mul_22(Z4, Z4e, 8, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64\n" \
"	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))\n" \
"#endif\n" \
"void mul64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_64();\n" \
"	__global const RNS * restrict const zpk = &zp[k64 + i64 + i16];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k64 + i64 + i16];\n" \
"	__global const RNS_W * const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	mul_4(Z4, Z4e, 16, zpk, zpke, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);\n" \
"	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128\n" \
"	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))\n" \
"#endif\n" \
"void mul128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_128();\n" \
"	__global const RNS * restrict const zpk = &zp[k128 + i128 + i32];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k128 + i128 + i32];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	mul_22(Z4, Z4e, 32, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256\n" \
"	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))\n" \
"#endif\n" \
"void mul256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_256();\n" \
"	__global const RNS * restrict const zpk = &zp[k256 + i256 + i64];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k256 + i256 + i64];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);\n" \
"	forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	mul_4(Z4, Z4e, 64, zpk, zpke, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);\n" \
"	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"	backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 512 / 4\n" \
"	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_512();\n" \
"	__global const RNS * restrict const zpk = &zp[k512 + i128];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k512 + i128];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);\n" \
"	forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	mul_22(Z4, Z4e, 128, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"	backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 1024 / 4\n" \
"	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_1024();\n" \
"	__global const RNS * restrict const zpk = &zp[k1024 + i256];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k1024 + i256];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);\n" \
"	forward_4(64, Zi64, Zi64e, w, we, j / 64);\n" \
"	forward_4(16, Zi16, Zi16e, w, we, j / 16);\n" \
"	forward_4(4, Zi4, Zi4e, w, we, j / 4);\n" \
"	mul_4(Z4, Z4e, 256, zpk, zpke, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);\n" \
"	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);\n" \
"	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);\n" \
"	backward_4(64, Zi64, Zi64e, wi, wie, j / 64);\n" \
"	backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"#if MAX_WORK_GROUP_SIZE >= 2048 / 4\n" \
"	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))\n" \
"#endif\n" \
"void mul2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,\n" \
"	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)\n" \
"{\n" \
"	DECLARE_VAR_2048();\n" \
"	__global const RNS * restrict const zpk = &zp[k2048 + i512];\n" \
"	__global const RNSe * restrict const zpke = &zpe[k2048 + i512];\n" \
"	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];\n" \
"	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];\n" \
"\n" \
"	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);\n" \
"	forward_4(128, Zi128, Zi128e, w, we, j / 128);\n" \
"	forward_4(32, Zi32, Zi32e, w, we, j / 32);\n" \
"	forward_4(8, Zi8, Zi8e, w, we, j / 8);\n" \
"	forward_4(2, Zi2, Zi2e, w, we, j / 2);\n" \
"	mul_22(Z4, Z4e, 512, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);\n" \
"	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);\n" \
"	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);\n" \
"	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);\n" \
"	backward_4(128, Zi128, Zi128e, wi, wie, j / 128);\n" \
"	backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);\n" \
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
"__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))\n" \
"void normalize1(__global RNS2 * restrict const z, __global RNS2e * restrict const ze, __global int64 * restrict const c,\n" \
"	const uint32 b, const uint32 b_inv, const int b_s, const int32 dup)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;\n" \
"	__global RNS2 * restrict const zi = &z[2 * gid];\n" \
"	__global RNS2e * restrict const zie = &ze[2 * gid];\n" \
"	__local int64 cl[NORM_WG_SZ];\n" \
"\n" \
"	// Not converted into Montgomery form such that output is converted out of Montgomery form\n" \
"	const RNS norm = (RNS)(NORM1, NORM2);\n" \
"	const RNSe norme = (RNSe)(NORM3);\n" \
"\n" \
"	const RNS2 u01 = mul2(zi[0], norm), u23 = mul2(zi[1], norm);\n" \
"	const RNS2e u01e = mul2e(zie[0], norme), u23e = mul2e(zie[1], norme);\n" \
"\n" \
"	int32_4 r;\n" \
"	int96 l0 = garner3(u01.s0, u01.s1, u01e.s0); if (dup != 0) l0 = int96_add(l0, l0);\n" \
"	int96 f96 = l0; r.s0 = reduce96(&f96, b, b_inv, b_s);\n" \
"	int96 l1 = garner3(u01.s2, u01.s3, u01e.s1); if (dup != 0) l1 = int96_add(l1, l1);\n" \
"	f96 = int96_add(f96, l1); r.s1 = reduce96(&f96, b, b_inv, b_s);\n" \
"	int96 l2 = garner3(u23.s0, u23.s1, u23e.s0); if (dup != 0) l2 = int96_add(l2, l2);\n" \
"	f96 = int96_add(f96, l2); r.s2 = reduce96(&f96, b, b_inv, b_s);\n" \
"	int96 l3 = garner3(u23.s2, u23.s3, u23e.s1); if (dup != 0) l3 = int96_add(l3, l3);\n" \
"	f96 = int96_add(f96, l3); r.s3 = reduce96(&f96, b, b_inv, b_s);\n" \
"\n" \
"	int64 f = (int64)(f96.s0);\n" \
"	cl[lid] = f;\n" \
"\n" \
"	if (lid == NORM_WG_SZ - 1)\n" \
"	{\n" \
"		const sz_t i = (gid / NORM_WG_SZ + 1) % (NSIZE_4 / NORM_WG_SZ);\n" \
"		c[i] = (i == 0) ? -f : f;\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	f = (lid == 0) ? 0 : cl[lid - 1];\n" \
"	f += r.s0; r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s1; r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s2; r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s3; r.s3 = (sz_t)(f);\n" \
"\n" \
"	zi[0] = (RNS2)(toRNS(r.s0), toRNS(r.s1)); zi[1] = (RNS2)(toRNS(r.s2), toRNS(r.s3));\n" \
"	zie[0] = (RNS2e)(toRNSe(r.s0), toRNSe(r.s1)); zie[1] = (RNS2e)(toRNSe(r.s2), toRNSe(r.s3));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const int64 * restrict const c, \n" \
"	const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0);\n" \
"	__global RNS * restrict const zi = &z[NORM_WG_SZ * 4 * gid];\n" \
"	__global RNSe * restrict const zie = &ze[NORM_WG_SZ * 4 * gid];\n" \
"\n" \
"	int64 f = c[gid];\n" \
"\n" \
"	for (sz_t j = 0; j < 3; ++j)\n" \
"	{\n" \
"		f += geti_P3(zie[j]);\n" \
"		const int32 r = reduce64(&f, b, b_inv, b_s);\n" \
"		zi[j] = toRNS(r); zie[j] = toRNSe(r);\n" \
"		if (f == 0) return;\n" \
"	}\n" \
"	f += geti_P3(zie[3]);\n" \
"	zi[3] = toRNS((int32)(f)); zie[3] = toRNSe((int32)(f));\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))\n" \
"void mulscalar(__global RNS * restrict const z, __global RNSe * restrict const ze, __global int64 * restrict const c,\n" \
"	const uint32 b, const uint32 b_inv, const int b_s, const int32 a)\n" \
"{\n" \
"	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;\n" \
"	__global RNS * restrict const zi = &z[4 * gid];\n" \
"	__global RNSe * restrict const zie = &ze[4 * gid];\n" \
"	__local int64 cl[NORM_WG_SZ];\n" \
"\n" \
"	int32_4 r;\n" \
"	int64 f = geti_P3(zie[0]) * (int64)(a);\n" \
"	r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += geti_P3(zie[1]) * (int64)(a);\n" \
"	r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += geti_P3(zie[2]) * (int64)(a);\n" \
"	r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += geti_P3(zie[3]) * (int64)(a);\n" \
"	r.s3 = reduce64(&f, b, b_inv, b_s);\n" \
"\n" \
"	cl[lid] = f;\n" \
"\n" \
"	if (lid == NORM_WG_SZ - 1)\n" \
"	{\n" \
"		const sz_t i = (gid / NORM_WG_SZ + 1) % (NSIZE_4 / NORM_WG_SZ);\n" \
"		c[i] = (i == 0) ? -f : f;\n" \
"	}\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	f = (lid == 0) ? 0 : cl[lid - 1];\n" \
"	f += r.s0; r.s0 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s1; r.s1 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s2; r.s2 = reduce64(&f, b, b_inv, b_s);\n" \
"	f += r.s3; r.s3 = (sz_t)(f);\n" \
"\n" \
"	zi[0] = toRNS(r.s0); zi[1] = toRNS(r.s1); zi[2] = toRNS(r.s2); zi[3] = toRNS(r.s3);\n" \
"	zie[0] = toRNSe(r.s0); zie[1] = toRNSe(r.s1); zie[2] = toRNSe(r.s2); zie[3] = toRNSe(r.s3);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void set(__global RNS2 * restrict const z, __global RNS2e * restrict const ze, const uint32 a)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	const uint32 ai = (idx == 0) ? a : 0;\n" \
"	z[idx] = (RNS2)(ai, ai, 0, 0);\n" \
"	ze[idx] = (RNS2e)(ai, 0);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global RNS2 * restrict const z, __global RNS2e * restrict const ze, const sz_t dst, const sz_t src)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	z[dst + idx] = z[src + idx];\n" \
"	ze[dst + idx] = ze[src + idx];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copyp(__global RNS2 * restrict const zp, __global RNS2e * restrict const zpe,\n" \
"		   __global const RNS2 * restrict const z, __global const RNS2e * restrict const ze, const sz_t src)\n" \
"{\n" \
"	const sz_t idx = (sz_t)get_global_id(0);\n" \
"	zp[idx] = z[src + idx];\n" \
"	zpe[idx] = ze[src + idx];\n" \
"}\n" \
"";
