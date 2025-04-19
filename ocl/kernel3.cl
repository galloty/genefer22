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

typedef uint	sz_t;
typedef uint	uint32;
typedef int		int32;
typedef ulong	uint64;
typedef long	int64;

#if !defined(LNSIZE)
#define LNSIZE		16
#define NSIZE_4		16384u
#define P1			4194304001u
#define P2			4076863489u
#define P3			3942645761u
#define Q1			100663297u
#define Q2			218103809u
#define Q3			352321537u
#define R1			232465106u
#define R2			3444438393u
#define R3			3810498414u
#define NORM1		4193792001u
#define NORM2		4076365825u
#define NORM3		3941683201u
#define InvP2_P1	1797558821u
#define InvP3_P1	3075822917u
#define InvP3_P2	4076863457u
#define P1P2P3l		12816400126780112897ul
#define P1P2P3h		3654720002u
#define P1P2P3_2l	6408200063390056448ul
#define P1P2P3_2h	1827360001u
#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1
#define CHUNK64		8
#define CHUNK256	4
#define CHUNK1024	2
#define NORM_WG_SZ	64
#define MAX_WORK_GROUP_SIZE	256
#endif

#define P1P2	(P1 * (uint64)(P2))
#define P2P3	(P2 * (uint64)(P3))

// --- uint96/int96 ---

typedef struct { uint64 s0; uint32 s1; } uint96;
typedef struct { uint64 s0; int32 s1; } int96;

INLINE int96 int96_set_si(const int64 n) { int96 r; r.s0 = (ulong)n; r.s1 = (n < 0) ? -1 : 0; return r; }
INLINE uint96 uint96_set(const uint64 s0, const uint32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }

INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int32)(x.s1); return r; }
INLINE uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint32)(x.s1); return r; }

INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }

INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

INLINE int96 int96_neg(const int96 x)
{
	const int32 c = (x.s0 != 0) ? 1 : 0;
	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;
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
	const int32 c = (s0 < y.s0) ? 1 : 0;
	r.s0 = s0; r.s1 = x.s1 + y.s1 + c;
#endif
	return r;
}

INLINE uint96 uint96_add_64(const uint96 x, const uint64 y)
{
	uint96 r;
#ifdef PTX_ASM
	asm volatile ("add.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (r.s1) : "r" (x.s1));
#else
	const uint64 s0 = x.s0 + y;
	const uint32 c = (s0 < y) ? 1 : 0;
	r.s0 = s0; r.s1 = x.s1 + c;
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
	const uint32 c = (x.s0 < y.s0) ? 1 : 0;
	r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - c);
#endif
	return r;
}

INLINE uint96 uint96_mul_64_32(const uint64 x, const uint32 y)
{
	const uint64 l = (uint32)(x) * (uint64)(y), h = (x >> 32) * y + (l >> 32);
	uint96 r; r.s0 = (h << 32) | (uint32)(l); r.s1 = (uint32)(h >> 32);
	return r;
}

// --- mod arith ---

INLINE uint32 _addMod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 c = (lhs >= p - rhs) ? p : 0;
	return lhs + rhs - c;
}

INLINE uint32 _subMod(const uint32 lhs, const uint32 rhs, const uint32 p)
{
	const uint32 c = (lhs < rhs) ? p : 0;
	return lhs - rhs + c;
}

// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.

// Montgomery form of n is n * 2^32 mod p. q * p = 1 mod 2^32.

// r = lhs * rhs * 2^-32 mod p
// If lhs = x * 2^32 and rhs = y * 2^32 then r = (x * y) * 2^32 mod p.
// If lhs = x and rhs = y * 2^32 then r = x * y mod p.
INLINE uint32 _mulMonty(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 q)
{
	const uint32 t_lo = lhs * rhs, t_hi = mul_hi(lhs, rhs);
	const uint32 mp = mul_hi(t_lo * q, p);
	return _subMod(t_hi, mp, p);
}

// Conversion into Montgomery form
INLINE uint32 _toMonty(const uint32 n, const uint32 r2, const uint32 p, const uint32 q)
{
	// n * (2^32)^2 = (n * 2^32) * (1 * 2^32)
	return _mulMonty(n, r2, p, q);
}

// Conversion out of Montgomery form
// INLINE uint32 _fromMonty(const uint32 n, const uint32 p, const uint32 q)
// {
// 	// If n = x * 2^32 mod p then _mulMonty(n, 1, p, q) = x.
// 	const uint32 mp = mul_hi(n * q, p);
// 	return (mp != 0) ? p - mp : 0;
// }

INLINE uint32 add_P1(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P1); }
INLINE uint32 add_P2(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P2); }
INLINE uint32 add_P3(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P3); }

INLINE uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }
INLINE uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }
INLINE uint32 sub_P3(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P3); }

// Montgomery form
INLINE uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P1, Q1); }
INLINE uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P2, Q2); }
INLINE uint32 mul_P3(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P3, Q3); }

INLINE uint32 toMonty_P1(const uint32 lhs) { return _toMonty(lhs, R1, P1, Q1); }
INLINE uint32 toMonty_P2(const uint32 lhs) { return _toMonty(lhs, R2, P2, Q2); }
INLINE uint32 toMonty_P3(const uint32 lhs) { return _toMonty(lhs, R3, P3, Q3); }

// INLINE uint32 fromMonty_P1(const uint32 lhs) { return _fromMonty(lhs, P1, Q1); }
// INLINE uint32 fromMonty_P2(const uint32 lhs) { return _fromMonty(lhs, P2, Q2); }
// INLINE uint32 fromMonty_P3(const uint32 lhs) { return _fromMonty(lhs, P3, Q3); }

INLINE int32 geti_P3(const uint32 n) { return (n > P3 / 2) ? (int32)(n - P3) : (int32)(n); }

INLINE int96 garner3(const uint32 r1, const uint32 r2, const uint32 r3)
{
	const uint32 u13 = mul_P1(sub_P1(r1, r3), InvP3_P1);
	const uint32 u23 = mul_P2(sub_P2(r2, r3), InvP3_P2);
	const uint32 u123 = mul_P1(sub_P1(u13, u23), InvP2_P1);
	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (ulong)P3 + r3);
	const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);
	const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);
	return r;
}

// --- RNS/RNSe ---

typedef uint2	RNS;
typedef RNS		RNS_W;
typedef uint	RNSe;
typedef RNSe	RNS_We;

INLINE RNS toRNS(const int32 i) { return ((RNS)(i, i) + ((i < 0) ? (RNS)(P1, P2) : (RNS)(0, 0))); }

INLINE RNS add(const RNS lhs, const RNS rhs) { return (RNS)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }
INLINE RNS sub(const RNS lhs, const RNS rhs) { return (RNS)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }
INLINE RNS mul(const RNS lhs, const RNS rhs) { return (RNS)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }

INLINE RNS sqr(const RNS lhs) { return mul(lhs, lhs); }

INLINE RNS mulW(const RNS lhs, const RNS_W w) { return mul(lhs, w); }

INLINE RNS toMonty(const RNS lhs) { return (RNS)(toMonty_P1(lhs.s0), toMonty_P2(lhs.s1)); }

INLINE RNSe toRNSe(const int i) { return ((RNSe)(i) + ((i < 0) ? (RNSe)(P3) : (RNSe)(0))); }

INLINE RNSe adde(const RNSe lhs, const RNSe rhs) { return (RNSe)(add_P3(lhs, rhs)); }
INLINE RNSe sube(const RNSe lhs, const RNSe rhs) { return (RNSe)(sub_P3(lhs, rhs)); }
INLINE RNSe mule(const RNSe lhs, const RNSe rhs) { return (RNSe)(mul_P3(lhs, rhs)); }

INLINE RNSe sqre(const RNSe lhs) { return mule(lhs, lhs); }

INLINE RNSe mulWe(const RNSe lhs, const RNS_We w) { return mule(lhs, w); }

INLINE RNSe toMontye(const RNSe lhs) { return (RNSe)toMonty_P3(lhs); }

// --- transform/macro ---

#define FWD2(z0, z1, w) { const RNS t = mulW(z1, w); z1 = sub(z0, t); z0 = add(z0, t); }
#define FWD2e(z0e, z1e, we) { const RNSe t = mulWe(z1e, we); z1e = sube(z0e, t); z0e = adde(z0e, t); }

#define BCK2(z0, z1, wi) { const RNS t = sub(z0, z1); z0 = add(z0, z1); z1 = mulW(t, wi); }
#define BCK2e(z0e, z1e, wie) { const RNSe t = sube(z0e, z1e); z0e = adde(z0e, z1e); z1e = mulWe(t, wie); }

#define SQR2(z0, z1, w) { const RNS t = sqr(mulW(z1, w)); z1 = mul(add(z0, z0), z1); z0 = add(sqr(z0), t); }
#define SQR2e(z0e, z1e, we) { const RNSe t = sqre(mulWe(z1e, we)); z1e = mule(adde(z0e, z0e), z1e); z0e = adde(sqre(z0e), t); }
#define SQR2N(z0, z1, w) { const RNS t = sqr(mulW(z1, w)); z1 = mul(add(z0, z0), z1); z0 = sub(sqr(z0), t); }
#define SQR2Ne(z0e, z1e, we) { const RNSe t = sqre(mulWe(z1e, we)); z1e = mule(adde(z0e, z0e), z1e); z0e = sube(sqre(z0e), t); }

#define MUL2(z0, z1, zp0, zp1, w) { const RNS t = mul(mulW(z1, w), mulW(zp1, w)); z1 = add(mul(z0, zp1), mul(zp0, z1)); z0 = add(mul(z0, zp0), t); }
#define MUL2e(z0e, z1e, zp0e, zp1e, we) { const RNSe t = mule(mulWe(z1e, we), mulWe(zp1e, we)); z1e = adde(mule(z0e, zp1e), mule(zp0e, z1e)); z0e = adde(mule(z0e, zp0e), t); }
#define MUL2N(z0, z1, zp0, zp1, w) { const RNS t = mul(mulW(z1, w), mulW(zp1, w)); z1 = add(mul(z0, zp1), mul(zp0, z1)); z0 = sub(mul(z0, zp0), t); }
#define MUL2Ne(z0e, z1e, zp0e, zp1e, we) { const RNSe t = mule(mulWe(z1e, we), mulWe(zp1e, we)); z1e = adde(mule(z0e, zp1e), mule(zp0e, z1e)); z0e = sube(mule(z0e, zp0e), t); }

// --- transform/inline ---

INLINE void _loadg(RNS zl[4], __global const RNS * restrict const z, const size_t s) { for (size_t l = 0; l < 4; ++l) zl[l] = z[l * s]; }
INLINE void _loadge(RNSe zle[4], __global const RNSe * restrict const ze, const size_t s) { for (size_t l = 0; l < 4; ++l) zle[l] = ze[l * s]; }
INLINE void _loadl(RNS zl[4], __local const RNS * restrict const Z, const size_t s) { for (size_t l = 0; l < 4; ++l) zl[l] = Z[l * s]; }
INLINE void _loadle(RNSe zle[4], __local const RNSe * restrict const Ze, const size_t s) { for (size_t l = 0; l < 4; ++l) zle[l] = Ze[l * s]; }
INLINE void _storeg(__global RNS * restrict const z, const size_t s, const RNS zl[4]) { for (size_t l = 0; l < 4; ++l) z[l * s] = zl[l]; }
INLINE void _storege(__global RNSe * restrict const ze, const size_t s, const RNSe zle[4]) { for (size_t l = 0; l < 4; ++l) ze[l * s] = zle[l]; }
INLINE void _storel(__local RNS * restrict const Z, const size_t s, const RNS zl[4]) { for (size_t l = 0; l < 4; ++l) Z[l * s] = zl[l]; }
INLINE void _storele(__local RNSe * restrict const Ze, const size_t s, const RNSe zle[4]) { for (size_t l = 0; l < 4; ++l) Ze[l * s] = zle[l]; }

INLINE void forward_4(const sz_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	const RNS_W w1 = w[j], w2 = w[2 * j + 0], w3 = w[2 * j + 1];
	const RNS_We w1e = we[j], w2e = we[2 * j + 0], w3e = we[2 * j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, m); _loadle(zle, Ze, m);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	FWD2(zl[0], zl[1], w2); FWD2(zl[2], zl[3], w3); FWD2e(zle[0], zle[1], w2e); FWD2e(zle[2], zle[3], w3e);
	_storel(Z, m, zl); _storele(Ze, m, zle);
}

INLINE void forward_4i(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	const RNS_W w1 = w[j], w2 = w[2 * j + 0], w3 = w[2 * j + 1];
	const RNS_We w1e = we[j], w2e = we[2 * j + 0], w3e = we[2 * j + 1];
	RNS zl[4]; RNSe zle[4]; _loadg(zl, z, mg); _loadge(zle, ze, mg);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	FWD2(zl[0], zl[1], w2); FWD2(zl[2], zl[3], w3); FWD2e(zle[0], zle[1], w2e); FWD2e(zle[2], zle[3], w3e);
	_storel(Z, ml, zl); _storele(Ze, ml, zle);
}

INLINE void forward_4i_0(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	const RNS_W w1 = w[1], w2 = w[2], w3 = w[3];
	const RNS_We w1e = we[1], w2e = we[2], w3e = we[3];
	RNS zl[4]; RNSe zle[4]; _loadg(zl, z, mg); _loadge(zle, ze, mg);
	zl[0] = toMonty(zl[0]); zl[1] = toMonty(zl[1]); zle[0] = toMontye(zle[0]); zle[1] = toMontye(zle[1]);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	FWD2(zl[0], zl[1], w2); FWD2(zl[2], zl[3], w3); FWD2e(zle[0], zle[1], w2e); FWD2e(zle[2], zle[3], w3e);
	_storel(Z, ml, zl); _storele(Ze, ml, zle);
}

INLINE void forward_4o(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	const sz_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	const RNS_W w1 = w[j], w2 = w[2 * j + 0], w3 = w[2 * j + 1];
	const RNS_We w1e = we[j], w2e = we[2 * j + 0], w3e = we[2 * j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, ml); _loadle(zle, Ze, ml);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	FWD2(zl[0], zl[1], w2); FWD2(zl[2], zl[3], w3); FWD2e(zle[0], zle[1], w2e); FWD2e(zle[2], zle[3], w3e);
	_storeg(z, mg, zl); _storege(ze, mg, zle);
}

INLINE void backward_4(const sz_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)
{
	const RNS_W wi1 = wi[j], wi2 = wi[2 * j + 0], wi3 = wi[2 * j + 1];
	const RNS_We wi1e = wie[j], wi2e = wie[2 * j + 0], wi3e = wie[2 * j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, m); _loadle(zle, Ze, m);
	BCK2(zl[0], zl[1], wi2); BCK2(zl[2], zl[3], wi3); BCK2e(zle[0], zle[1], wi2e); BCK2e(zle[2], zle[3], wi3e);
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);
	_storel(Z, m, zl); _storele(Ze, m, zle);
}

INLINE void backward_4i(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,
	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)
{
	const RNS_W wi1 = wi[j], wi2 = wi[2 * j + 0], wi3 = wi[2 * j + 1];
	const RNS_We wi1e = wie[j], wi2e = wie[2 * j + 0], wi3e = wie[2 * j + 1];
	RNS zl[4]; RNSe zle[4]; _loadg(zl, z, mg); _loadge(zle, ze, mg);
	BCK2(zl[0], zl[1], wi2); BCK2(zl[2], zl[3], wi3); BCK2e(zle[0], zle[1], wi2e); BCK2e(zle[2], zle[3], wi3e);
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);
	_storel(Z, ml, zl); _storele(Ze, ml, zle);
}

INLINE void backward_4o(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	const sz_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,
	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)
{
	const RNS_W wi1 = wi[j], wi2 = wi[2 * j + 0], wi3 = wi[2 * j + 1];
	const RNS_We wi1e = wie[j], wi2e = wie[2 * j + 0], wi3e = wie[2 * j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, ml); _loadle(zle, Ze, ml);
	BCK2(zl[0], zl[1], wi2); BCK2(zl[2], zl[3], wi3); BCK2e(zle[0], zle[1], wi2e); BCK2e(zle[2], zle[3], wi3e);
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);
	_storeg(z, mg, zl); _storege(ze, mg, zle);
}

INLINE void write_4(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	__local const RNS * restrict const Z, __local const RNSe * restrict const Ze)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0 * mg] = Z[0]; z[1 * mg] = Z[1]; z[2 * mg] = Z[2]; z[3 * mg] = Z[3];
	ze[0 * mg] = Ze[0]; ze[1 * mg] = Ze[1]; ze[2 * mg] = Ze[2]; ze[3 * mg] = Ze[3];
}

INLINE void fwd2write_4(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	__local const RNS * restrict const Z, __local const RNSe * restrict const Ze, const RNS_W w1, const RNS_We w1e)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	_storeg(z, mg, zl); _storege(ze, mg, zle);
}

INLINE void square_22(__local RNS * restrict const Z, __local RNSe * restrict const Ze, const RNS_W w0, const RNS_We w0e)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);
	SQR2(zl[0], zl[1], w0); SQR2N(zl[2], zl[3], w0); SQR2e(zle[0], zle[1], w0e); SQR2Ne(zle[2], zle[3], w0e);
	_storel(Z, 1, zl); _storele(Ze, 1, zle);
}

INLINE void square_4(__local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const RNS_W w1, const RNS_W wi1, const RNS_W w0, const RNS_We w1e, const RNS_We wi1e, const RNS_We w0e)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	SQR2(zl[0], zl[1], w0); SQR2N(zl[2], zl[3], w0); SQR2e(zle[0], zle[1], w0e); SQR2Ne(zle[2], zle[3], w0e);
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);
	_storel(Z, 1, zl); _storele(Ze, 1, zle);
}

INLINE void mul_22(__local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe, const RNS_W w0, const RNS_We w0e)
{
	RNS zpl[4]; RNSe zple[4]; _loadg(zpl, zp, mg); _loadge(zple, zpe, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);
	MUL2(zl[0], zl[1], zpl[0], zpl[1], w0); MUL2N(zl[2], zl[3], zpl[2], zpl[3], w0);
	MUL2e(zle[0], zle[1], zple[0], zple[1], w0e); MUL2Ne(zle[2], zle[3], zple[2], zple[3], w0e);
	_storel(Z, 1, zl); _storele(Ze, 1, zle);
}

INLINE void mul_4(__local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe, 
	const RNS_W w1, const RNS_W wi1, const RNS_W w0, const RNS_We w1e, const RNS_We wi1e, const RNS_We w0e)
{
	RNS zpl[4]; RNSe zple[4]; _loadg(zpl, zp, mg); _loadge(zple, zpe, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; RNSe zle[4]; _loadl(zl, Z, 1); _loadle(zle, Ze, 1);
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); FWD2e(zle[0], zle[2], w1e); FWD2e(zle[1], zle[3], w1e);
	MUL2(zl[0], zl[1], zpl[0], zpl[1], w0); MUL2N(zl[2], zl[3], zpl[2], zpl[3], w0);
	MUL2e(zle[0], zle[1], zple[0], zple[1], w0e); MUL2Ne(zle[2], zle[3], zple[2], zple[3], w0e);
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1); BCK2e(zle[0], zle[2], wi1e); BCK2e(zle[1], zle[3], wi1e);
	_storel(Z, 1, zl); _storele(Ze, 1, zle);
}

// --- transform ---

#define DECLARE_VAR(B_N, CHUNK_N) \
	__local RNS Z[4 * B_N * CHUNK_N]; \
	__local RNSe Ze[4 * B_N * CHUNK_N]; \
	\
	/* threadIdx < B_N */ \
	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \
	__local RNS * const Zi = &Z[chunk_idx]; \
	__local RNSe * const Zie = &Ze[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	sz_t sj = s + idx_m;

#define DECLARE_VAR_FORWARD() \
	__global RNS * __restrict__ const zi = &z[ki]; \
	__global RNSe * __restrict__ const zie = &ze[ki]; \
	__global RNS * __restrict__ const zo = &z[ko]; \
	__global RNSe * __restrict__ const zoe = &ze[ko];

#define DECLARE_VAR_BACKWARD() \
	__global RNS * __restrict__ const zi = &z[ko]; \
	__global RNSe * __restrict__ const zie = &ze[ko]; \
	__global RNS * __restrict__ const zo = &z[ki]; \
	__global RNSe * __restrict__ const zoe = &ze[ki]; \
	const sz_t n_4 = NSIZE_4; \
	__global const RNS_W * restrict const wi = &w[4 * n_4]; \
	__global const RNS_We * restrict const wie = &we[4 * n_4];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(B_N * CHUNK_N, &Z[i], &Ze[i], B_N << lm, zi, zie, w, we, sj / B_N);

#define FORWARD_I_0(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_0(B_N * CHUNK_N, &Z[i], &Ze[i], B_N << lm, zi, zie, w, we);

#define FORWARD_O(CHUNK_N) \
	forward_4o((sz_t)1 << lm, zo, zoe, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &Zie[CHUNK_N * 4 * threadIdx], w, we, sj / 1);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], &Zie[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, zie, wi, wie, sj / 1);

#define BACKWARD_O(B_N, CHUNK_N) \
	backward_4o(B_N << lm, zo, zoe, B_N * CHUNK_N, &Z[i], &Ze[i], wi, wie, sj / B_N);

// -----------------

#define B_64	(64 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void forward64(__global RNS * restrict const z, __global RNSe * restrict const ze,
	 __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	FORWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], w, we, sj / 4);
	FORWARD_O(CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void backward64(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	BACKWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], wi, wie, sj / 4);
	BACKWARD_O(B_64, CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void forward64_0(__global RNS * restrict const z, __global RNSe * restrict const ze,
	 __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	const int lm = LNSIZE - 6; const unsigned int s = 64 / 4;
	FORWARD_I_0(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], w, we, sj / 4);
	FORWARD_O(CHUNK64);
}

// -----------------

#define B_256	(256 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void forward256(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	FORWARD_I(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], w, we, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], w, we, sj / 4);
	FORWARD_O(CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void backward256(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	BACKWARD_I(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], wi, wie, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], wi, wie, sj / 16);
	BACKWARD_O(B_256, CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void forward256_0(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	const int lm = LNSIZE - 8; const unsigned int s = 256 / 4;
	FORWARD_I_0(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], &Zie[CHUNK256 * k16], w, we, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], &Zie[CHUNK256 * k4], w, we, sj / 4);
	FORWARD_O(CHUNK256);
}

// -----------------

#define B_1024	(1024 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	FORWARD_I(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], w, we, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], w, we, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], w, we, sj / 4);
	FORWARD_O(CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void backward1024(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	BACKWARD_I(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], wi, wie, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], wi, wie, sj / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], wi, wie, sj / 64);
	BACKWARD_O(B_1024, CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024_0(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	const int lm = LNSIZE - 10; const unsigned int s = 1024 / 4;
	FORWARD_I_0(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], &Zie[CHUNK1024 * k64], w, we, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], &Zie[CHUNK1024 * k16], w, we, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], &Zie[CHUNK1024 * k4], w, we, sj / 4);
	FORWARD_O(CHUNK1024);
}

// -----------------

#define DECLARE_VAR_32() \
	__local RNS Z[32 * BLK32]; \
	__local RNSe Ze[32 * BLK32]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (32 / 4 * BLK32), group_id = gid / (32 / 4 * BLK32); \
	const sz_t k32 = group_id * 32 * BLK32, i = local_id; \
	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \
	\
	__global RNS * restrict const zk = &z[k32 + i32 + i8]; \
	__global RNSe * restrict const zke = &ze[k32 + i32 + i8]; \
	__local RNS * const Z32 = &Z[i32]; \
	__local RNSe * const Z32e = &Ze[i32]; \
	__local RNS * const Zi8 = &Z32[i8]; \
	__local RNSe * const Zi8e = &Z32e[i8]; \
	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \
	__local RNS * const Zi2 = &Z32[i2]; \
	__local RNSe * const Zi2e = &Z32e[i2]; \
	__local RNS * const Z4 = &Z32[4 * i8]; \
	__local RNSe * const Z4e = &Z32e[4 * i8];

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
#endif
void square32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_32();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);
}

#define DECLARE_VAR_64() \
	__local RNS Z[64 * BLK64]; \
	__local RNSe Ze[64 * BLK64]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (64 / 4 * BLK64), group_id = gid / (64 / 4 * BLK64); \
	const sz_t k64 = group_id * 64 * BLK64, i = local_id; \
	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \
	\
	__global RNS * restrict const zk = &z[k64 + i64 + i16]; \
	__global RNSe * restrict const zke = &ze[k64 + i64 + i16]; \
	__local RNS * const Z64 = &Z[i64]; \
	__local RNSe * const Z64e = &Ze[i64]; \
	__local RNS * const Zi16 = &Z64[i16]; \
	__local RNSe * const Zi16e = &Z64e[i16]; \
	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \
	__local RNS * const Zi4 = &Z64[i4]; \
	__local RNSe * const Zi4e = &Z64e[i4]; \
	__local RNS * const Z4 = &Z64[4 * i16]; \
	__local RNSe * const Z4e = &Z64e[4 * i16];

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#endif
void square64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_64();
	__global const RNS_W * const wi = &w[4 * NSIZE_4];
	__global const RNS_We * const wie = &we[4 * NSIZE_4];

	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	square_4(Z4, Z4e, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);
}

#define DECLARE_VAR_128() \
	__local RNS Z[128 * BLK128]; \
	__local RNSe Ze[128 * BLK128]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (128 / 4 * BLK128), group_id = gid / (128 / 4 * BLK128); \
	const sz_t k128 = group_id * 128 * BLK128, i = local_id; \
	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \
	\
	__global RNS * restrict const zk = &z[k128 + i128 + i32]; \
	__global RNSe * restrict const zke = &ze[k128 + i128 + i32]; \
	__local RNS * const Z128 = &Z[i128]; \
	__local RNSe * const Z128e = &Ze[i128]; \
	__local RNS * const Zi32 = &Z128[i32]; \
	__local RNSe * const Zi32e = &Z128e[i32]; \
	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \
	__local RNS * const Zi8 = &Z128[i8]; \
	__local RNSe * const Zi8e = &Z128e[i8]; \
	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \
	__local RNS * const Zi2 = &Z128[i2]; \
	__local RNSe * const Zi2e = &Z128e[i2]; \
	__local RNS * const Z4 = &Z128[4 * i32]; \
	__local RNSe * const Z4e = &Z128e[4 * i32];

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
#endif
void square128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_128();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);
}

#define DECLARE_VAR_256() \
	__local RNS Z[256 * BLK256]; \
	__local RNSe Ze[256 * BLK256]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (256 / 4 * BLK256), group_id = gid / (256 / 4 * BLK256); \
	const sz_t k256 = group_id * 256 * BLK256, i = local_id; \
	const sz_t i256 = 0, i64 = i; \
	\
	__global RNS * restrict const zk = &z[k256 + i256 + i64]; \
	__global RNSe * restrict const zke = &ze[k256 + i256 + i64]; \
	__local RNS * const Z256 = &Z[i256]; \
	__local RNSe * const Z256e = &Ze[i256]; \
	__local RNS * const Zi64 = &Z256[i64]; \
	__local RNSe * const Zi64e = &Z256e[i64]; \
	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \
	__local RNS * const Zi16 = &Z256[i16]; \
	__local RNSe * const Zi16e = &Z256e[i16]; \
	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \
	__local RNS * const Zi4 = &Z256[i4]; \
	__local RNSe * const Zi4e = &Z256e[i4]; \
	__local RNS * const Z4 = &Z256[4 * i64]; \
	__local RNSe * const Z4e = &Z256e[4 * i64];

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
#endif
void square256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_256();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	square_4(Z4, Z4e, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);
}

#define DECLARE_VAR_512() \
	__local RNS Z[512]; \
	__local RNSe Ze[512]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (512 / 4), group_id = gid / (512 / 4); \
	const sz_t k512 = group_id * 512, i128 = local_id; \
	\
	__global RNS * restrict const zk = &z[k512 + i128]; \
	__global RNSe * restrict const zke = &ze[k512 + i128]; \
	__local RNS * const Zi128 = &Z[i128]; \
	__local RNSe * const Zi128e = &Ze[i128]; \
	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \
	__local RNS * const Zi32 = &Z[i32]; \
	__local RNSe * const Zi32e = &Ze[i32]; \
	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \
	__local RNS * const Zi8 = &Z[i8]; \
	__local RNSe * const Zi8e = &Ze[i8]; \
	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \
	__local RNS * const Zi2 = &Z[i2]; \
	__local RNSe * const Zi2e = &Ze[i2]; \
	__local RNS * const Z4 = &Z[4 * i128]; \
	__local RNSe * const Z4e = &Ze[4 * i128];

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void square512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_512();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);
}

#define DECLARE_VAR_1024() \
	__local RNS Z[1024]; \
	__local RNSe Ze[1024]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (1024 / 4), group_id = gid / (1024 / 4); \
	const sz_t k1024 = group_id * 1024, i256 = local_id; \
	\
	__global RNS * restrict const zk = &z[k1024 + i256]; \
	__global RNSe * restrict const zke = &ze[k1024 + i256]; \
	__local RNS * const Zi256 = &Z[i256]; \
	__local RNSe * const Zi256e = &Ze[i256]; \
	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \
	__local RNS * const Zi64 = &Z[i64]; \
	__local RNSe * const Zi64e = &Ze[i64]; \
	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \
	__local RNS * const Zi16 = &Z[i16]; \
	__local RNSe * const Zi16e = &Ze[i16]; \
	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \
	__local RNS * const Zi4 = &Z[i4]; \
	__local RNSe * const Zi4e = &Ze[i4]; \
	__local RNS * const Z4 = &Z[4 * i256]; \
	__local RNSe * const Z4e = &Ze[4 * i256];

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void square1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_1024();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);
	forward_4(64, Zi64, Zi64e, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	square_4(Z4, Z4e, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4(64, Zi64, Zi64e, wi, wie, j / 64);
	backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);
}

#define DECLARE_VAR_2048() \
	__local RNS Z[2048]; \
	__local RNSe Ze[2048]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (2048 / 4), group_id = gid / (2048 / 4); \
	const sz_t k2048 = group_id * 2048, i512 = local_id; \
	\
	__global RNS * restrict const zk = &z[k2048 + i512]; \
	__global RNSe * restrict const zke = &ze[k2048 + i512]; \
	__local RNS * const Zi512 = &Z[i512]; \
	__local RNSe * const Zi512e = &Ze[i512]; \
	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \
	__local RNS * const Zi128 = &Z[i128]; \
	__local RNSe * const Zi128e = &Ze[i128]; \
	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \
	__local RNS * const Zi32 = &Z[i32]; \
	__local RNSe * const Zi32e = &Ze[i32]; \
	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \
	__local RNS * const Zi8 = &Z[i8]; \
	__local RNSe * const Zi8e = &Ze[i8]; \
	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \
	__local RNS * const Zi2 = &Z[i2]; \
	__local RNSe * const Zi2e = &Ze[i2]; \
	__local RNS * const Z4 = &Z[4 * i512]; \
	__local RNSe * const Z4e = &Ze[4 * i512];

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void square2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_2048();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);
	forward_4(128, Zi128, Zi128e, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4(128, Zi128, Zi128e, wi, wie, j / 128);
	backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(8, zk, zke, Z4, Z4e);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	fwd2write_4(16, zk, zke, Z4, Z4e, w[j], we[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(32, zk, zke, Z4, Z4e);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	fwd2write_4(64, zk, zke, Z4, Z4e, w[j], we[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void fwd512p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(128, zk, zke, Z4, Z4e);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void fwd1024p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);
	forward_4(64, Zi64, Zi64e, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	fwd2write_4(256, zk, zke, Z4, Z4e, w[j], we[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void fwd2048p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);
	forward_4(128, Zi128, Zi128e, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(512, zk, zke, Z4, Z4e);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
#endif
void mul32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_32();
	__global const RNS * restrict const zpk = &zp[k32 + i32 + i8];
	__global const RNSe * restrict const zpke = &zpe[k32 + i32 + i8];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	mul_22(Z4, Z4e, 8, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#endif
void mul64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_64();
	__global const RNS * restrict const zpk = &zp[k64 + i64 + i16];
	__global const RNSe * restrict const zpke = &zpe[k64 + i64 + i16];
	__global const RNS_W * const wi = &w[4 * NSIZE_4];
	__global const RNS_We * const wie = &we[4 * NSIZE_4];

	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	mul_4(Z4, Z4e, 16, zpk, zpke, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
#endif
void mul128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_128();
	__global const RNS * restrict const zpk = &zp[k128 + i128 + i32];
	__global const RNSe * restrict const zpke = &zpe[k128 + i128 + i32];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	mul_22(Z4, Z4e, 32, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
#endif
void mul256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_256();
	__global const RNS * restrict const zpk = &zp[k256 + i256 + i64];
	__global const RNSe * restrict const zpke = &zpe[k256 + i256 + i64];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	mul_4(Z4, Z4e, 64, zpk, zpke, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void mul512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_512();
	__global const RNS * restrict const zpk = &zp[k512 + i128];
	__global const RNSe * restrict const zpke = &zpe[k512 + i128];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	mul_22(Z4, Z4e, 128, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void mul1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_1024();
	__global const RNS * restrict const zpk = &zp[k1024 + i256];
	__global const RNSe * restrict const zpke = &zpe[k1024 + i256];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);
	forward_4(64, Zi64, Zi64e, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	mul_4(Z4, Z4e, 256, zpk, zpke, w[j], wi[j], w[NSIZE_4 + j], we[j], wie[j], we[NSIZE_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4(64, Zi64, Zi64e, wi, wie, j / 64);
	backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void mul2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_2048();
	__global const RNS * restrict const zpk = &zp[k2048 + i512];
	__global const RNSe * restrict const zpke = &zpe[k2048 + i512];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];
	__global const RNS_We * restrict const wie = &we[4 * NSIZE_4];

	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);
	forward_4(128, Zi128, Zi128e, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	mul_22(Z4, Z4e, 512, zpk, zpke, w[NSIZE_4 + j], we[NSIZE_4 + j]);
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4(128, Zi128, Zi128e, wi, wie, j / 128);
	backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);
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
	const uint32 t_l = (uint32)(t) % (1u << 29);

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
	const uint32 t_l = (uint32)(t.s0) % (1u << 29);

	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint32 d_l, r_l = barrett(((uint64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint64 d = ((uint64)(d_h) << 29) | d_l;

	const bool s = int96_is_neg(*f);
	*f = int96_set_si(s ? -(int64)(d) : (int64)(d));
	return s ? -(int32)(r_l) : (int32)(r_l);
}

__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))
void normalize1(__global RNS * restrict const z, __global RNSe * restrict const ze, __global int64 * restrict const c,
	const uint32 b, const uint32 b_inv, const int b_s, const int32 dup)
{
	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;
	__global RNS * restrict const zi = &z[4 * gid];
	__global RNSe * restrict const zie = &ze[4 * gid];
	__local int64 cl[NORM_WG_SZ];

	// Not converted into Montgomery form such that output is converted out of Montgomery form
	const RNS norm = (RNS)(NORM1, NORM2);
	const RNSe norme = (RNSe)(NORM3);

	int96 f96 = int96_set_si(0);
	int32 r[4];

	for (sz_t j = 0; j < 4; ++j)
	{
		const RNS u = mul(zi[j], norm); const RNSe ue = mule(zie[j], norme);
		int96 l = garner3(u.s0, u.s1, ue);
		if (dup != 0) l = int96_add(l, l);
		f96 = int96_add(f96, l);
		r[j] = reduce96(&f96, b, b_inv, b_s);
	}

	int64 f = (int64)(f96.s0);
	cl[lid] = f;

	if (lid == NORM_WG_SZ - 1)
	{
		const sz_t i = (gid / NORM_WG_SZ + 1) % (NSIZE_4 / NORM_WG_SZ);
		c[i] = (i == 0) ? -f : f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	f = (lid == 0) ? 0 : cl[lid - 1];
	f += r[0]; r[0] = reduce64(&f, b, b_inv, b_s);
	f += r[1]; r[1] = reduce64(&f, b, b_inv, b_s);
	f += r[2]; r[2] = reduce64(&f, b, b_inv, b_s);
	f += r[3]; r[3] = (sz_t)(f);

	for (sz_t j = 0; j < 4; ++j) { zi[j] = toRNS(r[j]); zie[j] = toRNSe(r[j]); }
}

__kernel
void normalize2(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const int64 * restrict const c, 
	const uint32 b, const uint32 b_inv, const int b_s)
{
	const sz_t gid = (sz_t)get_global_id(0);
	__global RNS * restrict const zi = &z[NORM_WG_SZ * 4 * gid];
	__global RNSe * restrict const zie = &ze[NORM_WG_SZ * 4 * gid];

	int64 f = c[gid];

	for (sz_t j = 0; j < 3; ++j)
	{
		f += geti_P3(zie[j]);
		const int32 r = reduce64(&f, b, b_inv, b_s);
		zi[j] = toRNS(r); zie[j] = toRNSe(r);
		if (f == 0) return;
	}
	f += geti_P3(zie[3]);
	zi[3] = toRNS((int32)(f)); zie[3] = toRNSe((int32)(f));
}

__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))
void mulscalar(__global RNS * restrict const z, __global RNSe * restrict const ze, __global int64 * restrict const c,
	const uint32 b, const uint32 b_inv, const int b_s, const int32 a)
{
	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;
	__global RNS * restrict const zi = &z[4 * gid];
	__global RNSe * restrict const zie = &ze[4 * gid];
	__local int64 cl[NORM_WG_SZ];

	int64 f = 0;
	int32 r[4];

	for (sz_t j = 0; j < 4; ++j)
	{
		f += geti_P3(zie[j]) * (int64)(a);
		r[j] = reduce64(&f, b, b_inv, b_s);
	}

	cl[lid] = f;

	if (lid == NORM_WG_SZ - 1)
	{
		const sz_t i = (gid / NORM_WG_SZ + 1) % (NSIZE_4 / NORM_WG_SZ);
		c[i] = (i == 0) ? -f : f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	f = (lid == 0) ? 0 : cl[lid - 1];
	f += r[0]; r[0] = reduce64(&f, b, b_inv, b_s);
	f += r[1]; r[1] = reduce64(&f, b, b_inv, b_s);
	f += r[2]; r[2] = reduce64(&f, b, b_inv, b_s);
	f += r[3]; r[3] = (sz_t)(f);

	for (sz_t j = 0; j < 4; ++j)  { zi[j] = toRNS(r[j]); zie[j] = toRNSe(r[j]); }
}

__kernel
void set(__global RNS * restrict const z, __global RNSe * restrict const ze, const uint32 a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const uint32 ai = (idx == 0) ? a : 0;
	z[idx] = (RNS)(ai, ai);
	ze[idx] = (RNSe)(ai);
}

__kernel
void copy(__global RNS * restrict const z, __global RNSe * restrict const ze, const sz_t dst, const sz_t src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[dst + idx] = z[src + idx];
	ze[dst + idx] = ze[src + idx];
}

__kernel
void copyp(__global RNS * restrict const zp, __global RNSe * restrict const zpe,
		   __global const RNS * restrict const z, __global const RNSe * restrict const ze, const sz_t src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	zp[idx] = z[src + idx];
	zpe[idx] = ze[src + idx];
}
