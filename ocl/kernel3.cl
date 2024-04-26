/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

typedef uint	sz_t;

#define P1P2	(P1 * (ulong)P2)
#define P2P3	(P2 * (ulong)P3)

// --- uint96/int96 ---

typedef struct { ulong s0; uint s1; } uint96;
typedef struct { ulong s0; int s1; } int96;

inline int96 int96_set_si(const long n) { int96 r; r.s0 = (ulong)n; r.s1 = (n < 0) ? -1 : 0; return r; }
inline uint96 uint96_set(const ulong s0, const uint s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }

inline int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int)x.s1; return r; }
inline uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint)x.s1; return r; }

inline bool int96_is_neg(const int96 x) { return (x.s1 < 0); }

inline bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

inline int96 int96_neg(const int96 x)
{
	const int c = (x.s0 != 0) ? 1 : 0;
	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;
	return r;
}

inline uint96 int96_abs(const int96 x)
{
	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;
	return int96_u(t);
}

inline int96 int96_add(const int96 x, const int96 y)
{
	int96 r;
#ifdef PTX_ASM
	asm volatile ("add.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y.s0));
	asm volatile ("addc.s32 %0, %1, %2;" : "=r" (r.s1) : "r" (x.s1), "r" (y.s1));
#else
	const ulong s0 = x.s0 + y.s0;
	const int c = (s0 < y.s0) ? 1 : 0;
	r.s0 = s0; r.s1 = x.s1 + y.s1 + c;
#endif
	return r;
}

inline uint96 uint96_add_64(const uint96 x, const ulong y)
{
	uint96 r;
#ifdef PTX_ASM
	asm volatile ("add.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (r.s1) : "r" (x.s1));
#else
	const ulong s0 = x.s0 + y;
	const uint c = (s0 < y) ? 1 : 0;
	r.s0 = s0; r.s1 = x.s1 + c;
#endif
	return r;
}

inline int96 uint96_subi(const uint96 x, const uint96 y)
{
	int96 r;
#ifdef PTX_ASM
	asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l" (r.s0) : "l" (x.s0), "l" (y.s0));
	asm volatile ("subc.s32 %0, %1, %2;" : "=r" (r.s1) : "r" (x.s1), "r" (y.s1));
#else
	const uint c = (x.s0 < y.s0) ? 1 : 0;
	r.s0 = x.s0 - y.s0; r.s1 = (int)(x.s1 - y.s1 - c);
#endif
	return r;
}

inline uint96 uint96_mul_64_32(const ulong x, const uint y)
{
	const ulong l = (uint)x * (ulong)y, h = (x >> 32) * y + (l >> 32);
	uint96 r; r.s0 = (h << 32) | (uint)l; r.s1 = (uint)(h >> 32);
	return r;
}

// --- mod arith ---

inline uint _addMod(const uint lhs, const uint rhs, const uint p)
{
	const uint c = (lhs >= p - rhs) ? p : 0;
	return lhs + rhs - c;
}

inline uint _subMod(const uint lhs, const uint rhs, const uint p)
{
	const uint c = (lhs < rhs) ? p : 0;
	return lhs - rhs + c;
}

// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.

// Montgomery form (lhs, rhs and output): if 0 <= r < p then f is r * 2^32 mod p
inline uint _mulMonty(const uint lhs, const uint rhs, const uint p, const uint q)
{
	const uint t_lo = lhs * rhs, t_hi = mul_hi(lhs, rhs);
	const uint mp = mul_hi(t_lo * q, p);
	return _subMod(t_hi, mp, p);
}

// Conversion into Montgomery form
inline uint _toMonty(const uint n, const uint r2, const uint p, const uint q)
{
	// n * (2^32)^2 = (n * 2^32) * (1 * 2^32)
	return _mulMonty(n, r2, p, q);
}

// Conversion out of Montgomery form
// inline uint _fromMonty(const uint n, const uint p, const uint q)
// {
// 	// REDC(n * 2^32, 1)
// 	const uint mp = mul_hi(n * q, p);
// 	return (mp != 0) ? p - mp : 0;
// }

inline uint add_P1(const uint lhs, const uint rhs) { return _addMod(lhs, rhs, P1); }
inline uint add_P2(const uint lhs, const uint rhs) { return _addMod(lhs, rhs, P2); }
inline uint add_P3(const uint lhs, const uint rhs) { return _addMod(lhs, rhs, P3); }

inline uint sub_P1(const uint lhs, const uint rhs) { return _subMod(lhs, rhs, P1); }
inline uint sub_P2(const uint lhs, const uint rhs) { return _subMod(lhs, rhs, P2); }
inline uint sub_P3(const uint lhs, const uint rhs) { return _subMod(lhs, rhs, P3); }

// Montgomery form
inline uint mul_P1(const uint lhs, const uint rhs) { return _mulMonty(lhs, rhs, P1, Q1); }
inline uint mul_P2(const uint lhs, const uint rhs) { return _mulMonty(lhs, rhs, P2, Q2); }
inline uint mul_P3(const uint lhs, const uint rhs) { return _mulMonty(lhs, rhs, P3, Q3); }

inline uint toMonty_P1(const uint lhs) { return _toMonty(lhs, R1, P1, Q1); }
inline uint toMonty_P2(const uint lhs) { return _toMonty(lhs, R2, P2, Q2); }
inline uint toMonty_P3(const uint lhs) { return _toMonty(lhs, R3, P3, Q3); }

// inline uint fromMonty_P1(const uint lhs) { return _fromMonty(lhs, P1, Q1); }
// inline uint fromMonty_P2(const uint lhs) { return _fromMonty(lhs, P2, Q2); }
// inline uint fromMonty_P3(const uint lhs) { return _fromMonty(lhs, P3, Q3); }

inline int geti_P1(const uint n) { return (n > P1 / 2) ? (int)(n - P1) : (int)n; }

inline int96 garner3(const uint r1, const uint r2, const uint r3)
{
	const uint u13 = mul_P1(sub_P1(r1, r3), InvP3_P1);
	const uint u23 = mul_P2(sub_P2(r2, r3), InvP3_P2);
	const uint u123 = mul_P1(sub_P1(u13, u23), InvP2_P1);
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

inline RNS toRNS(const int i) { return ((RNS)(i, i) + ((i < 0) ? (RNS)(P1, P2) : (RNS)(0, 0))); }

inline RNS add(const RNS lhs, const RNS rhs) { return (RNS)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }
inline RNS sub(const RNS lhs, const RNS rhs) { return (RNS)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }
inline RNS mul(const RNS lhs, const RNS rhs) { return (RNS)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }

inline RNS sqr(const RNS lhs) { return mul(lhs, lhs); }

inline RNS mulW(const RNS lhs, const RNS_W w) { return mul(lhs, w); }

inline RNS toMonty(const RNS lhs) { return (RNS)(toMonty_P1(lhs.s0), toMonty_P2(lhs.s1)); }

inline RNSe toRNSe(const int i) { return ((RNSe)(i) + ((i < 0) ? (RNSe)(P3) : (RNSe)(0))); }

inline RNSe adde(const RNSe lhs, const RNSe rhs) { return (RNSe)(add_P3(lhs, rhs)); }
inline RNSe sube(const RNSe lhs, const RNSe rhs) { return (RNSe)(sub_P3(lhs, rhs)); }
inline RNSe mule(const RNSe lhs, const RNSe rhs) { return (RNSe)(mul_P3(lhs, rhs)); }

inline RNSe sqre(const RNSe lhs) { return mule(lhs, lhs); }

inline RNSe mulWe(const RNSe lhs, const RNS_We w) { return mule(lhs, w); }

inline RNSe toMontye(const RNSe lhs) { return (RNSe)toMonty_P3(lhs); }

// --- transform/inline ---

inline void forward_4(const sz_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	__global const RNS_W * restrict const w_j = &w[j];
	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];
	__global const RNS_We * restrict const we_j = &we[j];
	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0 * m], u2 = mulW(Z[2 * m], w1), u1 = Z[1 * m], u3 = mulW(Z[3 * m], w1);
	const RNSe u0e = Ze[0 * m], u2e = mulWe(Ze[2 * m], w1e), u1e = Ze[1 * m], u3e = mulWe(Ze[3 * m], w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = mulWe(adde(u1e, u3e), w2e), v3e = mulWe(sube(u1e, u3e), w3e);
	Z[0 * m] = add(v0, v1); Z[1 * m] = sub(v0, v1); Z[2 * m] = add(v2, v3); Z[3 * m] = sub(v2, v3);
	Ze[0 * m] = adde(v0e, v1e); Ze[1 * m] = sube(v0e, v1e); Ze[2 * m] = adde(v2e, v3e); Ze[3 * m] = sube(v2e, v3e);
}

inline void forward_4i(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	__global const RNS_W * restrict const w_j = &w[j];
	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];
	__global const RNS_We * restrict const we_j = &we[j];
	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];
	__global const RNS * const z2mg = &z[2 * mg];
	const RNS u0 = z[0], u2 = mulW(z2mg[0], w1), u1 = z[mg], u3 = mulW(z2mg[mg], w1);
	__global const RNSe * const z2mge = &ze[2 * mg];
	const RNSe u0e = ze[0], u2e = mulWe(z2mge[0], w1e), u1e = ze[mg], u3e = mulWe(z2mge[mg], w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = mulWe(adde(u1e, u3e), w2e), v3e = mulWe(sube(u1e, u3e), w3e);
	Z[0 * ml] = add(v0, v1); Z[1 * ml] = sub(v0, v1); Z[2 * ml] = add(v2, v3); Z[3 * ml] = sub(v2, v3);
	Ze[0 * ml] = adde(v0e, v1e); Ze[1 * ml] = sube(v0e, v1e); Ze[2 * ml] = adde(v2e, v3e); Ze[3 * ml] = sube(v2e, v3e);
}

inline void forward_4i_0(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	__global const RNS_W * restrict const w_j = &w[j];
	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];
	__global const RNS_We * restrict const we_j = &we[j];
	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];
	__global const RNS * const z2mg = &z[2 * mg];
	// const RNS u0 = z[0], u2 = mulW(z2mg[0], w1), u1 = z[mg], u3 = mulW(z2mg[mg], w1);
	const RNS u0 = toMonty(z[0]), u2 = mulW(toMonty(z2mg[0]), w1), u1 = toMonty(z[mg]), u3 = mulW(toMonty(z2mg[mg]), w1);
	__global const RNSe * const z2mge = &ze[2 * mg];
	// const RNSe u0e = ze[0], u2e = mulWe(z2mge[0], w1e), u1e = ze[mg], u3e = mulWe(z2mge[mg], w1e);
	const RNSe u0e = toMontye(ze[0]), u2e = mulWe(toMontye(z2mge[0]), w1e), u1e = toMontye(ze[mg]), u3e = mulWe(toMontye(z2mge[mg]), w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = mulWe(adde(u1e, u3e), w2e), v3e = mulWe(sube(u1e, u3e), w3e);
	Z[0 * ml] = add(v0, v1); Z[1 * ml] = sub(v0, v1); Z[2 * ml] = add(v2, v3); Z[3 * ml] = sub(v2, v3);
	Ze[0 * ml] = adde(v0e, v1e); Ze[1 * ml] = sube(v0e, v1e); Ze[2 * ml] = adde(v2e, v3e); Ze[3 * ml] = sube(v2e, v3e);
}

inline void forward_4o(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	const sz_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we, const sz_t j)
{
	__global const RNS_W * restrict const w_j = &w[j];
	const RNS_W w1 = w_j[0], w2 = w_j[j], w3 = w_j[j + 1];
	__global const RNS_We * restrict const we_j = &we[j];
	const RNS_We w1e = we_j[0], w2e = we_j[j], w3e = we_j[j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0 * ml], u2 = mulW(Z[2 * ml], w1), u1 = Z[1 * ml], u3 = mulW(Z[3 * ml], w1);
	const RNSe u0e = Ze[0 * ml], u2e = mulWe(Ze[2 * ml], w1e), u1e = Ze[1 * ml], u3e = mulWe(Ze[3 * ml], w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = mulW(add(u1, u3), w2), v3 = mulW(sub(u1, u3), w3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = mulWe(adde(u1e, u3e), w2e), v3e = mulWe(sube(u1e, u3e), w3e);
	__global RNS * const z2mg = &z[2 * mg];
	z[0] = add(v0, v1); z[mg] = sub(v0, v1); z2mg[0] = add(v2, v3); z2mg[mg] = sub(v2, v3);
	__global RNSe * const z2mge = &ze[2 * mg];
	ze[0] = adde(v0e, v1e); ze[mg] = sube(v0e, v1e); z2mge[0] = adde(v2e, v3e); z2mge[mg] = sube(v2e, v3e);
}

inline void backward_4(const sz_t m, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)
{
	__global const RNS_W * restrict const wi_j = &wi[j];
	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];
	__global const RNS_We * restrict const wie_j = &wie[j];
	const RNS_We wi1e = wie_j[0], wi2e = wie_j[j], wi3e = wie_j[j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0 * m], u1 = Z[1 * m], u2 = Z[2 * m], u3 = Z[3 * m];
	const RNSe u0e = Ze[0 * m], u1e = Ze[1 * m], u2e = Ze[2 * m], u3e = Ze[3 * m];
	const RNS v0 = add(u0, u1), v1 = mulW(sub(u0, u1), wi2), v2 = add(u2, u3), v3 = mulW(sub(u2, u3), wi3);
	const RNSe v0e = adde(u0e, u1e), v1e = mulWe(sube(u0e, u1e), wi2e), v2e = adde(u2e, u3e), v3e = mulWe(sube(u2e, u3e), wi3e);
	Z[0 * m] = add(v0, v2); Z[2 * m] = mulW(sub(v0, v2), wi1); Z[1 * m] = add(v1, v3); Z[3 * m] = mulW(sub(v1, v3), wi1);
	Ze[0 * m] = adde(v0e, v2e); Ze[2 * m] = mulWe(sube(v0e, v2e), wi1e); Ze[1 * m] = adde(v1e, v3e); Ze[3 * m] = mulWe(sube(v1e, v3e), wi1e);
}

inline void backward_4i(const sz_t ml, __local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze,
	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)
{
	__global const RNS_W * restrict const wi_j = &wi[j];
	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];
	__global const RNS_We * restrict const wie_j = &wie[j];
	const RNS_We wi1e = wie_j[0], wi2e = wie_j[j], wi3e = wie_j[j + 1];
	__global const RNS * const z2mg = &z[2 * mg];
	const RNS u0 = z[0], u1 = z[mg], u2 = z2mg[0], u3 = z2mg[mg];
	__global const RNSe * const z2mge = &ze[2 * mg];
	const RNSe u0e = ze[0], u1e = ze[mg], u2e = z2mge[0], u3e = z2mge[mg];
	const RNS v0 = add(u0, u1), v1 = mulW(sub(u0, u1), wi2), v2 = add(u2, u3), v3 = mulW(sub(u2, u3), wi3);
	const RNSe v0e = adde(u0e, u1e), v1e = mulWe(sube(u0e, u1e), wi2e), v2e = adde(u2e, u3e), v3e = mulWe(sube(u2e, u3e), wi3e);
	Z[0 * ml] = add(v0, v2); Z[2 * ml] = mulW(sub(v0, v2), wi1); Z[1 * ml] = add(v1, v3); Z[3 * ml] = mulW(sub(v1, v3), wi1);
	Ze[0 * ml] = adde(v0e, v2e); Ze[2 * ml] = mulWe(sube(v0e, v2e), wi1e); Ze[1 * ml] = adde(v1e, v3e); Ze[3 * ml] = mulWe(sube(v1e, v3e), wi1e);
}

inline void backward_4o(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	const sz_t ml, __local const RNS * restrict const Z, __local const RNSe * restrict const Ze,
	__global const RNS_W * restrict const wi, __global const RNS_We * restrict const wie, const sz_t j)
{
	__global const RNS_W * restrict const wi_j = &wi[j];
	const RNS_W wi1 = wi_j[0], wi2 = wi_j[j], wi3 = wi_j[j + 1];
	__global const RNS_We * restrict const wie_j = &wie[j];
	const RNS_We wi1e = wie_j[0], wi2e = wie_j[j], wi3e = wie_j[j + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0 * ml], u1 = Z[1 * ml], u2 = Z[2 * ml], u3 = Z[3 * ml];
	const RNSe u0e = Ze[0 * ml], u1e = Ze[1 * ml], u2e = Ze[2 * ml], u3e = Ze[3 * ml];
	const RNS v0 = add(u0, u1), v1 = mulW(sub(u0, u1), wi2), v2 = add(u2, u3), v3 = mulW(sub(u2, u3), wi3);
	const RNSe v0e = adde(u0e, u1e), v1e = mulWe(sube(u0e, u1e), wi2e), v2e = adde(u2e, u3e), v3e = mulWe(sube(u2e, u3e), wi3e);
	__global RNS * const z2mg = &z[2 * mg];
	z[0] = add(v0, v2); z2mg[0] = mulW(sub(v0, v2), wi1); z[mg] = add(v1, v3); z2mg[mg] = mulW(sub(v1, v3), wi1);
	__global RNSe * const z2mge = &ze[2 * mg];
	ze[0] = adde(v0e, v2e); z2mge[0] = mulWe(sube(v0e, v2e), wi1e); ze[mg] = adde(v1e, v3e); z2mge[mg] = mulWe(sube(v1e, v3e), wi1e);
}

inline void write_4(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	__local const RNS * restrict const Z, __local const RNSe * restrict const Ze)
{
	__global RNS * const z2mg = &z[2 * mg];
	__global RNSe * const z2mge = &ze[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];
	ze[0] = Ze[0]; ze[mg] = Ze[1]; z2mge[0] = Ze[2]; z2mge[mg] = Ze[3];
}

inline void fwd2write_4(const sz_t mg, __global RNS * restrict const z, __global RNSe * restrict const ze,
	__local const RNS * restrict const Z, __local const RNSe * restrict const Ze, const RNS_W w1, const RNS_We w1e)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0], u2 = mulW(Z[2], w1), u1 = Z[1], u3 = mulW(Z[3], w1);
	const RNSe u0e = Ze[0], u2e = mulWe(Ze[2], w1e), u1e = Ze[1], u3e = mulWe(Ze[3], w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = add(u1, u3), v3 = sub(u1, u3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = adde(u1e, u3e), v3e = sube(u1e, u3e);
	__global RNS * const z2mg = &z[2 * mg];
	z[0] = v0; z2mg[0] = v2; z[mg] = v1; z2mg[mg] = v3;
	__global RNSe * const z2mge = &ze[2 * mg];
	ze[0] = v0e; z2mge[0] = v2e; ze[mg] = v1e; z2mge[mg] = v3e;
}

inline void square_22(__local RNS * restrict const Z, __local RNSe * restrict const Ze, const RNS_W w0, const RNS_We w0e)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];
	const RNSe u0e = Ze[0], u1e = Ze[1], u2e = Ze[2], u3e = Ze[3];
	Z[0] = add(sqr(u0), sqr(mulW(u1, w0))); Z[1] = mul(add(u0, u0), u1);
	Ze[0] = adde(sqre(u0e), sqre(mulWe(u1e, w0e))); Ze[1] = mule(adde(u0e, u0e), u1e);
	Z[2] = sub(sqr(u2), sqr(mulW(u3, w0))); Z[3] = mul(add(u2, u2), u3);
	Ze[2] = sube(sqre(u2e), sqre(mulWe(u3e, w0e))); Ze[3] = mule(adde(u2e, u2e), u3e);
}

inline void square_4(__local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const RNS_W w1, const RNS_W w1i, const RNS_W w0, const RNS_We w1e, const RNS_We w1ie, const RNS_We w0e)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0], u2 = mulW(Z[2], w1), u1 = Z[1], u3 = mulW(Z[3], w1);
	const RNSe u0e = Ze[0], u2e = mulWe(Ze[2], w1e), u1e = Ze[1], u3e = mulWe(Ze[3], w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = add(u1, u3), v3 = sub(u1, u3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = adde(u1e, u3e), v3e = sube(u1e, u3e);
	const RNS s0 = add(sqr(v0), sqr(mulW(v1, w0))), s1 = mul(add(v0, v0), v1);
	const RNSe s0e = adde(sqre(v0e), sqre(mulWe(v1e, w0e))), s1e = mule(adde(v0e, v0e), v1e);
	const RNS s2 = sub(sqr(v2), sqr(mulW(v3, w0))), s3 = mul(add(v2, v2), v3);
	const RNSe s2e = sube(sqre(v2e), sqre(mulWe(v3e, w0e))), s3e = mule(adde(v2e, v2e), v3e);
	Z[0] = add(s0, s2); Z[2] = mulW(sub(s0, s2), w1i); Z[1] = add(s1, s3); Z[3] = mulW(sub(s1, s3), w1i);
	Ze[0] = adde(s0e, s2e); Ze[2] = mulWe(sube(s0e, s2e), w1ie); Ze[1] = adde(s1e, s3e); Ze[3] = mulWe(sube(s1e, s3e), w1ie);
}

inline void mul_22(__local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze, const RNS_W w0, const RNS_We w0e)
{
	__global const RNS * const z2mg = &z[2 * mg];
	const RNS u0p = z[0], u1p = z[mg], u2p = z2mg[0], u3p = z2mg[mg];
	__global const RNSe * const z2mge = &ze[2 * mg];
	const RNSe u0pe = ze[0], u1pe = ze[mg], u2pe = z2mge[0], u3pe = z2mge[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];
	const RNSe u0e = Ze[0], u1e = Ze[1], u2e = Ze[2], u3e = Ze[3];
	Z[0] = add(mul(u0, u0p), mul(mulW(u1, w0), mulW(u1p, w0)));
	Z[1] = add(mul(u0, u1p), mul(u0p, u1));
	Ze[0] = adde(mule(u0e, u0pe), mule(mulWe(u1e, w0e), mulWe(u1pe, w0e)));
	Ze[1] = adde(mule(u0e, u1pe), mule(u0pe, u1e));
	Z[2] = sub(mul(u2, u2p), mul(mulW(u3, w0), mulW(u3p, w0)));
	Z[3] = add(mul(u2, u3p), mul(u2p, u3));
	Ze[2] = sube(mule(u2e, u2pe), mule(mulWe(u3e, w0e), mulWe(u3pe, w0e)));
	Ze[3] = adde(mule(u2e, u3pe), mule(u2pe, u3e));
}

inline void mul_4(__local RNS * restrict const Z, __local RNSe * restrict const Ze,
	const sz_t mg, __global const RNS * restrict const z, __global const RNSe * restrict const ze, 
	const RNS_W w1, const RNS_W w1i, const RNS_W w0, const RNS_We w1e, const RNS_We w1ie, const RNS_We w0e)
{
	__global const RNS * const z2mg = &z[2 * mg];
	const RNS v0p = z[0], v1p = z[mg], v2p = z2mg[0], v3p = z2mg[mg];
	__global const RNSe * const z2mge = &ze[2 * mg];
	const RNSe v0pe = ze[0], v1pe = ze[mg], v2pe = z2mge[0], v3pe = z2mge[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	const RNS u0 = Z[0], u2 = mulW(Z[2], w1), u1 = Z[1], u3 = mulW(Z[3], w1);
	const RNSe u0e = Ze[0], u2e = mulWe(Ze[2], w1e), u1e = Ze[1], u3e = mulWe(Ze[3], w1e);
	const RNS v0 = add(u0, u2), v2 = sub(u0, u2), v1 = add(u1, u3), v3 = sub(u1, u3);
	const RNSe v0e = adde(u0e, u2e), v2e = sube(u0e, u2e), v1e = adde(u1e, u3e), v3e = sube(u1e, u3e);
	const RNS s0 = add(mul(v0, v0p), mul(mulW(v1, w0), mulW(v1p, w0)));
	const RNS s1 = add(mul(v0, v1p), mul(v0p, v1));
	const RNSe s0e = adde(mule(v0e, v0pe), mule(mulWe(v1e, w0e), mulWe(v1pe, w0e)));
	const RNSe s1e = adde(mule(v0e, v1pe), mule(v0pe, v1e));
	const RNS s2 = sub(mul(v2, v2p), mul(mulW(v3, w0), mulW(v3p, w0)));
	const RNS s3 = add(mul(v2, v3p), mul(v2p, v3));
	const RNSe s2e = sube(mule(v2e, v2pe), mule(mulWe(v3e, w0e), mulWe(v3pe, w0e)));
	const RNSe s3e = adde(mule(v2e, v3pe), mule(v2pe, v3e));
	Z[0] = add(s0, s2); Z[2] = mulW(sub(s0, s2), w1i); Z[1] = add(s1, s3); Z[3] = mulW(sub(s1, s3), w1i);
	Ze[0] = adde(s0e, s2e); Ze[2] = mulWe(sube(s0e, s2e), w1ie); Ze[1] = adde(s1e, s3e); Ze[3] = mulWe(sube(s1e, s3e), w1ie);
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
	const sz_t n_4 = (sz_t)get_global_size(0); \
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
	forward_4i_0(B_N * CHUNK_N, &Z[i], &Ze[i], B_N << lm, zi, zie, w, we, sj / B_N);

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

__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
void forward64(__global RNS * restrict const z, __global RNSe * restrict const ze,
	 __global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	FORWARD_I(B_64, CHUNK64);

	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], w, we, sj / 4);

	FORWARD_O(CHUNK64);
}

__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
void backward64(__global RNS * restrict const z, __global RNSe * restrict const ze,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we,
	const int lm, const unsigned int s)
{
	BACKWARD_I(B_64, CHUNK64);

	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], &Zie[CHUNK64 * k4], wi, wie, sj / 4);

	BACKWARD_O(B_64, CHUNK64);
}

__kernel __attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
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

__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
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

__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
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

__kernel // __attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
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

__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
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

__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
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

__kernel // __attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
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
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k32 = (sz_t)get_group_id(0) * 32 * BLK32, i = (sz_t)get_local_id(0); \
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

__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
void square32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);
}

#define DECLARE_VAR_64() \
	__local RNS Z[64 * BLK64]; \
	__local RNSe Ze[64 * BLK64]; \
	\
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k64 = (sz_t)get_group_id(0) * 64 * BLK64, i = (sz_t)get_local_id(0); \
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

__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
void square64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	__global const RNS_W * const wi = &w[4 * n_4];
	__global const RNS_We * const wie = &we[4 * n_4];
	square_4(Z4, Z4e, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);
}

#define DECLARE_VAR_128() \
	__local RNS Z[128 * BLK128]; \
	__local RNSe Ze[128 * BLK128]; \
	\
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k128 = (sz_t)get_group_id(0) * 128 * BLK128, i = (sz_t)get_local_id(0); \
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

__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
void square128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);
}

#define DECLARE_VAR_256() \
	__local RNS Z[256 * BLK256]; \
	__local RNSe Ze[256 * BLK256]; \
	\
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k256 = (sz_t)get_group_id(0) * 256 * BLK256, i = (sz_t)get_local_id(0); \
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

__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
void square256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	square_4(Z4, Z4e, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);
}

#define DECLARE_VAR_512() \
	__local RNS Z[512]; \
	__local RNSe Ze[512]; \
	\
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k512 = (sz_t)get_group_id(0) * 512, i128 = (sz_t)get_local_id(0); \
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

__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))
void square512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);
}

#define DECLARE_VAR_1024() \
	__local RNS Z[1024]; \
	__local RNSe Ze[1024]; \
	\
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k1024 = (sz_t)get_group_id(0) * 1024, i256 = (sz_t)get_local_id(0); \
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

__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
void square1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);
	forward_4(64, Zi64, Zi64e, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	square_4(Z4, Z4e, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4(64, Zi64, Zi64e, wi, wie, j / 64);
	backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);
}

#define DECLARE_VAR_2048() \
	__local RNS Z[2048]; \
	__local RNSe Ze[2048]; \
	\
	const sz_t n_4 = (sz_t)get_global_size(0), idx = (sz_t)get_global_id(0), j = n_4 + idx; \
	\
	const sz_t k2048 = (sz_t)get_group_id(0) * 2048, i512 = (sz_t)get_local_id(0); \
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

__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
void square2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);
	forward_4(128, Zi128, Zi128e, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	square_22(Z4, Z4e, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4(128, Zi128, Zi128e, wi, wie, j / 128);
	backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);
}

// -----------------

__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
void fwd32p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(8, zk, zke, Z4, Z4e);
}

__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
void fwd64p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	fwd2write_4(16, zk, zke, Z4, Z4e, w[j], we[j]);
}

__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
void fwd128p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(32, zk, zke, Z4, Z4e);
}

__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
void fwd256p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	fwd2write_4(64, zk, zke, Z4, Z4e, w[j], we[j]);
}

__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))
void fwd512p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	write_4(128, zk, zke, Z4, Z4e);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
void fwd1024p(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);
	forward_4(64, Zi64, Zi64e, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	fwd2write_4(256, zk, zke, Z4, Z4e, w[j], we[j]);
}

__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
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

__kernel __attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
void mul32(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, Zi8e, 8, zk, zke, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	__global const RNS * restrict const zpk = &zp[k32 + i32 + i8];
	__global const RNSe * restrict const zpke = &zpe[k32 + i32 + i8];
	mul_22(Z4, Z4e, 8, zpk, zpke, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4o(8, zk, zke, 8, Zi8, Zi8e, wi, wie, j / 8);
}

__kernel __attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
void mul64(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, Zi16e, 16, zk, zke, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	__global const RNS_W * const wi = &w[4 * n_4];
	__global const RNS_We * const wie = &we[4 * n_4];
	__global const RNS * restrict const zpk = &zp[k64 + i64 + i16];
	__global const RNSe * restrict const zpke = &zpe[k64 + i64 + i16];
	mul_4(Z4, Z4e, 16, zpk, zpke, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4o(16, zk, zke, 16, Zi16, Zi16e, wi, wie, j / 16);
}

__kernel __attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
void mul128(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, Zi32e, 32, zk, zke, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	__global const RNS * restrict const zpk = &zp[k128 + i128 + i32];
	__global const RNSe * restrict const zpke = &zpe[k128 + i128 + i32];
	mul_22(Z4, Z4e, 32, zpk, zpke, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4o(32, zk, zke, 32, Zi32, Zi32e, wi, wie, j / 32);
}

__kernel __attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
void mul256(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, Zi64e, 64, zk, zke, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	__global const RNS * restrict const zpk = &zp[k256 + i256 + i64];
	__global const RNSe * restrict const zpke = &zpe[k256 + i256 + i64];
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	mul_4(Z4, Z4e, 64, zpk, zpke, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4o(64, zk, zke, 64, Zi64, Zi64e, wi, wie, j / 64);
}

__kernel __attribute__((reqd_work_group_size(512 / 4, 1, 1)))
void mul512(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, Zi128e, 128, zk, zke, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	__global const RNS * restrict const zpk = &zp[k512 + i128];
	__global const RNSe * restrict const zpke = &zpe[k512 + i128];
	mul_22(Z4, Z4e, 128, zpk, zpke, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4o(128, zk, zke, 128, Zi128, Zi128e, wi, wie, j / 128);
}

__kernel __attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
void mul1024(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, Zi256e, 256, zk, zke, w, we, j / 256);
	forward_4(64, Zi64, Zi64e, w, we, j / 64);
	forward_4(16, Zi16, Zi16e, w, we, j / 16);
	forward_4(4, Zi4, Zi4e, w, we, j / 4);
	__global const RNS * restrict const zpk = &zp[k1024 + i256];
	__global const RNSe * restrict const zpke = &zpe[k1024 + i256];
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	mul_4(Z4, Z4e, 256, zpk, zpke, w[j], wi[j], w[n_4 + j], we[j], wie[j], we[n_4 + j]);
	backward_4(4, Zi4, Zi4e, wi, wie, j / 4);
	backward_4(16, Zi16, Zi16e, wi, wie, j / 16);
	backward_4(64, Zi64, Zi64e, wi, wie, j / 64);
	backward_4o(256, zk, zke, 256, Zi256, Zi256e, wi, wie, j / 256);
}

__kernel // __attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
void mul2048(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const RNS * restrict const zp, __global const RNSe * restrict const zpe,
	__global const RNS_W * restrict const w, __global const RNS_We * restrict const we)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, Zi512e, 512, zk, zke, w, we, j / 512);
	forward_4(128, Zi128, Zi128e, w, we, j / 128);
	forward_4(32, Zi32, Zi32e, w, we, j / 32);
	forward_4(8, Zi8, Zi8e, w, we, j / 8);
	forward_4(2, Zi2, Zi2e, w, we, j / 2);
	__global const RNS * restrict const zpk = &zp[k2048 + i512];
	__global const RNSe * restrict const zpke = &zpe[k2048 + i512];
	mul_22(Z4, Z4e, 512, zpk, zpke, w[n_4 + j], we[n_4 + j]);
	__global const RNS_W * restrict const wi = &w[4 * n_4];
	__global const RNS_We * restrict const wie = &we[4 * n_4];
	backward_4(2, Zi2, Zi2e, wi, wie, j / 2);
	backward_4(8, Zi8, Zi8e, wi, wie, j / 8);
	backward_4(32, Zi32, Zi32e, wi, wie, j / 32);
	backward_4(128, Zi128, Zi128e, wi, wie, j / 128);
	backward_4o(512, zk, zke, 512, Zi512, Zi512e, wi, wie, j / 512);
}

// -----------------

inline uint barrett(const ulong a, const uint b, const uint b_inv, const int b_s, uint * a_p)
{
	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.
	// b < 2^31, alpha = 2^29 => a < 2^29 b
	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31
	// b_inv = [2^{s + 32} / b]
	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31
	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].
	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])
	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,
	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.

	const uint d = mul_hi((uint)(a >> b_s), b_inv), r = (uint)a - d * b;
	const bool o = (r >= b);
	*a_p = o ? d + 1 : d;
	return o ? r - b : r;
}

inline int reduce64(long * f, const uint b, const uint b_inv, const int b_s)
{
	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b
	const ulong t = abs(*f);
	const ulong t_h = t >> 29;
	const uint t_l = (uint)t & ((1u << 29) - 1);

	uint d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint d_l, r_l = barrett(((ulong)r_h << 29) | t_l, b, b_inv, b_s, &d_l);
	const ulong d = ((ulong)d_h << 29) | d_l;

	const bool s = (*f < 0);
	*f = s ? -(long)d : (long)d;
	return s ? -(int)r_l : (int)r_l;
}

inline int reduce96(int96 * f, const uint b, const uint b_inv, const int b_s)
{
	const uint96 t = int96_abs(*f);
	const ulong t_h = ((ulong)t.s1 << (64 - 29)) | (t.s0 >> 29);
	const uint t_l = (uint)t.s0 & ((1u << 29) - 1);

	uint d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint d_l, r_l = barrett(((ulong)r_h << 29) | t_l, b, b_inv, b_s, &d_l);
	const ulong d = ((ulong)d_h << 29) | d_l;

	const bool s = int96_is_neg(*f);
	*f = int96_set_si(s ? -(long)d : (long)d);
	return s ? -(int)r_l : (int)r_l;
}

__kernel
void normalize1(__global RNS * restrict const z, __global RNSe * restrict const ze, __global long * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const unsigned int blk = abs(sblk);
	__global RNS * restrict const zi = &z[blk * idx];
	__global RNSe * restrict const zie = &ze[blk * idx];

	prefetch(zi, (size_t)blk);
	prefetch(zie, (size_t)blk);

	// Not converted into Montgomery form such that output is converted out of Montgomery form
	const RNS norm = (RNS)(NORM1, NORM2);
	const RNSe norme = (RNSe)(NORM3);

	int96 f = int96_set_si(0);

	sz_t j = 0;
	do
	{
		const RNS zj = mul(zi[j], norm);
		const RNSe zje = mule(zie[j], norme);
		int96 l = garner3(zj.s0, zj.s1, zje);
		if (sblk < 0) l = int96_add(l, l);
		f = int96_add(f, l);

		const int r = reduce96(&f, b, b_inv, b_s);
		zi[j] = toRNS(r); zie[j] = toRNSe(r);

		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	c[i] = (i == 0) ? -(long)f.s0 : (long)f.s0;
}

__kernel
void mul1(__global RNS * restrict const z, __global RNSe * restrict const ze, __global long * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global RNS * restrict const zi = &z[blk * idx];
	__global RNSe * restrict const zie = &ze[blk * idx];

	prefetch(zi, (size_t)blk);
	prefetch(zie, (size_t)blk);

	long f = 0;

	sz_t j = 0;
	do
	{
		f += geti_P1(zi[j].s0) * (long)a;
		const int r = reduce64(&f, b, b_inv, b_s);
		zi[j] = toRNS(r); zie[j] = toRNSe(r);
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	c[i] = (i == 0) ? -f : f;
}

__kernel
void normalize2(__global RNS * restrict const z, __global RNSe * restrict const ze, __global const long * restrict const c, 
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global RNS * restrict const zi = &z[blk * idx];
	__global RNSe * restrict const zie = &ze[blk * idx];

	long f = c[idx];

	sz_t j = 0;
	do
	{
		f += geti_P1(zi[j].s0);
		const int r = reduce64(&f, b, b_inv, b_s);
		zi[j] = toRNS(r); zie[j] = toRNSe(r);
		if (f == 0) return;
		++j;
	} while (j != blk - 1);

	const int r = (int)f;
	zi[blk - 1] = add(zi[blk - 1], toRNS(r));
	zie[blk - 1] = adde(zie[blk - 1], toRNSe(r));
}

__kernel
void set(__global RNS * restrict const z, __global RNSe * restrict const ze, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[idx] = (idx == 0) ? toRNS(a) : (RNS)(0, 0);
	ze[idx] = (idx == 0) ? toRNSe(a) : (RNSe)(0);
}

__kernel
void copy(__global RNS * restrict const z, __global RNSe * restrict const ze, const unsigned int dst, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[dst + idx] = z[src + idx];
	ze[dst + idx] = ze[src + idx];
}

__kernel
void copyp(__global RNS * restrict const zp, __global RNSe * restrict const zpe,
		   __global const RNS * restrict const z, __global const RNSe * restrict const ze, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	zp[idx] = z[src + idx];
	zpe[idx] = ze[src + idx];
}
