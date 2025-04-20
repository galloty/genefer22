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

typedef uint	sz_t;
typedef uint	uint32;
typedef int		int32;
typedef ulong	uint64;
typedef long	int64;
typedef uint2	uint32_2;
typedef uint4	uint32_4;
typedef int4	int32_4;

#if !defined(LNSIZE)
#define LNSIZE		16
#define NSIZE_4		16384u
#define P1			4194304001u
#define P2			4076863489u
#define Q1			100663297u
#define Q2			218103809u
#define R1			232465106u
#define R2			3444438393u
#define NORM1		4193792001u
#define NORM2		4076365825u
#define InvP2_P1	1797558821u
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
	const uint64 t = lhs * (uint64)(rhs);
	const uint32 lo = (uint32)(t), hi = (uint32)(t >> 32);
	const uint32 mp = mul_hi(lo * q, p);
	return _subMod(hi, mp, p);
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

INLINE uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }
INLINE uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }

// Montgomery form
INLINE uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P1, Q1); }
INLINE uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P2, Q2); }

INLINE uint32 toMonty_P1(const uint32 lhs) { return _toMonty(lhs, R1, P1, Q1); }
INLINE uint32 toMonty_P2(const uint32 lhs) { return _toMonty(lhs, R2, P2, Q2); }

// INLINE uint32 fromMonty_P1(const uint32 lhs) { return _fromMonty(lhs, P1, Q1); }
// INLINE uint32 fromMonty_P2(const uint32 lhs) { return _fromMonty(lhs, P2, Q2); }

INLINE int32 geti_P1(const uint32 r) { return (r > P1 / 2) ? (int32)(r - P1) : (int32)(r); }

INLINE int64 garner2(const uint32 r1, const uint32 r2)
{
	const uint32 u12 = mul_P1(sub_P1(r1, r2), InvP2_P1);
	const uint64 n = r2 + u12 * (uint64)(P2);
	return (n > P1P2 / 2) ? (int64)(n - P1P2) : (int64)(n);
}

// --- RNS ---

typedef uint32_2	RNS;
typedef uint32_4	RNS2;
typedef uint32_2	RNS_W;
typedef uint32_4	RNS_W2;

INLINE RNS toRNS(const int32 i) { return ((RNS)(i, i) + ((i < 0) ? (RNS)(P1, P2) : (RNS)(0, 0))); }

INLINE RNS add(const RNS lhs, const RNS rhs) { return (RNS)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }
INLINE RNS sub(const RNS lhs, const RNS rhs) { return (RNS)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }
INLINE RNS mul(const RNS lhs, const RNS rhs) { return (RNS)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }

INLINE RNS sqr(const RNS lhs) { return mul(lhs, lhs); }

INLINE RNS mulW(const RNS lhs, const RNS_W w) { return mul(lhs, w); }

INLINE RNS toMonty(const RNS lhs) { return (RNS)(toMonty_P1(lhs.s0), toMonty_P2(lhs.s1)); }

INLINE RNS2 mul2(const RNS2 lhs, const RNS rhs) { return (RNS2)(mul(lhs.s01, rhs), mul(lhs.s23, rhs)); }

// --- transform/macro ---

#define FWD2(z0, z1, w) { const RNS t = mulW(z1, w); z1 = sub(z0, t); z0 = add(z0, t); }
#define BCK2(z0, z1, wi) { const RNS t = sub(z0, z1); z0 = add(z0, z1); z1 = mulW(t, wi); }

#define SQR2(z0, z1, w) { const RNS t = sqr(mulW(z1, w)); z1 = mul(add(z0, z0), z1); z0 = add(sqr(z0), t); }
#define SQR2N(z0, z1, w) { const RNS t = sqr(mulW(z1, w)); z1 = mul(add(z0, z0), z1); z0 = sub(sqr(z0), t); }

#define MUL2(z0, z1, zp0, zp1, w) { const RNS t = mul(mulW(z1, w), mulW(zp1, w)); z1 = add(mul(z0, zp1), mul(zp0, z1)); z0 = add(mul(z0, zp0), t); }
#define MUL2N(z0, z1, zp0, zp1, w) { const RNS t = mul(mulW(z1, w), mulW(zp1, w)); z1 = add(mul(z0, zp1), mul(zp0, z1)); z0 = sub(mul(z0, zp0), t); }

#define DECLARE_W(j)	const RNS_W w1 = w[j]; const RNS_W2 w2 = ((__global const RNS_W2 *)w)[j];
#define DECLARE_WI(j)	const RNS_W wi1 = wi[j]; const RNS_W2 wi2 = ((__global const RNS_W2 *)wi)[j];

#define FORWARD4() \
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1); \
	FWD2(zl[0], zl[1], w2.s01); FWD2(zl[2], zl[3], w2.s23);

#define BACKWARD4() \
	BCK2(zl[0], zl[1], wi2.s01); BCK2(zl[2], zl[3], wi2.s23); \
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1);

#define FORWARD22() \
	FWD2(zl[0], zl[2], w1); FWD2(zl[1], zl[3], w1);

#define BACKWARD22() \
	BCK2(zl[0], zl[2], wi1); BCK2(zl[1], zl[3], wi1);

#define SQUARE22() \
	SQR2(zl[0], zl[1], w0); SQR2N(zl[2], zl[3], w0);

#define MUL22() \
	MUL2(zl[0], zl[1], zpl[0], zpl[1], w0); MUL2N(zl[2], zl[3], zpl[2], zpl[3], w0);

// --- transform/inline ---

INLINE void _loadg(RNS zl[4], __global const RNS * restrict const z, const size_t s) { for (size_t l = 0; l < 4; ++l) zl[l] = z[l * s]; }
INLINE void _loadl(RNS zl[4], __local const RNS * restrict const Z, const size_t s) { for (size_t l = 0; l < 4; ++l) zl[l] = Z[l * s]; }
INLINE void _storeg(__global RNS * restrict const z, const size_t s, const RNS zl[4]) { for (size_t l = 0; l < 4; ++l) z[l * s] = zl[l]; }
INLINE void _storel(__local RNS * restrict const Z, const size_t s, const RNS zl[4]) { for (size_t l = 0; l < 4; ++l) Z[l * s] = zl[l]; }

INLINE void forward_4(const sz_t m, __local RNS * restrict const Z, __global const RNS_W * restrict const w, const sz_t j)
{
	DECLARE_W(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, m);
	FORWARD4();
	_storel(Z, m, zl);
}

INLINE void forward_4i(const sz_t ml, __local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z,
	__global const RNS_W * restrict const w, const sz_t j)
{
	DECLARE_W(j);
	RNS zl[4]; _loadg(zl, z, mg);
	FORWARD4();
	_storel(Z, ml, zl);
}

INLINE void forward_4i_0(const sz_t ml, __local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z,
	__global const RNS_W * restrict const w)
{
	DECLARE_W(1);
	RNS zl[4]; _loadg(zl, z, mg);
	zl[0] = toMonty(zl[0]); zl[1] = toMonty(zl[1]);
	FORWARD4();
	_storel(Z, ml, zl);
}

INLINE void forward_4o(const sz_t mg, __global RNS * restrict const z, const sz_t ml, __local const RNS * restrict const Z,
	__global const RNS_W * restrict const w, const sz_t j)
{
	DECLARE_W(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, ml);
	FORWARD4();
	_storeg(z, mg, zl);
}

INLINE void backward_4(const sz_t m, __local RNS * restrict const Z, __global const RNS_W * restrict const wi, const sz_t j)
{
	DECLARE_WI(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, m);
	BACKWARD4();
	_storel(Z, m, zl);
}

INLINE void backward_4i(const sz_t ml, __local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const z,
	__global const RNS_W * restrict const wi,const sz_t j)
{
	DECLARE_WI(j);
	RNS zl[4]; _loadg(zl, z, mg);
	BACKWARD4();
	_storel(Z, ml, zl);
}

INLINE void backward_4o(const sz_t mg, __global RNS * restrict const z, const sz_t ml, __local const RNS * restrict const Z,
	__global const RNS_W * restrict const wi, const sz_t j)
{
	DECLARE_WI(j);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, ml);
	BACKWARD4();
	_storeg(z, mg, zl);
}

INLINE void write_4(const sz_t mg, __global RNS * restrict const z, __local const RNS * restrict const Z)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0 * mg] = Z[0]; z[1 * mg] = Z[1]; z[2 * mg] = Z[2]; z[3 * mg] = Z[3];
}

INLINE void fwd2write_4(const sz_t mg, __global RNS * restrict const z, __local const RNS * restrict const Z, const RNS_W w1)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, 1);
	FORWARD22();
	_storeg(z, mg, zl);
}

INLINE void square_22(__local RNS * restrict const Z, const RNS_W w0)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, 1);
	SQUARE22();
	_storel(Z, 1, zl);
}

INLINE void square_4(__local RNS * restrict const Z, const RNS_W w1, const RNS_W wi1, const RNS_W w0)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, 1);
	FORWARD22();
	SQUARE22();
	BACKWARD22();
	_storel(Z, 1, zl);
}

INLINE void mul_22(__local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const zp, const RNS_W w0)
{
	RNS zpl[4]; _loadg(zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, 1);
	MUL22();
	_storel(Z, 1, zl);
}

INLINE void mul_4(__local RNS * restrict const Z, const sz_t mg, __global const RNS * restrict const zp,
	const RNS_W w1, const RNS_W wi1, const RNS_W w0)
{
	RNS zpl[4]; _loadg(zpl, zp, mg);
	barrier(CLK_LOCAL_MEM_FENCE);
	RNS zl[4]; _loadl(zl, Z, 1);
	FORWARD22();
	MUL22();
	BACKWARD22();
	_storel(Z, 1, zl);
}

// --- transform ---

#define DECLARE_VAR(B_N, CHUNK_N) \
	__local RNS Z[4 * B_N * CHUNK_N]; \
	\
	/* threadIdx < B_N */ \
	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \
	__local RNS * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	sz_t sj = s + idx_m;

#define DECLARE_VAR_FORWARD() \
	__global RNS * __restrict__ const zi = &z[ki]; \
	__global RNS * __restrict__ const zo = &z[ko];

#define DECLARE_VAR_BACKWARD() \
	__global RNS * __restrict__ const zi = &z[ko]; \
	__global RNS * __restrict__ const zo = &z[ki]; \
	const sz_t n_4 = NSIZE_4; \
	__global const RNS_W * restrict const wi = &w[4 * n_4];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);

#define FORWARD_I_0(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_0(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w);

#define FORWARD_O(CHUNK_N) \
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], w, sj / 1);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sj / 1);

#define BACKWARD_O(B_N, CHUNK_N) \
	backward_4o(B_N << lm, zo, B_N * CHUNK_N, &Z[i], wi, sj / B_N);

// -----------------

#define B_64	(64 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void forward64(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	FORWARD_O(CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void backward64(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], wi, sj / 4);
	BACKWARD_O(B_64, CHUNK64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void forward64_0(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	const int lm = LNSIZE - 6; const unsigned int s = 64 / 4;
	FORWARD_I_0(B_64, CHUNK64);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);
	FORWARD_O(CHUNK64);
}

// -----------------

#define B_256	(256 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void forward256(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	FORWARD_O(CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void backward256(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_256, CHUNK256);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sj / 16);
	BACKWARD_O(B_256, CHUNK256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void forward256_0(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	const int lm = LNSIZE - 8; const unsigned int s = 256 / 4;
	FORWARD_I_0(B_256, CHUNK256);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	FORWARD_O(CHUNK256);
}

// -----------------

#define B_1024	(1024 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)
{
	FORWARD_I(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	FORWARD_O(CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void backward1024(__global RNS * restrict const z, __global const RNS_W * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_1024, CHUNK1024);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], wi, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], wi, sj / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], wi, sj / 64);
	BACKWARD_O(B_1024, CHUNK1024);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024_0(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	const int lm = LNSIZE - 10; const unsigned int s = 1024 / 4;
	FORWARD_I_0(B_1024, CHUNK1024);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64 );
	forward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	forward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	forward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	FORWARD_O(CHUNK1024);
}

// -----------------

#define DECLARE_VAR_32() \
	__local RNS Z[32 * BLK32]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (32 / 4 * BLK32), group_id = gid / (32 / 4 * BLK32); \
	const sz_t k32 = group_id * 32 * BLK32, i = local_id; \
	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \
	\
	__global RNS * restrict const zk = &z[k32 + i32 + i8]; \
	__local RNS * const Z32 = &Z[i32]; \
	__local RNS * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \
	__local RNS * const Zi2 = &Z32[i2]; \
	__local RNS * const Z4 = &Z32[4 * i8];

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
#endif
void square32(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_32();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4o(8, zk, 8, Zi8, wi, j / 8);
}

#define DECLARE_VAR_64() \
	__local RNS Z[64 * BLK64]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (64 / 4 * BLK64), group_id = gid / (64 / 4 * BLK64); \
	const sz_t k64 = group_id * 64 * BLK64, i = local_id; \
	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \
	\
	__global RNS * restrict const zk = &z[k64 + i64 + i16]; \
	__local RNS * const Z64 = &Z[i64]; \
	__local RNS * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \
	__local RNS * const Zi4 = &Z64[i4]; \
	__local RNS * const Z4 = &Z64[4 * i16];

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#endif
void square64(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_64();
	__global const RNS_W * const wi = &w[4 * NSIZE_4];

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w[j], wi[j], w[NSIZE_4 + j]);
	backward_4(4, Zi4, wi, j / 4);
	backward_4o(16, zk, 16, Zi16, wi, j / 16);
}

#define DECLARE_VAR_128() \
	__local RNS Z[128 * BLK128]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (128 / 4 * BLK128), group_id = gid / (128 / 4 * BLK128); \
	const sz_t k128 = group_id * 128 * BLK128, i = local_id; \
	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \
	\
	__global RNS * restrict const zk = &z[k128 + i128 + i32]; \
	__local RNS * const Z128 = &Z[i128]; \
	__local RNS * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \
	__local RNS * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \
	__local RNS * const Zi2 = &Z128[i2]; \
	__local RNS * const Z4 = &Z128[4 * i32];

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
#endif
void square128(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_128();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4(8, Zi8, wi, j / 8);
	backward_4o(32, zk, 32, Zi32, wi, j / 32);
}

#define DECLARE_VAR_256() \
	__local RNS Z[256 * BLK256]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (256 / 4 * BLK256), group_id = gid / (256 / 4 * BLK256); \
	const sz_t k256 = group_id * 256 * BLK256, i = local_id; \
	const sz_t i256 = 0, i64 = i; \
	\
	__global RNS * restrict const zk = &z[k256 + i256 + i64]; \
	__local RNS * const Z256 = &Z[i256]; \
	__local RNS * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \
	__local RNS * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \
	__local RNS * const Zi4 = &Z256[i4]; \
	__local RNS * const Z4 = &Z256[4 * i64];

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
#endif
void square256(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_256();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w[j], wi[j], w[NSIZE_4 + j]);
	backward_4(4, Zi4, wi, j / 4);
	backward_4(16, Zi16, wi, j / 16);
	backward_4o(64, zk, 64, Zi64, wi, j / 64);
}

#define DECLARE_VAR_512() \
	__local RNS Z[512]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (512 / 4), group_id = gid / (512 / 4); \
	const sz_t k512 = group_id * 512, i128 = local_id; \
	\
	__global RNS * restrict const zk = &z[k512 + i128]; \
	__local RNS * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \
	__local RNS * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \
	__local RNS * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \
	__local RNS * const Zi2 = &Z[i2]; \
	__local RNS * const Z4 = &Z[4 * i128];

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void square512(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_512();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4(8, Zi8, wi, j / 8);
	backward_4(32, Zi32, wi, j / 32);
	backward_4o(128, zk, 128, Zi128, wi, j / 128);
}

#define DECLARE_VAR_1024() \
	__local RNS Z[1024]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (1024 / 4), group_id = gid / (1024 / 4); \
	const sz_t k1024 = group_id * 1024, i256 = local_id; \
	\
	__global RNS * restrict const zk = &z[k1024 + i256]; \
	__local RNS * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \
	__local RNS * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \
	__local RNS * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \
	__local RNS * const Zi4 = &Z[i4]; \
	__local RNS * const Z4 = &Z[4 * i256];

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void square1024(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_1024();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w[j], wi[j], w[NSIZE_4 + j]);
	backward_4(4, Zi4, wi, j / 4);
	backward_4(16, Zi16, wi, j / 16);
	backward_4(64, Zi64, wi, j / 64);
	backward_4o(256, zk, 256, Zi256, wi, j / 256);
}

#define DECLARE_VAR_2048() \
	__local RNS Z[2048]; \
	\
	const sz_t gid = (sz_t)get_global_id(0), j = NSIZE_4 + gid; \
	const sz_t local_id = gid % (2048 / 4), group_id = gid / (2048 / 4); \
	const sz_t k2048 = group_id * 2048, i512 = local_id; \
	\
	__global RNS * restrict const zk = &z[k2048 + i512]; \
	__local RNS * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \
	__local RNS * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \
	__local RNS * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \
	__local RNS * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \
	__local RNS * const Zi2 = &Z[i2]; \
	__local RNS * const Z4 = &Z[4 * i512];

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void square2048(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_2048();
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4(8, Zi8, wi, j / 8);
	backward_4(32, Zi32, wi, j / 32);
	backward_4(128, Zi128, wi, j / 128);
	backward_4o(512, zk, 512, Zi512, wi, j / 512);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(8, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2write_4(16, zk, Z4, w[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(32, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2write_4(64, zk, Z4, w[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void fwd512p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(128, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void fwd1024p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2write_4(256, zk, Z4, w[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void fwd2048p(__global RNS * restrict const z, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(512, zk, Z4);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((reqd_work_group_size(32 / 4 * BLK32, 1, 1)))
#endif
void mul32(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_32();
	__global const RNS * restrict const zpk = &zp[k32 + i32 + i8];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 8, zpk, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4o(8, zk, 8, Zi8, wi, j / 8);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((reqd_work_group_size(64 / 4 * BLK64, 1, 1)))
#endif
void mul64(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_64();
	__global const RNS * restrict const zpk = &zp[k64 + i64 + i16];
	__global const RNS_W * const wi = &w[4 * NSIZE_4];

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 16, zpk, w[j], wi[j], w[NSIZE_4 + j]);
	backward_4(4, Zi4, wi, j / 4);
	backward_4o(16, zk, 16, Zi16, wi, j / 16);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((reqd_work_group_size(128 / 4 * BLK128, 1, 1)))
#endif
void mul128(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_128();
	__global const RNS * restrict const zpk = &zp[k128 + i128 + i32];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 32, zpk, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4(8, Zi8, wi, j / 8);
	backward_4o(32, zk, 32, Zi32, wi, j / 32);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((reqd_work_group_size(256 / 4 * BLK256, 1, 1)))
#endif
void mul256(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_256();
	__global const RNS * restrict const zpk = &zp[k256 + i256 + i64];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 64, zpk, w[j], wi[j], w[NSIZE_4 + j]);
	backward_4(4, Zi4, wi, j / 4);
	backward_4(16, Zi16, wi, j / 16);
	backward_4o(64, zk, 64, Zi64, wi, j / 64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void mul512(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_512();
	__global const RNS * restrict const zpk = &zp[k512 + i128];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 128, zpk, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4(8, Zi8, wi, j / 8);
	backward_4(32, Zi32, wi, j / 32);
	backward_4o(128, zk, 128, Zi128, wi, j / 128);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void mul1024(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_1024();
	__global const RNS * restrict const zpk = &zp[k1024 + i256];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 256, zpk, w[j], wi[j], w[NSIZE_4 + j]);
	backward_4(4, Zi4, wi, j / 4);
	backward_4(16, Zi16, wi, j / 16);
	backward_4(64, Zi64, wi, j / 64);
	backward_4o(256, zk, 256, Zi256, wi, j / 256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void mul2048(__global RNS * restrict const z, __global const RNS * restrict const zp, __global const RNS_W * restrict const w)
{
	DECLARE_VAR_2048();
	__global const RNS * restrict const zpk = &zp[k2048 + i512];
	__global const RNS_W * restrict const wi = &w[4 * NSIZE_4];

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 512, zpk, w[NSIZE_4 + j]);
	backward_4(2, Zi2, wi, j / 2);
	backward_4(8, Zi8, wi, j / 8);
	backward_4(32, Zi32, wi, j / 32);
	backward_4(128, Zi128, wi, j / 128);
	backward_4o(512, zk, 512, Zi512, wi, j / 512);
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

__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))
void normalize1(__global RNS2 * restrict const z, __global int64 * restrict const c,
	const uint32 b, const uint32 b_inv, const int b_s, const int32 dup)
{
	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;
	__global RNS2 * restrict const zi = &z[2 * gid];
	__local int64 cl[NORM_WG_SZ];

	// Not converted into Montgomery form such that output is converted out of Montgomery form
	const RNS norm = (RNS)(NORM1, NORM2);

	const RNS2 u01 = mul2(zi[0], norm), u23 = mul2(zi[1], norm);

	int32_4 r;
	int64 l0 = garner2(u01.s0, u01.s1); if (dup != 0) l0 += l0;
	int64 f = l0; r.s0 = reduce64(&f, b, b_inv, b_s);
	int64 l1 = garner2(u01.s2, u01.s3); if (dup != 0) l1 += l1;
	f += l1; r.s1 = reduce64(&f, b, b_inv, b_s);
	int64 l2 = garner2(u23.s0, u23.s1); if (dup != 0) l2 += l2;
	f += l2; r.s2 = reduce64(&f, b, b_inv, b_s);
	int64 l3 = garner2(u23.s2, u23.s3); if (dup != 0) l3 += l3;
	f += l3; r.s3 = reduce64(&f, b, b_inv, b_s);

	cl[lid] = f;

	if (lid == NORM_WG_SZ - 1)
	{
		const sz_t i = (gid / NORM_WG_SZ + 1) % (NSIZE_4 / NORM_WG_SZ);
		c[i] = (i == 0) ? -f : f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	f = (lid == 0) ? 0 : cl[lid - 1];
	f += r.s0; r.s0 = reduce64(&f, b, b_inv, b_s);
	f += r.s1; r.s1 = reduce64(&f, b, b_inv, b_s);
	f += r.s2; r.s2 = reduce64(&f, b, b_inv, b_s);
	f += r.s3; r.s3 = (sz_t)(f);

	zi[0] = (RNS2)(toRNS(r.s0), toRNS(r.s1)); zi[1] = (RNS2)(toRNS(r.s2), toRNS(r.s3));
}

__kernel
void normalize2(__global RNS * restrict const z, __global const int64 * restrict const c, 
	const uint32 b, const uint32 b_inv, const int b_s)
{
	const sz_t gid = (sz_t)get_global_id(0);
	__global RNS * restrict const zi = &z[NORM_WG_SZ * 4 * gid];

	int64 f = c[gid];

	for (sz_t j = 0; j < 3; ++j)
	{
		f += geti_P1(zi[j].s0);
		const int32 r = reduce64(&f, b, b_inv, b_s);
		zi[j] = toRNS(r);
		if (f == 0) return;
	}
	f += geti_P1(zi[3].s0);
	zi[3] = toRNS((int32)(f));
}

__kernel __attribute__((reqd_work_group_size(NORM_WG_SZ, 1, 1)))
void mulscalar(__global RNS * restrict const z, __global int64 * restrict const c,
	const uint32 b, const uint32 b_inv, const int b_s, const int32 a)
{
	const sz_t gid = (sz_t)get_global_id(0), lid = gid % NORM_WG_SZ;
	__global RNS * restrict const zi = &z[4 * gid];
	__local int64 cl[NORM_WG_SZ];

	int32_4 r;
	int64 f = geti_P1(zi[0].s0) * (int64)(a);
	r.s0 = reduce64(&f, b, b_inv, b_s);
	f += geti_P1(zi[1].s0) * (int64)(a);
	r.s1 = reduce64(&f, b, b_inv, b_s);
	f += geti_P1(zi[2].s0) * (int64)(a);
	r.s2 = reduce64(&f, b, b_inv, b_s);
	f += geti_P1(zi[3].s0) * (int64)(a);
	r.s3 = reduce64(&f, b, b_inv, b_s);

	cl[lid] = f;

	if (lid == NORM_WG_SZ - 1)
	{
		const sz_t i = (gid / NORM_WG_SZ + 1) % (NSIZE_4 / NORM_WG_SZ);
		c[i] = (i == 0) ? -f : f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	f = (lid == 0) ? 0 : cl[lid - 1];
	f += r.s0; r.s0 = reduce64(&f, b, b_inv, b_s);
	f += r.s1; r.s1 = reduce64(&f, b, b_inv, b_s);
	f += r.s2; r.s2 = reduce64(&f, b, b_inv, b_s);
	f += r.s3; r.s3 = (sz_t)(f);

	zi[0] = toRNS(r.s0); zi[1] = toRNS(r.s1); zi[2] = toRNS(r.s2); zi[3] = toRNS(r.s3);
}

__kernel
void set(__global RNS2 * restrict const z, const uint32 a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const uint32 ai = (idx == 0) ? a : 0;
	z[idx] = (RNS2)(ai, ai, 0, 0);
}

__kernel
void copy(__global RNS2 * restrict const z, const sz_t dst, const sz_t src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[dst + idx] = z[src + idx];
}

__kernel
void copyp(__global RNS2 * restrict const zp, __global const RNS2 * restrict const z, const sz_t src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	zp[idx] = z[src + idx];
}
