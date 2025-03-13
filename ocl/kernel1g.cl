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

#ifndef NSIZE_2
#define NSIZE_2		524288
#define	NSIZE_4		262144
#define	SNORM61		42
#define BLK32		8	// 4KB
#define BLK64		4
#define BLK128		2
#define BLK256		1
#define CHUNK64		4	// 4KB
#define CHUNK256	2
#define CHUNK1024	1
#define MAX_WORK_GROUP_SIZE	1024
#endif

typedef uint	sz_t;

// --- Z/(2^61 - 1)Z ---

#define	M61		0x1ffffffffffffffful

INLINE ulong _add61(const ulong lhs, const ulong rhs)
{
	const ulong t = lhs + rhs;
	const ulong c = ((uint)(t >> 32) > (uint)(M61 >> 32)) ? M61 : 0;	// t > M61 ?
	return t - c;
}

INLINE ulong _sub61(const ulong lhs, const ulong rhs)
{
	const ulong t = lhs - rhs;
	const ulong c = ((int)(t >> 32) < 0) ? M61 : 0;	// t < 0 ?
	return t + c;
}

INLINE ulong _mul61(const ulong lhs, const ulong rhs)
{
	const ulong lo = lhs * rhs, hi = mul_hi(lhs, rhs);
	const ulong lo61 = lo & M61, hi61 = (lo >> 61) | (hi << 3);
	return _add61(lo61, hi61);
}

INLINE ulong _lshift61(const ulong lhs, const int s)
{
	const ulong lo = lhs << s, hi = lhs >> (64 - s);
	const ulong lo61 = lo & M61, hi61 = (lo >> 61) | (hi << 3);
	return _add61(lo61, hi61);
}

INLINE long get_int61(const ulong n) { return ((uint)(n >> 32) > (uint)((M61 / 2) >> 32)) ? (long)(n - M61) : (long)(n); }
INLINE ulong set_int61(const int i) { return (i < 0) ? ((ulong)(i) + M61) : (ulong)(i); }

// --- GF((2^61 - 1)^2) ---

typedef ulong2	GF61;

INLINE GF61 toGF61(const int ia, const int ib) { return (GF61)(set_int61(ia), set_int61(ib)); }

INLINE GF61 add61(const GF61 lhs, const GF61 rhs) { return (GF61)(_add61(lhs.s0, rhs.s0), _add61(lhs.s1, rhs.s1)); }
INLINE GF61 sub61(const GF61 lhs, const GF61 rhs) { return (GF61)(_sub61(lhs.s0, rhs.s0), _sub61(lhs.s1, rhs.s1)); }
INLINE GF61 lshift61(const GF61 lhs, const int s) { return (GF61)(_lshift61(lhs.s0, s), _lshift61(lhs.s1, s)); }

INLINE GF61 sqr61(const GF61 lhs)
{
	const ulong t = _mul61(lhs.s0, lhs.s1);
	return (GF61)(_sub61(_mul61(lhs.s0, lhs.s0), _mul61(lhs.s1, lhs.s1)), _add61(t, t));
}
INLINE GF61 mul61(const GF61 lhs, const GF61 rhs)
{
	return (GF61)(_sub61(_mul61(lhs.s0, rhs.s0), _mul61(lhs.s1, rhs.s1)), _add61(_mul61(lhs.s1, rhs.s0), _mul61(lhs.s0, rhs.s1)));
}
INLINE GF61 mulconj61(const GF61 lhs, const GF61 rhs)
{
	return (GF61)(_add61(_mul61(lhs.s0, rhs.s0), _mul61(lhs.s1, rhs.s1)), _sub61(_mul61(lhs.s1, rhs.s0), _mul61(lhs.s0, rhs.s1)));
}

INLINE GF61 addi61(const GF61 lhs, const GF61 rhs) { return (GF61)(_sub61(lhs.s0, rhs.s1), _add61(lhs.s1, rhs.s0)); }
INLINE GF61 subi61(const GF61 lhs, const GF61 rhs) { return (GF61)(_add61(lhs.s0, rhs.s1), _sub61(lhs.s1, rhs.s0)); }

// --- transform/inline ---

INLINE void forward_4io(const sz_t m, __global GF61 * restrict const z, __global const GF61 * restrict const w, const sz_t j)
{
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	const GF61 u0 = z[0 * m], u1 = mul61(z[1 * m], w2), u2 = mul61(z[2 * m], w1), u3 = mul61(z[3 * m], w3);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	z[0 * m] = add61(v0, v1); z[1 * m] = sub61(v0, v1); z[2 * m] = addi61(v2, v3); z[3 * m] = subi61(v2, v3);
}

INLINE void backward_4io(const sz_t m, __global GF61 * restrict const z, __global const GF61 * restrict const w, const sz_t j)
{
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	const GF61 u0 = z[0 * m], u1 = z[1 * m], u2 = z[2 * m], u3 = z[3 * m];
	const GF61 v0 = add61(u0, u1), v1 = sub61(u0, u1), v2 = add61(u2, u3), v3 = sub61(u3, u2);
	z[0 * m] = add61(v0, v2); z[2 * m] = mulconj61(sub61(v0, v2), w1);
	z[1 * m] = mulconj61(addi61(v1, v3), w2); z[3 * m] = mulconj61(subi61(v1, v3), w3);
}

INLINE void square_22io(__global GF61 * restrict const z, const GF61 w)
{
	const GF61 u0 = z[0], u1 = z[1], u2 = z[2], u3 = z[3];
	z[0] = add61(sqr61(u0), mul61(sqr61(u1), w)); z[1] = mul61(add61(u0, u0), u1);
	z[2] = sub61(sqr61(u2), mul61(sqr61(u3), w)); z[3] = mul61(add61(u2, u2), u3);
}

INLINE void square_4io(__global GF61 * restrict const z, const GF61 w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = z[0], u1 = z[1], u2 = mul61(z[2], w), u3 = mul61(z[3], w);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	const GF61 s0 = add61(sqr61(v0), mul61(sqr61(v1), w)), s1 = mul61(add61(v0, v0), v1);
	const GF61 s2 = sub61(sqr61(v2), mul61(sqr61(v3), w)), s3 = mul61(add61(v2, v2), v3);
	z[0] = add61(s0, s2); z[2] = mulconj61(sub61(s0, s2), w);
	z[1] = add61(s1, s3); z[3] = mulconj61(sub61(s1, s3), w);
}

// -----------------

INLINE void forward_4(const sz_t m, __local GF61 * restrict const Z, __global const GF61 * restrict const w, const sz_t j)
{
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0 * m], u1 = mul61(Z[1 * m], w2), u2 = mul61(Z[2 * m], w1), u3 = mul61(Z[3 * m], w3);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	Z[0 * m] = add61(v0, v1); Z[1 * m] = sub61(v0, v1); Z[2 * m] = addi61(v2, v3); Z[3 * m] = subi61(v2, v3);
}

INLINE void forward_4i(const sz_t ml, __local GF61 * restrict const Z, const sz_t mg, __global const GF61 * restrict const z, __global const GF61 * restrict const w, const sz_t j)
{
	__global const GF61 * const z2mg = &z[2 * mg];
	const GF61 z0 = z[0], z2 = z2mg[0], z1 = z[mg], z3 = z2mg[mg];
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	const GF61 u0 = z0, u1 = mul61(z1, w2), u2 = mul61(z2, w1), u3 = mul61(z3, w3);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	Z[0 * ml] = add61(v0, v1); Z[1 * ml] = sub61(v0, v1); Z[2 * ml] = addi61(v2, v3); Z[3 * ml] = subi61(v2, v3);
}

INLINE void forward_4o(const sz_t mg, __global GF61 * restrict const z, const sz_t ml, __local const GF61 * restrict const Z, __global const GF61 * restrict const w, const sz_t j)
{
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0 * ml], u1 = mul61(Z[1 * ml], w2), u2 = mul61(Z[2 * ml], w1), u3 = mul61(Z[3 * ml], w3);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	__global GF61 * const z2mg = &z[2 * mg];
	z[0] = add61(v0, v1); z[mg] = sub61(v0, v1); z2mg[0] = addi61(v2, v3); z2mg[mg] = subi61(v2, v3);
}

INLINE void backward_4(const sz_t m, __local GF61 * restrict const Z, __global const GF61 * restrict const w, const sz_t j)
{
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0 * m], u1 = Z[1 * m], u2 = Z[2 * m], u3 = Z[3 * m];
	const GF61 v0 = add61(u0, u1), v1 = sub61(u0, u1), v2 = add61(u2, u3), v3 = sub61(u3, u2);
	Z[0 * m] = add61(v0, v2); Z[2 * m] = mulconj61(sub61(v0, v2), w1);
	Z[1 * m] = mulconj61(addi61(v1, v3), w2); Z[3 * m] = mulconj61(subi61(v1, v3), w3);
}

INLINE void backward_4i(const sz_t ml, __local GF61 * restrict const Z, const sz_t mg, __global const GF61 * restrict const z, __global const GF61 * restrict const w,const sz_t j)
{
	__global const GF61 * const z2mg = &z[2 * mg];
	const GF61 u0 = z[0], u1 = z[mg], u2 = z2mg[0], u3 = z2mg[mg];
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	const GF61 v0 = add61(u0, u1), v1 = sub61(u0, u1), v2 = add61(u2, u3), v3 = sub61(u3, u2);
	Z[0 * ml] = add61(v0, v2); Z[2 * ml] = mulconj61(sub61(v0, v2), w1);
	Z[1 * ml] = mulconj61(addi61(v1, v3), w2); Z[3 * ml] = mulconj61(subi61(v1, v3), w3);
}

INLINE void backward_4o(const sz_t mg, __global GF61 * restrict const z, const sz_t ml, __local const GF61 * restrict const Z, __global const GF61 * restrict const w, const sz_t j)
{
	const GF61 w1 = w[j], w2 = w[NSIZE_2 + j], w3 = w[2 * NSIZE_2 + j];
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0 * ml], u1 = Z[1 * ml], u2 = Z[2 * ml], u3 = Z[3 * ml];
	const GF61 v0 = add61(u0, u1), v1 = sub61(u0, u1), v2 = add61(u2, u3), v3 = sub61(u3, u2);
	__global GF61 * const z2mg = &z[2 * mg];
	z[0] = add61(v0, v2); z2mg[0] = mulconj61(sub61(v0, v2), w1);
	z[mg] = mulconj61(addi61(v1, v3), w2); z2mg[mg] = mulconj61(subi61(v1, v3), w3);
}

INLINE void write_4(const sz_t mg, __global GF61 * restrict const z, __local const GF61 * restrict const Z)
{
	__global GF61 * const z2mg = &z[2 * mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	z[0] = Z[0]; z[mg] = Z[1]; z2mg[0] = Z[2]; z2mg[mg] = Z[3];
}

INLINE void fwd2write_4(const sz_t mg, __global GF61 * restrict const z, __local const GF61 * restrict const Z, const GF61 w1)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0], u2 = mul61(Z[2], w1), u1 = Z[1], u3 = mul61(Z[3], w1);
	const GF61 v0 = add61(u0, u2), v2 = sub61(u0, u2), v1 = add61(u1, u3), v3 = sub61(u1, u3);
	__global GF61 * const z2mg = &z[2 * mg];
	z[0] = v0; z2mg[0] = v2; z[mg] = v1; z2mg[mg] = v3;
}

INLINE void square_22(__local GF61 * restrict const Z, const GF61 w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];
	Z[0] = add61(sqr61(u0), mul61(sqr61(u1), w)); Z[1] = mul61(add61(u0, u0), u1);
	Z[2] = sub61(sqr61(u2), mul61(sqr61(u3), w)); Z[3] = mul61(add61(u2, u2), u3);
}

INLINE void square_4(__local GF61 * restrict const Z, const GF61 w)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0], u1 = Z[1], u2 = mul61(Z[2], w), u3 = mul61(Z[3], w);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	const GF61 s0 = add61(sqr61(v0), mul61(sqr61(v1), w)), s1 = mul61(add61(v0, v0), v1);
	const GF61 s2 = sub61(sqr61(v2), mul61(sqr61(v3), w)), s3 = mul61(add61(v2, v2), v3);
	Z[0] = add61(s0, s2); Z[2] = mulconj61(sub61(s0, s2), w);
	Z[1] = add61(s1, s3); Z[3] = mulconj61(sub61(s1, s3), w);
}

INLINE void mul_22(__local GF61 * restrict const Z, const sz_t mg, __global const GF61 * restrict const z, const GF61 w)
{
	__global const GF61 * const z2mg = &z[2 * mg];
	const GF61 u0p = z[0], u1p = z[mg], u2p = z2mg[0], u3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0], u1 = Z[1], u2 = Z[2], u3 = Z[3];
	Z[0] = add61(mul61(u0, u0p), mul61(mul61(u1, u1p), w));
	Z[1] = add61(mul61(u0, u1p), mul61(u0p, u1));
	Z[2] = sub61(mul61(u2, u2p), mul61(mul61(u3, u3p), w));
	Z[3] = add61(mul61(u2, u3p), mul61(u2p, u3));
}

INLINE void mul_4(__local GF61 * restrict const Z, const sz_t mg, __global const GF61 * restrict const z, const GF61 w)
{
	__global const GF61 * const z2mg = &z[2 * mg];
	const GF61 v0p = z[0], v1p = z[mg], v2p = z2mg[0], v3p = z2mg[mg];
	barrier(CLK_LOCAL_MEM_FENCE);
	const GF61 u0 = Z[0], u1 = Z[1], u2 = mul61(Z[2], w), u3 = mul61(Z[3], w);
	const GF61 v0 = add61(u0, u2), v1 = add61(u1, u3), v2 = sub61(u0, u2), v3 = sub61(u1, u3);
	const GF61 s0 = add61(mul61(v0, v0p), mul61(mul61(v1, v1p), w));
	const GF61 s1 = add61(mul61(v0, v1p), mul61(v0p, v1));
	const GF61 s2 = sub61(mul61(v2, v2p), mul61(mul61(v3, v3p), w));
	const GF61 s3 = add61(mul61(v2, v3p), mul61(v2p, v3));
	Z[0] = add61(s0, s2); Z[2] = mulconj61(sub61(s0, s2), w);
	Z[1] = add61(s1, s3); Z[3] = mulconj61(sub61(s1, s3), w);
}

// --- transform ---

__kernel
void forward4(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx >> lm, k = 3 * (j << lm) + idx;
	forward_4io((sz_t)(1) << lm, &z[k], w, s + j);
}

__kernel
void backward4(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const sz_t j = idx >> lm, k = 3 * (j << lm) + idx;
	backward_4io((sz_t)(1) << lm, &z[k], w, s + j);
}

__kernel
void square22(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0), n_4 = (sz_t)get_global_size(0);
	const sz_t j = idx, k = 4 * idx;
	square_22io(&z[k], w[n_4 + j]);
}

__kernel
void square4(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	const sz_t idx = (sz_t)get_global_id(0), n_4 = (sz_t)get_global_size(0);
	const sz_t j = idx, k = 4 * idx;
	square_4io(&z[k], w[n_4 + j]);
}

// -----------------

#define DECLARE_VAR(B_N, CHUNK_N) \
	__local GF61 Z[4 * B_N * CHUNK_N]; \
	\
	/* threadIdx < B_N */ \
	const sz_t i = (sz_t)get_local_id(0), chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = (sz_t)get_group_id(0) * CHUNK_N + chunk_idx; \
	__local GF61 * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	sz_t sj = s + idx_m;

#define DECLARE_VAR_FORWARD() \
	__global GF61 * __restrict__ const zi = &z[ki]; \
	__global GF61 * __restrict__ const zo = &z[ko];

#define DECLARE_VAR_BACKWARD() \
	__global GF61 * __restrict__ const zi = &z[ko]; \
	__global GF61 * __restrict__ const zo = &z[ki];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);

#define FORWARD_O(CHUNK_N) \
	forward_4o((sz_t)1 << lm, zo, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], w, sj / 1);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, w, sj / 1);

#define BACKWARD_O(B_N, CHUNK_N) \
	backward_4o(B_N << lm, zo, B_N * CHUNK_N, &Z[i], w, sj / B_N);

// -----------------

#define B_64	(64 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_64 * CHUNK64
	__attribute__((reqd_work_group_size(B_64 * CHUNK64, 1, 1)))
#endif
void forward64(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
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
void backward64(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_64, CHUNK64);

	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4);

	BACKWARD_O(B_64, CHUNK64);
}

// -----------------

#define B_256	(256 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_256 * CHUNK256
	__attribute__((reqd_work_group_size(B_256 * CHUNK256, 1, 1)))
#endif
void forward256(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
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
void backward256(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_256, CHUNK256);

	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16);

	BACKWARD_O(B_256, CHUNK256);
}

// -----------------

#define B_1024	(1024 / 4)

__kernel
#if MAX_WORK_GROUP_SIZE >= B_1024 * CHUNK1024
	__attribute__((reqd_work_group_size(B_1024 * CHUNK1024, 1, 1)))
#endif
void forward1024(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
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
void backward1024(__global GF61 * restrict const z, __global const GF61 * restrict const w, const int lm, const unsigned int s)
{
	BACKWARD_I(B_1024, CHUNK1024);

	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4);
	backward_4(4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4);
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16);
	backward_4(16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16);
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64);
	backward_4(64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64);

	BACKWARD_O(B_1024, CHUNK1024);
}

// -----------------

#define DECLARE_VAR_32() \
	__local GF61 Z[32 * BLK32]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k32 = (sz_t)get_group_id(0) * 32 * BLK32, i = (sz_t)get_local_id(0); \
	const sz_t i32 = (i & (sz_t)~(32 / 4 - 1)) * 4, i8 = i % (32 / 4); \
	\
	__global GF61 * restrict const zk = &z[k32 + i32 + i8]; \
	__local GF61 * const Z32 = &Z[i32]; \
	__local GF61 * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & (sz_t)~(4 * 2 - 1)) + (i8 % 2); \
	__local GF61 * const Zi2 = &Z32[i2]; \
	__local GF61 * const Z4 = &Z32[4 * i8];

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void square32(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4o(8, zk, 8, Zi8, w, j / 8);
}

#define DECLARE_VAR_64() \
	__local GF61 Z[64 * BLK64]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k64 = (sz_t)get_group_id(0) * 64 * BLK64, i = (sz_t)get_local_id(0); \
	const sz_t i64 = (i & (sz_t)~(64 / 4 - 1)) * 4, i16 = i % (64 / 4); \
	\
	__global GF61 * restrict const zk = &z[k64 + i64 + i16]; \
	__local GF61 * const Z64 = &Z[i64]; \
	__local GF61 * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & (sz_t)~(4 * 4 - 1)) + (i16 % 4); \
	__local GF61 * const Zi4 = &Z64[i4]; \
	__local GF61 * const Z4 = &Z64[4 * i16];

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void square64(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w[j]);
	backward_4(4, Zi4, w, j / 4);
	backward_4o(16, zk, 16, Zi16, w, j / 16);
}

#define DECLARE_VAR_128() \
	__local GF61 Z[128 * BLK128]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k128 = (sz_t)get_group_id(0) * 128 * BLK128, i = (sz_t)get_local_id(0); \
	const sz_t i128 = (i & (sz_t)~(128 / 4 - 1)) * 4, i32 = i % (128 / 4); \
	\
	__global GF61 * restrict const zk = &z[k128 + i128 + i32]; \
	__local GF61 * const Z128 = &Z[i128]; \
	__local GF61 * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & (sz_t)~(4 * 8 - 1)) + (i32 % 8); \
	__local GF61 * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & (sz_t)~(4 * 2 - 1)) + (i32 % 2); \
	__local GF61 * const Zi2 = &Z128[i2]; \
	__local GF61 * const Z4 = &Z128[4 * i32];

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void square128(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4(8, Zi8, w, j / 8);
	backward_4o(32, zk, 32, Zi32, w, j / 32);
}

#define DECLARE_VAR_256() \
	__local GF61 Z[256 * BLK256]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k256 = (sz_t)get_group_id(0) * 256 * BLK256, i = (sz_t)get_local_id(0); \
	const sz_t i256 = 0, i64 = i; \
	\
	__global GF61 * restrict const zk = &z[k256 + i256 + i64]; \
	__local GF61 * const Z256 = &Z[i256]; \
	__local GF61 * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & (sz_t)~(4 * 16 - 1)) + (i64 % 16); \
	__local GF61 * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & (sz_t)~(4 * 4 - 1)) + (i64 % 4); \
	__local GF61 * const Zi4 = &Z256[i4]; \
	__local GF61 * const Z4 = &Z256[4 * i64];

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void square256(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_256();

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w[j]);
	backward_4(4, Zi4, w, j / 4);
	backward_4(16, Zi16, w, j / 16);
	backward_4o(64, zk, 64, Zi64, w, j / 64);
}

#define DECLARE_VAR_512() \
	__local GF61 Z[512]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k512 = (sz_t)get_group_id(0) * 512, i128 = (sz_t)get_local_id(0); \
	\
	__global GF61 * restrict const zk = &z[k512 + i128]; \
	__local GF61 * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i128) & (sz_t)~(4 * 32 - 1)) + (i128 % 32); \
	__local GF61 * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i128) & (sz_t)~(4 * 8 - 1)) + (i128 % 8); \
	__local GF61 * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i128) & (sz_t)~(4 * 2 - 1)) + (i128 % 2); \
	__local GF61 * const Zi2 = &Z[i2]; \
	__local GF61 * const Z4 = &Z[4 * i128];

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void square512(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_512();

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4(8, Zi8, w, j / 8);
	backward_4(32, Zi32, w, j / 32);
	backward_4o(128, zk, 128, Zi128, w, j / 128);
}

#define DECLARE_VAR_1024() \
	__local GF61 Z[1024]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k1024 = (sz_t)get_group_id(0) * 1024, i256 = (sz_t)get_local_id(0); \
	\
	__global GF61 * restrict const zk = &z[k1024 + i256]; \
	__local GF61 * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i256) & (sz_t)~(4 * 64 - 1)) + (i256 % 64); \
	__local GF61 * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i256) & (sz_t)~(4 * 16 - 1)) + (i256 % 16); \
	__local GF61 * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i256) & (sz_t)~(4 * 4 - 1)) + (i256 % 4); \
	__local GF61 * const Zi4 = &Z[i4]; \
	__local GF61 * const Z4 = &Z[4 * i256];

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void square1024(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_1024();

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	square_4(Z4, w[j]);
	backward_4(4, Zi4, w, j / 4);
	backward_4(16, Zi16, w, j / 16);
	backward_4(64, Zi64, w, j / 64);
	backward_4o(256, zk, 256, Zi256, w, j / 256);
}

#define DECLARE_VAR_2048() \
	__local GF61 Z[2048]; \
	\
	const sz_t idx = (sz_t)get_global_id(0), j = NSIZE_4 + idx; \
	\
	const sz_t k2048 = (sz_t)get_group_id(0) * 2048, i512 = (sz_t)get_local_id(0); \
	\
	__global GF61 * restrict const zk = &z[k2048 + i512]; \
	__local GF61 * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & (sz_t)~(4 * 128 - 1)) + (i512 % 128); \
	__local GF61 * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & (sz_t)~(4 * 32 - 1)) + (i512 % 32); \
	__local GF61 * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & (sz_t)~(4 * 8 - 1)) + (i512 % 8); \
	__local GF61 * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & (sz_t)~(4 * 2 - 1)) + (i512 % 2); \
	__local GF61 * const Zi2 = &Z[i2]; \
	__local GF61 * const Z4 = &Z[4 * i512];

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void square2048(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_2048();

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	square_22(Z4, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4(8, Zi8, w, j / 8);
	backward_4(32, Zi32, w, j / 32);
	backward_4(128, Zi128, w, j / 128);
	backward_4o(512, zk, 512, Zi512, w, j / 512);
}

// -----------------

__kernel
#if MAX_WORK_GROUP_SIZE >= 32 / 4 * BLK32
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void fwd32p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_32();

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(8, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void fwd64p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_64();

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	fwd2write_4(16, zk, Z4, w[j]);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void fwd128p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
{
	DECLARE_VAR_128();

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	write_4(32, zk, Z4);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void fwd256p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
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
void fwd512p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
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
void fwd1024p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
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
void fwd2048p(__global GF61 * restrict const z, __global const GF61 * restrict const w)
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
	__attribute__((work_group_size_hint(32 / 4 * BLK32, 1, 1)))
#endif
void mul32(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_32();
	__global const GF61 * restrict const zpk = &zp[k32 + i32 + i8];

	forward_4i(8, Zi8, 8, zk, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 8, zpk, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4o(8, zk, 8, Zi8, w, j / 8);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 64 / 4 * BLK64
	__attribute__((work_group_size_hint(64 / 4 * BLK64, 1, 1)))
#endif
void mul64(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_64();
	__global const GF61 * restrict const zpk = &zp[k64 + i64 + i16];

	forward_4i(16, Zi16, 16, zk, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 16, zpk, w[j]);
	backward_4(4, Zi4, w, j / 4);
	backward_4o(16, zk, 16, Zi16, w, j / 16);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 128 / 4 * BLK128
	__attribute__((work_group_size_hint(128 / 4 * BLK128, 1, 1)))
#endif
void mul128(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_128();
	__global const GF61 * restrict const zpk = &zp[k128 + i128 + i32];

	forward_4i(32, Zi32, 32, zk, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 32, zpk, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4(8, Zi8, w, j / 8);
	backward_4o(32, zk, 32, Zi32, w, j / 32);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 256 / 4 * BLK256
	__attribute__((work_group_size_hint(256 / 4 * BLK256, 1, 1)))
#endif
void mul256(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_256();
	__global const GF61 * restrict const zpk = &zp[k256 + i256 + i64];

	forward_4i(64, Zi64, 64, zk, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 64, zpk, w[j]);
	backward_4(4, Zi4, w, j / 4);
	backward_4(16, Zi16, w, j / 16);
	backward_4o(64, zk, 64, Zi64, w, j / 64);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 512 / 4
	__attribute__((reqd_work_group_size(512 / 4, 1, 1)))
#endif
void mul512(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_512();
	__global const GF61 * restrict const zpk = &zp[k512 + i128];

	forward_4i(128, Zi128, 128, zk, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 128, zpk, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4(8, Zi8, w, j / 8);
	backward_4(32, Zi32, w, j / 32);
	backward_4o(128, zk, 128, Zi128, w, j / 128);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 1024 / 4
	__attribute__((reqd_work_group_size(1024 / 4, 1, 1)))
#endif
void mul1024(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_1024();
	__global const GF61 * restrict const zpk = &zp[k1024 + i256];

	forward_4i(256, Zi256, 256, zk, w, j / 256);
	forward_4(64, Zi64, w, j / 64);
	forward_4(16, Zi16, w, j / 16);
	forward_4(4, Zi4, w, j / 4);
	mul_4(Z4, 256, zpk, w[j]);
	backward_4(4, Zi4, w, j / 4);
	backward_4(16, Zi16, w, j / 16);
	backward_4(64, Zi64, w, j / 64);
	backward_4o(256, zk, 256, Zi256, w, j / 256);
}

__kernel
#if MAX_WORK_GROUP_SIZE >= 2048 / 4
	__attribute__((reqd_work_group_size(2048 / 4, 1, 1)))
#endif
void mul2048(__global GF61 * restrict const z, __global const GF61 * restrict const zp, __global const GF61 * restrict const w)
{
	DECLARE_VAR_2048();
	__global const GF61 * restrict const zpk = &zp[k2048 + i512];

	forward_4i(512, Zi512, 512, zk, w, j / 512);
	forward_4(128, Zi128, w, j / 128);
	forward_4(32, Zi32, w, j / 32);
	forward_4(8, Zi8, w, j / 8);
	forward_4(2, Zi2, w, j / 2);
	mul_22(Z4, 512, zpk, w[j]);
	backward_4(2, Zi2, w, j / 2);
	backward_4(8, Zi8, w, j / 8);
	backward_4(32, Zi32, w, j / 32);
	backward_4(128, Zi128, w, j / 128);
	backward_4o(512, zk, 512, Zi512, w, j / 512);
}

// -----------------

INLINE uint barrett(const ulong a, const uint b, const uint b_inv, const int b_s, uint * a_p)
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

	const uint d = mul_hi((uint)(a >> b_s), b_inv), r = (uint)a - d * b;
	const bool o = (r >= b);
	*a_p = o ? d + 1 : d;
	return o ? r - b : r;
}

INLINE int reduce64(long * f, const uint b, const uint b_inv, const int b_s)
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

__kernel
void normalize1(__global GF61 * restrict const z, __global long2 * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const int sblk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const unsigned int blk = abs(sblk);
	__global GF61 * restrict const zi = &z[blk * idx];

	prefetch(zi, (size_t)blk);

	long f0 = 0, f1 = 0;

	sz_t j = 0;
	do
	{
		const GF61 u = lshift61(zi[j], SNORM61);
		long l0 = get_int61(u.s0), l1 = get_int61(u.s1);
		if (sblk < 0) { l0 += l0; l1 += l1; }
		f0 += l0; f1 += l1;
		const int r0 = reduce64(&f0, b, b_inv, b_s), r1 = reduce64(&f1, b, b_inv, b_s);
		zi[j] = toGF61(r0, r1);
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	if (i == 0) { const long t = f0; f0 = -f1; f1 = t; }	// a_n = -a_0
	c[i] = (long2)(f0, f1);
}

__kernel
void mul1(__global GF61 * restrict const z, __global long2 * restrict const c,
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global GF61 * restrict const zi = &z[blk * idx];

	prefetch(zi, (size_t)blk);

	long f0 = 0, f1 = 0;

	sz_t j = 0;
	do
	{
		const GF61 u = zi[j];
		long l0 = get_int61(u.s0), l1 = get_int61(u.s1);
		l0 *= a; l1 *= a;
		f0 += l0; f1 += l1;
		const int r0 = reduce64(&f0, b, b_inv, b_s), r1 = reduce64(&f1, b, b_inv, b_s);
		zi[j] = toGF61(r0, r1);
		++j;
	} while (j != blk);

	const sz_t i = (idx + 1) & ((sz_t)get_global_size(0) - 1);
	if (i == 0) { const long t = f0; f0 = -f1; f1 = t; }	// a_n = -a_0
	c[i] = (long2)(f0, f1);
}

__kernel
void normalize2(__global GF61 * restrict const z, __global const long2 * restrict const c, 
	const unsigned int b, const unsigned int b_inv, const int b_s, const unsigned int blk)
{
	const sz_t idx = (sz_t)get_global_id(0);
	__global GF61 * restrict const zi = &z[blk * idx];

	long f0 = c[idx].s0, f1 = c[idx].s1;

	sz_t j = 0;
	do
	{
		const GF61 u = zi[j];
		const long l0 = get_int61(u.s0), l1 = get_int61(u.s1);
		f0 += l0; f1 += l1;
		const int r0 = reduce64(&f0, b, b_inv, b_s), r1 = reduce64(&f1, b, b_inv, b_s);
		zi[j] = toGF61(r0, r1);
		if ((f0 == 0) && (f1 == 0)) return;
		++j;
	} while (j != blk - 1);

	const GF61 r = toGF61((int)(f0), (int)(f1));
	zi[blk - 1] = add61(zi[blk - 1], r);
}

__kernel
void set(__global GF61 * restrict const z, const int a)
{
	const sz_t idx = (sz_t)get_global_id(0);
	const int ai = (idx == 0) ? a : 0;
	z[idx] = toGF61(ai, 0);
}

__kernel
void copy(__global GF61 * restrict const z, const unsigned int dst, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	z[dst + idx] = z[src + idx];
}

__kernel
void copyp(__global GF61 * restrict const zp, __global const GF61 * restrict const z, const unsigned int src)
{
	const sz_t idx = (sz_t)get_global_id(0);
	zp[idx] = z[src + idx];
}
