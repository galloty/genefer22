/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#if defined(__aarch64__)

#if defined(__ARM_FEATURE_SVE) && (__ARM_FEATURE_SVE_BITS == 512)	// 512-bit SVE

#include <arm_sve.h>

typedef svfloat64_t simd512d __attribute__((arm_sve_vector_bits(512)));
typedef svuint64_t simd512u __attribute__((arm_sve_vector_bits(512)));

inline bool is_zero_512d(const simd512d v)
{
	return (svadda_f64(svcmpeq_f64(svptrue_b64(), v, svdup_f64(0.0)), -8.0, svdup_f64(1.0)) == 0.0);
}

inline simd512d abs_512d(const simd512d v) { return svabs_f64_x(svptrue_b64(), v); }

inline simd512d max_512d(const simd512d v0, const simd512d v1) { return svmax_f64_x(svptrue_b64(), v0, v1); }
inline double reduce_max_512d(const simd512d v) { return svmaxv_f64(svptrue_b64(), v); }

inline simd512d round_512d(const simd512d v) { return svrinta_f64_x(svptrue_b64(), v); }

inline void transpose_512d(simd512d & v0, simd512d & v1, simd512d & v2, simd512d & v3, simd512d & v4, simd512d & v5, simd512d & v6, simd512d & v7)
{
	const simd512d r0 = svzip1_f64(v0, v4), r4 = svzip2_f64(v0, v4);
	const simd512d r1 = svzip1_f64(v1, v5), r5 = svzip2_f64(v1, v5);
	const simd512d r2 = svzip1_f64(v2, v6), r6 = svzip2_f64(v2, v6);
	const simd512d r3 = svzip1_f64(v3, v7), r7 = svzip2_f64(v3, v7);
	const simd512d t0 = svzip1_f64(r0, r2), t2 = svzip2_f64(r0, r2);
	const simd512d t1 = svzip1_f64(r1, r3), t3 = svzip2_f64(r1, r3);
	const simd512d t4 = svzip1_f64(r4, r6), t6 = svzip2_f64(r4, r6);
	const simd512d t5 = svzip1_f64(r5, r7), t7 = svzip2_f64(r5, r7);
	v0 = svzip1_f64(t0, t1); v1 = svzip2_f64(t0, t1);
	v2 = svzip1_f64(t2, t3); v3 = svzip2_f64(t2, t3);
	v4 = svzip1_f64(t4, t5); v5 = svzip2_f64(t4, t5);
	v6 = svzip1_f64(t6, t7); v7 = svzip2_f64(t6, t7);
}

inline void interleave_512d(simd512d & v0, simd512d & v1)
{
	const simd512d v0_lo = svtbl_f64(v0, (simd512u){0, 1, 2, 3, 16, 16, 16, 16});
	const simd512d v0_hi = svtbl_f64(v0, (simd512u){4, 5, 6, 7, 16, 16, 16, 16});
	const simd512d v1_lo = svtbl_f64(v1, (simd512u){16, 16, 16, 16, 0, 1, 2, 3});
	const simd512d v1_hi = svtbl_f64(v1, (simd512u){16, 16, 16, 16, 4, 5, 6, 7});
	v0 = v0_lo + v1_lo; v1 = v0_hi + v1_hi;
}

#endif

#elif defined(__AVX512F__)	// AVX-512

#include <immintrin.h>

typedef __m512d simd512d;

inline bool is_zero_512d(const simd512d v) { return (_mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_NEQ_OQ) == 0); }

inline simd512d abs_512d(const simd512d v) { return _mm512_abs_pd(v); }

inline simd512d max_512d(const simd512d v0, const simd512d v1) { return _mm512_max_pd(v0, v1); }
inline double reduce_max_512d(const simd512d v) { return _mm512_reduce_max_pd(v); }

inline simd512d round_512d(const simd512d v) { return _mm512_roundscale_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }

inline void transpose_512d(simd512d & v0, simd512d & v1, simd512d & v2, simd512d & v3, simd512d & v4, simd512d & v5, simd512d & v6, simd512d & v7)
{
	const simd512d r0 = _mm512_unpacklo_pd(v0, v1), r1 = _mm512_unpackhi_pd(v0, v1);
	const simd512d r2 = _mm512_unpacklo_pd(v2, v3), r3 = _mm512_unpackhi_pd(v2, v3);
	const simd512d r4 = _mm512_unpacklo_pd(v4, v5), r5 = _mm512_unpackhi_pd(v4, v5);
	const simd512d r6 = _mm512_unpacklo_pd(v6, v7), r7 = _mm512_unpackhi_pd(v6, v7);
	const simd512d t0 = _mm512_shuffle_f64x2(r0, r2, _MM_SHUFFLE(2, 0, 2, 0)), t2 = _mm512_shuffle_f64x2(r0, r2, _MM_SHUFFLE(3, 1, 3, 1));
	const simd512d t1 = _mm512_shuffle_f64x2(r1, r3, _MM_SHUFFLE(2, 0, 2, 0)), t3 = _mm512_shuffle_f64x2(r1, r3, _MM_SHUFFLE(3, 1, 3, 1));
	const simd512d t4 = _mm512_shuffle_f64x2(r4, r6, _MM_SHUFFLE(2, 0, 2, 0)), t6 = _mm512_shuffle_f64x2(r4, r6, _MM_SHUFFLE(3, 1, 3, 1));
	const simd512d t5 = _mm512_shuffle_f64x2(r5, r7, _MM_SHUFFLE(2, 0, 2, 0)), t7 = _mm512_shuffle_f64x2(r5, r7, _MM_SHUFFLE(3, 1, 3, 1));
	v0 = _mm512_shuffle_f64x2(t0, t4, _MM_SHUFFLE(2, 0, 2, 0)); v4 = _mm512_shuffle_f64x2(t0, t4, _MM_SHUFFLE(3, 1, 3, 1));
	v1 = _mm512_shuffle_f64x2(t1, t5, _MM_SHUFFLE(2, 0, 2, 0)); v5 = _mm512_shuffle_f64x2(t1, t5, _MM_SHUFFLE(3, 1, 3, 1));
	v2 = _mm512_shuffle_f64x2(t2, t6, _MM_SHUFFLE(2, 0, 2, 0)); v6 = _mm512_shuffle_f64x2(t2, t6, _MM_SHUFFLE(3, 1, 3, 1));
	v3 = _mm512_shuffle_f64x2(t3, t7, _MM_SHUFFLE(2, 0, 2, 0)); v7 = _mm512_shuffle_f64x2(t3, t7, _MM_SHUFFLE(3, 1, 3, 1));
}

inline void interleave_512d(simd512d & v0, simd512d & v1)
{
	const simd512d t = _mm512_shuffle_f64x2(v0, v1, _MM_SHUFFLE(2, 3, 2, 3));
	v0 = _mm512_shuffle_f64x2(v0, v1, _MM_SHUFFLE(0, 1, 0, 1)); v1 = t;
}

#endif