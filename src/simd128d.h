/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#if defined(__aarch64__)

#include <arm_neon.h>

typedef float64x2_t simd128d;

inline simd128d set_pd(const double h, const double l)
{
	const double __attribute__((aligned(16))) data[2] = { l, h };
	return vld1q_f64((const float64_t *)data);
}

inline simd128d set1_pd(const double f) { return vdupq_n_f64(f); }

inline bool is_zero_pd(const simd128d v)
{
	const uint64x2_t mask = vceqzq_f64(v);
	return ((vgetq_lane_u64(mask, 0) & vgetq_lane_u64(mask, 1)) != 0);
}

inline simd128d abs_pd(const simd128d v) { return vabsq_f64(v); }

inline simd128d max_pd(const simd128d v0, const simd128d v1) { return vmaxq_f64(v0, v1); }

inline simd128d round_pd(const simd128d v) { return vrndnq_f64(v); }

inline simd128d unpacklo_pd(const simd128d v0, const simd128d v1) { return vzip1q_f64(v0, v1); }
inline simd128d unpackhi_pd(const simd128d v0, const simd128d v1) { return vzip2q_f64(v0, v1); }

#else	// SSE2/SSE4.1

#include <immintrin.h>

typedef __m128d simd128d;

inline simd128d set_pd(const double h, const double l) { return _mm_set_pd(h, l); }

inline simd128d set1_pd(const double f) { return _mm_set1_pd(f); }

inline bool is_zero_pd(const simd128d v) { return (_mm_movemask_pd(_mm_cmpneq_pd(v, _mm_setzero_pd())) == 0); }

inline simd128d abs_pd(const simd128d v) { return _mm_andnot_pd(_mm_set1_pd(-0.0), v); }

inline simd128d max_pd(const simd128d v0, const simd128d v1) { return _mm_max_pd(v0, v1); }

inline simd128d round_pd(const simd128d v)
{
#if defined(__SSE4_1__)
	return _mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else // SSE2
	const __m128d signMask = _mm_set1_pd(-0.0), C52 = _mm_set1_pd(4503599627370496.0);  // 2^52
	const __m128d ar = _mm_andnot_pd(signMask, v);
	const __m128d ir = _mm_or_pd(_mm_sub_pd(_mm_add_pd(ar, C52), C52), _mm_and_pd(signMask, v));
	const __m128d mr = _mm_cmpge_pd(ar, C52);
	return _mm_or_pd(_mm_and_pd(mr, v), _mm_andnot_pd(mr, ir));
#endif
}

inline simd128d unpacklo_pd(const simd128d v0, const simd128d v1) { return _mm_unpacklo_pd(v0, v1); }
inline simd128d unpackhi_pd(const simd128d v0, const simd128d v1) { return _mm_unpackhi_pd(v0, v1); }

#endif