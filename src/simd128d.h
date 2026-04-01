/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#if defined(__aarch64__)

#if defined(__ARM_FEATURE_SVE)
#if __ARM_FEATURE_SVE_BITS == 128	// 128-bit SVE

#include <arm_sve.h>

typedef svfloat64_t simd128d __attribute__((arm_sve_vector_bits(128)));

inline simd128d addmul_128d(const simd128d v0, const simd128d v1, const simd128d v2)
{
#if defined(__clang__)
	return svmla_f64_x(svptrue_b64(), v0, v1, v2);
#else
	return v0 + v1 * v2;
#endif
}

inline simd128d submul_128d(const simd128d v0, const simd128d v1, const simd128d v2)
{
#if defined(__clang__)
	return svmls_f64_x(svptrue_b64(), v0, v1, v2);
#else
	return v0 - v1 * v2;
#endif
}

inline bool is_zero_128d(const simd128d v)
{
	return (svadda_f64(svcmpeq_f64(svptrue_b64(), v, svdup_f64(0.0)), -2.0, svdup_f64(1.0)) == 0.0);
}

inline simd128d abs_128d(const simd128d v) { return svabs_f64_x(svptrue_b64(), v); }

inline simd128d max_128d(const simd128d v0, const simd128d v1) { return svmax_f64_x(svptrue_b64(), v0, v1); }
inline double reduce_max_128d(const simd128d v) { return svmaxv_f64(svptrue_b64(), v); }

inline simd128d round_128d(const simd128d v) { return svrinta_f64_x(svptrue_b64(), v); }

inline void transpose_128d(simd128d & v0, simd128d & v1)
{
	const simd128d t = svzip2_f64(v0, v1);
	v0 = svzip1_f64(v0, v1); v1 = t;
}

#endif
#else	// NEON

#include <arm_neon.h>

typedef float64x2_t simd128d;

inline simd128d addmul_128d(const simd128d v0, const simd128d v1, const simd128d v2)
{
#if defined(__clang__)
	return vfmaq_f64(v0, v1, v2);
#else
	return v0 + v1 * v2;
#endif
}

inline simd128d submul_128d(const simd128d v0, const simd128d v1, const simd128d v2)
{
#if defined(__clang__)
	return vfmsq_f64(v0, v1, v2);
#else
	return v0 - v1 * v2;
#endif
}

inline bool is_zero_128d(const simd128d v) { const uint64x2_t mask = vceqzq_f64(v); return ((mask[0] & mask[1]) != 0); }

inline simd128d abs_128d(const simd128d v) { return vabsq_f64(v); }

inline simd128d max_128d(const simd128d v0, const simd128d v1) { return vmaxq_f64(v0, v1); }
inline double reduce_max_128d(const simd128d v) { return std::max(v[0], v[1]); }

inline simd128d round_128d(const simd128d v) { return vrndnq_f64(v); }

inline void transpose_128d(simd128d & v0, simd128d & v1)
{
	const simd128d t = vzip2q_f64(v0, v1);
	v0 = vzip1q_f64(v0, v1); v1 = t;
}

#endif

#else	// SSE2/SSE4.1

#include <immintrin.h>

typedef __m128d simd128d;

inline simd128d addmul_128d(const simd128d v0, const simd128d v1, const simd128d v2) { return v0 + v1 * v2; }
inline simd128d submul_128d(const simd128d v0, const simd128d v1, const simd128d v2) { return v0 - v1 * v2; }

inline bool is_zero_128d(const simd128d v) { return (_mm_movemask_pd(_mm_cmpneq_pd(v, _mm_setzero_pd())) == 0); }

inline simd128d abs_128d(const simd128d v)
{
	const long long m = (long long)(uint64_t(1) << 63);
	const simd128d mask = _mm_castsi128_pd((__m128i){m, m});	// _mm_set1_pd(-0.0);
	return _mm_andnot_pd(mask, v);
}

inline simd128d max_128d(const simd128d v0, const simd128d v1) { return _mm_max_pd(v0, v1); }
inline double reduce_max_128d(const simd128d v) { return std::max(v[0], v[1]); }

inline simd128d round_128d(const simd128d v)
{
#if defined(__SSE4_1__)
	return _mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else // SSE2
	const long long m = (long long)(uint64_t(1) << 63);
	const simd128d mask = _mm_castsi128_pd((__m128i){m, m});	// _mm_set1_pd(-0.0);
	const simd128d C52 = _mm_set1_pd(4503599627370496.0);		// 2^52
	const simd128d ar = _mm_andnot_pd(mask, v);
	const simd128d ir = _mm_or_pd(_mm_sub_pd(_mm_add_pd(ar, C52), C52), _mm_and_pd(mask, v));
	const simd128d mr = _mm_cmpge_pd(ar, C52);
	return _mm_or_pd(_mm_and_pd(mr, v), _mm_andnot_pd(mr, ir));
#endif
}

inline void transpose_128d(simd128d & v0, simd128d & v1)
{
	const simd128d t = _mm_unpackhi_pd(v0, v1);
	v0 = _mm_unpacklo_pd(v0, v1); v1 = t;
}

#endif