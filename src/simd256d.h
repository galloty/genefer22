/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#if defined(__AVX__)

#include <immintrin.h>

typedef __m256d simd256d;

inline bool is_zero_256d(const simd256d v) { return (_mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_NEQ_OQ)) == 0); }

inline simd256d abs_256d(const simd256d v)
{
	const long long m = (long long)(uint64_t(1) << 63);
	const simd256d mask = _mm256_castsi256_pd((__m256i){m, m, m, m});	// _mm256_set1_pd(-0.0);
	return _mm256_andnot_pd(mask, v);
}

inline simd256d max_256d(const simd256d v0, const simd256d v1) { return _mm256_max_pd(v0, v1); }

inline simd256d round_256d(const simd256d v) { return _mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }

inline void transpose_256d(simd256d & v0, simd256d & v1, simd256d & v2, simd256d & v3)
{
	const simd256d r0 = _mm256_shuffle_pd(v0, v1, 0b0000), r1 = _mm256_shuffle_pd(v0, v1, 0b1111);
	const simd256d r2 = _mm256_shuffle_pd(v2, v3, 0b0000), r3 = _mm256_shuffle_pd(v2, v3, 0b1111);
	v0 = _mm256_permute2f128_pd(r0, r2, _MM_SHUFFLE(0, 2, 0, 0));
	v2 = _mm256_permute2f128_pd(r0, r2, _MM_SHUFFLE(0, 3, 0, 1));
	v1 = _mm256_permute2f128_pd(r1, r3, _MM_SHUFFLE(0, 2, 0, 0));
	v3 = _mm256_permute2f128_pd(r1, r3, _MM_SHUFFLE(0, 3, 0, 1));
}

#endif