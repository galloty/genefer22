/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define transformCPU_namespace	transformCPU_sse2
#include "transformCPUf64.h"
#include "transformCPUf64s.h"

transform * transform::create_sse2(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError)
{
#if defined(DTRANSFORM) || defined(IBDTRANSFORM)
	return transformCPU_sse2::create_transformCPUf64<2>(b, n, num_threads, num_regs, checkError);
#elif defined(SBDTRANSFORM)
	return transformCPU_sse2::create_transformCPUf64s<2>(b, n, num_threads, num_regs, checkError);
#else
	if (n > 17) return transformCPU_sse2::create_transformCPUf64<2>(b, n, num_threads, num_regs, checkError);
	else return transformCPU_sse2::create_transformCPUf64s<2>(b, n, num_threads, num_regs, checkError);
#endif
}
