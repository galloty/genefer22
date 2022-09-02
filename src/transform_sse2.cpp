/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#define transformCPU_namespace	transformCPU_sse2
#include "transformCPUf64.h"

transform * transform::create_sse2(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs)
{
	return transformCPU_sse2::create_transformCPUf64<2>(b, n, num_threads, num_regs);
}
