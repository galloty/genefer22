/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformCPU.h"

transform * transform::create_fma(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t num_threads, const size_t num_regs)
{
	return create_transformCPU<4>(b, n, isBoinc, num_threads, num_regs);
}
