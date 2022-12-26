/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformCPUi32.h"

transform * transform::create_i32(const uint32_t b, const uint32_t n, const size_t num_regs)
{
	return new transformCPUi32(b, n, num_regs);
}
