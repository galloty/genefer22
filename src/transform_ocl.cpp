/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformGPU.h"

transform * transform::create_ocl(const uint32_t b, const uint32_t n, const size_t device, const size_t num_regs)
{
	transform * const pTransform = new transformGPU(b, n, device, num_regs);
	return pTransform;
}
