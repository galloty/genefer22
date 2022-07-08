/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformCPU.h"

Transform * Transform::create_512(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs)
{
	Transform * transform = nullptr;
	if (n == (1 << 10))      transform = new TransformCPU<(1 << 10), 8>(b, num_threads, num_regs);
	else if (n == (1 << 11)) transform = new TransformCPU<(1 << 11), 8>(b, num_threads, num_regs);
	else if (n == (1 << 12)) transform = new TransformCPU<(1 << 12), 8>(b, num_threads, num_regs);
	else if (n == (1 << 13)) transform = new TransformCPU<(1 << 13), 8>(b, num_threads, num_regs);
	else if (n == (1 << 14)) transform = new TransformCPU<(1 << 14), 8>(b, num_threads, num_regs);
	else if (n == (1 << 15)) transform = new TransformCPU<(1 << 15), 8>(b, num_threads, num_regs);
	else if (n == (1 << 16)) transform = new TransformCPU<(1 << 16), 8>(b, num_threads, num_regs);
	else if (n == (1 << 17)) transform = new TransformCPU<(1 << 17), 8>(b, num_threads, num_regs);
	else if (n == (1 << 18)) transform = new TransformCPU<(1 << 18), 8>(b, num_threads, num_regs);
	else if (n == (1 << 19)) transform = new TransformCPU<(1 << 19), 8>(b, num_threads, num_regs);
	else if (n == (1 << 20)) transform = new TransformCPU<(1 << 20), 8>(b, num_threads, num_regs);
	else if (n == (1 << 21)) transform = new TransformCPU<(1 << 21), 8>(b, num_threads, num_regs);
	else if (n == (1 << 22)) transform = new TransformCPU<(1 << 22), 8>(b, num_threads, num_regs);
	if (transform == nullptr) throw std::runtime_error("exponent is not supported");

	return transform;
}
