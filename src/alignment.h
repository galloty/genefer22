/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

inline void * align_new(const size_t size, const size_t alignment, const size_t offset = 0)
{
	char * const alloc_ptr = new char[size + alignment + offset + sizeof(size_t)];
	const size_t addr = size_t(alloc_ptr) + alignment + sizeof(size_t);
	size_t * const ptr = (size_t *)(addr - addr % alignment + offset);
	ptr[-1] = size_t(alloc_ptr);
	return (void *)(ptr);
}

inline void align_delete(void * const ptr)
{
	char * const alloc_ptr = (char *)((size_t *)(ptr))[-1];
	delete[] alloc_ptr;
}
