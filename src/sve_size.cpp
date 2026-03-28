/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "transform.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

size_t transform::get_sve_size()
{
	uint64_t n = 0;
#ifdef __ARM_FEATURE_SVE
	n = svcntb() * 8;
#endif
	return n;
}
