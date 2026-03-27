/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#include "transform.h"

size_t transform::get_sve_size()
{
#ifdef __ARM_FEATURE_SVE
	return svcntb() * 8;
#else
	return 0;
#endif
}
