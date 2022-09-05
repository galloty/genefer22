/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformGPU.h"

transform * transform::create_ocl(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
{
	// NTT2/3 limits
	// for (uint32_t m = 10; m <= 22; ++m)
	// {
	// 	uint32_t b_l = uint32_t(sqrt(P1 * uint64_t(P2) / (2 << m)));
	// 	if (b_l % 2 == 1) --b_l;
	// 	const int t1 = (b_l * uint64_t(b_l) >= (P1 * uint64_t(P2) / (2 << m))) ? 3 : 2;
	// 	const int t2 = ((b_l + 2) * uint64_t(b_l + 2) >= (P1 * uint64_t(P2) / (2 << m))) ? 3 : 2;
	// 	std::cout << m << ": " << b_l << ", " << t1 << ", " << t2 << std::endl;
	// }

	transform * pTransform = nullptr;
	if (b * uint64_t(b) >= (P1_32 * uint64_t(P2_32) / 2) / (1 << n))
	{
		pTransform = new transformGPU<3>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else
	{
		pTransform = new transformGPU<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	return pTransform;
}
