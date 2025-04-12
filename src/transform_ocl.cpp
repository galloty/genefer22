/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformGPU.h"
#include "transformGPUs.h"

transform * transform::create_ocl(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
{
	// NTT limits
	// for (uint32_t m = 12; m <= 23; ++m)
	// {
	// 	{
	// 		const double p12s = P1S * double(P2S);
	// 		uint64_t b_l = static_cast<uint64_t>(sqrt(p12s / (2 << m)));
	// 		if (b_l % 2 == 1) --b_l;
	// 		const int t1 = (b_l * double(b_l) >= (p12s / (2 << m))) ? 3 : 2;
	// 		const int t2 = ((b_l + 2) * double(b_l + 2) >= (p12s / (2 << m))) ? 3 : 2;
	// 		std::cout << m << ": " << b_l << ", " << t1 << ", " << t2;
	// 	}
	// 	{
	// 		const double p12 = P1_32 * double(P2_32);
	// 		uint64_t b_l = static_cast<uint32_t>(sqrt(p12 / (2 << m)));
	// 		if (b_l % 2 == 1) --b_l;
	// 		const int t1 = (b_l * double(b_l) >= (p12 / (2 << m))) ? 3 : 2;
	// 		const int t2 = ((b_l + 2) * double(b_l + 2) >= (p12 / (2 << m))) ? 3 : 2;
	// 		std::cout << " / " << b_l << ", " << t1 << ", " << t2;
	// 	}
	// 	{
	// 		const double p123 = P1S * double(P2S) * P3S;
	// 		uint64_t b_l = static_cast<uint64_t>(sqrt(p123 / (2 << m)));
	// 		if (b_l % 2 == 1) --b_l;
	// 		const int t1 = (b_l * double(b_l) >= (p123 / (2 << m))) ? 3 : 2;
	// 		const int t2 = ((b_l + 2) * double(b_l + 2) >= (p123 / (2 << m))) ? 3 : 2;
	// 		std::cout << " / " << b_l << ", " << t1 << ", " << t2 << std::endl;
	// 	}
	// }

	// transform * pTransform = nullptr;
	// if (b * static_cast<uint64_t>(b) < (P1S * static_cast<uint64_t>(P2S) / 2) / (size_t(1) << n))
	// {
	// 	pTransform = new transformGPUs<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	// }
	// else
	// {
	// 	pTransform = new transformGPUs<3>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	// }
	// return pTransform;

	transform * pTransform = nullptr;
	if (b * static_cast<uint64_t>(b) < (P1_32 * static_cast<uint64_t>(P2_32) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPU<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else
	{
		pTransform = new transformGPU<3>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	return pTransform;
}
