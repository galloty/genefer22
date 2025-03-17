/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformGPU.h"
#include "transformGPUm.h"

transform * transform::create_ocl(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
{
	// NTT limits
	// for (uint32_t m = 5; m <= 23; ++m)
	// {
	// 	{
	// 		const uint64_t m61 = (uint64_t(1) << 61) - 1, m31 = (uint32_t(1) << 31) - 1, p1 = 127 * (uint32_t(1) << 24) + 1;
	// 		const double M = m31 * double(p1);
	// 		uint64_t b_l = static_cast<uint64_t>(sqrt(M / (2 << m)));
	// 		if (b_l % 2 == 1) --b_l;
	// 		const int t1 = (b_l * static_cast<uint64_t>(b_l) >= (M / (2 << m))) ? 3 : 2;
	// 		const int t2 = ((b_l + 2) * static_cast<uint64_t>(b_l + 2) >= (M / (2 << m))) ? 3 : 2;
	// 		std::cout << m << ": " << b_l << ", " << t1 << ", " << t2;
	// 	}
	// 	{
	// 		const uint64_t p12 = P1_32 * static_cast<uint64_t>(P2_32);
	// 		uint32_t b_l = static_cast<uint32_t>(sqrt(p12 / (2 << m)));
	// 		if (b_l % 2 == 1) --b_l;
	// 		const int t1 = (b_l * static_cast<uint64_t>(b_l) >= (p12 / (2 << m))) ? 3 : 2;
	// 		const int t2 = ((b_l + 2) * static_cast<uint64_t>(b_l + 2) >= (p12 / (2 << m))) ? 3 : 2;
	// 		std::cout << " / " << b_l << ", " << t1 << ", " << t2 << std::endl;
	// 	}
	// }

	// return new transformGPU<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	// return new transformGPU<3>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	// return new transformGPUg(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	return new transformGPUm<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);

	transform * pTransform = nullptr;

	/*if (b * static_cast<uint64_t>(b) < (M31 * static_cast<uint64_t>(P1M)  / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUm<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else*/ if (b * static_cast<uint64_t>(b) < (P1_32 * static_cast<uint64_t>(P2_32) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPU<2>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else
	{
		pTransform = new transformGPU<3>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	return pTransform;
}
