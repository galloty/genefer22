/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transformGPU.h"

// inline uint64_t bt12(const double pp_2N)
// {
// 	uint64_t b = static_cast<uint64_t>(sqrt(pp_2N));
// 	if (b % 2 == 1) --b;
// 	if ((b * double(b) >= pp_2N) || ((b + 2) * double(b + 2) < pp_2N)) std::cout << "ERROR! ";
// 	return b;
// }

transform * transform::create_ocl(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
{
	// NTT limits
	// for (int m = 11; m <= 23; ++m)
	// {
	// 	std::cout << m << ": ";
	// 	std::cout << bt12(P1S * double(P2S) / 2 / (1 << m));
	// 	std::cout << ", " << bt12(P1U * double(P2U) / 2 / (1 << m));
	// 	std::cout << ", " << std::min(bt12(P1S * double(P2S) * P3S / 2 / (1 << m)), uint64_t(P3S / 2));
	// 	std::cout << ", " << std::min(bt12(P1U * double(P2U) * P3U / 2 / (1 << m)), uint64_t(P3U / 2));
	// 	std::cout << "." << std::endl;
	// }

	transform * pTransform = nullptr;
	if (b * static_cast<uint64_t>(b) < (P1S * static_cast<uint64_t>(P2S) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUs<2, false>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else if (b * static_cast<uint64_t>(b) < (P1U * static_cast<uint64_t>(P2U) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUs<2, true>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else if (b <= 1000000000)
	{
		pTransform = new transformGPUs<3, false>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	else
	{
		pTransform = new transformGPUs<3, true>(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
	}
	return pTransform;
}
