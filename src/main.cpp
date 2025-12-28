/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

#if defined(BOINC)
#include "version.h"
#endif

#if defined(GPU)
#include "ocl.h"
#endif
#include "genefer.h"

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		genefer::getInstance().quit();
	}

private:
#if defined(_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

public:
	application()
	{
#if defined(_WIN32)	
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~application() {}

	static application & getInstance()
	{
		static std::unique_ptr<application, deleter> pInstance(new application());
		return *pInstance;
	}

private:
	static std::string header(const std::vector<std::string> & args, const bool nl = false)
	{
		const char * const sysver =
#if defined(_WIN64)
			"win64";
#elif defined(_WIN32)
			"win32";
#elif defined(__linux__)
#if defined(__x86_64)
			"linux x64";
#elif defined(__aarch64__)
			"linux arm64";
#else
			"linux x86";
#endif
#elif defined(__APPLE__)
#if defined(__aarch64__)
			"macOS arm64";
#else
			"macOS x64";
#endif
#else
			"unknown";
#endif

		std::ostringstream ssc;
#if defined(__clang__)
		ssc << ", clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
		ssc << ", gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif

#if defined(BOINC)
		ssc << ", boinc-" << BOINC_VERSION_STRING;
#endif

		const char * const name = 
#if defined(CYCLO)
			"cyclo";
#else
			"genefer";
#endif

		const char * const ext =
#if defined(GPU)
			"g";
#else
			"";
#endif

		std::ostringstream ss;
		ss << name << ext << " version 25.12.0 (" << sysver << ssc.str() << ")" << std::endl;
		ss << "Copyright (c) 2022, Yves Gallot" << std::endl;
		ss << name << " is free source code, under the MIT license." << std::endl;
		if (nl)
		{
			ss << std::endl << "Command line: '";
			bool first = true;
			for (const std::string & arg : args)
			{
				if (first) first = false; else ss << " ";
				ss << arg;
			}
			ss << "'" << std::endl << std::endl;
		}
		return ss.str();
	}

private:
	static std::string usage()
	{
		const char * const name = 
#if defined(CYCLO)
			"cyclo";
#else
			"genefer";
#endif

		const char * const ext =
#if defined(GPU)
			"g";
#else
			"";
#endif
		std::ostringstream ss;
		ss << "Usage: " << name << ext << " [options]  options may be specified in any order" << std::endl;
		ss << "  -n <n>                      exponent of the GFN (12 <= n <= 23)" << std::endl;
		ss << "  -b <b>                      base of the GFN (1024 <= b <= 2000000000)" << std::endl;
		ss << "  -q                          quick test" << std::endl;
		ss << "  -p                          full test: a proof is generated" << std::endl;
		ss << "  -s                          server job: convert the proof into a certificate and a 64-bit key" << std::endl;
		ss << "  -c                          check certificate: a 64-bit key is generated (must be identical to server key)" << std::endl;
		ss << "  -e                          perform a deterministic test (Brillhart, Lehmer, Selfridge: Theorem 1)" << std::endl;
		ss << "  -h                          validate and bench your hardware" << std::endl;
#if defined(GPU)
		ss << "  -d <n> or --device <n>      set the device number (default 0)" << std::endl;
#else
		ss << "  -t <n> or --nthreads <n>    set the number of threads (default: one thread, 0: all logical cores)" << std::endl;
#if !defined(__aarch64__)
		ss << "  -x <implementation>         set a specific implementation (sse2, sse4, avx, fma, 512)" << std::endl;
#endif
#endif
		ss << "  -f <filename>               main filename (without extension) of input and output files" << std::endl;
		ss << "  -v or -V                    print the startup banner and exit" << std::endl;
#if defined(BOINC)
		ss << "  -boinc                      operate as a BOINC client app" << std::endl;
#endif
		ss << std::endl;
		return ss.str();
	}

public:
	void run(int argc, char * argv[])
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

		bool bBoinc = false;
#if defined(BOINC)
		for (const std::string & arg : args) if (arg == "-boinc") bBoinc = true;
#endif
		pio::getInstance().setBoinc(bBoinc);

#if defined(GPU)
		cl_platform_id boinc_platform_id = 0;
		cl_device_id boinc_device_id = 0;
#endif
		if (bBoinc)
		{
			BOINC_OPTIONS boinc_options;
			boinc_options_defaults(boinc_options);
			boinc_options.direct_process_action = 0;
#if defined(GPU)
			boinc_options.normal_thread_priority = 1;
#endif
			const int retval = boinc_init_options(&boinc_options);
			if (retval != 0)
			{
				std::ostringstream ss; ss << "boinc_init returned " << retval;
				throw std::runtime_error(ss.str());
			}
		}

		// if -v or -V then print header and exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V')))
			{
				pio::print(header(args));
				if (bBoinc) boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		pio::print(header(args, true));

		uint32_t b = 0, n = 0;
		genefer::EMode mode = genefer::EMode::None;
		bool oldfashion = false;
		size_t device = 0, nthreads = 1;
#if defined(BOINC) && defined(GPU)
		bool ext_device = false;
#endif
		std::string mainFilename = "", impl = "";
		const int depth = 7;

		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if ((arg.substr(0, 2) == "-b") && (arg.substr(0, 3) != "-bo"))
			{
				const std::string bstr = ((arg == "-b") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				b = static_cast<uint32_t>(std::atoi(bstr.c_str()));
#if !defined(CYCLO)
				if (b % 2 != 0) throw std::runtime_error("b must be even");
#endif
				if (b < 1024) throw std::runtime_error("b < 1024 is not supported");
				if (b > 2000000000) throw std::runtime_error("b > 2000000000 is not supported");
				if ((b == 0) || ((b & (~b + 1)) == b)) throw std::runtime_error("b must not be a power of two");
			}
			if (arg.substr(0, 2) == "-n")
			{
				const std::string nstr = ((arg == "-n") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				n = static_cast<uint32_t>(std::atoi(nstr.c_str()));
				if (n < 12) throw std::runtime_error("n < 12 is not supported");
				if (n > 23) throw std::runtime_error("n > 23 is not supported");
			}
			if (arg.substr(0, 2) == "-q")
			{
				const std::string qstr = ((arg == "-q") && (i + 1 < size)) ? args[i + 1] : arg.substr(2);
				const auto flex = qstr.find('^');
				if (flex != std::string::npos)
				{
					const auto end = qstr.find('+');
					if (end != std::string::npos)
					{
						const uint32_t c = static_cast<uint32_t>(std::atoi(qstr.substr(0, flex).c_str()));
						const uint32_t m = static_cast<uint32_t>(std::atoi(qstr.substr(flex + 1, end - (flex + 1)).c_str()));
						for (uint32_t lm = 12; lm <= 22; ++lm)
						{
							if (m == (static_cast<uint32_t>(1) << lm))
							{
								b = c; n = lm; oldfashion = true;
								if ((arg == "-q") && (i + 1 < size)) ++i;
							}
						}
					}
				}
				if (mode != genefer::EMode::None) throw std::runtime_error("-q used with an incompatible option (-p, -s, -c, -e, -h)");
				mode = genefer::EMode::Quick;
			}
			if (arg.substr(0, 2) == "-p")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-p used with an incompatible option (-q, -s, -c, -e, -h)");
				mode = genefer::EMode::Proof;
			}
			if (arg.substr(0, 2) == "-s")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-s used with an incompatible option (-q, -p, -c, -e, -h)");
				mode = genefer::EMode::Server;
			}
			if (arg.substr(0, 2) == "-c")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-c used with an incompatible option (-q, -p, -s, -e, -h)");
				mode = genefer::EMode::Check;
			}
			if (arg.substr(0, 2) == "-e")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-e used with an incompatible option (-q, -p, -s, -c, -h)");
				mode = genefer::EMode::Prime;
			}
			if (arg.substr(0, 2) == "-h")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-h used with an incompatible option (-q, -p, -s, -e, -c)");
				mode = genefer::EMode::Bench;
			}
			if (arg.substr(0, 2) == "-f")
			{
				mainFilename = ((arg == "-f") && (i + 1 < size)) ? args[++i] : arg.substr(2);
			}
			if (arg.substr(0, 2) == "-d")
			{
				const std::string dstr = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = size_t(std::atoi(dstr.c_str()));
#if defined(BOINC) && defined(GPU)
				ext_device = true;
#endif
			}
			if (arg.substr(0, 8) == "--device")
			{
				const std::string dstr = ((arg == "--device") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = size_t(std::atoi(dstr.c_str()));
#if defined(BOINC) && defined(GPU)
				ext_device = true;
#endif
			}
			if (arg.substr(0, 2) == "-t")
			{
				const std::string ntstr = ((arg == "-t") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				const int nt = std::atoi(ntstr.c_str());
				if (nt > 64) pio::error("number of threads > 64");
				nthreads = size_t(std::min(nt, 64));
			}
			if (arg.substr(0, 10) == "--nthreads")
			{
				const std::string ntstr = ((arg == "--nthreads") && (i + 1 < size)) ? args[++i] : arg.substr(10);
				const int nt = std::atoi(ntstr.c_str());
				if (nt > 64) pio::error("number of threads > 64");
				nthreads = size_t(std::min(nt, 64));
			}
#if !defined(__aarch64__)
			if (arg.substr(0, 2) == "-x")
			{
				impl = ((arg == "-x") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				if ((impl != "i32") && (impl != "sse2") && (impl != "sse4") && (impl != "avx") && (impl != "fma") && (impl != "512"))
				{
					pio::error("implementation is not valid");
					impl = "";
				}
			}
#endif
		}

#if defined(BOINC) && defined(GPU)
		if (bBoinc && !boinc_is_standalone() && !ext_device)
		{
			const int err = boinc_get_opencl_ids(argc, argv, 0, &boinc_device_id, &boinc_platform_id);
			if ((err != 0) || (boinc_device_id == 0) || (boinc_platform_id == 0))
			{
				std::ostringstream ss; ss << "boinc_get_opencl_ids() failed, err = " << err;
				pio::error(ss.str());
				// Continue using default OpenCL device
				boinc_device_id = 0; boinc_platform_id = 0;
			}
		}
#endif

		genefer & g = genefer::getInstance();
		g.setBoinc(bBoinc);
#if defined(GPU)
		g.setBoincParam(boinc_platform_id, boinc_device_id);
#endif
		g.setFilename(mainFilename);

		if ((mode == genefer::EMode::Bench) || (mode == genefer::EMode::Limit))
		{
			for (size_t n = 16; n <= 23; ++n)
			{
				if (g.check(0, n, mode, device, nthreads, impl, depth) != genefer::EReturn::Success) return;
			}
			return;
		}

		if ((mode == genefer::EMode::None) || (b == 0) || (n == 0))
		{
			/*// internal test

			// if (g.check(1999992578, 10, genefer::EMode::Quick, device, nthreads, "i32", 5) != genefer::EReturn::Success) return;
			// if (g.check(1999997802, 11, genefer::EMode::Proof, device, nthreads, "i32", 5) != genefer::EReturn::Success) return;
			// if (g.check(1999997802, 11, genefer::EMode::Server, device, nthreads, "i32", 5) != genefer::EReturn::Success) return;
			// if (g.check(1999997802, 11, genefer::EMode::Check, device, nthreads, "i32", 5) != genefer::EReturn::Success) return;
			// if (g.check(1999999266, 12, genefer::EMode::Quick, device, nthreads, "i32", 6) != genefer::EReturn::Success) return;

			// if (g.check(33141254, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(64602916, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(999995486, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(1999997802, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(33160956, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 9E2D51DA6C7C3F54
			// if (g.check(64611980, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 6E2E19E3382BC8E8
			// if (g.check(1000000000, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 43C6CC5326E5C77F
			// if (g.check(2000000000, 11, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 1A16EAEE2487902D

			// if (g.check(23445612, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(45686464, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(999999618, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(1999999266, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(23448336, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 81CFF004ACE85CAA
			// if (g.check(45687570, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 6C3EA0FBFFEE2C34
			// if (g.check(1000000000, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 4E43A5B93273C649
			// if (g.check(2000000000, 12, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// E5F3EA34F6C68EBD

			// if (g.check(16558530, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(32303796, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(999955696, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(1999941378, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(16580478, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// A9976E31FA1E8211
			// if (g.check(32305990, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 541D9DF2A13A01BE
			// if (g.check(1000000000, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// AD5434699AA36C6F
			// if (g.check(2000000000, 13, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 39E2B6791634579D

			// if (g.check(11709684, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(22833026, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(999944006, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			// if (g.check(1999947588, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;
			if (g.check(11724168, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 063922D606F7ED14
			if (g.check(22843784, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 45DAA25E59EAC127
			if (g.check(1000000000, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 47A9DFAD0217519A
			if (g.check(2000000000, 14, genefer::EMode::Quick, device, nthreads, impl, 5) != genefer::EReturn::Success) return;	// 434063FE41B764E2

			return;

			// static const size_t count = 20 - 12 + 1;
			// static constexpr uint32_t bp[count] = { 1534, 30406, 67234, 70906, 48594, 62722, 24518, 75898, 919444 };	// gfn
			// static constexpr uint32_t bp[count] = { 1999999266, 1999941378, 1154623840, 326160660, 1010036096, 123910270, 16769618, 4896418, 1963736 };	// gfn
			// static constexpr uint32_t bp[count] = { 45687570, 32305990, 22843784, 16152994, 11421892, 8076496, 5710946, 4038248, 2855472 };	// gfn NTT-2 limits
			// static constexpr uint32_t bp[count] = { 484, 22, 5164, 7726, 13325, 96873, 192098, 712012, 123447 };	// cyclo
			// static constexpr uint32_t bp[count] = { 2005838, 1805064, 1401068, 1276943, 1090383, 984522, 192098, 712012, 123447 };	// cyclo
			// for (size_t i = 0; i < count; ++i)
			// {
				// g.check(bp[i] + 0, 12 + i, genefer::EMode::Quick, device, nthreads, impl, depth);
				// if (g.check(bp[i] + 0, 12 + i, genefer::EMode::Quick, device, nthreads, impl, depth) != genefer::EReturn::Success) return;

				// if (g.check(bp[i] + 0, 12 + i, genefer::EMode::Proof, device, nthreads, impl, depth) != genefer::EReturn::Success) return;
				// if (g.check(bp[i] + 0, 12 + i, genefer::EMode::Server, device, nthreads, impl, depth) != genefer::EReturn::Success) return;
				// if (g.check(bp[i] + 0, 12 + i, genefer::EMode::Check, device, nthreads, impl, depth) != genefer::EReturn::Success) return;

				// if (g.check(bp[i] + 0, 12 + i, genefer::EMode::Prime, device, nthreads, impl, depth) != genefer::EReturn::Success) return;
			// }*/

			pio::print(usage());
#if defined(GPU)
			platform pfm;
			if (pfm.displayDevices() == 0) throw std::runtime_error("No OpenCL device");
#else
			g.displaySupportedImplementations();
#endif
			return;
		}

		const genefer::EReturn ret = g.check(b, n, mode, device, nthreads, impl, depth, oldfashion);
		if (bBoinc)
		{
			if (ret == genefer::EReturn::Success) boinc_finish(BOINC_SUCCESS);
			if (ret == genefer::EReturn::Failed) boinc_finish(EXIT_CHILD_FAILED);
		}

		if (ret == genefer::EReturn::Aborted)
		{
			std::ostringstream ss; ss << std::endl;
			pio::print(ss.str());
		}
	}
};

int main(int argc, char * argv[])
{
	std::setvbuf(stderr, nullptr, _IONBF, 0);	// no buffer

	try
	{
		application & app = application::getInstance();
		app.run(argc, argv);
	}
	catch (const std::runtime_error & e)
	{
		pio::error(e.what(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
