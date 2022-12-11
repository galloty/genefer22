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

#if defined(GPU)
		const char * const ext = "g";
#else
		const char * const ext = "";
#endif

		std::ostringstream ss;
		ss << "genefer" << ext << " version 22.12.0 (" << sysver << ssc.str() << ")" << std::endl;
		ss << "Copyright (c) 2022, Yves Gallot" << std::endl;
		ss << "genefer is free source code, under the MIT license." << std::endl;
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
#if defined(GPU)
		const char * const ext = "g";
#else
		const char * const ext = "";
#endif
		std::ostringstream ss;
		ss << "Usage: genefer" << ext << " [options]  options may be specified in any order" << std::endl;
		ss << "  -n <n>                      the exponent of the GFN (12 <= n <= 23)" << std::endl;
		ss << "  -b <b>                      the base of the GFN (2 <= b <= 2G)" << std::endl;
		ss << "  -q                          quick test" << std::endl;
		ss << "  -p                          full test: a proof is generated" << std::endl;
		ss << "  -s                          convert the proof into a certificate and a 64-bit key (server job)" << std::endl;
		ss << "  -c                          check the certificate: a 64-bit key is generated (must be identical to server key)" << std::endl;
		ss << "  -h                          validate and bench your hardware" << std::endl;
#if defined(GPU)
		ss << "  -d <n> or --device <n>      set the device number (default 0)" << std::endl;
#else
		ss << "  -t <n> or --nthreads <n>    set the number of threads (default: one thread, 0: all logical cores)" << std::endl;
		ss << "  -x <implementation>         set a specific implementation (i32, sse2, sse4, avx, fma, 512)" << std::endl;
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

		// if -v or -V then print header to stderr and exit
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
				if (b % 2 != 0) throw std::runtime_error("b must be even");
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
				if (mode != genefer::EMode::None) throw std::runtime_error("-q used with an incompatible option (-p, -s, -c, -h)");
				mode = genefer::EMode::Quick;
			}
			if (arg.substr(0, 2) == "-p")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-p used with an incompatible option (-q, -s, -c, -h)");
				mode = genefer::EMode::Proof;
			}
			if (arg.substr(0, 2) == "-s")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-s used with an incompatible option (-q, -p, -c, -h)");
				mode = genefer::EMode::Server;
			}
			if (arg.substr(0, 2) == "-c")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-c used with an incompatible option (-q, -p, -s, -h)");
				mode = genefer::EMode::Check;
			}
			if (arg.substr(0, 2) == "-h")
			{
				if (mode != genefer::EMode::None) throw std::runtime_error("-h used with an incompatible option (-q, -p, -s, -c)");
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
				nthreads = size_t(std::min(std::atoi(ntstr.c_str()), 64));
			}
			if (arg.substr(0, 10) == "--nthreads")
			{
				const std::string ntstr = ((arg == "--nthreads") && (i + 1 < size)) ? args[++i] : arg.substr(10);
				nthreads = size_t(std::min(std::atoi(ntstr.c_str()), 64));
			}
			if (arg.substr(0, 2) == "-x")
			{
				impl = ((arg == "-x") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				if ((impl != "i32") && (impl != "sse2") && (impl != "sse4") && (impl != "avx") && (impl != "fma") && (impl != "512"))
				{
					throw std::runtime_error("implementation is not valid");
				}
			}
		}

#if defined(BOINC) && defined(GPU)
		if (bBoinc && !boinc_is_standalone() && !ext_device)
		{
			const int err = boinc_get_opencl_ids(argc, argv, 0, &boinc_device_id, &boinc_platform_id);
			if ((err != 0) || (boinc_device_id == 0) || (boinc_platform_id == 0))
			{
				std::ostringstream ss; ss << std::endl << "error: boinc_get_opencl_ids() failed err = " << err;
				throw std::runtime_error(ss.str());
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
			for (size_t n = 15; n <= 22; ++n)
			{
				if (!g.check(0, n, mode, device, nthreads, impl, depth)) return;
			}
			if (mode == genefer::EMode::Bench)
			{
				if (!g.check(0, 23, mode, device, nthreads, impl, depth)) return;	// DYFL
				if (!g.check(0, 24, mode, device, nthreads, impl, depth)) return;	// 23
			}
			return;
		}

		if ((mode == genefer::EMode::None) || (b == 0) || (n == 0))
		{
			// internal test
			/*static const size_t count = 20 - 12 + 1;
#if defined(GPU)
			static constexpr uint32_t bp[count] = { 1999999266, 1999941378, 699995450, 302257864,
													167811262, 113521888, 15859176, 4896418, 1059094 };
#else
			static constexpr uint32_t bp[count] = { 500002286, 380018796, 290067480, 220129842,
													169277952, 114340846, 15913772, 4896418, 1951734 };
#endif

			// if (!g.check(1999992578, 10, genefer::EMode::Quick, device, nthreads, "i32", 5)) return;
			// if (!g.check(1999997802, 11, genefer::EMode::Proof, device, nthreads, "i32", 5)) return;
			// if (!g.check(1999997802, 11, genefer::EMode::Server, device, nthreads, "i32", 5)) return;
			// if (!g.check(1999997802, 11, genefer::EMode::Check, device, nthreads, "i32", 5)) return;
			// if (!g.check(1999999266, 12, genefer::EMode::Quick, device, nthreads, "i32", 6)) return;
			// return;
			for (size_t i = 0; i < count; ++i)
			{
				// if (!g.check(bp[i] + 0, 12 + i, genefer::EMode::Quick, device, nthreads, impl, depth)) return;

				if (!g.check(bp[i] + 0, 12 + i, genefer::EMode::Proof, device, nthreads, impl, depth)) return;
				if (!g.check(bp[i] + 0, 12 + i, genefer::EMode::Server, device, nthreads, impl, depth)) return;
				if (!g.check(bp[i] + 0, 12 + i, genefer::EMode::Check, device, nthreads, impl, depth)) return;
			}*/

			pio::print(usage());
#if defined(GPU)
			platform pfm;
			if (pfm.displayDevices() == 0) throw std::runtime_error("No OpenCL device");
#else
			g.displaySupportedImplementations();
#endif
			return;
		}

		g.check(b, n, mode, device, nthreads, impl, depth, oldfashion);

		if (bBoinc) boinc_finish(EXIT_SUCCESS);
	}
};

int main(int argc, char * argv[])
{
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
