/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined (_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

#if defined (GPU)
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
#if defined (_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

public:
	application()
	{
#if defined (_WIN32)	
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
#ifdef __x86_64
			"linux64";
#else
			"linux32";
#endif
#elif defined(__APPLE__)
			"macOS";
#else
			"unknown";
#endif

		std::ostringstream ssc;
#if defined(__GNUC__)
		ssc << " gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(__clang__)
		ssc << " clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#endif

#ifdef GPU
		const char * const ext = "g";
#else
		const char * const ext = "";
#endif

		std::ostringstream ss;
		ss << "genefer22" << ext << " 0.1.0 " << sysver << ssc.str() << std::endl;
		ss << "Copyright (c) 2022, Yves Gallot" << std::endl;
		ss << "genefer22 is free source code, under the MIT license." << std::endl;
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
#ifdef GPU
		const char * const ext = "g";
#else
		const char * const ext = "";
#endif
		std::ostringstream ss;
		ss << "Usage: genefer22" << ext << " [options]  options may be specified in any order" << std::endl;
		ss << "  -n <n>                        the exponent of the GFN (14 <= n <= 22)" << std::endl;
		ss << "  -b <b>                        the base of the GFN (2 <= b <= 2G)" << std::endl;
		ss << "  -q                            quick test (default)" << std::endl;
		ss << "  -p                            full test: a proof is generated" << std::endl;
		ss << "  -s                            convert the proof into a certificate and a 64-bit key (server job)" << std::endl;
		ss << "  -c                            check the certificate: a 64-bit key is generated (must be identical to server key)" << std::endl;
#ifdef GPU
		ss << "  -d <n> or --device <n>        set the device number (default 0)" << std::endl;
#else
		ss << "  -t <n> or --nthreads <n>      set the number of threads (default: one thread per logical core)" << std::endl;
		ss << "  -x <implementation>           set a specific implementation (sse2, sse4, avx, fma, 512)" << std::endl;
#endif
		ss << "  -v or -V                      print the startup banner and exit" << std::endl;
#ifdef BOINC
		ss << "  -boinc                  operate as a BOINC client app" << std::endl;
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
#ifdef BOINC
		for (const std::string & arg : args) if (arg == "-boinc") bBoinc = true;
#endif
		pio::getInstance().setBoinc(bBoinc);

#if defined(GPU)
		cl_platform_id boinc_platform_id = 0;
		cl_device_id boinc_device_id = 0;
#endif
		if (bBoinc)
		{
			const int retval = boinc_init();
			if (retval != 0)
			{
				std::ostringstream ss; ss << "boinc_init returned " << retval;
				throw std::runtime_error(ss.str());
			}
#if defined(BOINC) && defined(GPU)
			if (!boinc_is_standalone())
			{
				const int err = boinc_get_opencl_ids(argc, argv, 0, &boinc_device_id, &boinc_platform_id);
				if ((err != 0) || (boinc_device_id == 0) || (boinc_platform_id == 0))
				{
					std::ostringstream ss; ss << std::endl << "error: boinc_get_opencl_ids() failed err = " << err;
					throw std::runtime_error(ss.str());
				}
			}
#endif
		}

		// if -v or -V then print header to stderr and exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V')))
			{
				pio::error(header(args));
				if (bBoinc) boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		pio::print(header(args, true));

		if (args.empty())
		{
			pio::print(usage());
#if defined (GPU)
			platform pfm;
			if (pfm.displayDevices() == 0) throw std::runtime_error("No OpenCL device");
#endif
			// return;
		}

		uint32_t b = 0, n = 0;
		size_t device = 0, nthreads = 1;	// 0;
		std::string impl;
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-b")
			{
				const std::string bstr = ((arg == "-b") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				b = std::atoi(bstr.c_str());
				if (b % 2 != 0) throw std::runtime_error("b must be even");
				if (b > 2000000000) throw std::runtime_error("b > 2000000000 is not supported");
				if ((b == 0) || ((b & (~b + 1)) != b)) throw std::runtime_error("b must not be a power of two");
			}
			if (arg.substr(0, 2) == "-n")
			{
				const std::string nstr = ((arg == "-n") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				n = std::atoi(nstr.c_str());
				if (n < 15) throw std::runtime_error("n < 15 is not supported");
				if (n > 22) throw std::runtime_error("n > 22 is not supported");
			}
			if (arg.substr(0, 2) == "-d")
			{
				const std::string dstr = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = std::atoi(dstr.c_str());
			}
			if (arg.substr(0, 2) == "--device")
			{
				const std::string dstr = ((arg == "--device") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				device = std::atoi(dstr.c_str());
			}
			if (arg.substr(0, 2) == "-t")
			{
				const std::string ntstr = ((arg == "-t") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				nthreads = std::min(std::atoi(ntstr.c_str()), 64);
			}
			if (arg.substr(0, 10) == "--nthreads")
			{
				const std::string ntstr = ((arg == "--nthreads") && (i + 1 < size)) ? args[++i] : arg.substr(10);
				nthreads =  std::min(std::atoi(ntstr.c_str()), 64);
			}
			if (arg.substr(0, 2) == "-x")
			{
				impl = ((arg == "-x") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				if ((impl != "sse2") && (impl != "sse4") && (impl != "avx") && (impl != "fma") && (impl != "512")) throw std::runtime_error("implementation is not valid");
			}
		}

		genefer & g = genefer::getInstance();
		g.setBoinc(bBoinc);
#if defined(GPU)
		g.setBoincParam(boinc_platform_id, boinc_device_id);
#endif

		if ((b != 0) && (n != 0))
		{
			g.check(b, n, device, nthreads, impl, 5);
		}
		else
		{
			static const size_t count = 20 - 10 + 1;
			static constexpr uint32_t bp[count] = { 399998298, 399998572, 399987078, 399992284, 250063168,
													200295018, 167811262, 112719374, 15417192, 4896418, 1059094 };

			// 10: 5, 11: 5, 12: 5, 13: 6, 14: 6

			for (size_t i = 0; i < count; ++i)
			{
				if (!g.check(bp[i] + 0, 10 + i, device, nthreads, impl, 5)) break;
			}

			// size_t i = 4;
			// for (int d = 5; d <= 8; ++d)
			// {
			// 	if (!g.check(bp[i], 10 + i, device, nthreads, impl, d)) break;
			// }
		}

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
		std::ostringstream ss; ss << std::endl << "error: " << e.what() << "." << std::endl;
		pio::error(ss.str(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

