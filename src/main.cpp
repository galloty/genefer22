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

#include "genefer.h"
#include "pio.h"

#if defined (_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

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

		std::ostringstream ss;
		ss << "genefer22 0.1.0 " << sysver << ssc.str() << std::endl;
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
		std::ostringstream ss;
		ss << "Usage: genefer22 [options]  options may be specified in any order" << std::endl;
		ss << "  -n <n>                        the exponent of the GFN (14 <= n <= 22)" << std::endl;
		ss << "  -b <b>                        the base of the GFN (2 <= b <= 2G)" << std::endl;
		ss << "  -q                            quick test (default)" << std::endl;
		ss << "  -p                            full test: a proof is generated" << std::endl;
		ss << "  -s                            convert the proof into a certificate and a 64-bit key (server job)" << std::endl;
		ss << "  -c                            check the certificate: a 64-bit key is generated (must be identical to server key)" << std::endl;
		ss << "  -t <n> or --nthreads <n>      set the number of threads (default: one thread per logical core)" << std::endl;
		ss << "  -x <implementation>           set a specific implementation (sse2, sse4, avx, fma, 512)" << std::endl;
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

		if (bBoinc)
		{
			const int retval = boinc_init();
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
				pio::error(header(args));
				if (bBoinc) boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		pio::print(header(args, true));

		if (args.empty())
		{
			pio::print(usage());
			// return;
		}

		uint32_t b = 0, n = 0;
		size_t nthreads = 1;	// 0;
		std::string impl;
		bool qTest = false;
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-q")
			{
				const std::string exp = ((arg == "-q") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				auto b_end = exp.find('^');
				if (b_end != std::string::npos) b = std::atoi(exp.substr(0, b_end).c_str());
				auto n_start = exp.find('^'), n_end = exp.find('+');
				if ((n_start != std::string::npos) && (n_end != std::string::npos)) n = std::atoi(exp.substr(n_start + 1, n_end).c_str());
				if (b % 2 != 0) throw std::runtime_error("b must be even");
				if (b > 2000000000) throw std::runtime_error("b > 2000000000 is not supported");
				if ((n == 0) || ((n & (~n + 1)) != n)) throw std::runtime_error("exponent must be a power of two");
				if (n > (1 << 22)) throw std::runtime_error("n > 22 is not supported");
				qTest = true;
			}
			if (arg.substr(0, 2) == "-t")
			{
				const std::string nt = ((arg == "-t") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				nthreads = std::min(std::atoi(nt.c_str()), 64);
			}
			if (arg.substr(0, 10) == "--nthreads")
			{
				const std::string nt = ((arg == "--nthreads") && (i + 1 < size)) ? args[++i] : arg.substr(10);
				nthreads =  std::min(std::atoi(nt.c_str()), 64);
			}
			if (arg.substr(0, 2) == "-x")
			{
				impl = ((arg == "-x") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				if ((impl != "sse2") && (impl != "sse4") && (impl != "avx") && (impl != "fma") && (impl != "512")) throw std::runtime_error("implementation is not valid");
			}
		}

		genefer & g = genefer::getInstance();
		g.setBoinc(bBoinc);

		if (qTest)
		{
			g.check(b, n, nthreads, impl, 5);
		}
		else
		{
			static const size_t count = 20 - 10 + 1;
			static constexpr uint32_t bp[count] = { 399998298, 399998572, 399987078, 399992284, 250063168,
													200295018, 167811262, 112719374, 15417192, 4896418, 1059094 };

			// 10: 5, 11: 5, 12: 5, 13: 6, 14: 6

			for (size_t i = 0; i < count; ++i)
			{
				if (!g.check(bp[i] + 2, 1 << (10 + i), nthreads, impl, 5)) break;
			}

			// size_t i = 4;
			// for (int d = 5; d <= 8; ++d)
			// {
			// 	if (!g.check(bp[i], 1 << (10 + i), nthreads, impl, d)) break;
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

