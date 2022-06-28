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
		ss << "  -q \"b^{2^n}+1\"          test quick expression (Fermat probable prime)" << std::endl;
		ss << "  -v or -V                print the startup banner and immediately exit" << std::endl;
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

		// if (args.empty())
		// {
		// 	pio::print(usage());
		// 	return;
		// }

		uint32_t b = 0, n = 0;
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
		}

		genefer & g = genefer::getInstance();
		g.setBoinc(bBoinc);

		if (qTest)
		{
			g.check(b, n);
		}
		else
		{
			static const size_t count = 5;
			// static constexpr uint32_t bp[count] = { 399998298, 399998572, 399987078, 399992284, 300084246 };
			static constexpr uint32_t bc[count] = { 399998300, 399998574, 399987080, 399992286, 300000000 };
			static const char * const res[count] = { "5a82277cc9c6f782", "1907ebae0c183e35", "dced858499069664", "3c918e0f87815627", "978bc600c793bae1" };

			for (size_t i = 0; i < count; ++i)
			{
				if (!g.check(bc[i], 1 << (10 + i), res[i])) break;
			}
			// // g.check(399998298, 1 << 10, "");
			// g.check(399998300, 1 << 10, "5a82277cc9c6f782");
			// // g.check(399998572, 1 << 11, "");
			// g.check(399998574, 1 << 11, "1907ebae0c183e35");
			// // g.check(399987078, 1 << 12, "");
			// g.check(399987080, 1 << 12, "dced858499069664");
			// // g.check(399992284, 1 << 13, "");
			// g.check(399992286, 1 << 13, "3c918e0f87815627");
			// // g.check(300084246, 1 << 14, "");
			// g.check(300000000, 1 << 14, "978bc600c793bae1");

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

	// std::cerr << "genefer22: search for Generalized Fermat primes" << std::endl;
	// std::cerr << " Copyright (c) 2022, Yves Gallot" << std::endl;
	// std::cerr << " genefer22 is free source code, under the MIT license." << std::endl << std::endl;

	// try
	// {
	// 	genefer g;
	// 	// g.check(399998298, 1 << 10, "");
	// 	g.check(399998300, 1 << 10, "5a82277cc9c6f782");
	// 	// g.check(399998572, 1 << 11, "");
	// 	g.check(399998574, 1 << 11, "1907ebae0c183e35");
	// 	// g.check(399987078, 1 << 12, "");
	// 	g.check(399987080, 1 << 12, "dced858499069664");
	// 	// g.check(399992284, 1 << 13, "");
	// 	g.check(399992286, 1 << 13, "3c918e0f87815627");
	// 	// g.check(300084246, 1 << 14, "");
	// 	g.check(300000000, 1 << 14, "978bc600c793bae1");
	// }
	// catch (const std::runtime_error & e)
	// {
	// 	std::ostringstream ss; ss << std::endl << "error: " << e.what() << ".";
	// 	std::cerr << ss.str() << std::endl;
	// 	return EXIT_FAILURE;
	// }

	// return EXIT_SUCCESS;
}
