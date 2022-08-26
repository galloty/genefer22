/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <thread>
#include <chrono>

#include "boinc.h"

class pio
{
private:
	struct deleter { void operator()(const pio * const p) { delete p; } };

public:
	pio() {}
	virtual ~pio() {}

	static pio & getInstance()
	{
		static std::unique_ptr<pio, deleter> pInstance(new pio());
		return *pInstance;
	}

public:
	void setBoinc(const bool isBoinc) { _isBoinc = isBoinc; }

private:
	bool _isBoinc = false;

private:
	// print: console: cout, boinc: stderr
	void _print(const std::string & str) const
	{
		if (_isBoinc) { std::fprintf(stderr, "%s", str.c_str()); std::fflush(stderr); }
		else { std::cout << str; }
	}

private:
	// display: console: cout, boinc: -
	void _display(const std::string & str) const
	{
		if (!_isBoinc) { std::cout << str << std::flush; }
	}

private:
	// error: normal: cerr, boinc: stderr
	void _error(const std::string & str, const bool fatal) const
	{
		std::ostringstream ss; ss << std::endl << "Error: " << str << "." << std::endl;
		if (_isBoinc)
		{
			std::fprintf(stderr, "%s", ss.str().c_str()); std::fflush(stderr);
			if (fatal)
			{
				// delay five minutes before reporting to the host in order to slow down the error rate.
				std::this_thread::sleep_for(std::chrono::minutes(5));
				boinc_finish(EXIT_FAILURE);
			}
		}
		else
		{
			std::cerr << ss.str().c_str();
			if (fatal) exit(EXIT_FAILURE);
		}
	}

private:
	// result: normal: 'results.txt' file, boinc: 'out' file
	bool _result(const std::string & str) const
	{
		if (_isBoinc)
		{
			FILE * const out_file = _open("out", "a");
			if (out_file == nullptr) throw std::runtime_error("Cannot write results to out file");
			std::fprintf(out_file, "%s", str.c_str());
			std::fclose(out_file);
			return true;
		}
		std::ofstream resFile("results.txt", std::ios::app);
		if (!resFile.is_open()) return false;
		resFile << str;
		resFile.close();
		return true;
	}

private:
	FILE * _open(const char * const filename, const char * const mode) const
	{
		if (_isBoinc)
		{
			char path[512];
			boinc_resolve_filename(filename, path, sizeof(path));
			return boinc_fopen(path, mode);
		}
		return std::fopen(filename, mode);
	}

public:
	static void print(const std::string & str) { getInstance()._print(str); }
	static void display(const std::string & str) { getInstance()._display(str); }
	static void error(const std::string & str, const bool fatal = false) { getInstance()._error(str, fatal); }
	static bool result(const std::string & str) { return getInstance()._result(str); }

	static FILE * open(const char * const filename, const char * const mode) { return getInstance()._open(filename, mode); }
};
