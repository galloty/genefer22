/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
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
	// print: console: stdout, boinc: stderr
	void _print(const std::string & str) const
	{
		if (_isBoinc) { std::cerr << str << std::flush; }
		else { std::cout << str; }
	}

private:
	// display: console: stdout, boinc: -
	void _display(const std::string & str) const
	{
		if (!_isBoinc) { std::cout << str << std::flush; }
	}

private:
	// error: normal: stderr, boinc: stderr
	void _error(const std::string & str, const bool fatal) const
	{
		if (fatal) std::cerr << std::endl;
		std::cerr << "Error: " << str << "." << std::endl << std::flush;
		if (fatal)
		{
			if (_isBoinc)
			{
				// delay five minutes before reporting to the host in order to slow down the error rate.
				std::this_thread::sleep_for(std::chrono::minutes(5));
				boinc_finish(EXIT_FAILURE);
			}
			else
			{
				exit(EXIT_FAILURE);
			}
		}
	}

private:
	// result: normal: 'results.txt' file
	bool _result(const std::string & str) const
	{
		FILE * const out_file = _open("results.txt", "a");
		if (out_file == nullptr)
		{
			if (_isBoinc) throw std::runtime_error("Cannot write 'results.txt' file");
			return false;
		}
		std::fprintf(out_file, "%s", str.c_str());
		std::fclose(out_file);
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

		// resolve filename as Boinc does
		std::string path = filename;
		std::ifstream linkFile(filename);
		if (linkFile.is_open())
		{
			std::string line; std::getline(linkFile, line);
			linkFile.close();
			std::string::size_type first = line.find("<soft_link>"), last = line.find("</soft_link>");
			if ((first != std::string::npos) && (last != std::string::npos))
			{
				first += 11;
				path = line.substr(first, last - first);
			}
		}

		return std::fopen(path.c_str(), mode);
	}

public:
	static void print(const std::string & str) { getInstance()._print(str); }
	static void display(const std::string & str) { getInstance()._display(str); }
	static void error(const std::string & str, const bool fatal = false) { getInstance()._error(str, fatal); }
	static bool result(const std::string & str) { return getInstance()._result(str); }

	static FILE * open(const char * const filename, const char * const mode) { return getInstance()._open(filename, mode); }
};
