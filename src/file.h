/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <iomanip>

#include <gmp.h>

// #if defined(_WIN32)
// #include <IO.h>
// #else
// #include <unistd.h>
// #endif

#include "pio.h"

class file
{
private:
	const std::string _filename;
	FILE * const _cFile;
	const bool _fatal;
	// const bool _isSync;

public:
	file(const std::string & filename, const char * const mode, const bool fatal)
		: _filename(filename), _cFile(pio::open(filename.c_str(), mode)), _fatal(fatal) //, _isSync(std::string(mode) == "wb")
	{
		if (_cFile == nullptr) error("cannot open file");
	}

	file(const std::string & filename)
		: _filename(filename), _cFile(pio::open(filename.c_str(), "rb")), _fatal(false) //, _isSync(false)
	{
		// _cFile may be null
	}

	virtual ~file()
	{
		if (_cFile != nullptr)
		{
// 			if (_isSync)
// 			{
// #if defined(_WIN32)
// 				_commit(_fileno(_cFile));
// #else
// 				fsync(fileno(_cFile));
// #endif
// 			}
			if (std::fclose(_cFile) != 0) error("cannot close file");
		}
	}

	void error(const std::string & str) const
	{
		std::ostringstream ss; ss << _filename << ": " << str << std::endl;
		pio::error(ss.str(), _fatal);
	}

	bool exists() const { return (_cFile != nullptr); }

	bool read(char * const ptr, const size_t size)
	{
		const size_t ret = std::fread(ptr , sizeof(char), size, _cFile);
		if (ret == size * sizeof(char)) return true;
		std::fclose(_cFile);
		error("failure of a read operation");
		return false;
	}

	bool write(const char * const ptr, const size_t size)
	{
		const size_t ret = std::fwrite(ptr , sizeof(char), size, _cFile);
		if (ret == size * sizeof(char)) return true;
		std::fclose(_cFile);
		error("failure of a write operation");
		return false;
	}

	bool read(mpz_t & z)
	{
		if (mpz_inp_raw(z, _cFile) != 0) return true;
		std::fclose(_cFile);
		error("failure of a read operation");
		return false;
	}

	bool write(const mpz_t & z)
	{
		if (mpz_out_raw(_cFile, z) != 0) return true;
		std::fclose(_cFile);
		error("failure of a write operation");
		return false;
	}

	bool print(const char * const str)
	{
		const int ret = std::fprintf(_cFile, "%s", str);
		if (ret >= 0) return true;
		std::fclose(_cFile);
		error("failure of a print operation");
		return false;
	}
};
