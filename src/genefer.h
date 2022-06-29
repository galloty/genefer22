/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <iostream>
#include <stdexcept>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <chrono>
#include <thread>

#include <gmp.h>
#include <omp.h>

#include "transform.h"

class genefer
{
private:
	struct deleter { void operator()(const genefer * const p) { delete p; } };

public:
	genefer() {}
	virtual ~genefer() {}

	static genefer & getInstance()
	{
		static std::unique_ptr<genefer, deleter> pInstance(new genefer());
		return *pInstance;
	}

public:
	void quit() { _quit = true; }
	void setBoinc(const bool isBoinc) { _isBoinc = isBoinc; }

protected:
	volatile bool _quit = false;
private:
	int _n = 0;
	bool _isBoinc = false;

public:
	bool check(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const std::string & exp_residue = "")
	{
		if (nthreads != 0) omp_set_num_threads(int(nthreads));
		size_t num_threads = 0;
#pragma omp parallel
		{
#pragma omp single
			num_threads = size_t(omp_get_num_threads());
		}

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, n);

		Transform * transform = nullptr;
		std::string ttype;

		if (__builtin_cpu_supports("avx512f") && (impl.empty() || (impl == "512")))
		{
			transform = Transform::create_512(b, n, num_threads);
			ttype = "512";
		}
		else if (__builtin_cpu_supports("fma") && (impl.empty() || (impl == "fma")))
		{
			transform = Transform::create_fma(b, n, num_threads);
			ttype = "fma";
		}
		else if (__builtin_cpu_supports("avx") && (impl.empty() || (impl == "avx")))
		{
			transform = Transform::create_avx(b, n, num_threads);
			ttype = "avx";
		}
		else if (__builtin_cpu_supports("sse4.1") && (impl.empty() || (impl == "sse4")))
		{
			transform = Transform::create_sse4(b, n, num_threads);
			ttype = "sse4";
		}
		else if (__builtin_cpu_supports("sse2") && (impl.empty() || (impl == "sse2")))
		{
			transform = Transform::create_sse2(b, n, num_threads);
			ttype = "sse2";
		}
		else
		{
			if (impl.empty()) throw std::runtime_error("processor must support sse2");
			std::ostringstream ss; ss << impl << " is not supported";
			throw std::runtime_error(ss.str());
		}

		std::cout << "Using " << ttype << " implementation, " << num_threads << " thread(s)." << std::endl;

		auto t0 = std::chrono::steady_clock::now();

		double err = 0;
		for (int i = int(mpz_sizeinbase(exponent, 2)) - 1; i >= 0; --i)
		{
			const double e = transform->squareDup(mpz_tstbit(exponent, i) != 0);
			err  = std::max(err, e);
			if (_quit) break;
		}

		mpz_clear(exponent);

		if (_quit) return false;

		const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		uint64_t res;
		const bool isPrp = transform->isOne(res);
		char residue[30];
		sprintf(residue, "%016" PRIx64, res);

		std::cout << b << "^" << n << " + 1";
		if (isPrp) std::cout << " is prime";
		std::cout << ", err = " << err << ", " << time << " sec";
		if (!isPrp & (std::string(residue) != exp_residue)) std::cout << ", res = " << residue << " [" << exp_residue << "]";
		std::cout << "." << std::endl;

		return true;
	}
};
