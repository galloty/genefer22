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
	// transform * _transform = nullptr;

public:
	bool check(const uint32_t b, const uint32_t n, const std::string & exp_residue = "")
	{
		omp_set_num_threads(2);
		size_t num_threads = 0;
#pragma omp parallel
		{
#pragma omp single
			num_threads = size_t(omp_get_num_threads());
		}
		std::cout << num_threads << " thread(s)." << std::endl;

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, n);

		ComplexITransform * t = nullptr;
		if (n == (1 << 10))      t = new CZIT_CPU_vec_mt<(1 << 10), 4>(b, num_threads);
		else if (n == (1 << 11)) t = new CZIT_CPU_vec_mt<(1 << 11), 4>(b, num_threads);
		else if (n == (1 << 12)) t = new CZIT_CPU_vec_mt<(1 << 12), 4>(b, num_threads);
		else if (n == (1 << 13)) t = new CZIT_CPU_vec_mt<(1 << 13), 4>(b, num_threads);
		else if (n == (1 << 14)) t = new CZIT_CPU_vec_mt<(1 << 14), 4>(b, num_threads);
		if (t == nullptr) throw std::runtime_error("exponent is not supported");

		auto t0 = std::chrono::steady_clock::now();

		double err = 0;
		for (int i = int(mpz_sizeinbase(exponent, 2)) - 1; i >= 0; --i)
		{
			const double e = t->squareDup(mpz_tstbit(exponent, i) != 0);
			err  = std::max(err, e);
			if (_quit) break;
		}

		mpz_clear(exponent);

		if (_quit) return false;

		const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		uint64_t res;
		const bool isPrp = t->isOne(res);
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
