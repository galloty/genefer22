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

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, n);

		auto t0 = std::chrono::steady_clock::now();

		static const size_t L = 8;
		const size_t esize = mpz_sizeinbase(exponent, 2);
		const size_t B = (esize - 1) / L + 1, o = L * B - esize;

		transform->set(1);
		size_t k = 0;
		for (size_t j = 1; j <= esize; ++j)
		{
			transform->squareDup(mpz_tstbit(exponent, esize - j) != 0);

			if ((j + o) % B == 0)
			{
				// std::cout << k << ", " << i << std::endl;
				transform->copy(2 + k, 0);
				++k;
			}
			if (_quit) break;
		}
		if (_quit) return false;
		if (k != L) throw std::runtime_error("internal error");

		// r2, ..., r9: cert[1], ..., cert[L]

		transform->set(1);
		for (size_t i = 1; i <= L - 1; ++i)
		{
			transform->copy(1, 1 + i);
			transform->initMultiplicand();
			transform->mul();
		}
		transform->copy(11, 0);	// r11: prod1

		transform->copy(0, 2);
		for (size_t i = 1; i <= L - 1; ++i)
		{
			transform->copy(1, 2 + i);
			transform->initMultiplicand();
			transform->mul();
		}
		transform->copy(12, 0);	// r12: prod2

		transform->copy(0, 11);
		for (size_t i = 0; i < B; ++i)
		{
			transform->squareDup(false);
		}
		transform->copy(11, 0);	// r11: prod1^(2^B)

		mpz_t t; mpz_init_set_ui(t, 0);
		for (size_t j = 1; j <= B - o; ++j)
		{
			mpz_mul_2exp(t, t, 1);
			mpz_add_ui(t, t, mpz_tstbit(exponent, esize - j));
		}
		mpz_t res; mpz_init_set(res, t);
		for (size_t i = 1; i <= L - 1; ++i)
		{
			mpz_set_ui(t, 0);
			for (size_t j = i * B - o + 1; j <= i * B - o + B; ++j)
			{
				mpz_mul_2exp(t, t, 1);
				mpz_add_ui(t, t, mpz_tstbit(exponent, esize - j));
			}
			mpz_add(res, res, t);
		}
		mpz_clear(t);

		transform->set(1);
		for (int i = int(mpz_sizeinbase(res, 2)) - 1; i >= 0; --i)
		{
			transform->squareDup(mpz_tstbit(res, i) != 0);
			if (_quit) break;
		}
		mpz_clear(res);

		// r0: 2^res

		transform->copy(1, 11);
		transform->initMultiplicand();
		transform->mul();

		// r0: prod1^(2^B) * 2^res

		// r0 ?= r12
		int32_t * const zi0 = new int32_t[n];
		transform->getZi(zi0);
		transform->unbalance(zi0);
		int32_t * const zi12 = new int32_t[n];
		transform->copy(0, 12);
		transform->getZi(zi12);
		transform->unbalance(zi12);
		bool isEqual = true;
		for (size_t k = 0; k < n; ++k) isEqual &= (zi0[k] == zi12[k]);

		std::cout << "Check: " << (isEqual ? "OK" : "NOK") << std::endl;

		transform->copy(0, 9);

		// RTL
		// transform->set(2);
		// for (size_t i = 0; i < n; ++i)
		// {
		// 	//transform->pow(b);
		// 	transform->copy(1, 0);
		// 	transform->initMultiplicand();
		// 	for (int j = 31 - __builtin_clz(b) - 1; j >= 0; --j)
		// 	{
		// 		transform->squareDup(false);
		// 		if ((b & (uint32_t(1) << j)) != 0)
		// 		{
		// 			transform->mul();
		// 		}
		// 	}
		// 	if (_quit) break;
		// }

		mpz_clear(exponent);

		if (_quit) return false;

		const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		const double err = transform->getError();
		uint64_t res64;
		const bool isPrp = transform->isOne(res64);
		char residue[30];
		sprintf(residue, "%016" PRIx64, res64);

		std::cout << b << "^" << n << " + 1";
		if (isPrp) std::cout << " is prime";
		std::cout << ", err = " << err << ", " << time << " sec";
		if (!isPrp & (std::string(residue) != exp_residue)) std::cout << ", res = " << residue << " [" << exp_residue << "]";
		std::cout << "." << std::endl;

		return true;
	}
};
