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

private:
	static void power(Transform * const transform, const size_t reg, const mpz_t & e)
	{
		transform->copy(1, reg);
		transform->initMultiplicand();
		transform->set(1);
		for (size_t j = 0, esize = mpz_sizeinbase(e, 2); j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			transform->squareDup(false);
			if (mpz_tstbit(e, i) != 0) transform->mul();
		}
	}

public:
	bool check(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const std::string & exp_residue = "")
	{
		const int depth = 5;
		const size_t L = size_t(1) << depth;

		if (nthreads != 0) omp_set_num_threads(int(nthreads));
		size_t num_threads = 0;
#pragma omp parallel
		{
#pragma omp single
			num_threads = size_t(omp_get_num_threads());
		}

		Transform * transform = nullptr;
		std::string ttype;
		const size_t num_regs = L + 2;

		if (__builtin_cpu_supports("avx512f") && (impl.empty() || (impl == "512")))
		{
			transform = Transform::create_512(b, n, num_threads, num_regs);
			ttype = "512";
		}
		else if (__builtin_cpu_supports("fma") && (impl.empty() || (impl == "fma")))
		{
			transform = Transform::create_fma(b, n, num_threads, num_regs);
			ttype = "fma";
		}
		else if (__builtin_cpu_supports("avx") && (impl.empty() || (impl == "avx")))
		{
			transform = Transform::create_avx(b, n, num_threads, num_regs);
			ttype = "avx";
		}
		else if (__builtin_cpu_supports("sse4.1") && (impl.empty() || (impl == "sse4")))
		{
			transform = Transform::create_sse4(b, n, num_threads, num_regs);
			ttype = "sse4";
		}
		else if (__builtin_cpu_supports("sse2") && (impl.empty() || (impl == "sse2")))
		{
			transform = Transform::create_sse2(b, n, num_threads, num_regs);
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

		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		std::cout << "depth = " << depth << ", ";

		gint * cert_ptr[depth + 1];

		double time = 0;
{
		const auto t0 = std::chrono::steady_clock::now();

		transform->set(1);
		for (size_t j = 0; j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			transform->squareDup(mpz_tstbit(exponent, i) != 0);
			// if (j == 0) x.add(1);	// error
			// if (i == 0) x.add(1);	// error
			if (i % B == 0) transform->copy(2 + i / B, 0);
			if (_quit) return false;
		}

		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "test: " << std::chrono::duration<double>(t1 - t0).count() << ", ";

		// cert[0] = ckpt[0];
		transform->copy(0, 2);
		cert_ptr[0] = transform->getInt();

		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], cert_ptr[0]->gethash32());

		for (int k = 1; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// ckpt[i] = ckpt[i].pow(w[0]);
			power(transform, 2 + i, w[0]);
			transform->copy(2 + i, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// ckpt[i].mul(ckpt[i + 2 * j].pow(w[j]));
				power(transform, 2 + i + 2 * j, w[j]);
				transform->copy(1, 2 + i);
				transform->initMultiplicand();
				transform->mul();
				transform->copy(2 + i, 0);
			}
			// cert[k] = ckpt[size_t(1) << (depth - k)];
			transform->copy(0, 2 + i);
			cert_ptr[k] = transform->getInt();

			if (i > 1)
			{
				const uint32_t q = cert_ptr[k]->gethash32();
				for (size_t j = 0; j < L / 2; j += i) mpz_mul_ui(w[i / 2 + j], w[j], q);
			}

			if (_quit) return false;
		}

		for (size_t i = 0; i < L / 2; ++i) mpz_clear(w[i]);
		delete[] w;

		const auto t2 = std::chrono::steady_clock::now();
		std::cout << "cert: " << std::chrono::duration<double>(t2 - t1).count() << ", ";
		time += std::chrono::duration<double>(t2 - t0).count();
}

		mpz_t p2; mpz_init(p2);
		int64_t hv1srv;
{
		const auto t0 = std::chrono::steady_clock::now();

		mpz_t * const w = new mpz_t[L]; for (size_t i = 0; i < L; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], cert_ptr[0]->gethash32());

		// Mod v1 = Mod(cert[0]).pow(w[0]);
		transform->setInt(cert_ptr[0]);
		power(transform, 0, w[0]);
		transform->copy(2, 0);	// v1: reg = 2

		// v2 = Mod(1);
		transform->set(1);
		transform->copy(3, 0);	// v2: reg = 3

		for (int k = 1; k <= depth; ++k)
		{
			// const Mod & mu = cert[k];
			// const uint32_t q = mu.gethash32();
			const uint32_t q = cert_ptr[k]->gethash32();
			transform->setInt(cert_ptr[k]);
			transform->copy(4, 0);	// mu: reg = 4

			mpz_t mq; mpz_init_set_ui(mq, q);

			// v1.mul(mu.pow(mq));
			power(transform, 4, mq);
			transform->copy(1, 2);
			transform->initMultiplicand();
			transform->mul();
			transform->copy(2, 0);

			// v2 = v2.pow(mq); v2.mul(mu);
			power(transform, 3, mq);
			transform->copy(1, 4);
			transform->initMultiplicand();
			transform->mul();
			transform->copy(3, 0);

			mpz_clear(mq);

			const size_t i = size_t(1) << (depth - k);
			for (size_t j = 0; j < L; j += 2 * i) mpz_mul_ui(w[i + j], w[j], q);

			if (_quit) return false;
		}
		// hv1 = v1.gethash64();
		transform->copy(0, 2);
		gint * v1 = transform->getInt();
		hv1srv = v1->getResidue();
		delete v1;

		mpz_set_ui(p2, 0);
		mpz_t e, t; mpz_init_set(e, exponent); mpz_init(t);
		for (size_t i = 0; i < L; i++)
		{
			mpz_mod_2exp(t, e, B);
			mpz_addmul(p2, t, w[i]);
			mpz_div_2exp(e, e, B);
		}
		mpz_clear(e); mpz_clear(t);

		for (size_t i = 0; i < L; ++i) mpz_clear(w[i]);
		delete[] w;

		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "srv: " << std::chrono::duration<double>(t1 - t0).count() << ", ";	// << (cert[0].isone() ? "prime" : "composite") << ", ";
		time += std::chrono::duration<double>(t1 - t0).count();
}
		// v2: reg = 3
		int64_t hv1val;
{
		const auto t0 = std::chrono::steady_clock::now();

		// Mod t = v2; for (size_t i = 0; i < B; ++i) t.square_dup(false);
		transform->copy(0, 3);
		for (size_t i = 0; i < B; ++i)
		{
			transform->squareDup(false);
			if (_quit) return false;
		}
		transform->copy(3, 0);

		// t.mul(Mod(2).pow(p2));
		transform->set(2);
		power(transform, 0, p2);
		transform->copy(1, 3);
		transform->initMultiplicand();
		transform->mul();

		// hv1v = t.gethash64();
		gint * v1 = transform->getInt();
		hv1val = v1->getResidue();
		delete v1;

		const auto t1 = std::chrono::steady_clock::now();;
		std::cout << "valid: " << std::chrono::duration<double>(t1 - t0).count() << ", ";
		time += std::chrono::duration<double>(t1 - t0).count();
}

		mpz_clear(p2);

		std::cout << "proof " << ((hv1srv == hv1val) ? "ok" : "failed") << ": " << std::hex << hv1srv << std::dec << std::endl;

		transform->setInt(cert_ptr[0]);

		for (size_t i = 0; i <= depth; ++i) delete cert_ptr[i];

		mpz_clear(exponent);

		const double err = transform->getError();
		uint64_t res64; const bool isPrp = transform->isOne(res64);
		char residue[30]; sprintf(residue, "%016" PRIx64, res64);

		std::cout << b << "^" << n << " + 1 is " << (isPrp ? "prime" : "composite") << ", err = " << err << ", " << time << " sec";
		if (!isPrp & (std::string(residue) != exp_residue)) std::cout << ", res = " << residue << " [" << exp_residue << "]";
		std::cout << "." << std::endl;

		return true;
	}
};
