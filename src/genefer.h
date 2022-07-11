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
#include <chrono>
#include <thread>
#include <cmath>

#include <gmp.h>
#include <omp.h>

#include "transform.h"
#include "timer.h"

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
	bool _isBoinc = false;
	Transform * _transform = nullptr;

private:
	static size_t set_num_threads(const size_t nthreads)
	{
		if (nthreads != 0) omp_set_num_threads(int(nthreads));
		size_t num_threads = 0;
#pragma omp parallel
		{
#pragma omp single
			num_threads = size_t(omp_get_num_threads());
		}
		return num_threads;
	}

	void createTransform(const uint32_t b, const uint32_t n, const size_t num_threads, const std::string & impl, const size_t num_regs)
	{
		if (_transform != nullptr) delete _transform;
		std::string ttype;

		if (__builtin_cpu_supports("avx512f") && (impl.empty() || (impl == "512")))
		{
			_transform = Transform::create_512(b, n, num_threads, num_regs);
			ttype = "512";
		}
		else if (__builtin_cpu_supports("fma") && (impl.empty() || (impl == "fma")))
		{
			_transform = Transform::create_fma(b, n, num_threads, num_regs);
			ttype = "fma";
		}
		else if (__builtin_cpu_supports("avx") && (impl.empty() || (impl == "avx")))
		{
			_transform = Transform::create_avx(b, n, num_threads, num_regs);
			ttype = "avx";
		}
		else if (__builtin_cpu_supports("sse4.1") && (impl.empty() || (impl == "sse4")))
		{
			_transform = Transform::create_sse4(b, n, num_threads, num_regs);
			ttype = "sse4";
		}
		else if (__builtin_cpu_supports("sse2") && (impl.empty() || (impl == "sse2")))
		{
			_transform = Transform::create_sse2(b, n, num_threads, num_regs);
			ttype = "sse2";
		}
		else
		{
			if (impl.empty()) throw std::runtime_error("processor must support sse2");
			std::ostringstream ss; ss << impl << " is not supported";
			throw std::runtime_error(ss.str());
		}

		std::cout << "Using " << ttype << " implementation, " << num_threads << " thread(s)." << std::endl;
	}

	void deleteTransform()
	{
		if (_transform != nullptr)
		{
			delete _transform;
			_transform = nullptr;
		}
	}

	void power(const size_t reg, const uint32_t e) const
	{
		Transform * const transform = _transform;
		transform->initMultiplicand(reg);
		transform->set(1);
		for (size_t j = 0, esize = 32 - __builtin_clzl(e); j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			transform->squareDup(false);
			if ((e & (uint32_t(1) << i)) != 0) transform->mul();
		}
	}

	void power(const size_t reg, const mpz_t & e) const
	{
		Transform * const transform = _transform;
		transform->initMultiplicand(reg);
		transform->set(1);
		for (size_t j = 0, esize = mpz_sizeinbase(e, 2); j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			transform->squareDup(false);
			if (mpz_tstbit(e, i) != 0) transform->mul();
		}
	}

	bool test(const int depth, const mpz_t & exponent, gint cert[], double & testTime, double & checkTime, double & certTime)
	{
		Transform * const transform = _transform;

		const auto t0 = timer::currentTime();

		const size_t L = size_t(1) << depth;
		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		transform->set(1);
		transform->copy(L + 2, 0);	// prod1
		for (size_t j = 0; j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			transform->squareDup(mpz_tstbit(exponent, i) != 0);
			// if (j == 0) x.add(1);	// error
			// if (i == 0) x.add(1);	// error
			if (i % B == 0)
			{
				const size_t reg = 2 + i / B;	// ckpt[i]
 				transform->copy(reg, 0);
				if (reg != 0)
				{
					transform->mul(L + 2);	// prod1
					transform->copy(L + 2, 0);
					transform->copy(0, reg);
				}
			}
			if (_quit) return false;
		}

		const auto t1 = timer::currentTime();
		testTime = timer::diffTime(t1, t0);

		// Gerbicz-Li error checking

		transform->copy(0, L + 2);
		transform->mul(2);
		transform->copy(L + 3, 0);	// prod2

		transform->copy(0, L + 2);
		for (size_t i = 0; i < B; ++i)
		{
			transform->squareDup(false);
		}
		transform->copy(L + 2, 0);	// prod1^{2^B}

		mpz_t res; mpz_init_set_ui(res, 0);
		mpz_t e, t; mpz_init_set(e, exponent); mpz_init(t);
		for (size_t i = 0; i < L; i++)
		{
			mpz_mod_2exp(t, e, B);
			mpz_add(res, res, t);
			mpz_div_2exp(e, e, B);
		}
		mpz_clear(e); mpz_clear(t);

		// 2^res
		transform->set(1);
		for (size_t j = 0, ressize = mpz_sizeinbase(res, 2); j < ressize; ++j)
		{
			const size_t i = ressize - 1 - j;
			transform->squareDup(mpz_tstbit(res, i) != 0);
			if (_quit) return false;
		}
		mpz_clear(res);

		// prod1^{2^B} * 2^res
		transform->mul(L + 2);

		// prod1^{2^B} * 2^res ?= prod2
		gint v1; transform->getInt(v1);
		const uint64_t h1 = v1.gethash64();
		v1.clear();
		transform->copy(0, L + 3);
		gint v2; transform->getInt(v2);
		const uint64_t h2 = v2.gethash64();
		v2.clear();

		const auto t2 = timer::currentTime();
		checkTime = timer::diffTime(t2, t1);

		// generate certificates

		// cert[0] = ckpt[0]
		transform->copy(0, 2);
		transform->getInt(cert[0]);

		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], cert[0].gethash32());

		for (int k = 1; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// ckpt[i] = ckpt[i]^w[0]
			power(2 + i, w[0]);
			transform->copy(2 + i, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// ckpt[i] *= ckpt[i + 2 * j]^w[j]
				power(2 + i + 2 * j, w[j]);
				transform->mul(2 + i);
				transform->copy(2 + i, 0);
			}
			// cert[k] = ckpt[i]
			transform->copy(0, 2 + i);
			transform->getInt(cert[k]);

			if (i > 1)
			{
				const uint32_t q = cert[k].gethash32();
				for (size_t j = 0; j < L / 2; j += i) mpz_mul_ui(w[i / 2 + j], w[j], q);
			}

			if (_quit) return false;
		}

		for (size_t i = 0; i < L / 2; ++i) mpz_clear(w[i]);
		delete[] w;

		const auto t3 = timer::currentTime();
		certTime = timer::diffTime(t3, t2);

		return h1 == h2;
	}

	uint64_t server(const int depth, const mpz_t & exponent, const gint cert[], gint & v2, mpz_t & p2, double & dt)
	{
		Transform * const transform = _transform;

		const auto t0 = timer::currentTime();

		const size_t L = size_t(1) << depth;
		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		mpz_t * const w = new mpz_t[L]; for (size_t i = 0; i < L; ++i) mpz_init(w[i]);

		// v1 = cert[0]^w[0], v1: reg = 2
		const uint32_t q = cert[0].gethash32();
		mpz_set_ui(w[0], q);
		transform->setInt(cert[0]);
		power(0, q);
		transform->copy(2, 0);

		// v2 = 1, v2: reg = 3
		transform->set(1);
		transform->copy(3, 0);

		for (int k = 1; k <= depth; ++k)
		{
			// mu = cert[k], mu: reg = 4
			const uint32_t q = cert[k].gethash32();
			transform->setInt(cert[k]);
			transform->copy(4, 0);

			// v1 = v1 * mu^q
			power(4, q);
			transform->mul(2);
			transform->copy(2, 0);

			// v2 = v2^q * mu
			power(3, q);
			transform->mul(4);
			transform->copy(3, 0);

			const size_t i = size_t(1) << (depth - k);
			for (size_t j = 0; j < L; j += 2 * i) mpz_mul_ui(w[i + j], w[j], q);

			if (_quit) return 0;
		}

		transform->copy(0, 3);
		transform->getInt(v2);

		// h1 = hash64(v1);
		transform->copy(0, 2);
		gint v1; transform->getInt(v1);
		const uint64_t h1 = v1.gethash64();
		v1.clear();

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

		dt = timer::diffTime(timer::currentTime(), t0);
		return h1;
	}

	uint64_t valid(const size_t B, const gint & v2, const mpz_t & p2, double & dt)
	{
		Transform * const transform = _transform;

		const auto t0 = timer::currentTime();

		// v2 = v2^{2^B}
		transform->setInt(v2);
		for (size_t i = 0; i < B; ++i)
		{
			transform->squareDup(false);
			if (_quit) return 0;
		}
		transform->copy(3, 0);

		// v1' = v2 * 2^p2
		transform->set(1);
		for (size_t j = 0, p2size = mpz_sizeinbase(p2, 2); j < p2size; ++j)
		{
			const size_t i = p2size - 1 - j;
			transform->squareDup(mpz_tstbit(p2, i) != 0);
			if (_quit) return 0;
		}
		transform->mul(3);

		// h1 = hash64(v1')
		gint v1; transform->getInt(v1);
		const uint64_t h1 = v1.gethash64();
		v1.clear();

		dt = timer::diffTime(timer::currentTime(), t0);
		return h1;
	}

public:
	bool check(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const int depth)
	{
		// const int depth = 5;
		const size_t L = size_t(1) << depth;

		const size_t num_threads = set_num_threads(nthreads);
		createTransform(b, n, num_threads, impl, 2 + L + 2);
		Transform * const transform = _transform;

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, n);

		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		std::cout << "depth = " << depth << ", L = " << L << ", B = " << B << ", ";

		gint cert[depth + 1];

		double testTime = 0, checkTime = 0, certTime = 0; const bool success = test(depth, exponent, cert, testTime, checkTime, certTime);
		if (!success) return false;
		std::cout << "test: " << testTime << ", check: " << (success ? "ok" : "FAILED") << " " << checkTime << ", cert: " << certTime << ", ";

		gint v2; mpz_t p2; mpz_init(p2);
		double serverTime = 0; const uint64_t hv1srv = server(depth, exponent, cert, v2, p2, serverTime);
		if (hv1srv == 0) return false;
		std::cout << "srv: " << serverTime << ", ";

		const bool isPrp = cert[0].isOne();

		for (int i = 0; i <= depth; ++i) cert[i].clear();
		mpz_clear(exponent);

		double validTime = 0; const uint64_t hv1val = valid(B, v2, p2, validTime);
		if (hv1val == 0) return false;
		std::cout << "valid: " << validTime << ", ";

		v2.clear();
		mpz_clear(p2);

		const double time = testTime + checkTime + certTime + serverTime + validTime;
		std::cout << "+" << std::rint((time - testTime) * 1000 / time) / 10 << "%, ";
		std::cout << "proof " << ((hv1srv == hv1val) ? "ok" : "FAILED") << ": " << std::hex << hv1srv << std::dec << std::endl;

		const double err = transform->getError();

		std::cout << b << "^" << n << " + 1 is " << (isPrp ? "prime" : "composite") << ", err = " << err << ", " << time << " sec." << std::endl;

		deleteTransform();

		return true;
	}
};
