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
#include <cmath>

#include <gmp.h>
#if !defined(GPU)
#include <omp.h>
#endif

#include "pio.h"
#if defined(GPU)
#include "ocl.h"
#endif
#include "transform.h"
#include "timer.h"

inline int ilog2_32(const uint32_t n) { return 31 - __builtin_clzl(n); }
inline int ilog2_64(const uint64_t n) { return 63 - __builtin_clzll(n); }

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
#if defined(GPU)
	void setBoincParam(const cl_platform_id platform_id, const cl_device_id device_id)
	{
		_boinc_platform_id = platform_id;
		_boinc_device_id = device_id;
	}
#endif

protected:
	volatile bool _quit = false;
private:
	bool _isBoinc = false;
#if defined(GPU)
	cl_platform_id _boinc_platform_id = 0;
	cl_device_id _boinc_device_id = 0;
#endif
	transform * _transform = nullptr;

private:
#ifdef GPU
	transform * createTransformGPU(const uint32_t b, const uint32_t n, const size_t device, const size_t num_regs)
	{
		deleteTransform();
		_transform = transform::create_gpu(b, n, _isBoinc, device, num_regs, _boinc_platform_id, _boinc_device_id);
		std::cout << "Using device " << device << "." << std::endl;
		return _transform;
	}
#else
	transform * createTransformCPU(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const size_t num_regs)
	{
		deleteTransform();

		if (nthreads != 0) omp_set_num_threads(int(nthreads));
		size_t num_threads = 0;
#pragma omp parallel
		{
#pragma omp single
			num_threads = size_t(omp_get_num_threads());
		}

		std::string ttype;
		_transform = transform::create_cpu(b, n, _isBoinc, num_threads, impl, num_regs, ttype);
		std::cout << "Using " << ttype << " implementation, " << num_threads << " thread(s), " << _transform->getMemSize() / (1 << 20) << " MB." << std::endl;
		return _transform;
	}
#endif

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
		transform * const pTransform = _transform;
		pTransform->initMultiplicand(reg);
		pTransform->set(1);
		for (size_t j = 0, esize = ilog2_32(e) + 1; j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			pTransform->squareDup(false);
			if ((e & (uint32_t(1) << i)) != 0) pTransform->mul();
		}
	}

	void power(const size_t reg, const mpz_t & e) const
	{
		transform * const pTransform = _transform;
		pTransform->initMultiplicand(reg);
		pTransform->set(1);
		for (size_t j = 0, esize = mpz_sizeinbase(e, 2); j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			pTransform->squareDup(false);
			if (mpz_tstbit(e, i) != 0) pTransform->mul();
		}
	}

	bool test(const int depth, const mpz_t & exponent, gint cert[], double & testTime, double & checkTime, double & certTime)
	{
		transform * const pTransform = _transform;

		const auto t0 = timer::currentTime();

		const size_t esize = mpz_sizeinbase(exponent, 2), L_GL = (2 << (ilog2_64(esize) / 2)), B_GL = ((esize - 1) / L_GL) + 1;
		std::cout << "L_GL = " << L_GL << ", B_GL = " << B_GL << ", ";

		const size_t L = size_t(1) << depth, B_PL = ((esize - 1) >> depth) + 1;

		pTransform->set(1);
		pTransform->copy(1, 0);	// prod1
		for (size_t j = 0; j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			pTransform->squareDup(mpz_tstbit(exponent, i) != 0);
			// if (j == 0) pTransform->add1();	// error
			// if (i == 0) pTransform->add1();	// error
			if (i % B_GL == 0)
			{
				if (i / B_GL != 0)
				{
	 				pTransform->copy(2, 0);
					pTransform->mul(1);	// prod1
					pTransform->copy(1, 0);
					pTransform->copy(0, 2);
				}
			}
			if (i % B_PL == 0)
			{
				const size_t reg = 3 + i / B_PL;	// ckpt[i]
 				pTransform->copy(reg, 0);
			}
			if (_quit) return false;
		}

		const auto t1 = timer::currentTime();
		testTime = timer::diffTime(t1, t0);

		// Gerbicz-Li error checking

		pTransform->mul(1);
		pTransform->copy(2, 0);	// prod2 = prod1 * result

		pTransform->copy(0, 1);
		for (size_t i = 0; i < B_GL; ++i)
		{
			pTransform->squareDup(false);
		}
		pTransform->copy(1, 0);	// prod1^{2^B}

		mpz_t res; mpz_init_set_ui(res, 0);
		mpz_t e, t; mpz_init_set(e, exponent); mpz_init(t);
		while (mpz_sgn(e) != 0)
		{
			mpz_mod_2exp(t, e, B_GL);
			mpz_add(res, res, t);
			mpz_div_2exp(e, e, B_GL);
		}
		mpz_clear(e); mpz_clear(t);

		// 2^res
		pTransform->set(1);
		for (size_t j = 0, ressize = mpz_sizeinbase(res, 2); j < ressize; ++j)
		{
			const size_t i = ressize - 1 - j;
			pTransform->squareDup(mpz_tstbit(res, i) != 0);
			if (_quit) return false;
		}
		mpz_clear(res);

		// prod1^{2^B} * 2^res
		pTransform->mul(1);

		// prod1^{2^B} * 2^res ?= prod2
		gint v1; pTransform->getInt(v1);
		const uint64_t h1 = v1.gethash64();
		v1.clear();
		pTransform->copy(0, 2);
		gint v2; pTransform->getInt(v2);
		const uint64_t h2 = v2.gethash64();
		v2.clear();

		const bool success = (h1 == h2);

		const auto t2 = timer::currentTime();
		checkTime = timer::diffTime(t2, t1);

		// generate certificates

		// cert[0] = ckpt[0]
		pTransform->copy(0, 3);
		pTransform->getInt(cert[0]);

		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], cert[0].gethash32());

		for (int k = 1; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// cert[k] = ckpt[i]^w[0]
			power(3 + i, w[0]);
			pTransform->copy(1, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// cert[k] *= ckpt[i + 2 * j]^w[j]
				power(3 + i + 2 * j, w[j]);
				pTransform->mul(1);
				pTransform->copy(1, 0);
			}
			pTransform->getInt(cert[k]);

			if (i > 1)
			{
				const uint32_t q = cert[k].gethash32();
				for (size_t j = 0; j < L / 2; j += i) mpz_mul_ui(w[i / 2 + j], w[j], q);
			}

			if (_quit) return false;
		}

		for (size_t i = 0; i < L / 2; ++i) mpz_clear(w[i]);
		delete[] w;

		certTime = timer::diffTime(timer::currentTime(), t2);

		return success;
	}

	uint64_t server(const int depth, const mpz_t & exponent, const gint cert[], gint & v2, mpz_t & p2, double & dt)
	{
		transform * const pTransform = _transform;

		const auto t0 = timer::currentTime();

		const size_t L = size_t(1) << depth;
		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		mpz_t * const w = new mpz_t[L]; for (size_t i = 0; i < L; ++i) mpz_init(w[i]);

		// v1 = cert[0]^w[0], v1: reg = 1
		const uint32_t q = cert[0].gethash32();
		mpz_set_ui(w[0], q);
		pTransform->setInt(cert[0]);
		power(0, q);
		pTransform->copy(1, 0);

		// v2 = 1, v2: reg = 2
		pTransform->set(1);
		pTransform->copy(2, 0);

		for (int k = 1; k <= depth; ++k)
		{
			// mu = cert[k], mu: reg = 3
			const uint32_t q = cert[k].gethash32();
			pTransform->setInt(cert[k]);
			pTransform->copy(3, 0);

			// v1 = v1 * mu^q
			power(3, q);
			pTransform->mul(1);
			pTransform->copy(1, 0);

			// v2 = v2^q * mu
			power(2, q);
			pTransform->mul(3);
			pTransform->copy(2, 0);

			const size_t i = size_t(1) << (depth - k);
			for (size_t j = 0; j < L; j += 2 * i) mpz_mul_ui(w[i + j], w[j], q);

			if (_quit) return 0;
		}

		pTransform->copy(0, 2);
		pTransform->getInt(v2);

		// h1 = hash64(v1);
		pTransform->copy(0, 1);
		gint v1; pTransform->getInt(v1);
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
		transform * const pTransform = _transform;

		const auto t0 = timer::currentTime();

		// v2 = v2^{2^B}
		pTransform->setInt(v2);
		for (size_t i = 0; i < B; ++i)
		{
			pTransform->squareDup(false);
			if (_quit) return 0;
		}
		pTransform->copy(2, 0);

		// v1' = v2 * 2^p2
		pTransform->set(1);
		for (size_t j = 0, p2size = mpz_sizeinbase(p2, 2); j < p2size; ++j)
		{
			const size_t i = p2size - 1 - j;
			pTransform->squareDup(mpz_tstbit(p2, i) != 0);
			if (_quit) return 0;
		}
		pTransform->mul(2);

		// h1 = hash64(v1')
		gint v1; pTransform->getInt(v1);
		const uint64_t h1 = v1.gethash64();
		v1.clear();

		dt = timer::diffTime(timer::currentTime(), t0);
		return h1;
	}

private:
	static double percent(const double num, const double den) { return std::rint(num * 1000 / den) / 10; }

public:
	bool check(const uint32_t b, const uint32_t n, const size_t device, const size_t nthreads, const std::string & impl, const int depth)
	{
		// const int depth = 5;
		const size_t L = size_t(1) << depth;

		const size_t num_regs = 3 + L;
#ifdef GPU
		(void)nthreads; (void)impl;
		transform * const pTransform = createTransformGPU(b, n, device, num_regs);
#else
		(void)device;
		transform * const pTransform = createTransformCPU(b, n, nthreads, impl, num_regs);
#endif

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, 1 << n);

		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		std::cout << "depth = " << depth << ", L_PL = " << L << ", B_PL = " << B << ", ";

		gint cert[depth + 1];

		double testTime = 0, checkTime = 0, certTime = 0; const bool success = test(depth, exponent, cert, testTime, checkTime, certTime);
		// if (!success) return false;

		gint v2; mpz_t p2; mpz_init(p2);
		double serverTime = 0; const uint64_t hv1srv = server(depth, exponent, cert, v2, p2, serverTime);
		// if (hv1srv == 0) return false;

		const bool isPrp = cert[0].isOne();

		for (int i = 0; i <= depth; ++i) cert[i].clear();
		mpz_clear(exponent);

		double validTime = 0; const uint64_t hv1val = valid(B, v2, p2, validTime);
		// if (hv1val == 0) return false;

		v2.clear();
		mpz_clear(p2);

		const double time = testTime + checkTime + certTime + serverTime + validTime;

		std::cout << "test: " << percent(testTime, time) << "%, check: " << (success ? "ok" : "FAILED") << " " << percent(checkTime, time)
				  << "%, cert: " << percent(certTime, time) << "%, srv: " << percent(serverTime, time) << "%, valid: " << percent(validTime, time) << "%, "
				  << "proof " << ((hv1srv == hv1val) ? "ok" : "FAILED") << ": " << std::hex << hv1srv << std::dec << std::endl;

		const double err = pTransform->getError();

		std::cout << b << "^{2^" << n << "} + 1 is " << (isPrp ? "prime" : "composite") << ", err = " << err << ", " << time << " sec." << std::endl;

		deleteTransform();

		return true;
	}
};
