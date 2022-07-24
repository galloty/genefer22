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
#include <omp.h>

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

	Transform * createTransform(const uint32_t b, const uint32_t n, const size_t num_threads, const std::string & impl, const size_t num_regs)
	{
		deleteTransform();
		std::string ttype;
		_transform = Transform::create(b, n, num_threads, impl, num_regs, ttype);
		std::cout << "Using " << ttype << " implementation, " << num_threads << " thread(s), " << _transform->getMemSize() / (1 << 20) << " MB." << std::endl;
		return _transform;
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
		for (size_t j = 0, esize = ilog2_32(e) + 1; j < esize; ++j)
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

		const size_t esize = mpz_sizeinbase(exponent, 2), L_GL = (2 << (ilog2_64(esize) / 2)), B_GL = ((esize - 1) / L_GL) + 1;
		std::cout << "L_GL = " << L_GL << ", B_GL = " << B_GL << ", ";

		const size_t L = size_t(1) << depth, B_PL = ((esize - 1) >> depth) + 1;

		transform->set(1);
		transform->copy(2, 0);	// prod1
		for (size_t j = 0; j < esize; ++j)
		{
			const size_t i = esize - 1 - j;
			transform->squareDup(mpz_tstbit(exponent, i) != 0);
			// if (j == 0) x.add(1);	// error
			// if (i == 0) x.add(1);	// error
			if (i % B_GL == 0)
			{
				if (i / B_GL != 0)
				{
	 				transform->copy(3, 0);
					transform->mul(2);	// prod1
					transform->copy(2, 0);
					transform->copy(0, 3);
				}
			}
			if (i % B_PL == 0)
			{
				const size_t reg = 4 + i / B_PL;	// ckpt[i]
 				transform->copy(reg, 0);
			}
			if (_quit) return false;
		}

		const auto t1 = timer::currentTime();
		testTime = timer::diffTime(t1, t0);

		// Gerbicz-Li error checking

		transform->mul(2);
		transform->copy(3, 0);	// prod2 = prod1 * result

		transform->copy(0, 2);
		for (size_t i = 0; i < B_GL; ++i)
		{
			transform->squareDup(false);
		}
		transform->copy(2, 0);	// prod1^{2^B}

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
		transform->set(1);
		for (size_t j = 0, ressize = mpz_sizeinbase(res, 2); j < ressize; ++j)
		{
			const size_t i = ressize - 1 - j;
			transform->squareDup(mpz_tstbit(res, i) != 0);
			if (_quit) return false;
		}
		mpz_clear(res);

		// prod1^{2^B} * 2^res
		transform->mul(2);

		// prod1^{2^B} * 2^res ?= prod2
		gint v1; transform->getInt(v1);
		const uint64_t h1 = v1.gethash64();
		v1.clear();
		transform->copy(0, 3);
		gint v2; transform->getInt(v2);
		const uint64_t h2 = v2.gethash64();
		v2.clear();

		const bool success = (h1 == h2);

		const auto t2 = timer::currentTime();
		checkTime = timer::diffTime(t2, t1);

		// generate certificates

		// cert[0] = ckpt[0]
		transform->copy(0, 4);
		transform->getInt(cert[0]);

		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], cert[0].gethash32());

		for (int k = 1; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// cert[k] = ckpt[i]^w[0]
			power(4 + i, w[0]);
			transform->copy(2, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// cert[k] *= ckpt[i + 2 * j]^w[j]
				power(4 + i + 2 * j, w[j]);
				transform->mul(2);
				transform->copy(2, 0);
			}
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

		certTime = timer::diffTime(timer::currentTime(), t2);

		return success;
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

private:
	static double percent(const double num, const double den) { return std::rint(num * 1000 / den) / 10; }

public:
	bool check(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const int depth)
	{
		// const int depth = 5;
		const size_t L = size_t(1) << depth;

		const size_t num_threads = set_num_threads(nthreads), num_regs = 4 + L;
		Transform * const transform = createTransform(b, n, num_threads, impl, num_regs);

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, n);

		const size_t esize = mpz_sizeinbase(exponent, 2), B = ((esize - 1) >> depth) + 1;

		std::cout << "depth = " << depth << ", L_PL = " << L << ", B_PL = " << B << ", ";

		gint cert[depth + 1];

		double testTime = 0, checkTime = 0, certTime = 0; const bool success = test(depth, exponent, cert, testTime, checkTime, certTime);
		if (!success) return false;

		gint v2; mpz_t p2; mpz_init(p2);
		double serverTime = 0; const uint64_t hv1srv = server(depth, exponent, cert, v2, p2, serverTime);
		if (hv1srv == 0) return false;

		const bool isPrp = cert[0].isOne();

		for (int i = 0; i <= depth; ++i) cert[i].clear();
		mpz_clear(exponent);

		double validTime = 0; const uint64_t hv1val = valid(B, v2, p2, validTime);
		if (hv1val == 0) return false;

		v2.clear();
		mpz_clear(p2);

		const double time = testTime + checkTime + certTime + serverTime + validTime;

		std::cout << "test: " << percent(testTime, time) << "%, check: " << (success ? "ok" : "FAILED") << " " << percent(checkTime, time)
				  << "%, cert: " << percent(certTime, time) << "%, srv: " << percent(serverTime, time) << "%, valid: " << percent(validTime, time) << "%, "
				  << "proof " << ((hv1srv == hv1val) ? "ok" : "FAILED") << ": " << std::hex << hv1srv << std::dec << std::endl;

		const double err = transform->getError();

		std::cout << b << "^" << n << " + 1 is " << (isPrp ? "prime" : "composite") << ", err = " << err << ", " << time << " sec." << std::endl;

		deleteTransform();

		return true;
	}
};
