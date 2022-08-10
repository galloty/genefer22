/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <chrono>

#include <gmp.h>
#if !defined(GPU)
#include <omp.h>
#endif

#include "pio.h"
#include "file.h"
#if defined(GPU)
#include "ocl.h"
#endif
#include "transform.h"

inline int ilog2_32(const uint32_t n) { return 31 - __builtin_clzl(n); }
inline int ilog2_64(const uint64_t n) { return 63 - __builtin_clzll(n); }

class genefer
{
public:
	enum class EMode { None, Quick, Proof, Server, Check }; 

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
#if defined(GPU)
	void createTransformGPU(const uint32_t b, const uint32_t n, const size_t device, const size_t num_regs)
	{
		deleteTransform();
		_transform = transform::create_gpu(b, n, _isBoinc, device, num_regs, _boinc_platform_id, _boinc_device_id);
		std::ostringstream ss; ss << "Using " << _transform->getMemSize() / (1 << 20) << " MB." << std::endl;
		pio::print(ss.str());
	}
#else
	void createTransformCPU(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const size_t num_regs)
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
		std::ostringstream ss; ss << "Using " << ttype << " implementation, " << num_threads << " thread(s), " << _transform->getMemSize() / (1 << 20) << " MB." << std::endl;
		pio::print(ss.str());
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

	static std::string proofFilename(const uint32_t b, const uint32_t n)
	{
		std::ostringstream ss; ss << "g" << n << "_" << b << ".proof";
		return ss.str();
	}

	static std::string certFilename(const uint32_t b, const uint32_t n)
	{
		std::ostringstream ss; ss << "g" << n << "_" << b << ".cert";
		return ss.str();
	}

	static std::string uint64toString(const uint64_t u64)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u64;
		return ss.str();
	}

	static std::string formatTime(const double time)
	{
		uint64_t seconds = uint64_t(time), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::ostringstream ss;
		ss << std::setfill('0') << std::setw(2) <<  hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
		return ss.str();
	}

	static int printProgress(const char * const mode, const double elapsedTime, const int i0, const int i, int & id,
							 const std::chrono::high_resolution_clock::time_point & t, std::chrono::high_resolution_clock::time_point & td,
							 const double percent_start = 0, const double percent_end = 100)
	{
		if (id == i) return 1;
		const double mulTime = elapsedTime / (id - i), estimatedTime = mulTime * i;
		const double percent = (i0 - i) * (percent_end - percent_start) / i0 + percent_start;
		const int dcount = std::max(int(0.2 / mulTime), 2);
		std::ostringstream ss; ss << "\33[2K" << mode << ": " << std::setprecision(3) << percent << "% done, "
								  << formatTime(estimatedTime) << " remaining, " << mulTime * 1e3 << " ms/bit.        \r";
		pio::display(ss.str());
		td = t; id = i;
		return dcount;
	}

	void power(const size_t reg, const uint32_t e) const
	{
		transform * const pTransform = _transform;
		pTransform->initMultiplicand(reg);
		pTransform->set(1);
		for (int i = ilog2_32(e); i >= 0; --i)
		{
			pTransform->squareDup(false);
			if ((e & (uint32_t(1) << i)) != 0) pTransform->mul();
		}
	}

	void power(const size_t reg, const mpz_t & e) const
	{
		transform * const pTransform = _transform;
		pTransform->initMultiplicand(reg);
		pTransform->set(1);
		for (int i = int(mpz_sizeinbase(e, 2) - 1); i >= 0; --i)
		{
			pTransform->squareDup(false);
			if (mpz_tstbit(e, i) != 0) pTransform->mul();
		}
	}

	static int B_GerbiczLi(const size_t esize)
	{
		const size_t L = (2 << (ilog2_64(esize) / 2));
		return int((esize - 1) / L) + 1;
	}

	static int B_PietrzakLi(const size_t esize, const int depth)
	{
		return int((esize - 1) >> depth) + 1;
	}

	// out: reg_0 is 2^exponent and reg_1 is d(t)
	bool prp(const mpz_t & exponent, const int B_GL, const int B_PL, double & testTime)
	{
		transform * const pTransform = _transform;

		const auto t0 = std::chrono::high_resolution_clock::now();

		pTransform->set(1);
		pTransform->copy(1, 0);	// d(t)
		const int i0 = int(mpz_sizeinbase(exponent, 2) - 1);
		auto td = std::chrono::high_resolution_clock::now();
		int id = i0, dcount = 100;
		for (int i = i0; i >= 0; --i)
		{
			pTransform->squareDup(mpz_tstbit(exponent, i) != 0);
			// if (i == int(mpz_sizeinbase(exponent, 2) - 1)) pTransform->add1();	// => invalid
			// if (i == 0) pTransform->add1();	// => invalid
			if (i % B_GL == 0)
			{
				if (i / B_GL != 0)
				{
					pTransform->copy(2, 0);
					pTransform->mul(1);	// d(t)
					pTransform->copy(1, 0);
					pTransform->copy(0, 2);
				}
			}
			if ((B_PL != 0) && (i % B_PL == 0))
			{
				const size_t reg = 3 + i / B_PL;	// ckpt[i]
				pTransform->copy(reg, 0);
			}
			if (i % dcount == 0)
			{
				const auto t = std::chrono::high_resolution_clock::now();
				const double elapsedTime = (t - td).count() * 1e-9;
				if (elapsedTime >= 1) dcount = printProgress("Test", elapsedTime, i0, i, id, t, td);
			}
			if (_quit) return false;
		}

		testTime = (std::chrono::high_resolution_clock::now() - t0).count() * 1e-9;
		return true;
	}

	// Gerbicz-Li error checking
	// in: reg_0 is 2^exponent and reg_1 is d(t)
	// out: return valid/invalid
	bool GL(const mpz_t & exponent, const int B_GL, double & validTime)
	{
		transform * const pTransform = _transform;

		const auto t0 = std::chrono::high_resolution_clock::now();

		// d(t + 1) = d(t) * result
		pTransform->mul(1);
		pTransform->copy(2, 0);

		// d(t)^{2^B}
		pTransform->copy(0, 1);
		{
			const int i0 = B_GL - 1;
			auto td = std::chrono::high_resolution_clock::now();
			int id = i0, dcount = 100;
			for (int i = i0; i >= 0; --i)
			{
				pTransform->squareDup(false);
				if (i % dcount == 0)
				{
					const auto t = std::chrono::high_resolution_clock::now();
					const double elapsedTime = (t - td).count() * 1e-9;
					if (elapsedTime >= 1) dcount = printProgress("Valid", elapsedTime, i0, i, id, t, td, 0, 50);
				}
				if (_quit) return false;
			}
		}
		pTransform->copy(1, 0);

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
		{
			const int i0 = int(mpz_sizeinbase(res, 2) - 1);
			auto td = std::chrono::high_resolution_clock::now();
			int id = i0, dcount = 100;
			for (int i = i0; i >= 0; --i)
			{
				pTransform->squareDup(mpz_tstbit(res, i) != 0);
				if (i % dcount == 0)
				{
					const auto t = std::chrono::high_resolution_clock::now();
					const double elapsedTime = (t - td).count() * 1e-9;
					if (elapsedTime >= 1) dcount = printProgress("Valid", elapsedTime, i0, i, id, t, td, 50, 100);
				}
				if (_quit) return false;
			}
		}
		mpz_clear(res);

		// d(t)^{2^B} * 2^res
		pTransform->mul(1);

		// d(t)^{2^B} * 2^res ?= d(t + 1)
		gint v(pTransform->getSize(), pTransform->getB());
		pTransform->getInt(v);
		const uint64_t h1 = v.gethash64();
		pTransform->copy(0, 2);
		pTransform->getInt(v);
		const uint64_t h2 = v.gethash64();

		const bool success = (h1 == h2);

		validTime = (std::chrono::high_resolution_clock::now() - t0).count() * 1e-9;
		return success;
	}

	// (Pietrzak-Li proof generation
	// in: reg_{3 + i}, i in [0; 2^L[, // ckpt[i]
	// out: proof file
	bool PL(const uint32_t b, const uint32_t n, const int depth, double & proofTime)
	{
		transform * const pTransform = _transform;

		const auto t0 = std::chrono::high_resolution_clock::now();

		file proofFile(proofFilename(b, n).c_str(), "wb");
	
		// generate mu_k
		gint mu_k(pTransform->getSize(), pTransform->getB());

		// mu[0] = ckpt[0]
		pTransform->copy(0, 3);
		pTransform->getInt(mu_k);
		mu_k.write(proofFile);

		const size_t L = size_t(1) << depth;
		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], mu_k.gethash32());

		const int l0 = (1 << depth) - depth - 1;
		auto td = std::chrono::high_resolution_clock::now();
		int ld = l0;
		for (int k = 1, l = l0; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// mu[k] = ckpt[i]^w[0]
			power(3 + i, w[0]);
			pTransform->copy(1, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// mu[k] *= ckpt[i + 2 * j]^w[j]
				power(3 + i + 2 * j, w[j]);
				pTransform->mul(1);
				pTransform->copy(1, 0);
				--l;
				if (_quit) return false;
			}

			pTransform->getInt(mu_k);
			mu_k.write(proofFile);

			if (i > 1)
			{
				const uint32_t q = mu_k.gethash32();
				for (size_t j = 0; j < L / 2; j += i) mpz_mul_ui(w[i / 2 + j], w[j], q);
			}

			const auto t = std::chrono::high_resolution_clock::now();
			const double elapsedTime = (t - td).count() * 1e-9;
			if (elapsedTime >= 1) printProgress("Proof", elapsedTime, l0, l, ld, t, td, 0, 100);

			if (_quit) return false;
		}

		for (size_t i = 0; i < L / 2; ++i) mpz_clear(w[i]);
		delete[] w;

		proofTime = (std::chrono::high_resolution_clock::now() - t0).count() * 1e-9;
		return true;
	}

	bool quick(const mpz_t & exponent, double & testTime, double & validTime, bool & isPrp)
	{
		const int B_GL = B_GerbiczLi(mpz_sizeinbase(exponent, 2));

		if (!prp(exponent, B_GL, 0, testTime)) return false;
		{
			gint res(_transform->getSize(), _transform->getB()); _transform->getInt(res);
			isPrp = res.isOne();
		}
		if (!GL(exponent, B_GL, validTime)) return false;
		return true;
	}

	bool proof(const uint32_t b, const uint32_t n, const mpz_t & exponent, const int depth, double & testTime, double & validTime, double & proofTime)
	{
		const size_t esize = mpz_sizeinbase(exponent, 2);
		const int B_GL = B_GerbiczLi(esize), B_PL = B_PietrzakLi(esize, depth);

		if (!prp(exponent, B_GL, B_PL, testTime)) return false;
		if (!GL(exponent, B_GL, validTime)) return false;
		if (!PL(b, n, depth, proofTime)) return false;
		return true;
	}

	bool server(const uint32_t b, const uint32_t n, const mpz_t & exponent, const int depth, double & time, uint64_t & skey)
	{
		transform * const pTransform = _transform;

		const auto t0 = std::chrono::high_resolution_clock::now();

		const size_t L = size_t(1) << depth, esize = mpz_sizeinbase(exponent, 2);
		const int B = B_PietrzakLi(esize, depth);

		mpz_t * const w = new mpz_t[L]; for (size_t i = 0; i < L; ++i) mpz_init(w[i]);

		gint g(pTransform->getSize(), pTransform->getB());

		file proofFile(proofFilename(b, n).c_str(), "rb");

		// v1 = mu[0]^w[0], v1: reg = 1
		g.read(proofFile);
		const uint32_t q = g.gethash32();
		mpz_set_ui(w[0], q);
		pTransform->setInt(g);
		power(0, q);
		pTransform->copy(1, 0);

		// v2 = 1, v2: reg = 2
		pTransform->set(1);
		pTransform->copy(2, 0);

		for (int k = 1; k <= depth; ++k)
		{
			// mu[k]: reg = 3
			g.read(proofFile);
			const uint32_t q = g.gethash32();
			pTransform->setInt(g);
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

			if (_quit) return false;
		}

		// skey = hash64(v1);
		pTransform->copy(0, 1);
		gint v(pTransform->getSize(), pTransform->getB()); pTransform->getInt(v);
		skey = v.gethash64();

		mpz_t p2; mpz_init_set_ui(p2, 0);
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

		// v2
		pTransform->copy(0, 2);
		pTransform->getInt(g);

		file certFile(certFilename(b, n).c_str(), "wb");
		certFile.write(reinterpret_cast<const char *>(&B), sizeof(B));
		g.write(certFile);
		certFile.write(p2);

		mpz_clear(p2);

		time = (std::chrono::high_resolution_clock::now() - t0).count() * 1e-9;
		return true;
	}

	bool check(const uint32_t b, const uint32_t n, double & time, uint64_t & ckey)
	{
		transform * const pTransform = _transform;

		const auto t0 = std::chrono::high_resolution_clock::now();

		file certFile(certFilename(b, n).c_str(), "rb");
		int B = 0; certFile.read(reinterpret_cast<char *>(&B), sizeof(B));
		gint v2(pTransform->getSize(), pTransform->getB()); v2.read(certFile);
		mpz_t p2; mpz_init(p2); certFile.read(p2);

		// v2 = v2^{2^B}
		pTransform->setInt(v2);
		{
			const int i0 = B - 1;
			auto td = std::chrono::high_resolution_clock::now();
			int id = i0, dcount = 100;
			for (int i = i0; i >= 0; --i)
			{
				pTransform->squareDup(false);
				if (i % dcount == 0)
				{
					const auto t = std::chrono::high_resolution_clock::now();
					const double elapsedTime = (t - td).count() * 1e-9;
					if (elapsedTime >= 1) dcount = printProgress("Check", elapsedTime, i0, i, id, t, td, 0, 50);
				}
				if (_quit) return false;
			}
		}
		pTransform->copy(2, 0);

		// v1' = v2 * 2^p2
		pTransform->set(1);
		{
			const int i0 = int(mpz_sizeinbase(p2, 2) - 1);
			auto td = std::chrono::high_resolution_clock::now();
			int id = i0, dcount = 100;
			for (int i = i0; i >= 0; --i)
			{
				pTransform->squareDup(mpz_tstbit(p2, i) != 0);
				if (i % dcount == 0)
				{
					const auto t = std::chrono::high_resolution_clock::now();
					const double elapsedTime = (t - td).count() * 1e-9;
					if (elapsedTime >= 1) dcount = printProgress("Check", elapsedTime, i0, i, id, t, td, 50, 100);
				}
				if (_quit) return false;
			}
		}
		pTransform->mul(2);

		mpz_clear(p2);

		// ckey = hash64(v1')
		gint v(pTransform->getSize(), pTransform->getB()); pTransform->getInt(v);
		ckey = v.gethash64();

		time = (std::chrono::high_resolution_clock::now() - t0).count() * 1e-9;
		return true;
	}

private:
	static double percent(const double num, const double den) { return std::rint(num * 1000 / den) / 10; }

public:
	bool check(const uint32_t b, const uint32_t n, const EMode mode, const size_t device, const size_t nthreads, const std::string & impl, const int depth)
	{
		size_t num_regs;
		if (mode == EMode::Quick) num_regs = 3;
		if (mode == EMode::Proof) num_regs = 3 + (size_t(1) << depth);
		if (mode == EMode::Server) num_regs = 4;
		if (mode == EMode::Check) num_regs = 3;
		if (mode == EMode::None) num_regs = 3 + (size_t(1) << depth);

#if defined(GPU)
		(void)nthreads; (void)impl;
		createTransformGPU(b, n, device, num_regs);
#else
		(void)device;
		createTransformCPU(b, n, nthreads, impl, num_regs);
#endif

		bool success = true;

		if (mode == EMode::Check)
		{
			double time = 0;
			uint64_t ckey = 0;
			success = check(b, n, time, ckey);
			std::ostringstream ss; ss << b << "^{2^" << n << "} + 1: ";
			if (success) ss << "checked, time = " << formatTime(time) << ", ckey = " << uint64toString(ckey) << ".";
			else if (!_quit) ss << "check failed!";
			else ss << "terminated.";
			ss << std::endl; pio::print(ss.str());
		}
		else
		{
			mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, 1 << n);

			if (mode == EMode::Quick)
			{
				double testTime = 0, validTime = 0;
				bool isPrp = false;
				success = quick(exponent, testTime, validTime, isPrp);
				std::ostringstream ss; ss << "\33[2K" << b << "^{2^" << n << "} + 1";
				if (success) ss << " is " << (isPrp ? "a probable prime" : "composite") << ", time = " << formatTime(testTime + validTime) << ".";
				else if (!_quit) ss << ": validation failed!";
				else ss << ": terminated.";
				ss << std::endl; pio::print(ss.str());
				if (success)
				{
					std::ostringstream ss; ss << b << "^{2^" << n << "} + 1 is " << (isPrp ? "a probable prime" : "composite") << "." << std::endl;
					pio::result(ss.str());
				}
			}
			else if (mode == EMode::Proof)
			{
				double testTime = 0, validTime = 0, proofTime = 0;
				success = proof(b, n, exponent, depth, testTime, validTime, proofTime);
				std::ostringstream ss; ss << "\33[2K" << b << "^{2^" << n << "} + 1: ";
				if (success) ss << "proof file is generated, time = " << formatTime(testTime + validTime + proofTime) << ".";
				else if (!_quit) ss << "validation failed!";
				else ss << "terminated.";
				ss << std::endl; pio::print(ss.str());
			}
			else if (mode == EMode::Server)
			{
				double time = 0;
				uint64_t skey = 0;
				success = server(b, n, exponent, depth, time, skey);
				std::ostringstream ss; ss << b << "^{2^" << n << "} + 1: ";
				if (success) ss << "certificate file is generated, time = " << formatTime(time) << ", skey = " << uint64toString(skey) << ".";
				else if (!_quit) ss << "generation failed!";
				else ss << "terminated.";
				ss << std::endl; pio::print(ss.str());
			}
			else if (mode == EMode::Check)	// no exponent!
			{
				double time = 0;
				uint64_t ckey = 0;
				success = check(b, n, time, ckey);
				std::ostringstream ss; ss << b << "^{2^" << n << "} + 1: ";
				if (success) ss << "checked, time = " << formatTime(time) << ", ckey = " << uint64toString(ckey) << ".";
				else if (!_quit) ss << "check failed!";
				else ss << "terminated.";
				ss << std::endl; pio::print(ss.str());
			}

			mpz_clear(exponent);
		}

		deleteTransform();

		return success;
	}
};
