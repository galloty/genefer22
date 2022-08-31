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
#include <thread>
#include <chrono>

#include <gmp.h>
#if !defined(GPU)
#include <omp.h>
#endif

#include "pio.h"
#include "file.h"
#include "timer.h"
#if defined(GPU)
#include "ocl.h"
#endif
#include "transform.h"

inline int ilog2_32(const uint32_t n) { return 31 - __builtin_clz(n); }

class genefer
{
public:
	enum class EMode { None, Quick, Proof, Server, Check, Bench, Limit }; 

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

protected:
	volatile bool _quit = false;
private:
	bool _isBoinc = false;
#if defined(GPU)
	cl_platform_id _boinc_platform_id = 0;
	cl_device_id _boinc_device_id = 0;
#endif
	transform * _transform = nullptr;
	gint * _gi = nullptr;
	std::string _mainFilename;
	int _print_range = 0, _print_i = 0;

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
	void setFilename(const std::string & mainFilename) { _mainFilename = mainFilename; }

private:
#if defined(GPU)
	void createTransformGPU(const uint32_t b, const uint32_t n, const size_t device, const size_t num_regs, const bool verbose = true)
	{
		deleteTransform();
		_transform = transform::create_gpu(b, n, _isBoinc, device, num_regs, _boinc_platform_id, _boinc_device_id, verbose);
		if (verbose)
		{
			std::ostringstream ss; ss << ", data size: " << std::setprecision(3) << _transform->getMemSize() / (1024 * 1024.0) << " MB." << std::endl;
			pio::print(ss.str());
		}
	}
#else
	void createTransformCPU(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const size_t num_regs, const bool verbose = true)
	{
		deleteTransform();

		if (nthreads > 1) omp_set_num_threads(int(nthreads));
		size_t num_threads = 1;
		if (nthreads != 1)
		{
#pragma omp parallel
			{
#pragma omp single
				num_threads = size_t(omp_get_num_threads());
			}
		}

		std::string ttype;
		_transform = transform::create_cpu(b, n, num_threads, impl, num_regs, ttype);
		if (verbose)
		{
			std::ostringstream ss; ss << "Using " << ttype << " implementation, " << num_threads << " thread(s), data size: "
									  << std::setprecision(3) << _transform->getCacheSize() / (1024 * 1024.0) << " MB." << std::endl;
			pio::print(ss.str());
		}
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

	std::string contextFilename() const { return _mainFilename + ".ctx"; }
	std::string proofFilename() const { return _mainFilename + ".proof"; }
	std::string sfvFilename() const { return _mainFilename + ".sfv"; }
	std::string certFilename() const { return _mainFilename + ".cert"; }
	std::string ckeyFilename() const { return _mainFilename + ".ckey"; }
	std::string skeyFilename() const { return _mainFilename + ".skey"; }
	std::string ckptFilename(const size_t i) const
	{
		std::ostringstream ss; ss << _mainFilename << "_" << i << ".ckpt";
		return ss.str();
	}

	static std::string uint64toString(const uint64_t u)
	{
		std::stringstream ss; ss << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u;
		return ss.str();
	}

	static std::string gfn(const uint32_t b, const uint32_t n)
	{
		std::ostringstream ss; ss << b << "^{2^" << n << "} + 1";
		return ss.str();
	}

	static std::string gfnStatus(const bool isPrp, const uint64_t res64, const uint64_t old64, const double time)
	{
		std::ostringstream ss; ss << " is ";
		if (isPrp) ss << "a probable prime"; else ss << "composite, res64 = " << uint64toString(res64) << ", old64 = " << uint64toString(old64);
		ss << ", time = " << timer::formatTime(time) << ".";
		return ss.str();
	}

	void initPrintProgress(const int i0, const int i_start)
	{
		_print_range = i0; _print_i = i_start;
		if (_isBoinc) boinc_fraction_done((i0 > i_start) ? double(i0 - i_start) / i0 : 0.0);
	}

	int printProgress(const double displayTime, const int i)
	{
		if (_print_i == i) return 1;
		const double mulTime = displayTime / (_print_i - i); _print_i = i;
		const double percent = double(_print_range - i) / _print_range;
		const int dcount = std::max(int(1.0 / mulTime), 2);
		if (_isBoinc) boinc_fraction_done(percent);
		else
		{
			const double estimatedTime = mulTime * i;
			std::ostringstream ss; ss << std::setprecision(3) << percent * 100.0 << "% done, " << timer::formatTime(estimatedTime)
									<< " remaining, " << mulTime * 1e3 << " ms/bit.        \r";
			pio::display(ss.str());
		}
		return dcount;
	}

	static void clearline() { pio::display("                                                \r"); }

	int readContext(const int where, int & i, double & elapsedTime)
	{
		file contextFile(contextFilename());
		if (!contextFile.exists()) return -1;
		int version = 0;
		if (!contextFile.read(reinterpret_cast<char *>(&version), sizeof(version))) return -2;
#if defined(GPU)
		version = -version;
#endif
		if (version != 1) return -2;
		int rwhere = 0;
		if (!contextFile.read(reinterpret_cast<char *>(&rwhere), sizeof(rwhere))) return -2;
		if (rwhere != where) return -2;
		if (!contextFile.read(reinterpret_cast<char *>(&i), sizeof(i))) return -2;
		if (!contextFile.read(reinterpret_cast<char *>(&elapsedTime), sizeof(elapsedTime))) return -2;
		const size_t num_regs = 2;
		return _transform->readContext(contextFile, num_regs) ? 0 : -2;
	}

	void saveContext(const int where, const int i, const double elapsedTime) const
	{
		file contextFile(contextFilename(), "wb", false);
		int version = 1;
#if defined(GPU)
		version = -version;
#endif
		if (!contextFile.write(reinterpret_cast<const char *>(&version), sizeof(version))) return;
		if (!contextFile.write(reinterpret_cast<const char *>(&where), sizeof(where))) return;
		if (!contextFile.write(reinterpret_cast<const char *>(&i), sizeof(i))) return;
		if (!contextFile.write(reinterpret_cast<const char *>(&elapsedTime), sizeof(elapsedTime))) return;
		const size_t num_regs = 2;
		_transform->saveContext(contextFile, num_regs);
	}

	static bool boincQuitRequest(const BOINC_STATUS & status)
	{
		if ((status.quit_request | status.abort_request | status.no_heartbeat) == 0) return false;

		std::ostringstream ss; ss << "Terminating because Boinc ";
		if (status.quit_request != 0) ss << "requested that we should quit.";
		else if (status.abort_request != 0) ss << "requested that we should abort.";
		else if (status.no_heartbeat != 0) ss << "heartbeat was lost.";
		ss << std::endl;
		pio::print(ss.str());
		return true;
	}

	void boincMonitor()
	{
		BOINC_STATUS status; boinc_get_status(&status);
		const bool bQuit = boincQuitRequest(status);
		if (bQuit) { quit(); return; }
			
		if (status.suspended != 0)
		{
			std::ostringstream ss_s; ss_s << "Boinc client is suspended." << std::endl;
			pio::print(ss_s.str());

			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boincQuitRequest(status)) { quit(); return; }
			}

			std::ostringstream ss_r; ss_r << "Boinc client is resumed." << std::endl;
			pio::print(ss_r.str());
		}
	}

	bool boincMonitor(const int where, const int i, watch & chrono)
	{
		BOINC_STATUS status; boinc_get_status(&status);
		const bool bQuit = boincQuitRequest(status);
		if (bQuit) quit();

		if (bQuit || (status.suspended != 0)) saveContext(where, i, chrono.getElapsedTime());
		if (bQuit) return true;
			
		if (status.suspended != 0)
		{
			std::ostringstream ss_s; ss_s << "Boinc client is suspended." << std::endl;
			pio::print(ss_s.str());

			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boincQuitRequest(status)) { quit(); return true; }
			}

			std::ostringstream ss_r; ss_r << "Boinc client is resumed." << std::endl;
			pio::print(ss_r.str());
		}

		if (boinc_time_to_checkpoint() != 0)
		{
			saveContext(where, i, chrono.getElapsedTime());
			boinc_checkpoint_completed();
		}
		return false;
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
		const size_t L = (2 << (ilog2_32(uint32_t(esize)) / 2));
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
		gint & gi = *_gi;

		int ri = 0; double restoredTime = 0;
		const int error = readContext(0, ri, restoredTime);
		const bool found = (error == 0);
		if (error < -1) pio::error("invalid context");

		if (!found)
		{
			pTransform->set(1);
			pTransform->copy(1, 0);	// d(t)
		}
		else
		{
			std::ostringstream ss; ss << "Resuming from a checkpoint." << std::endl;
			pio::print(ss.str());
			if (ri == -1)
			{
				testTime = 0;
				return true;
			}
		}

		watch chrono(found ? restoredTime : 0);
		const int i0 = int(mpz_sizeinbase(exponent, 2) - 1), i_start = found ? ri : i0;
		initPrintProgress(i0, i_start);
		int dcount = 100;

		for (int i = i_start; i >= 0; --i)
		{
			if (_quit)
			{
				saveContext(0, i, chrono.getElapsedTime());
				return false;
			}

			if (i % dcount == 0)
			{
				chrono.read(); const double displayTime = chrono.getDisplayTime();
				if (displayTime >= 10) { dcount = printProgress(displayTime, i); chrono.resetDisplayTime(); }

				if (_isBoinc) { if (boincMonitor(0, i, chrono)) return false; }
				else if (chrono.getRecordTime() > 600) { saveContext(0, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
			}

			pTransform->squareDup(mpz_tstbit(exponent, i) != 0);
			// if (i == int(mpz_sizeinbase(exponent, 2) - 1)) pTransform->add1();	// => invalid
			// if (i == 0) pTransform->add1();	// => invalid
			if ((i % B_GL == 0) && (i / B_GL != 0))
			{
				pTransform->copy(2, 0);
				pTransform->mul(1);	// d(t)
				pTransform->copy(1, 0);
				pTransform->copy(0, 2);
			}
			if ((B_PL != 0) && (i % B_PL == 0))
			{
				pTransform->getInt(gi);
				file ckptFile(ckptFilename(i / B_PL), "wb", true);
				gi.write(ckptFile);
				ckptFile.write_crc32();
			}
		}

		testTime = chrono.getElapsedTime();
		saveContext(0, -1, testTime);
		return true;
	}

	// Gerbicz-Li error checking
	// in: reg_0 is 2^exponent and reg_1 is d(t)
	// out: return valid/invalid
	bool GL(const mpz_t & exponent, const int B_GL, double & validTime)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		clearline(); pio::display("Validating...\r");

		watch chrono;

		// d(t + 1) = d(t) * result
		pTransform->mul(1);
		pTransform->copy(2, 0);

		// d(t)^{2^B}
		pTransform->copy(0, 1);
		for (int i = B_GL - 1; i >= 0; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) return false;
			pTransform->squareDup(false);
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
		for (int i = int(mpz_sizeinbase(res, 2)) - 1; i >= 0; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) { mpz_clear(res); return false; }

			pTransform->squareDup(mpz_tstbit(res, i) != 0);
		}

		mpz_clear(res);

		// d(t)^{2^B} * 2^res
		pTransform->mul(1);

		// d(t)^{2^B} * 2^res ?= d(t + 1)
		pTransform->getInt(gi);
		const uint64_t h1 = gi.gethash64();
		pTransform->copy(0, 2);
		pTransform->getInt(gi);
		const uint64_t h2 = gi.gethash64();

		const bool success = (h1 == h2);

		validTime = chrono.getElapsedTime();
		return success;
	}

	// (Pietrzak-Li proof generation
	// in: reg_{3 + i}, i in [0; 2^L[, // ckpt[i]
	// out: proof file
	bool PL(const int depth, double & proofTime)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		clearline(); pio::display("Generating proof...\r");

		watch chrono;

		file proofFile(proofFilename(), "wb", true);
		proofFile.write(reinterpret_cast<const char *>(&depth), sizeof(depth));

		// mu[0] = ckpt[0]
		{
			file ckptFile(ckptFilename(0), "rb", true);
			gi.read(ckptFile);
			ckptFile.check_crc32();
		}
		gi.write(proofFile);

		const size_t L = size_t(1) << depth;
		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], gi.gethash32());

		const int l0 = (1 << depth) - depth - 1;
// size_t s = 0;	// complexity

		for (int k = 1, l = l0; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// mu[k] = ckpt[i]^w[0]
			{
				file ckptFile(ckptFilename(i), "rb", true);
				gi.read(ckptFile);
				ckptFile.check_crc32();
				pTransform->setInt(gi);
			}
			power(0, w[0]);
// s += mpz_sizeinbase(w[0], 2);
			pTransform->copy(1, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// mu[k] *= ckpt[i + 2 * j]^w[j]
				{
					file ckptFile(ckptFilename(i + 2 * j), "rb", true);
					gi.read(ckptFile);
					ckptFile.check_crc32();
					pTransform->setInt(gi);
				}
				power(0, w[j]);
// s += mpz_sizeinbase(w[j], 2);
				pTransform->mul(1);
				pTransform->copy(1, 0);
				--l;

				if (_isBoinc) boincMonitor();
				if (_quit)
				{
					for (size_t i = 0; i < L / 2; ++i) mpz_clear(w[i]);
					delete[] w;
					return false;
				}
			}
// std::cout << k << ": " << s << ", " << 32 * k * (1 << (k - 1))<< std::endl;
			pTransform->getInt(gi);
			gi.write(proofFile);

			if (i > 1)
			{
				const uint32_t q = gi.gethash32();
				for (size_t j = 0; j < L / 2; j += i) mpz_mul_ui(w[i / 2 + j], w[j], q);
			}
		}

		proofFile.write_crc32();

		// if (_isBoinc)
		// {
		// 	file sfvFile(sfvFilename(), "w", false);
		// 	std::ostringstream ss; ss << proofFilename() << " " << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << proofFile.crc32() << std::endl;
		// 	sfvFile.print(ss.str().c_str());
		// }

		for (size_t i = 0; i < L / 2; ++i) mpz_clear(w[i]);
		delete[] w;

		proofTime = chrono.getElapsedTime();
		return true;
	}

	bool quick(const mpz_t & exponent, double & testTime, double & validTime, bool & isPrp, uint64_t & res64, uint64_t & old64)
	{
		const int B_GL = B_GerbiczLi(mpz_sizeinbase(exponent, 2));

		if (!prp(exponent, B_GL, 0, testTime)) return false;
		{
			gint & gi = *_gi;
			_transform->getInt(gi);
			isPrp = gi.isOne(res64, old64);
		}
		if (!GL(exponent, B_GL, validTime)) return false;
		return true;
	}

	bool proof(const mpz_t & exponent, const int depth, double & testTime, double & validTime, double & proofTime, bool & isPrp, uint64_t & res64, uint64_t & old64)
	{
		const size_t esize = mpz_sizeinbase(exponent, 2);
		const int B_GL = B_GerbiczLi(esize), B_PL = B_PietrzakLi(esize, depth);

		if (!prp(exponent, B_GL, B_PL, testTime)) return false;
		{
			gint & gi = *_gi;
			_transform->getInt(gi);
			isPrp = gi.isOne(res64, old64);
		}
		if (!GL(exponent, B_GL, validTime)) return false;
		if (!PL(depth, proofTime)) return false;
		return true;
	}

	bool server(const mpz_t & exponent, double & time, bool & isPrp, uint64_t & skey, uint64_t & res64, uint64_t & old64)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		watch chrono;

		file proofFile(proofFilename(), "rb", true);
		int depth = 0; proofFile.read(reinterpret_cast<char *>(&depth), sizeof(depth));

		const size_t L = size_t(1) << depth, esize = mpz_sizeinbase(exponent, 2);
		const int B = B_PietrzakLi(esize, depth);

		mpz_t * const w = new mpz_t[L]; for (size_t i = 0; i < L; ++i) mpz_init(w[i]);

		// v1 = mu[0]^w[0], v1: reg = 1
		gi.read(proofFile);
		isPrp = gi.isOne(res64, old64);
		const uint32_t q = gi.gethash32();
		mpz_set_ui(w[0], q);
		pTransform->setInt(gi);
		power(0, q);
		pTransform->copy(1, 0);

		// v2 = 1, v2: reg = 2
		pTransform->set(1);
		pTransform->copy(2, 0);

		for (int k = 1; k <= depth; ++k)
		{
			// mu[k]: reg = 3
			gi.read(proofFile);
			const uint32_t q = gi.gethash32();
			pTransform->setInt(gi);
			pTransform->copy(3, 0);

			// v1 = v1 * mu^q
			power(0, q);
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

		proofFile.check_crc32();

		// skey = hash64(v1);
		pTransform->copy(0, 1);
		pTransform->getInt(gi);
		skey = gi.gethash64();

		// v2
		pTransform->copy(0, 2);
		pTransform->getInt(gi);

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

		{
			file certFile(certFilename(), "wb", true);
			certFile.write(reinterpret_cast<const char *>(&B), sizeof(B));
			gi.write(certFile);
			certFile.write(p2);
			certFile.write_crc32();
		}

		mpz_clear(p2);

		time = chrono.getElapsedTime();
		return true;
	}

	bool check(double & time, uint64_t & ckey)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		int ri = 0; double restoredTime = 0;
		const int error = readContext(1, ri, restoredTime);
		const bool found = (error == 0);
		if (error < -1) pio::error("invalid context");
		if (found)
		{
			std::ostringstream ss; ss << "Resuming from a checkpoint." << std::endl;
			pio::print(ss.str());
		}

		int B = 0;
		mpz_t p2; mpz_init(p2);
		{
			file certFile(certFilename(), "rb", true);
			certFile.read(reinterpret_cast<char *>(&B), sizeof(B));
			gi.read(certFile);
			certFile.read(p2);
			certFile.check_crc32();
		}
		const int p2size = int(mpz_sizeinbase(p2, 2));

		watch chrono(found ? restoredTime : 0);
		const int i0 = p2size + B - 1;
		initPrintProgress(i0, found ? ri : i0);
		int dcount = 100;

		// v2 = v2^{2^B}
		if (!found || (ri >= p2size))
		{
			if (!found) pTransform->setInt(gi);
			for (int i = found ? ri : i0; i >= p2size; --i)
			{
				if (_quit)
				{
					saveContext(1, i, chrono.getElapsedTime());
					mpz_clear(p2);
					return false;
				}

				if (i % dcount == 0)
				{
					chrono.read(); const double displayTime = chrono.getDisplayTime();
					if (displayTime >= 10) { dcount = printProgress(displayTime, i); chrono.resetDisplayTime(); }
					if (_isBoinc) { if (boincMonitor(1, i, chrono)) return false; }
					else if (chrono.getRecordTime() > 600) { saveContext(1, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
				}

				pTransform->squareDup(false);
			}
			pTransform->copy(1, 0);
		}

		// v1' = v2 * 2^p2
		if (!found || (ri >= p2size)) pTransform->set(1);
		for (int i = found ? std::min(ri, p2size - 1) : p2size - 1; i >= 0; --i)
		{
			if (_quit)
			{
				saveContext(1, i, chrono.getElapsedTime());
				mpz_clear(p2);
				return false;
			}

			if (i % dcount == 0)
			{
				chrono.read(); const double displayTime = chrono.getDisplayTime();
				if (displayTime >= 10) { dcount = printProgress(displayTime, i); chrono.resetDisplayTime(); }
				if (_isBoinc) { if (boincMonitor(1, i, chrono)) return false; }
				else if (chrono.getRecordTime() > 600) { saveContext(1, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
			}

			pTransform->squareDup(mpz_tstbit(p2, i) != 0);
		}

		mpz_clear(p2);

		pTransform->mul(1);

		// ckey = hash64(v1')
		pTransform->getInt(gi);
		ckey = gi.gethash64();

		time = chrono.getElapsedTime();
		return true;
	}

	bool bench(const uint32_t m, const size_t device, const size_t nthreads, const std::string & impl)
	{
		static constexpr uint32_t bm[12] = { 500000000, 380000000, 290000000, 220000000, 160000000,
											 115000000, 16000000, 6000000, 2000000, 820000, 230000, 980000 };

		// NTT2 limits
		// static constexpr uint32_t bm[12] = { 46664208, 32996578, 23332104, 16498288, 11666052,
		// 									 8249144, 5833026, 4124572, 2916512, 2062286, 1458256, 1458256 };

		// DT limits
		// static constexpr uint32_t bm[12] = { 4200000, 3500000, 2800000, 2300000, 1900000,
		// 									 1600000, 1300000, 1100000, 880000, 730000, 600000, 600002 };

		const size_t num_regs = 3;

		const uint32_t b = bm[(m != 0) ? m - 12 : 11], n = (m != 0) ? m : 22;

#if defined(GPU)
		(void)nthreads; (void)impl;
		createTransformGPU(b, n, device, num_regs, m == 12);
#else
		(void)device;
		createTransformCPU(b, n, nthreads, impl, num_regs, m == 12);
#endif

		transform * const pTransform = _transform;

		_gi = new gint(1 << n, b);
		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, 3, 20);
		double testTime = 0, validTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
		const bool success = quick(exponent, testTime, validTime, isPrp, res64, old64);
		mpz_clear(exponent);
		std::remove(contextFilename().c_str());
		delete _gi; _gi = nullptr;

		pio::print(gfn(b, n));

		std::ostringstream ss;
		if (!success) ss << ": test failed!" << std::endl;
		else
		{
			static volatile bool _break;

			_break = false;
			std::thread oneSecond([=] { std::this_thread::sleep_for(std::chrono::seconds(5)); _break = true; }); oneSecond.detach();

			watch chrono(0);
			size_t i = 1;
			while (!_break)
			{
				pTransform->squareDup((i % 2) != 0);
				++i;
				if (_quit) break;
			}

			pTransform->copy(1, 0);	// synchro
			const double mulTime = chrono.getElapsedTime() / i, estimatedTime = mulTime * std::log2(b) * (1 << n);
			ss << ": " << std::setprecision(3) << timer::formatTime(estimatedTime) << ", " << mulTime * 1e3 << " ms/bit." << std::endl;
		}
		pio::print(ss.str());

		deleteTransform();
		return !_quit;
	}

	bool check_limit(const uint32_t n, const size_t device, const size_t nthreads, const std::string & impl)
	{
		const size_t num_regs = 3;

		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, 3, 1000);

		uint32_t b_min = 100000, b_max = 2000000000;
		while (b_max - b_min > 5000)
		{
			const uint32_t b = (b_min + b_max) / 4 * 2;

#if defined(GPU)
			(void)nthreads; (void)impl;
			createTransformGPU(b, n, device, num_regs, false);
#else
			(void)device;
			createTransformCPU(b, n, nthreads, impl, num_regs, false);
#endif

			_gi = new gint(1 << n, b);

			double testTime = 0, validTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
			const bool success = quick(exponent, testTime, validTime, isPrp, res64, old64);
			if (success) b_min = b;
			else if (!_quit) b_max = b;

			// std::ostringstream ss; ss << n << ", " << b << ": " << (success ? 1 : 0) << std::endl;
			// pio::print(ss.str());

			std::remove(contextFilename().c_str());
			delete _gi; _gi = nullptr;
			deleteTransform();

			if (_quit) break;
		}

		mpz_clear(exponent);

		const uint32_t b = (b_min + b_max) / 4 * 2;
		std::ostringstream ss; ss << n << ": " << b << std::endl;
		pio::print(ss.str());

		return !_quit;
	}

public:
	bool check(const uint32_t b, const uint32_t n, const EMode mode, const size_t device, const size_t nthreads, const std::string & impl, const int depth)
	{
		const bool emptyMainFilename = _mainFilename.empty();
		if (emptyMainFilename)
		{
			std::ostringstream ss; ss << "g" << n << "_" << b;
			_mainFilename = ss.str();
		}

		if (mode == EMode::Bench) return bench(n, device, nthreads, impl);
		if (mode == EMode::Limit) return check_limit(n, device, nthreads, impl);

		size_t num_regs;
		if (mode == EMode::Quick) num_regs = 3;
		else if (mode == EMode::Proof) num_regs = 3;
		else if (mode == EMode::Server) num_regs = 4;
		else if (mode == EMode::Check) num_regs = 2;
		else return false;

#if defined(GPU)
		(void)nthreads; (void)impl;
		createTransformGPU(b, n, device, num_regs);
#else
		(void)device;
		createTransformCPU(b, n, nthreads, impl, num_regs);

		static constexpr uint32_t bm[22 - 12 + 1] = { 500, 380, 290, 220, 160, 125, 94, 71, 54, 41, 31 };
		if (b > bm[n - 12] * 1000000)
		{
			std::ostringstream ss; ss << "Warning: b > " << bm[n - 12] << ",000,000: the test may fail." << std::endl;
			pio::print(ss.str());
		}
#endif

		_gi = new gint(1 << n, b);

		bool success = false;

		if (mode == EMode::Check)
		{
			double time = 0; uint64_t ckey = 0;
			success = check(time, ckey);
			if (success)
			{
				file ckeyFile(ckeyFilename(), "w", false);
				ckeyFile.print(uint64toString(ckey).c_str());
			}
			clearline();
			std::ostringstream ss; ss << gfn(b, n) << ": ";
			if (success) ss << "checked, time = " << timer::formatTime(time) << "." << std::endl << "ckey = " << uint64toString(ckey) << ".";
			else if (!_quit) ss << "check failed!";
			else ss << "terminated.";
			ss << std::endl; pio::print(ss.str());
			if (success && !_isBoinc)
			{
				std::remove(contextFilename().c_str());
			}
		}
		else
		{
			mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, 1 << n);

			if (mode == EMode::Quick)
			{
				double testTime = 0, validTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
				success = quick(exponent, testTime, validTime, isPrp, res64, old64);
				clearline();
				std::ostringstream ss; ss << gfn(b, n);
				if (success) ss << gfnStatus(isPrp, res64, old64, testTime + validTime);
				else if (!_quit) ss << ": validation failed!";
				else ss << ": terminated.";
				ss << std::endl; pio::print(ss.str());
				if (success)
				{
					pio::result(ss.str());
					if (!_isBoinc) std::remove(contextFilename().c_str());
				}
			}
			else if (mode == EMode::Proof)
			{
				double testTime = 0, validTime = 0, proofTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
				success = proof(exponent, depth, testTime, validTime, proofTime, isPrp, res64, old64);
				const double time = testTime + validTime + proofTime;
				clearline();
				std::ostringstream ss; ss << gfn(b, n) << ": ";
				if (success) ss << "proof file is generated, time = " << timer::formatTime(time) << ".";
				else if (!_quit) ss << "validation failed!";
				else ss << "terminated.";
				ss << std::endl; pio::print(ss.str());
				if (success)
				{
					std::ostringstream ssr; ssr << gfn(b, n) << gfnStatus(isPrp, res64, old64, time) << std::endl;
					pio::result(ssr.str());
					if (!_isBoinc)
					{
						for (size_t i = 0, L = size_t(1) << depth; i < L; ++i) std::remove(ckptFilename(i).c_str());
						std::remove(contextFilename().c_str());
					}
				}
			}
			else if (mode == EMode::Server)
			{
				double time = 0; uint64_t skey = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
				success = server(exponent, time, isPrp, skey, res64, old64);
				if (success)
				{
					file skeyFile(skeyFilename(), "w", false);
					skeyFile.print(uint64toString(skey).c_str());
				}
				std::ostringstream ss; ss << gfn(b, n);
				if (success) ss << gfnStatus(isPrp, res64, old64, time) << std::endl << "skey = " << uint64toString(skey) << ".";
				else if (!_quit) ss << ": generation failed!";
				else ss << ": terminated.";
				ss << std::endl; pio::print(ss.str());
			}

			mpz_clear(exponent);
		}

		delete _gi; _gi = nullptr;
		deleteTransform();
		if (emptyMainFilename) _mainFilename.clear();

		return success;
	}
};
