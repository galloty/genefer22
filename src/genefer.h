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
#include <ctime>
#include <sys/stat.h>

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
	bool _print_sr = true;

public:
	void quit() { if (_isBoinc) exit(EXIT_SUCCESS); _quit = true; }
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
	void createTransformGPU(const uint32_t b, const uint32_t n, const size_t device, const size_t num_regs,
							const bool verbose = true, const bool full = true)
	{
		deleteTransform();
		_transform = transform::create_gpu(b, n, _isBoinc, device, num_regs, _boinc_platform_id, _boinc_device_id, verbose);
		if (verbose)
		{
			std::ostringstream ss;
			if (full) ss << ", data size: " << std::setprecision(3) << _transform->getMemSize() / (1024 * 1024.0) << " MB";
			ss << "." << std::endl;
			pio::print(ss.str());
		}
	}
#else
	void createTransformCPU(const uint32_t b, const uint32_t n, const size_t nthreads, const std::string & impl, const size_t num_regs,
							const bool checkError, const bool verbose = true, const bool full = true)
	{
		deleteTransform();

		if (nthreads > 1) omp_set_num_threads(static_cast<int>(nthreads));
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
		_transform = transform::create_cpu(b, n, num_threads, impl, num_regs, checkError, ttype);
		if (verbose)
		{
			std::ostringstream ss; ss << "Using " << ttype << " implementation, " << num_threads << " thread(s)";
			if (full) ss << ", data size: " << std::setprecision(3) << _transform->getCacheSize() / (1024 * 1024.0) << " MB";
			ss << "." << std::endl;
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
	// std::string ckeyFilename() const { return _mainFilename + ".ckey"; }
	// std::string skeyFilename() const { return _mainFilename + ".skey"; }
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

	static std::string gfnStatus(const bool isPrp, const uint64_t pkey, const uint64_t ckey, const uint64_t res64, const uint64_t old64, const double error, const double time)
	{
		std::ostringstream ss; ss << " is ";
		if (isPrp) ss << "a probable prime"; else ss << "composite, res64 = " << uint64toString(res64) << ", old64 = " << uint64toString(old64);
		if (pkey != 0) ss << ", pkey = " << uint64toString(pkey);
		if (ckey != 0) ss << ", ckey = " << uint64toString(ckey);
		if (error != 0) ss << ", error = " << std::setprecision(4) << error;
		ss << ", time = " << timer::formatTime(time) << ".";
		return ss.str();
	}

	void initPrintProgress(const int i0, const int i_start)
	{
		_print_range = i0; _print_i = i_start;
		if (_isBoinc) boinc_fraction_done((i0 > i_start) ? static_cast<double>(i0 - i_start) / i0 : 0.0);
	}

	int printProgress(const double displayTime, const int i)
	{
		if (_print_i == i) return 1;
		const double mulTime = displayTime / (_print_i - i); _print_i = i;
		const double percent = static_cast<double>(_print_range - i) / _print_range;
		const int dcount = std::max(static_cast<int>(1.0 / mulTime), 2);
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

	int readContext(const int where, const bool fast_checkpoints, int & i, double & elapsedTime)
	{
		std::string ctxFile = contextFilename();
		struct stat s;
		if (stat(ctxFile.c_str(), &s) != 0) ctxFile += ".old";	// not found, try old one

		file contextFile(ctxFile);
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
		if (!_transform->readContext(contextFile, fast_checkpoints ? 0 : 2)) return -2;
		if (!contextFile.check_crc32()) return -2;
		return 0;
	}

	void saveContext(const int where, const bool fast_checkpoints, const int i, const double elapsedTime) const
	{
		const std::string ctxFile = contextFilename(), oldCtxFile = ctxFile + ".old", newCtxFile = ctxFile + ".new";

		std::remove(oldCtxFile.c_str());

		{
			file contextFile(newCtxFile, "wb", false);
			int version = 1;
#if defined(GPU)
			version = -version;
#endif
			if (!contextFile.write(reinterpret_cast<const char *>(&version), sizeof(version))) return;
			if (!contextFile.write(reinterpret_cast<const char *>(&where), sizeof(where))) return;
			if (!contextFile.write(reinterpret_cast<const char *>(&i), sizeof(i))) return;
			if (!contextFile.write(reinterpret_cast<const char *>(&elapsedTime), sizeof(elapsedTime))) return;
			_transform->saveContext(contextFile, fast_checkpoints ? 0 : 2);
			contextFile.write_crc32();
		}

		struct stat s;
		if ((stat(ctxFile.c_str(), &s) == 0) && (std::rename(ctxFile.c_str(), oldCtxFile.c_str()) != 0))	// file exists and cannot rename it
		{
			pio::error("cannot save context");
			return;
		}

		if (std::rename(newCtxFile.c_str(), ctxFile.c_str()) != 0)
		{
			pio::error("cannot save context");
			return;
		}
	}

	void clearContext() const
	{
		const std::string ctxFile = contextFilename();
		std::remove(ctxFile.c_str());
		std::remove(std::string(ctxFile + ".old").c_str());
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

	void printState(const bool suspended)
	{
		if (_print_sr)
		{
			std::ostringstream ss_s; ss_s << "Boinc client is " << (suspended ? "suspended." : "resumed.") << std::endl;
			pio::print(ss_s.str());
		}
		if (!suspended) _print_sr = false;
	}

	void boincMonitor()
	{
		BOINC_STATUS status; boinc_get_status(&status);
		if (boincQuitRequest(status)) exit(EXIT_SUCCESS);

		if (status.suspended != 0)
		{
			printState(true);
			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boincQuitRequest(status)) exit(EXIT_SUCCESS);
			}
			printState(false);
		}
	}

	bool boincMonitor(const int where, const bool fast_checkpoints, const int i, watch & chrono)
	{
		BOINC_STATUS status; boinc_get_status(&status);
		const bool bQuit = boincQuitRequest(status);

		if (bQuit || (status.suspended != 0)) saveContext(where, fast_checkpoints, i, chrono.getElapsedTime());
		if (bQuit) exit(EXIT_SUCCESS);

		if (status.suspended != 0)
		{
			printState(true);
			while (status.suspended != 0)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				boinc_get_status(&status);
				if (boincQuitRequest(status)) exit(EXIT_SUCCESS);
			}
			printState(false);
		}

		if (boinc_time_to_checkpoint() != 0)
		{
			saveContext(where, fast_checkpoints, i, chrono.getElapsedTime());
			boinc_checkpoint_completed();
		}
		return false;
	}

	void power(const size_t reg, const uint32_t e) const
	{
		transform * const pTransform = _transform;
		pTransform->initMultiplicand(reg);
		pTransform->set(1);
		if (e == 0) return;
		for (int i = ilog2_32(e); i >= 0; --i)
		{
			pTransform->squareDup(false);
			if ((e & (static_cast<uint32_t>(1) << i)) != 0) pTransform->mul();
		}
	}

	void powerz(const size_t reg, const mpz_t & e) const
	{
		transform * const pTransform = _transform;
		pTransform->initMultiplicand(reg);
		pTransform->set(1);
		if (mpz_sgn(e) == 0) return;
		for (int i = static_cast<int>(mpz_sizeinbase(e, 2) - 1); i >= 0; --i)
		{
			pTransform->squareDup(false);
			if (mpz_tstbit(e, mp_bitcnt_t(i)) != 0) pTransform->mul();
		}
	}

	static int B_GerbiczLi(const size_t esize)
	{
		const size_t L = (size_t(2) << (ilog2_32(static_cast<uint32_t>(esize)) / 2));
		return static_cast<int>((esize - 1) / L) + 1;
	}

	static int B_PietrzakLi(const size_t esize, const int depth)
	{
		return static_cast<int>((esize - 1) >> depth) + 1;
	}

	// out: reg_0 is 2^exponent and reg_1 is d(t)
	bool prp(const mpz_t & exponent, const int B_GL, const int B_PL, const bool fast_checkpoints, double & testTime)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		int ri = 0; double restoredTime = 0;
		const int error = readContext(0, fast_checkpoints, ri, restoredTime);
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
		const int i0 = static_cast<int>(mpz_sizeinbase(exponent, 2) - 1), i_start = found ? ri : i0;
		initPrintProgress(i0, i_start);
		int dcount = 100;

		for (int i = i_start; i >= 0; --i)
		{
			if (_quit)
			{
				saveContext(0, fast_checkpoints, i, chrono.getElapsedTime());
				return false;
			}

			if (i % dcount == 0)
			{
				chrono.read(); const double displayTime = chrono.getDisplayTime();
				if (displayTime >= 10) { dcount = printProgress(displayTime, i); chrono.resetDisplayTime(); }

				if (_isBoinc) { if (boincMonitor(0, fast_checkpoints, i, chrono)) return false; }
				else if (chrono.getRecordTime() > 600) { saveContext(0, fast_checkpoints, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
			}

			pTransform->squareDup(mpz_tstbit(exponent, mp_bitcnt_t(i)) != 0);
			// if (i == static_cast<int>(mpz_sizeinbase(exponent, 2) - 1)) pTransform->add1();	// => invalid
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
				if (fast_checkpoints) pTransform->copy(3 + size_t(i / B_PL), 0);
				else
				{
					pTransform->getInt(gi);
					file ckptFile(ckptFilename(size_t(i / B_PL)), "wb", true);
					gi.write(ckptFile);
					ckptFile.write_crc32();
				}
			}
		}

		testTime = chrono.getElapsedTime();
		saveContext(0, fast_checkpoints, -1, testTime);
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
			mpz_mod_2exp(t, e, static_cast<unsigned long int>(B_GL));
			mpz_add(res, res, t);
			mpz_div_2exp(e, e, static_cast<unsigned long int>(B_GL));
		}
		mpz_clear(e); mpz_clear(t);

		// 2^res
		pTransform->set(1);
		for (int i = static_cast<int>(mpz_sizeinbase(res, 2)) - 1; i >= 0; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) { mpz_clear(res); return false; }

			pTransform->squareDup(mpz_tstbit(res, mp_bitcnt_t(i)) != 0);
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
	// in: ckpt[i]
	// out: proof file, proof key
	bool PL(const int depth, const bool fast_checkpoints, double & proofTime, uint64_t & pkey)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		clearline(); pio::display("Generating proof...\r");

		watch chrono;

		file proofFile(proofFilename(), "wb", true);
		int version = 1;
		proofFile.write(reinterpret_cast<const char *>(&version), sizeof(version));
		proofFile.write(reinterpret_cast<const char *>(&depth), sizeof(depth));

		// mu[0] = ckpt[0]
		if (fast_checkpoints)
		{
			pTransform->copy(0, 3 + 0);
			pTransform->getInt(gi);
		}
		else
		{
			file ckptFile(ckptFilename(0), "rb", true);
			gi.read(ckptFile);
			ckptFile.check_crc32();
		}
		gi.write(proofFile);
		// v1 = mu[0]^w[0]
		const uint32_t q = gi.gethash32();
		pTransform->setInt(gi);
		power(0, q);
		pTransform->copy(2, 0);

		const size_t L = size_t(1) << depth;
		mpz_t * const w = new mpz_t[L / 2]; for (size_t i = 0; i < L / 2; ++i) mpz_init(w[i]);
		mpz_set_ui(w[0], q);

		const int l0 = (1 << depth) - depth - 1;
// size_t s = 0;	// complexity

		for (int k = 1, l = l0; k <= depth; ++k)
		{
			const size_t i = size_t(1) << (depth - k);

			// mu[k] = ckpt[i]^w[0]
			if (fast_checkpoints) pTransform->copy(0, 3 + i);
			else
			{
				file ckptFile(ckptFilename(i), "rb", true);
				gi.read(ckptFile);
				ckptFile.check_crc32();
				pTransform->setInt(gi);
			}
			powerz(0, w[0]);
// s += mpz_sizeinbase(w[0], 2);
			pTransform->copy(1, 0);

			for (size_t j = i; j < L / 2; j += i)
			{
				// mu[k] *= ckpt[i + 2 * j]^w[j]
				if (fast_checkpoints) pTransform->copy(0, 3 + i + 2 * j);
				else
				{
					file ckptFile(ckptFilename(i + 2 * j), "rb", true);
					gi.read(ckptFile);
					ckptFile.check_crc32();
					pTransform->setInt(gi);
				}
				powerz(0, w[j]);
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
			const uint32_t q = gi.gethash32();
			// v1 = v1 * mu[k]^w[k]
			power(0, q);
			pTransform->mul(2);
			pTransform->copy(2, 0);

			if (i > 1)
			{
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

		// pkey = hash64(v1);
		pTransform->copy(0, 2);
		pTransform->getInt(gi);
		pkey = gi.gethash64();

		proofTime = chrono.getElapsedTime();
		return true;
	}

	bool quick(const mpz_t & exponent, double & testTime, double & validTime, bool & isPrp, uint64_t & res64, uint64_t & old64)
	{
		const int B_GL = B_GerbiczLi(mpz_sizeinbase(exponent, 2));

		if (!prp(exponent, B_GL, 0, false, testTime)) return false;
		{
			gint & gi = *_gi;
			_transform->getInt(gi);
			isPrp = gi.isOne(res64, old64);
		}
		if (!GL(exponent, B_GL, validTime)) return false;
		return true;
	}

	bool proof(const mpz_t & exponent, const int depth, const bool fast_checkpoints, double & testTime, double & validTime, double & proofTime,
			   bool & isPrp, uint64_t & pkey, uint64_t & res64, uint64_t & old64)
	{
		const size_t esize = mpz_sizeinbase(exponent, 2);
		const int B_GL = B_GerbiczLi(esize), B_PL = B_PietrzakLi(esize, depth);

		if (!prp(exponent, B_GL, B_PL, fast_checkpoints, testTime)) return false;
		{
			gint & gi = *_gi;
			_transform->getInt(gi);
			isPrp = gi.isOne(res64, old64);
		}
		if (!GL(exponent, B_GL, validTime)) return false;
		if (!PL(depth, fast_checkpoints, proofTime, pkey)) return false;
		return true;
	}

	static uint32_t rand32(const uint32_t rmin, const uint32_t rmax) { return (static_cast<uint32_t>(std::rand()) % (rmax - rmin)) + rmin; }

	bool server(const mpz_t & exponent, double & time, bool & isPrp, uint64_t & pkey, uint64_t & ckey, uint64_t & res64, uint64_t & old64)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		watch chrono;

		file proofFile(proofFilename(), "rb", true);
		int version = 0; proofFile.read(reinterpret_cast<char *>(&version), sizeof(version));
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

			// v1 = v1 * mu[k]^w[k]
			power(0, q);
			pTransform->mul(1);
			pTransform->copy(1, 0);

			// v2 = v2^w[k] * mu[k]
			power(2, q);
			pTransform->mul(3);
			pTransform->copy(2, 0);

			const size_t i = size_t(1) << (depth - k);
			for (size_t j = 0; j < L; j += 2 * i) mpz_mul_ui(w[i + j], w[j], q);

			if (_quit) return false;
		}

		proofFile.check_crc32();

		// pkey = hash64(v1);
		pTransform->copy(0, 1);
		pTransform->getInt(gi);
		pkey = gi.gethash64();

		mpz_t p2, t; mpz_init_set_ui(p2, 0); mpz_init(t);
		mpz_t e; mpz_init_set(e, exponent);
		for (size_t i = 0; i < L; i++)
		{
			mpz_mod_2exp(t, e, static_cast<unsigned long int>(B));
			mpz_addmul(p2, t, w[i]);
			mpz_div_2exp(e, e, static_cast<unsigned long int>(B));
		}
		mpz_clear(e);

		for (size_t i = 0; i < L; ++i) mpz_clear(w[i]);
		delete[] w;

		// encode
		std::srand(static_cast<unsigned int>(std::time(nullptr))); std::rand();	// use current time as seed for random generator
		const uint32_t rnd1 = rand32(2, 64), rnd2 = rand32(16, 256), rnd3 = rand32(4, 65536);

		mpz_set_ui(t, rnd1);
		mpz_mul_2exp(t, t, static_cast<unsigned long int>(B));
		if (mpz_cmp(p2, t) > 0)
		{
			mpz_sub(p2, p2, t);
			pTransform->set(2);
			power(0, rnd1);
			pTransform->mul(2);
			pTransform->copy(2, 0);
		}

		power(1, rnd2);
		pTransform->copy(1, 0);
		pTransform->set(2);
		power(0, rnd3);
		pTransform->mul(1);
		pTransform->getInt(gi);
		ckey = gi.gethash64();

		power(2, rnd2);
		pTransform->getInt(gi);
		mpz_mul_ui(p2, p2, rnd2);
		mpz_add_ui(p2, p2, rnd3);

		{
			file certFile(certFilename(), "wb", true);
			int version = 1;
			certFile.write(reinterpret_cast<const char *>(&version), sizeof(version));
			certFile.write(reinterpret_cast<const char *>(&B), sizeof(B));
			gi.write(certFile);
			certFile.write(p2);
			certFile.write_crc32();
		}

		mpz_clear(p2);  mpz_clear(t);

		time = chrono.getElapsedTime();
		return true;
	}

	bool check(double & time, uint64_t & ckey)
	{
		transform * const pTransform = _transform;
		gint & gi = *_gi;

		int ri = 0; double restoredTime = 0;
		const int error = readContext(1, false, ri, restoredTime);
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
			int version = 0; certFile.read(reinterpret_cast<char *>(&version), sizeof(version));
			certFile.read(reinterpret_cast<char *>(&B), sizeof(B));
			gi.read(certFile);
			certFile.read(p2);
			certFile.check_crc32();
		}
		const int p2size = static_cast<int>(mpz_sizeinbase(p2, 2));

		// Gerbicz test for v2^{2^B} and Gerbicz-Li test for 2^p2
		const int L = B_GerbiczLi(static_cast<size_t>(B)), GL = B_GerbiczLi(static_cast<size_t>(p2size));

		watch chrono(found ? restoredTime : 0);
		const int i0 = p2size + B - 1;
		initPrintProgress(i0, found ? ri : i0);
		int dcount = 100;

		// v2 = v2^{2^B}
		if (!found || (ri >= p2size))
		{
			if (!found)
			{
				pTransform->setInt(gi);
				pTransform->copy(1, 0);	// d(t) = u(0)
			}
			for (int i = found ? ri : i0; i >= p2size; --i)
			{
				if (_quit)
				{
					saveContext(1, false, i, chrono.getElapsedTime());
					mpz_clear(p2);
					return false;
				}

				if (i % dcount == 0)
				{
					chrono.read(); const double displayTime = chrono.getDisplayTime();
					if (displayTime >= 10) { dcount = printProgress(displayTime, i); chrono.resetDisplayTime(); }
					if (_isBoinc) { if (boincMonitor(1, false, i, chrono)) return false; }
					else if (chrono.getRecordTime() > 600) { saveContext(1, false, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
				}

				const int j = i0 - i;
				if ((j % L == 0) && (j != 0))
				{
					pTransform->copy(2, 0);
					pTransform->mul(1);	// d(t) = d(t - 1) * u(t * L)
					pTransform->copy(1, 0);
					pTransform->copy(0, 2);
				}

				pTransform->squareDup(false);
				// if (i == i0) pTransform->add1();	// => invalid
				// if (i == p2size) pTransform->add1();	// => invalid
			}

			pTransform->copy(2, 0);	// v2

			// u((t + 1) * L)
			if (B % L != 0)
			{
				for (int i = L - (B % L); i > 0; --i)
				{
					if (_isBoinc) boincMonitor();
					if (_quit) { mpz_clear(p2); return false; }
					pTransform->squareDup(false);
				}
			}
			// d(t + 1) = d(t) * u((t + 1) * L)
			pTransform->mul(1);
			pTransform->copy(3, 0);

			// d(t)^{2^L}
			pTransform->copy(0, 1);
			for (int i = L; i > 0; --i)
			{
				if (_isBoinc) boincMonitor();
				if (_quit) { mpz_clear(p2); return false; }
				pTransform->squareDup(false);
			}
			pTransform->copy(1, 0);

			// u(0) * d(t)^{2^L}
			pTransform->setInt(gi);
			pTransform->mul(1);

			// u(0) * d(t)^{2^L} ?= d(t + 1)
			pTransform->getInt(gi);
			const uint64_t h1 = gi.gethash64();
			pTransform->copy(0, 3);
			pTransform->getInt(gi);
			const uint64_t h2 = gi.gethash64();

			if (h1 != h2) { mpz_clear(p2); return false; }
		}

		// 2^p2
		if (!found || (ri >= p2size))
		{
			pTransform->set(1);
			pTransform->copy(1, 0);	// d(t)
		}
		for (int i = found ? std::min(ri, p2size - 1) : p2size - 1; i >= 0; --i)
		{
			if (_quit)
			{
				saveContext(1, false, i, chrono.getElapsedTime());
				mpz_clear(p2);
				return false;
			}

			if (i % dcount == 0)
			{
				chrono.read(); const double displayTime = chrono.getDisplayTime();
				if (displayTime >= 10) { dcount = printProgress(displayTime, i); chrono.resetDisplayTime(); }
				if (_isBoinc) { if (boincMonitor(1, false, i, chrono)) return false; }
				else if (chrono.getRecordTime() > 600) { saveContext(1, false, i, chrono.getElapsedTime()); chrono.resetRecordTime(); }
			}

			pTransform->squareDup(mpz_tstbit(p2, mp_bitcnt_t(i)) != 0);
			// if (i == p2size - 1) pTransform->add1();	// => invalid
			// if (i == 0) pTransform->add1();	// => invalid

			if ((i % GL == 0) && (i / GL != 0))
			{
				pTransform->copy(3, 0);
				pTransform->mul(1);	// d(t)
				pTransform->copy(1, 0);
				pTransform->copy(0, 3);
			}
		}

		pTransform->copy(3, 0);

		// v1' = v2 * 2^p2
		pTransform->mul(2);

		// ckey = hash64(v1')
		pTransform->getInt(gi);
		ckey = gi.gethash64();

		// d(t + 1) = d(t) * result
		pTransform->copy(0, 3);
		pTransform->mul(1);
		pTransform->copy(2, 0);

		// d(t)^{2^GL}
		pTransform->copy(0, 1);
		for (int i = GL - 1; i >= 0; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) return false;
			pTransform->squareDup(false);
		}
		pTransform->copy(1, 0);

		mpz_t res, t; mpz_init_set_ui(res, 0); mpz_init(t);
		while (mpz_sgn(p2) != 0)
		{
			mpz_mod_2exp(t, p2, static_cast<unsigned long int>(GL));
			mpz_add(res, res, t);
			mpz_div_2exp(p2, p2, static_cast<unsigned long int>(GL));
		}
		mpz_clear(p2); mpz_clear(t);

		// 2^res
		pTransform->set(1);
		for (int i = static_cast<int>(mpz_sizeinbase(res, 2)) - 1; i >= 0; --i)
		{
			if (_isBoinc) boincMonitor();
			if (_quit) { mpz_clear(res); return false; }

			pTransform->squareDup(mpz_tstbit(res, mp_bitcnt_t(i)) != 0);
		}

		mpz_clear(res);

		// d(t)^{2^GL} * 2^res
		pTransform->mul(1);

		// d(t)^{2^GL} * 2^res ?= d(t + 1)
		pTransform->getInt(gi);
		const uint64_t h1 = gi.gethash64();
		pTransform->copy(0, 2);
		pTransform->getInt(gi);
		const uint64_t h2 = gi.gethash64();

		if (h1 != h2) return false;

		time = chrono.getElapsedTime();
		return true;
	}

	bool bench(const uint32_t m, const size_t device, const size_t nthreads, const std::string & impl)
	{
		static constexpr uint32_t bm[13] = { 500000000, 380000000, 290000000, 220000000, 170000000,
											 115000000, 16000000, 5500000, 2000000, 830000, 240000, 980000, 500000 };

		// NTT2 limits
		// static constexpr uint32_t bm[13] = { 46664208, 32996578, 23332104, 16498288, 11666052,
		// 									 8249144, 5833026, 4124572, 2916512, 2062286, 1458256, 1458256, 1031142 };

		// DT limits
		// static constexpr uint32_t bm[13] = { 4200000, 3500000, 2800000, 2300000, 1900000,
		// 									 1600000, 1300000, 1100000, 880000, 730000, 600000, 600002, 50000 };

		const size_t num_regs = 3;

		const uint32_t b = bm[m - 12], n = (m <= 22) ? m : m - 1;

#if defined(GPU)
		(void)nthreads; (void)impl;
		createTransformGPU(b, n, device, num_regs, m == 15, false);
#else
		(void)device;
		createTransformCPU(b, n, nthreads, impl, num_regs, false, m == 15, false);
#endif

		transform * const pTransform = _transform;

		_gi = new gint(size_t(1) << n, b);
		mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, 3, 20);
		double testTime = 0, validTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
		const bool success = quick(exponent, testTime, validTime, isPrp, res64, old64);
		mpz_clear(exponent);
		clearContext();
		delete _gi; _gi = nullptr;

		pio::print(gfn(b, n));

		std::ostringstream ss;
		if (!success) ss << ": test failed!" << std::endl;
		else
		{
			static volatile bool _break;

			_break = false;
			std::thread delay([=]() { std::this_thread::sleep_for(std::chrono::seconds(5)); _break = true; }); delay.detach();

			watch chrono(0);
			size_t i = 1;
			while (!_break)
			{
				pTransform->squareDup((i % 2) != 0);
				++i;
				if (_quit) break;
			}

			pTransform->copy(1, 0);	// synchro

			const size_t memsize = 
#if defined(GPU)
			_transform->getMemSize();
#else
			_transform->getCacheSize();
#endif

			const double mulTime = chrono.getElapsedTime() / i, estimatedTime = mulTime * std::log2(b) * (size_t(1) << n);
			ss << ": " << timer::formatTime(estimatedTime) << std::setprecision(3) << ", " << mulTime * 1e3 << " ms/bit, ";
			ss << "data size: " << memsize / (1024 * 1024.0) << " MB." << std::endl;
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
			createTransformCPU(b, n, nthreads, impl, num_regs, false, false);
#endif

			_gi = new gint(size_t(1) << n, b);

			double testTime = 0, validTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
			const bool success = quick(exponent, testTime, validTime, isPrp, res64, old64);
			if (success) b_min = b;
			else if (!_quit) b_max = b;

			// std::ostringstream ss; ss << n << ", " << b << ": " << (success ? 1 : 0) << std::endl;
			// pio::print(ss.str());

			clearContext();
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
	bool check(const uint32_t b, const uint32_t n, const EMode mode, const size_t device, const size_t nthreads, const std::string & impl, const int depth, const bool oldfashion = false)
	{
		const bool emptyMainFilename = _mainFilename.empty();
		if (emptyMainFilename)
		{
			std::ostringstream ss; ss << "g" << n << "_" << b;
			_mainFilename = ss.str();
		}

		if (mode == EMode::Bench) return bench(n, device, nthreads, impl);
		if (mode == EMode::Limit) return check_limit(n, device, nthreads, impl);

		bool fast_checkpoints =
#if defined(GPU)
			(n <= 17);
#else
			false;
#endif
		size_t num_regs;
		if (mode == EMode::Quick) num_regs = 3;
		else if (mode == EMode::Proof) num_regs = fast_checkpoints ? 3 + (size_t(1) << depth) : 3;
		else if (mode == EMode::Server) num_regs = 4;
		else if (mode == EMode::Check) num_regs = 4;
		else return false;

#if defined(GPU)
		(void)nthreads; (void)impl;
		createTransformGPU(b, n, device, num_regs);
#else
		static constexpr uint32_t bm[23 - 12 + 1] = { 500, 380, 290, 220, 160, 125, 94, 71, 54, 41, 31, 24 };
		bool checkError = false;
		if (impl != "i32")
		{
			if (b > bm[n - 12] * 1000000) checkError = true;
		}
		(void)device;
		createTransformCPU(b, n, nthreads, impl, num_regs, checkError);

		if (checkError)
		{
			std::ostringstream ss; ss << "Warning: b > " << bm[n - 12] << ",000,000: the test may fail." << std::endl;
			pio::print(ss.str());
		}
#endif

		_gi = new gint(size_t(1) << n, b);

		bool success = false;

		if (mode == EMode::Check)
		{
			double time = 0; uint64_t ckey = 0;
			success = check(time, ckey);
			const double error = _transform->getError();
			// if (success)
			// {
			// 	file ckeyFile(ckeyFilename(), "w", false);
			// 	ckeyFile.print(uint64toString(ckey).c_str());
			// }
			clearline();
			std::ostringstream ss; ss << gfn(b, n);
			if (success)
			{
				ss << " is checked, ckey = " << uint64toString(ckey);
				if (error != 0) ss << ", error = " << std::setprecision(4) << error;
				ss << ", time = " << timer::formatTime(time) << ".";
			}
			else if (!_quit) ss << ": check failed!";
			else ss << ": terminated.";
			ss << std::endl; pio::print(ss.str());
			if (success)
			{
				pio::result(ss.str());
				if (!_isBoinc) clearContext();
			}
		}
		else
		{
			mpz_t exponent; mpz_init(exponent); mpz_ui_pow_ui(exponent, b, static_cast<unsigned long int>(1) << n);

			if (mode == EMode::Quick)
			{
				double testTime = 0, validTime = 0; bool isPrp = false; uint64_t res64 = 0, old64 = 0;
				success = quick(exponent, testTime, validTime, isPrp, res64, old64);
				const double error = _transform->getError();
				clearline();
				if (oldfashion)
				{
					std::ostringstream ss; ss << b << "^" << (size_t(1) << n) << "+1";
					if (success)
					{
						ss << " is complete";
						if (error != 0) ss << ", err = " << std::setprecision(4) << error;
						ss << ", time = " << timer::formatTime(testTime + validTime) << ".";
					}
					else if (!_quit) ss << ": validation failed!";
					else ss << ": terminated.";
					ss << std::endl; pio::print(ss.str());
					pio::result(ss.str(), "genefer.log");
					if (success)
					{
						std::ostringstream ssres; ssres << std::hex << std::setfill('0') << std::setw(16) << old64;
						std::ostringstream ssr; ssr << b << "^" << (size_t(1) << n) << "+1 is ";
						if (isPrp) ssr << "a probable prime."; else ssr << "composite. (RES=" << ssres.str() << ")";
						ssr << " (" << static_cast<uint32_t>((size_t(1) << n) * log(static_cast<double>(b)) / log(10.0)) + 1 << " digits) (err = 0.0000) (time = "
							<< timer::formatTime(testTime + validTime) << ") ";
						time_t ltime; time(&ltime);
						ssr << std::string(asctime(localtime(&ltime))).substr(11, 8) << std::endl;
						pio::result(ssr.str(), "out");
						if (!_isBoinc) clearContext();
					}
				}
				else
				{
					std::ostringstream ss; ss << gfn(b, n);
					if (success) ss << gfnStatus(isPrp, 0, 0, res64, old64, error, testTime + validTime);
					else if (!_quit) ss << ": validation failed!";
					else ss << ": terminated.";
					ss << std::endl; pio::print(ss.str());
					if (success)
					{
						pio::result(ss.str());
						if (!_isBoinc) clearContext();
					}
				}
			}
			else if (mode == EMode::Proof)
			{
				double testTime = 0, validTime = 0, proofTime = 0; bool isPrp = false; uint64_t pkey = 0, res64 = 0, old64 = 0;
				success = proof(exponent, depth, fast_checkpoints, testTime, validTime, proofTime, isPrp, pkey, res64, old64);
				const double error = _transform->getError();
				const double time = testTime + validTime + proofTime;
				clearline();
				std::ostringstream ss; ss << gfn(b, n) << ": ";
				if (success)
				{
					ss << "proof file is generated";
					if (error != 0) ss << ", error = " << std::setprecision(4) << error;
					ss << ", time = " << timer::formatTime(time) << ".";
				}
				else if (!_quit) ss << "validation failed!";
				else ss << "terminated.";
				ss << std::endl; pio::print(ss.str());
				if (success)
				{
					std::ostringstream ssr; ssr << gfn(b, n) << gfnStatus(isPrp, pkey, 0, res64, old64, error, time) << std::endl;
					pio::result(ssr.str());
					if (!_isBoinc)
					{
						for (size_t i = 0, L = size_t(1) << depth; i < L; ++i) std::remove(ckptFilename(i).c_str());
						clearContext();
					}
				}
			}
			else if (mode == EMode::Server)
			{
				double time = 0; bool isPrp = false; uint64_t pkey = 0, ckey = 0, res64 = 0, old64 = 0;
				success = server(exponent, time, isPrp, pkey, ckey, res64, old64);
				const double error = _transform->getError();
				// if (success)
				// {
				// 	file skeyFile(skeyFilename(), "w", false);
				// 	skeyFile.print(uint64toString(pkey).c_str());
				// }
				std::ostringstream ss; ss << gfn(b, n);
				if (success) ss << gfnStatus(isPrp, pkey, ckey, res64, old64, error, time);
				else if (!_quit) ss << ": generation failed!";
				else ss << ": terminated.";
				ss << std::endl; pio::print(ss.str());
				if (success) pio::result(ss.str());
			}

			mpz_clear(exponent);
		}

		delete _gi; _gi = nullptr;
		deleteTransform();
		if (emptyMainFilename) _mainFilename.clear();

		return success;
	}

	void displaySupportedImplementations()
	{
		const std::string impls = transform::implementations();
		std::ostringstream ss; ss << "Supported implementations:" << impls << std::endl;
		pio::print(ss.str());
	}
};
