/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <fstream>

#include "ocl.h"
#include "transform.h"

#include "ocl/kernels.h"

// #define CHECK_ALL_FUNCTIONS		1
// #define CHECK_RADIX4_FUNCTIONS	1
#define CHECK_FUNC_1		1	// GFN-12&13&14: square22, square4, square32, forward4_0, forward4, forward64, forward256_0, forward256
// #define CHECK_FUNC_2		1	// GFN-12&13&14: square64, square128, forward64_0, forward1024
// #define CHECK_FUNC_3		1	// GFN-12&13&14: square256, square512, forward1024_0
// #define CHECK_FUNC_4		1	// GFN-12&13&14: square1024

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

#define	P1S			(127 * (uint32(1) << 24) + 1)
#define	Q1S			2164260865u		// p * q = 1 (mod 2^32)
#define	R1S			33554430u		// 2^32 mod p
// #define	RSQ1S		402124772u		// (2^32)^2 mod p
#define	H1S			167772150u		// Montgomery form of the primitive root 5
// #define	IM1S		200536044u		// MF of MF of I = 5^{(p - 1)/4} to convert input into MF
// #define	SQRTI1S		856006302u		// MF of 5^{(p - 1)/8}
// #define	ISQRTI1S	1626730317u		// MF of i * sqrt(i)

#define	P2S			(63 * (uint32(1) << 25) + 1)
#define	Q2S			2181038081u
#define	R2S			67108862u
// #define	RSQ2S		2111798781u
#define	H2S			335544310u		// MF of the primitive root 5
// #define	IM2S		1036950657u
// #define	SQRTI2S		338852760u
// #define	ISQRTI2S	1090446030u

class ZP
{
protected:
	cl_uint _n;

public:
	ZP() {}
	explicit ZP(const uint32 n) : _n(n) {}

	uint32 get() const { return _n; }
};

template<uint32 P, uint32 Q, uint32 R, uint32 H>
class ZPT : public ZP
{
private:
	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= P) ? P : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? P : 0); }

	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * Q) * uint64(P)) >> 32);
		return _sub(hi, mp);
	}

public:
	ZPT() {}
	explicit ZPT(const uint32 n) : ZP(n) {}

	int32 get_int() const { return (_n >= P / 2) ? int32(_n - P) : int32(_n); }
	ZPT & set_int(const int32 i) { _n = (i < 0) ? (uint32(i) + P) : uint32(i); return *this; }

	ZPT mul(const ZPT & rhs) const { return ZPT(_mul(_n, rhs._n)); }

	ZPT pow(const size_t e) const
	{
		if (e == 0) return ZPT(R);	// MF of one is R
		ZPT r = ZPT(R), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.mul(y); }
		r = r.mul(y);
		return r;
	}

	static const ZPT primroot_n(const uint32 n) { return ZPT(H).pow((P - 1) / n); }
	static ZPT norm(const uint32 n) { return ZPT(P - (P - 1) / n); }
};

typedef ZPT<P1S, Q1S, R1S, H1S> ZP1;
typedef ZPT<P2S, Q2S, R2S, H2S> ZP2;

// Warning: DECLARE_VAR_32/64/128/256 in kernerl.cl must be modified if BLKxx = 1 or != 1.

#define BLK32m		8		// local size =  4KB, workgroup size =  64
#define BLK64m		4		// local size =  4KB, workgroup size =  64
#define BLK128m		2		// local size =  4KB, workgroup size =  64
#define BLK256m		1		// local size =  4KB, workgroup size =  64
//      BLK512m		1		   local size =  8KB, workgroup size = 128
//      BLK1024m	1		   local size = 16KB, workgroup size = 256

#define CHUNK64m	4		// local size =  4KB, workgroup size =  64
#define CHUNK256m	2		// local size =  8KB, workgroup size = 128
#define CHUNK1024m	1		// local size = 16KB, workgroup size = 256

template<size_t RNS_SIZE>
class engines : public device
{
private:
	const size_t _n;
	const int _ln;
	const bool _isBoinc;
	const size_t _num_regs;
	cl_mem _z = nullptr, _zp = nullptr, _w = nullptr, _c = nullptr;
	cl_kernel _forward4 = nullptr, _backward4 = nullptr, _forward4_0 = nullptr;
	cl_kernel _square22 = nullptr, _square4 = nullptr, _fwd4p = nullptr, _mul22 = nullptr, _mul4 = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward64_0 = nullptr;
	cl_kernel _forward256 = nullptr, _backward256 = nullptr, _forward256_0 = nullptr;
	cl_kernel _forward1024 = nullptr, _backward1024 = nullptr, _forward1024_0 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr, _square512 = nullptr, _square1024 = nullptr;	//, _square2048 = nullptr;
	cl_kernel _normalize1 = nullptr, _normalize2 = nullptr, _mulscalar = nullptr;
	cl_kernel _fwd32p = nullptr, _fwd64p = nullptr, _fwd128p = nullptr, _fwd256p = nullptr, _fwd512p = nullptr, _fwd1024p = nullptr;	//, _fwd2048p = nullptr;
	cl_kernel _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr, _mul256 = nullptr, _mul512 = nullptr, _mul1024 = nullptr;	//, _mul2048 = nullptr;
	cl_kernel _set = nullptr, _copy = nullptr, _copyp = nullptr;
	splitter * _pSplit = nullptr;
	size_t _naLocalWS = 32, _nbLocalWS = 32, _baseModBlk = 16, _splitIndex = 0;
	bool _first = false;

public:
	engines(const platform & platform, const size_t d, const int ln, const bool isBoinc, const size_t num_regs, const bool verbose)
		: device(platform, d, verbose), _n(size_t(1) << ln), _ln(ln), _isBoinc(isBoinc), _num_regs(num_regs) {}
	virtual ~engines() {}

///////////////////////////////

public:
	void allocMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t n = _n;
		if (n != 0)
		{
			_z = _createBuffer(CL_MEM_READ_WRITE, RNS_SIZE * n * _num_regs * sizeof(ZP));
			_zp = _createBuffer(CL_MEM_READ_WRITE, RNS_SIZE * n * sizeof(ZP));
			_w = _createBuffer(CL_MEM_READ_ONLY, RNS_SIZE * n / 2 * sizeof(ZP));
			_c = _createBuffer(CL_MEM_READ_WRITE, n / 4 * sizeof(cl_long));
		}
	}

	void releaseMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_n != 0)
		{
			_releaseBuffer(_z); _releaseBuffer(_zp);
			_releaseBuffer(_w); _releaseBuffer(_c);
		}
	}

///////////////////////////////

private:
	cl_kernel createTransformKernel(const char * const kernelName, const bool isMultiplier = true)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_w);
		return kernel;
	}

	cl_kernel createNormalizeKernel(const char * const kernelName, const cl_uint b, const cl_uint b_inv, const cl_int b_s)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_c);
		_setKernelArg(kernel, 2, sizeof(cl_uint), &b);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &b_inv);
		_setKernelArg(kernel, 4, sizeof(cl_int), &b_s);
		return kernel;
	}

	cl_kernel createMulKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_w);
		return kernel;
	}

	cl_kernel createSetKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		return kernel;
	}

	cl_kernel createCopyKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		return kernel;
	}

	cl_kernel createCopypKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_z);
		return kernel;
	}

public:
	void createKernels(const uint32_t b)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_forward4 = createTransformKernel("forward4");
		_backward4 = createTransformKernel("backward4");
		_forward4_0 = createTransformKernel("forward4_0");

		_square22 = createTransformKernel("square22");
		_square4 = createTransformKernel("square4");
		_fwd4p = createTransformKernel("fwd4p", false);
		_mul22 = createMulKernel("mul22");
		_mul4 = createMulKernel("mul4");

		_forward64 = createTransformKernel("forward64");
		_backward64 = createTransformKernel("backward64");
		_forward64_0 = createTransformKernel("forward64_0");
		_forward256 = createTransformKernel("forward256");
		_backward256 = createTransformKernel("backward256");
		_forward256_0 = createTransformKernel("forward256_0");
		_forward1024 = createTransformKernel("forward1024");
		_backward1024 = createTransformKernel("backward1024");
		_forward1024_0 = createTransformKernel("forward1024_0");

		_square32 = createTransformKernel("square32");
		_square64 = createTransformKernel("square64");
		// _square128 = createTransformKernel("square128");
		// _square256 = createTransformKernel("square256");
		// _square512 = createTransformKernel("square512");
		// _square1024 = createTransformKernel("square1024");
		// // _square2048 = createTransformKernel("square2048");

		const cl_int b_s = static_cast<cl_int>(31 - __builtin_clz(b) - 1);
		const cl_uint b_inv = static_cast<cl_uint>((static_cast<uint64_t>(1) << (b_s + 32)) / b);
		_normalize1 = createNormalizeKernel("normalize1", static_cast<cl_uint>(b), b_inv, b_s);
		_normalize2 = createNormalizeKernel("normalize2", static_cast<cl_uint>(b), b_inv, b_s);
		_mulscalar = createNormalizeKernel("mulscalar", static_cast<cl_uint>(b), b_inv, b_s);

		_fwd32p = createTransformKernel("fwd32p", false);
		_fwd64p = createTransformKernel("fwd64p", false);
		// _fwd128p = createTransformKernel("fwd128p", false);
		// _fwd256p = createTransformKernel("fwd256p", false);
		// _fwd512p = createTransformKernel("fwd512p", false);
		// _fwd1024p = createTransformKernel("fwd1024p", false);
		// // _fwd2048p = createTransformKernel("fwd2048p", false);

		_mul32 = createMulKernel("mul32");
		_mul64 = createMulKernel("mul64");
		// _mul128 = createMulKernel("mul128");
		// _mul256 = createMulKernel("mul256");
		// _mul512 = createMulKernel("mul512");
		// _mul1024 = createMulKernel("mul1024");
		// // _mul2048 = createMulKernel("mul2048");

		_set = createSetKernel("set");
		_copy = createCopyKernel("copy");
		_copyp = createCopypKernel("copyp");

		_pSplit = new splitter(size_t(_ln), CHUNK256m, CHUNK1024m, sizeof(ZP), 10, getLocalMemSize(), getMaxWorkGroupSize());
	}

	void releaseKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		delete _pSplit;

		_releaseKernel(_forward4); _releaseKernel(_backward4); _releaseKernel(_forward4_0);
		_releaseKernel(_square22); _releaseKernel(_square4);
		_releaseKernel(_fwd4p); _releaseKernel(_mul22); _releaseKernel(_mul4);

		_releaseKernel(_forward64); _releaseKernel(_backward64); _releaseKernel(_forward64_0);
		_releaseKernel(_forward256); _releaseKernel(_backward256); _releaseKernel(_forward256_0);
		_releaseKernel(_forward1024); _releaseKernel(_backward1024); _releaseKernel(_forward1024_0);
		 
		_releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128); _releaseKernel(_square256);
		_releaseKernel(_square512); _releaseKernel(_square1024); //_releaseKernel(_square2048);
		_releaseKernel(_normalize1); _releaseKernel(_normalize2); _releaseKernel(_mulscalar);
		_releaseKernel(_fwd32p); _releaseKernel(_fwd64p); _releaseKernel(_fwd128p); _releaseKernel(_fwd256p);
		_releaseKernel(_fwd512p); _releaseKernel(_fwd1024p); //_releaseKernel(_fwd2048p);
		_releaseKernel(_mul32); _releaseKernel(_mul64); _releaseKernel(_mul128); _releaseKernel(_mul256);
		_releaseKernel(_mul512); _releaseKernel(_mul1024); //_releaseKernel(_mul2048);

		_releaseKernel(_set); _releaseKernel(_copy); _releaseKernel(_copyp);
	}

///////////////////////////////

	void readMemory_z(ZP * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP)); }
	void writeMemory_z(const ZP * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP)); }
	void writeMemory_w(const ZP * const wPtr, const size_t offset) { _writeBuffer(_w, wPtr, _n / 2 * sizeof(ZP), offset * _n / 2 * sizeof(ZP)); }

///////////////////////////////

private:
	void ek(cl_kernel & kernel, const size_t localWorkSize)
	{
		const size_t n_4 = _n / 4;
		_executeKernel(kernel, RNS_SIZE * n_4, localWorkSize);
	}

	void ek_fb(cl_kernel & kernel, const int lm, const size_t localWorkSize)
	{
		const size_t n_4 = _n / 4;
		const cl_int ilm = static_cast<cl_int>(lm);
		const cl_uint is = static_cast<cl_uint>(n_4 >> lm);
		_setKernelArg(kernel, 2, sizeof(cl_int), &ilm);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &is);
		_executeKernel(kernel, RNS_SIZE * n_4, localWorkSize);
	}

	void forward4(const int lm) { ek_fb(_forward4, lm, 0); }
	void backward4(const int lm) { ek_fb(_backward4, lm, 0); }
	void forward4_0() { ek(_forward4_0, 0); }
	void square22() { ek(_square22, 0); }
	void square4() { ek(_square4, 0); }
	void fwd4p() { ek(_fwd4p, 0); }
	void mul22() { ek(_mul22, 0); }
	void mul4() { ek(_mul4, 0); }

	void forward64(const int lm) { ek_fb(_forward64, lm, 64 / 4 * CHUNK64m); }
	void backward64(const int lm) { ek_fb(_backward64, lm, 64 / 4 * CHUNK64m); }
	void forward64_0() { ek(_forward64_0, 64 / 4 * CHUNK64m); }
	void forward256(const int lm) { ek_fb(_forward256, lm, 256 / 4 * CHUNK256m); }
	void backward256(const int lm) { ek_fb(_backward256, lm, 256 / 4 * CHUNK256m); }
	void forward256_0() { ek(_forward256_0, 256 / 4 * CHUNK256m); }
	void forward1024(const int lm) { ek_fb(_forward1024, lm, 1024 / 4 * CHUNK1024m); }
	void backward1024(const int lm) { ek_fb(_backward1024, lm, 1024 / 4 * CHUNK1024m); }
	void forward1024_0() { ek(_forward1024_0, 1024 / 4 * CHUNK1024m); }

	void square32() { ek(_square32, std::min(_n / 4, size_t(32 / 4 * BLK32m))); }
	void square64() { ek(_square64, std::min(_n / 4, size_t(64 / 4 * BLK64m))); }
	void square128() { ek(_square128, std::min(_n / 4, size_t(128 / 4 * BLK128m))); }
	void square256() { ek(_square256, std::min(_n / 4, size_t(256 / 4 * BLK256m))); }
	void square512() { ek(_square512, 512 / 4); }
	void square1024() { ek(_square1024, 1024 / 4); }
	// void square2048() { ek(_square2048, 2048 / 4); }

	void fwd32p() { ek(_fwd32p, std::min(_n / 4, size_t(32 / 4 * BLK32m))); }
	void fwd64p() { ek(_fwd64p, std::min(_n / 4, size_t(64 / 4 * BLK64m))); }
	void fwd128p() { ek(_fwd128p, std::min(_n / 4, size_t(128 / 4 * BLK128m))); }
	void fwd256p() { ek(_fwd256p, std::min(_n / 4, size_t(256 / 4 * BLK256m))); }
	void fwd512p() { ek(_fwd512p, 512 / 4); }
	void fwd1024p() { ek(_fwd1024p, 1024 / 4); }
	// void fwd2048p() { ek(_fwd2048p, 2048 / 4); }

	void mul32() { ek(_mul32, std::min(_n / 4, size_t(32 / 4 * BLK32m))); }
	void mul64() { ek(_mul64, std::min(_n / 4, size_t(64 / 4 * BLK64m))); }
	void mul128() { ek(_mul128, std::min(_n / 4, size_t(128 / 4 * BLK128m))); }
	void mul256() { ek(_mul256, std::min(_n / 4, size_t(256 / 4 * BLK256m))); }
	void mul512() { ek(_mul512, 512 / 4); }
	void mul1024() { ek(_mul1024, 1024 / 4); }
	// void mul2048() { ek(_mul2048, 2048 / 4); }

	void setTransformArgs(cl_kernel & kernel, const bool isMultiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
	}

	void forward4p(const int lm)
	{
		setTransformArgs(_forward4, false);
		forward4(lm);
		setTransformArgs(_forward4);
	}

	void forward64p(const int lm)
	{
		setTransformArgs(_forward64, false);
		forward64(lm);
		setTransformArgs(_forward64);
	}

	void forward256p(const int lm)
	{
		setTransformArgs(_forward256, false);
		forward256(lm);
		setTransformArgs(_forward256);
	}

	void forward1024p(const int lm)
	{
		setTransformArgs(_forward1024, false);
		forward1024(lm);
		setTransformArgs(_forward1024);
	}

	void forward4p_0()
	{
		setTransformArgs(_forward4_0, false);
		forward4_0();
		setTransformArgs(_forward4_0);
	}

	void forward64p_0()
	{
		setTransformArgs(_forward64_0, false);
		forward64_0();
		setTransformArgs(_forward64_0);
	}

	void forward256p_0()
	{
		setTransformArgs(_forward256_0, false);
		forward256_0();
		setTransformArgs(_forward256_0);
	}

	void forward1024p_0()
	{
		setTransformArgs(_forward1024_0, false);
		forward1024_0();
		setTransformArgs(_forward1024_0);
	}

private:
	void _mul(const size_t sIndex, const bool isSquare, const bool verbose)
	{
#if defined(CHECK_FUNC_1)
		if (_ln == 11) { forward256_0(); forward4(11 - 10); if (isSquare) square22(); else mul22(); backward4(11 - 10); backward256(11 - 8); return; }
		if (_ln == 12) { forward4_0(); forward256(12 - 10); if (isSquare) square4(); else mul4(); backward256(12 - 10); backward4(12 - 2); return; }
		if (_ln == 13) { forward4_0(); forward64(13 - 8); if (isSquare) square32(); else mul32(); backward64(13 - 8); backward4(13 - 2); return; }
#endif
#if defined(CHECK_FUNC_2)
		if (_ln == 11) { forward4_0(); forward4(11 - 4); if (isSquare) square128(); else mul128(); backward4(11 - 4); backward4_0(); return; }
		if (_ln == 12) { forward64_0(); if (isSquare) square64(); else mul64(); backward64_0(); return; }
		if (_ln == 13) { forward4_0(); forward1024(13 - 12); if (isSquare) square22(); else mul22(); backward1024(13 - 12); backward4_0(); return; }
#endif
#if defined(CHECK_FUNC_3)
		if (_ln == 11) { forward1024_0(); if (isSquare) square22(); else mul22(); backward1024_0(); return; }
		if (_ln == 12) { forward4_0(); forward4(12 - 4); if (isSquare) square256(); else mul256(); backward4(12 - 4); backward4_0(); return; }
		if (_ln == 13) { forward4_0(); forward4(13 - 4); if (isSquare) square512(); else mul512(); backward4(13 - 4); backward4_0(); return; }
#endif
#if defined(CHECK_FUNC_4)
		if (_ln == 11) { forward64_0(); if (isSquare) square32(); else mul32(); backward64_0(); return; }
		if (_ln == 12) { forward4_0(); if (isSquare) square1024(); else mul1024(); backward4_0(); return; }
		if (_ln == 13) { forward64_0(); if (isSquare) square128(); else mul128(); backward64_0(); return; }
#endif

		int lm = _ln;

#if defined(CHECK_RADIX4_FUNCTIONS)
		lm -= 2; forward4_0();
		while (lm > 2) { lm -= 2; forward4(lm); }
		if (isSquare) { if (lm == 1) square22(); else square4(); } else if (lm == 1) mul22(); else mul4();
		while (lm < _ln) { backward4(lm); lm += 2; }
		return;
#endif

		const splitter * const pSplit = _pSplit;
		const size_t s = pSplit->getPartSize(sIndex);

		for (size_t i = 1; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10)
			{
				lm -= 10;
				if (i != 1) forward1024(lm); else forward1024_0();
				if (verbose) std::cout << "forward1024 (" << lm << ") ";
			}
			else if (k == 8)
			{
				lm -= 8;
				if (i != 1) forward256(lm); else forward256_0();
				if (verbose) std::cout << "forward256 (" << lm << ") ";
			}
			else // if (k == 6)
			{
				lm -= 6;
				if (i != 1) forward64(lm); else forward64_0();
				if (verbose) std::cout << "forward64 (" << lm << ") ";
			}
		}

		// lm = split.GetPart(sIndex, s - 1);
		if (isSquare)
		{
			// if (lm == 11) square2048();
			if (lm == 10) square1024();
			else if (lm == 9) square512();
			else if (lm == 8) square256();
			else if (lm == 7) square128();
			else if (lm == 6) square64();
			else if (lm == 5) square32();
		}
		else
		{
			// if (lm == 11) mul2048();
			if (lm == 10) mul1024();
			else if (lm == 9) mul512();
			else if (lm == 8) mul256();
			else if (lm == 7) mul128();
			else if (lm == 6) mul64();
			else if (lm == 5) mul32();
		}
		if (verbose) std::cout << "square" << (1u << lm) << " ";

		for (size_t i = s - 1; i > 0; --i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10)
			{
				backward1024(lm);
				if (verbose) std::cout << "backward1024 (" << lm << ") ";
				lm += 10;
			}
			else if (k == 8)
			{
				backward256(lm);
				if (verbose) std::cout << "backward256 (" << lm << ") ";
				lm += 8;
			}
			else // if (k == 6)
			{
				backward64(lm);
				if (verbose) std::cout << "backward64 (" << lm << ") ";
				lm += 6;
			}
		}

		if (verbose) std::cout << std::endl;
	}

public:
	void square()
	{
#if defined(CHECK_ALL_FUNCTIONS)
		_mul(size_t(rand()) % _pSplit->getSize(), true, false);
#else
		_mul(_splitIndex, true, _first);
#endif
		if (_first) _first = false;
	}

	void mul()
	{
		_mul(_splitIndex, false, false);
	}

	void initMultiplicand(const size_t src)
	{
		const cl_uint isrc = static_cast<cl_uint>(src * RNS_SIZE * _n);
		_setKernelArg(_copyp, 2, sizeof(cl_uint), &isrc);
		_executeKernel(_copyp, RNS_SIZE * _n);

#if defined(CHECK_FUNC_1)
		if (_ln == 11) { forward256p_0(); forward4p(11 - 10); return; }
		if (_ln == 12) { forward4p_0(); forward256p(12 - 10); fwd4p(); return; }
		if (_ln == 13) { forward4p_0(); forward64p(13 - 8); fwd32p(); return; }
#endif
#if defined(CHECK_FUNC_2)
		if (_ln == 11) { forward4p_0(); forward4p(11 - 4); fwd128p(); return; }
		if (_ln == 12) { forward64p_0(); fwd64p(); return; }
		if (_ln == 13) { forward4p_0(); forward1024p(13 - 12); return; }
#endif
#if defined(CHECK_FUNC_3)
		if (_ln == 11) { forward1024p_0(); return; }
		if (_ln == 12) { forward4p_0(); forward4p(12 - 4); fwd256p(); return; }
		if (_ln == 13) { forward4p_0(); forward4p(13 - 4); fwd512p(); return; }
#endif
#if defined(CHECK_FUNC_4)
		if (_ln == 11) { forward64p_0(); fwd32p(); return; }
		if (_ln == 12) { forward4p_0(); fwd1024p(); return; }
		if (_ln == 13) { forward64p_0(); fwd128p(); return; }
#endif

		const splitter * const pSplit = _pSplit;
#if defined(CHECK_ALL_FUNCTIONS)
		_splitIndex = size_t(rand()) % pSplit->getSize();
#endif

		int lm = _ln;

#if defined(CHECK_RADIX4_FUNCTIONS)
		lm -= 2; forward4p_0();
		while (lm > 2) { lm -= 2; forward4p(lm); }
		if (lm == 2) fwd4p();
		return;
#endif

		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		for (size_t i = 1; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10)
			{
				lm -= 10;
				if (i != 1) forward1024p(lm); else forward1024p_0();
			}
			else if (k == 8)
			{
				lm -= 8;
				if (i != 1) forward256p(lm); else forward256p_0();
			}
			else // if (k == 6)
			{
				lm -= 6;
				if (i != 1) forward64p(lm); else forward64p_0();
			}
		}

		// lm = split.GetPart(sIndex, s - 1);
		// if (lm == 11) fwd2048p();
		if (lm == 10) fwd1024p();
		else if (lm == 9) fwd512p();
		else if (lm == 8) fwd256p();
		else if (lm == 7) fwd128p();
		else if (lm == 6) fwd64p();
		else if (lm == 5) fwd32p();
	}

	void set(const uint32_t a)
	{
		const cl_uint ia = static_cast<cl_uint>(a);
		_setKernelArg(_set, 1, sizeof(cl_uint), &ia);
		_executeKernel(_set, RNS_SIZE * _n);
	}

	void copy(const size_t dst, const size_t src)
	{
		const cl_uint idst = static_cast<cl_uint>(dst * RNS_SIZE *_n), isrc = static_cast<cl_uint>(src * RNS_SIZE *_n);
		_setKernelArg(_copy, 1, sizeof(cl_uint), &idst);
		_setKernelArg(_copy, 2, sizeof(cl_uint), &isrc);
		_executeKernel(_copy, RNS_SIZE * _n);
	}

public:
	void baseMod(const bool dup)
	{
		const cl_uint blk = static_cast<cl_uint>(_baseModBlk);
		const cl_int sblk = dup ? -static_cast<cl_int>(blk) : static_cast<cl_int>(blk);
		const size_t size = _n / blk;

		_setKernelArg(_normalize1, 5, sizeof(cl_int), &sblk);
		_executeKernel(_normalize1, size, std::min(size, _naLocalWS));

		_setKernelArg(_normalize2, 5, sizeof(cl_uint), &blk);
		_executeKernel(_normalize2, size, std::min(size, _nbLocalWS));
	}

public:
	void baseModMul(const int a)
	{
		baseMod(false);

		const cl_uint blk = static_cast<cl_uint>(_baseModBlk);
		const size_t size = _n / blk;
		const cl_int ia = static_cast<cl_int>(a);

		cl_uint index1 = 5;
		_setKernelArg(_mulscalar, index1++, sizeof(cl_int), &blk);
		_setKernelArg(_mulscalar, index1++, sizeof(cl_int), &ia);
		_executeKernel(_mulscalar, size, std::min(size, _naLocalWS));

		cl_uint index2 = 5;
		_setKernelArg(_normalize2, index2++, sizeof(cl_uint), &blk);
		_executeKernel(_normalize2, size, std::min(size, _nbLocalWS));
	}

private:
	void baseModTune(const size_t count, const size_t blk, const size_t n3aLocalWS, const size_t n3bLocalWS, const ZP * const Z)
	{
		const cl_uint cblk = static_cast<cl_uint>(blk);
		const cl_int sblk = static_cast<cl_int>(blk);
		const size_t size = _n / blk;

		for (size_t i = 0; i != count; ++i)
		{
			writeMemory_z(Z);

			_setKernelArg(_normalize1, 5, sizeof(cl_int), &sblk);
			_executeKernel(_normalize1, size, std::min(size, n3aLocalWS));

			_setKernelArg(_normalize2, 5, sizeof(cl_uint), &cblk);
			_executeKernel(_normalize2, size, std::min(size, n3bLocalWS));
		}
	}

private:
	void squareTune(const size_t count, const size_t sIndex, const ZP * const Z)
	{
		for (size_t j = 0; j != count; ++j)
		{
			writeMemory_z(Z);
			_mul(sIndex, true, false);
		}
	}

public:
	void tune(const uint32_t base)
	{
		const size_t n = _n;

		ZP * const Z = new ZP[RNS_SIZE * n];
		for (size_t i = 0; i != n; ++i)
		{
			Z[0 * n + i] = ZP1().set_int(static_cast<int32>((P1S - 1) * cos(i + 0.25)));
			if (RNS_SIZE >= 2) Z[1 * n + i] = ZP2().set_int(static_cast<int32>((P1S - 1) * cos(i + 0.33)));
			// if (RNS_SIZE >= 3) Z[2 * n + i] = ZP3().set_int(static_cast<int32>((P1S - 1) * cos(i + 0.47)));
		}

		setProfiling(true);

		resetProfiles();
		baseModTune(1, 16, 0, 0, Z);
		const cl_ulong time = getProfileTime();
		if (time == 0) { delete[] Z; setProfiling(false); return; }
		// 410 tests, 0.1 second = 10^8 ns
		const size_t count = std::min(std::max(size_t(100000000 / (410 * time)), size_t(2)), size_t(100));

		cl_ulong minT = cl_ulong(-1);

		size_t bMin = 4;
		while (bMin < log(n * static_cast<double>(base + 2)) / log(static_cast<double>(base))) bMin *= 2;

		const double maxSqr = n * (base * static_cast<double>(base));
		for (size_t b = bMin; b <= 64; b *= 2)
		{
			// Check convergence
			if (log(maxSqr) >= base * log(static_cast<double>(b))) continue;

			resetProfiles();
			baseModTune(count, b, 0, 0, Z);
			cl_ulong minT_b = getProfileTime();
#if defined(ocl_debug)
			// std::ostringstream ss; ss << "b = " << b << ", sa = 0, sb = 0, count = " << count << ", t = " << minT_b << "." << std::endl;
			// pio::display(ss.str());
#endif
			size_t minsa = 0, minsb = 0;

			for (size_t sa = 1; sa <= 256; sa *= 2)
			{
				for (size_t sb = 1; sb <= 256; sb *= 2)
				{
					resetProfiles();
					baseModTune(count, b, sa, sb, Z);
					const cl_ulong t = getProfileTime();
#if defined(ocl_debug)
					// std::ostringstream ss; ss << "b = " << b << ", sa = " << sa << ", sb = " << sb << ", count = " << count << ", t = " << t << "." << std::endl;
					// pio::display(ss.str());
#endif
					if (t < minT_b)
					{
						minT_b = t;
						minsa = sa;
						minsb = sb;
					}
				}
			}

			if (minT_b < minT)
			{
				minT = minT_b;
				_naLocalWS = minsa;
				_nbLocalWS = minsb;
				_baseModBlk = b;
			}
		}
#if defined(ocl_debug)
		{
			std::ostringstream ss; ss << "baseModBlk = " << _baseModBlk << ", WorkgroupSize1 = " << _naLocalWS << ", WorkgroupSize2 = " << _nbLocalWS << "." << std::endl;
			pio::display(ss.str());
		}
#endif

		const splitter * const pSplit = _pSplit;
		const size_t ns = pSplit->getSize();
		if (ns > 1)
		{
			cl_ulong minT = cl_ulong(-1);
			for (size_t i = 0; i < ns; ++i)
			{
				resetProfiles();
				squareTune(2, i, Z);
				const cl_ulong t = getProfileTime();

#if defined(ocl_debug)
				std::ostringstream ss; ss << "[" << i << "]";
				for (size_t j = 0, nps = pSplit->getPartSize(i); j < nps; ++j) ss << " " << pSplit->getPart(i, j);
				ss << ": " << t << std::endl;
				pio::display(ss.str());
#endif
				if (t < minT)
				{
					minT = t;
					_splitIndex = i;
				}
			}
		}
#if defined(ocl_debug)
		{
			std::ostringstream ss;
			for (size_t j = 0, nps = pSplit->getPartSize(_splitIndex); j < nps; ++j) ss << " " << pSplit->getPart(_splitIndex, j);
			ss << std::endl;
			pio::display(ss.str());
		}
#endif

		delete[] Z;

		setProfiling(false);
	}

public:
	void info()
	{
		std::ostringstream ss; ss << "split:";
		for (size_t i = 0, ns = _pSplit->getSize(); i < ns; ++i)
		{
			for (size_t j = 0, nps = _pSplit->getPartSize(i); j < nps; ++j) ss << " " << _pSplit->getPart(i, j);
			if (i == _splitIndex) ss << " *";
			ss << ",";
		}
		ss << " baseModBlk = " << _baseModBlk << ", WorkgroupSize1 = " << _naLocalWS << ", WorkgroupSize2 = " << _nbLocalWS << "." << std::endl;
		pio::display(ss.str());
	}
};

template<size_t RNS_SIZE>
class transformGPUs : public transform
{
private:
	const size_t _mem_size;
	const size_t _num_regs;
	ZP * const _z;
	engines<RNS_SIZE> * _pEngine = nullptr;

public:
	transformGPUs(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
				 const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
		: transform(size_t(1) << n, n, b, (RNS_SIZE == 2) ? EKind::NTT2s : EKind::NTT3s),
		_mem_size(RNS_SIZE * (size_t(1) << n) * num_regs * sizeof(ZP)), _num_regs(num_regs), _z(new ZP[RNS_SIZE * (size_t(1) << n) * num_regs])
	{
		const size_t size = getSize();

		const bool is_boinc_platform = isBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pEngine = new engines<RNS_SIZE>(eng_platform, is_boinc_platform ? 0 : device, static_cast<int>(n), isBoinc, num_regs, verbose);

		std::ostringstream src;

		src << "#define NSIZE\t" << (1u << n) << "u" << std::endl;
		src << "#define LNSZ\t" << n << std::endl;
		src << "#define NORM1\t" << ZP1::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM2\t" << ZP2::norm(uint32(size / 2)).get() << "u" << std::endl;

		src << "#define WOFFSET\t" << size / 2 << "u" << std::endl;

		src << "#define BLK32\t" << BLK32m << std::endl;
		src << "#define BLK64\t" << BLK64m << std::endl;
		src << "#define BLK128\t" << BLK128m << std::endl;
		src << "#define BLK256\t" << BLK256m << std::endl;

		src << "#define CHUNK64\t" << CHUNK64m << std::endl;
		src << "#define CHUNK256\t" << CHUNK256m << std::endl;
		src << "#define CHUNK1024\t" << CHUNK1024m << std::endl;

		src << "#define MAX_WORK_GROUP_SIZE\t" << _pEngine->getMaxWorkGroupSize() << std::endl << std::endl;

		if (isBoinc || !_pEngine->readOpenCL("ocl/kernels.cl", "src/ocl/kernels.h", "src_ocl_kernels", src)) src << src_ocl_kernels;

		_pEngine->loadProgram(src.str());
		_pEngine->allocMemory();
		_pEngine->createKernels(b);

		ZP1 * const wr1 = new ZP1[size / 2];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const ZP1 r_s = ZP1::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr1[s + j] = r_s.pow(bitRev(j, 2 * s) + 1);
			}
		}
		_pEngine->writeMemory_w(wr1, 0);
		delete[] wr1;

		ZP2 * const wr2 = new ZP2[size / 2];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const ZP2 r_s = ZP2::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				wr2[s + j] = r_s.pow(bitRev(j, 2 * s) + 1);
			}
		}
		_pEngine->writeMemory_w(wr2, 1);
		delete[] wr2;

		_pEngine->tune(b);
	}

	virtual ~transformGPUs()
	{
		_pEngine->releaseKernels();
		_pEngine->releaseMemory();
		_pEngine->clearProgram();
		delete _pEngine;

		delete[] _z;
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return 0; }

protected:
	void getZi(int32_t * const zi) const override
	{
		_pEngine->readMemory_z(_z);

		const size_t size = getSize();

		const ZP * const z = _z;
		for (size_t i = 0; i < size; ++i)
		{
			zi[i] = ZP1(z[i].get()).get_int();
		}
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size = getSize();

		ZP * const z = _z;
		for (size_t i = 0; i < size; ++i)
		{
			z[0 * size + i] = ZP1().set_int(zi[i]);
			if (RNS_SIZE >= 2) z[1 * size + i] = ZP2().set_int(zi[i]);
			// if (RNS_SIZE >= 3) z[2 * size + i] = ZP3().set_int(zi[i]);
		}
		_pEngine->writeMemory_z(z);
	}

public:
	bool readContext(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(getKind())) return false;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		if (!cFile.read(reinterpret_cast<char *>(_z), RNS_SIZE * size * num_regs * sizeof(ZP))) return false;
		_pEngine->writeMemory_z(_z, num_regs);

		return true;
	}

	void saveContext(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		_pEngine->readMemory_z(_z, num_regs);
		if (!cFile.write(reinterpret_cast<const char *>(_z), RNS_SIZE * size * num_regs * sizeof(ZP))) return;
	}

	void set(const uint32_t a) override
	{
		_pEngine->set(a);
	}

	void squareDup(const bool dup) override
	{
		_pEngine->square();
		_pEngine->baseMod(dup);
	}

	void squareMul(const int32_t a) override
	{
		if (a <= 2) squareDup(a == 2);
		else
		{
			_pEngine->square();
			_pEngine->baseModMul(a);
		}
	}

	void initMultiplicand(const size_t src) override
	{
		_pEngine->initMultiplicand(src);
	}

	void mul() override
	{
		_pEngine->mul();
		_pEngine->baseMod(false);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		_pEngine->copy(dst, src);
	}

	void info() const override
	{
		_pEngine->info();
	}
};
