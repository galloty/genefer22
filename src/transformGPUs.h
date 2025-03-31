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

// #define USE_WI	1

#define VSIZE	4

// #define CHECK_ALL_FUNCTIONS		1
// #define CHECK_RADIX4_FUNCTIONS	1
// #define CHECK_FUNC_1		1	// GFN-11&12&13: square22, square4, square32, forward4_0, forward64, forward256_0, forward256
// #define CHECK_FUNC_2		1	// GFN-11&12&13: square64, square128, forward4, forward64_0, forward1024_0
// #define CHECK_FUNC_3		1	// GFN-12&13: square256, square512
// #define CHECK_FUNC_4		1	// GFN-12&13&14: square1024, square2048, forward1024

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

#define	P1S			(127 * (uint32(1) << 24) + 1)
#define	Q1S			2164260865u		// p * q = 1 (mod 2^32)
#define	R1S			33554430u		// 2^32 mod p
// #define	RSQ1S		402124772u		// (2^32)^2 mod p
#define	H1S			100663290u		// Montgomery form of the primitive root 3
// #define	IM1S		1930170389u		// MF of MF of I = 3^{(p - 1)/4} to convert input into MF
// #define	SQRTI1S		1626730317u		// MF of 3^{(p - 1)/8}
// #define	ISQRTI1S	856006302u		// MF of i * sqrt(i)

#define	P2S			(63 * (uint32(1) << 25) + 1)
#define	Q2S			2181038081u
#define	R2S			67108862u
// #define	RSQ2S		2111798781u
#define	H2S			335544310u		// MF of the primitive root 5
// #define	IM2S		1036950657u
// #define	SQRTI2S		338852760u
// #define	ISQRTI2S	1090446030u

#define	P3S			(15 * (uint32(1) << 27) + 1)
#define	Q3S			2281701377u
#define	R3S			268435454u
// #define	RSQ3S		1172168163u
#define	H3S			268435390u		// MF of the primitive root 31
// #define	IM3S		734725699u
// #define	SQRTI3S		1032137103u
// #define	ISQRTI3S	1964242958u

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
typedef ZPT<P3S, Q3S, R3S, H3S> ZP3;

#if VSIZE == 4
#define LVSIZE	2
#elif VSIZE == 2
#define LVSIZE	1
#else
#define LVSIZE	0
#endif

// Warning: DECLARE_VAR_32/64/128/256 in kernerl.cl must be modified if BLKxx = 1 or != 1.

#define BLK32m		32		// local size =  4KB, workgroup size = 256
#define BLK64m		16		// local size =  4KB, workgroup size = 256
#define BLK128m		8		// local size =  4KB, workgroup size = 256
#define BLK256m		4		// local size =  4KB, workgroup size = 256
#define BLK512m		2		// local size =  4KB, workgroup size = 256
//      BLK1024m	1		   local size =  4KB, workgroup size = 256
//      BLK2048m	1		   local size =  8KB, workgroup size = 512

#define CHUNK64m	4		// local size =  VSIZE * 1KB, workgroup size = 64
#define CHUNK256m	2		// local size =  VSIZE * 2KB, workgroup size = 128
#define CHUNK1024m	1		// local size =  VSIZE * 4KB, workgroup size = 256

#define CREATE_TRANSFORM_KERNEL(name) _##name = createTransformKernel(#name);
#define CREATE_TRANSFORM_KERNELP(name) _##name = createTransformKernel(#name, false);
#define CREATE_MUL_KERNEL(name) _##name = createMulKernel(#name);
#define CREATE_NORMALIZE_KERNEL(name, b, b_inv, b_s) _##name = createNormalizeKernel(#name, b, b_inv, b_s);
#define CREATE_SETCOPY_KERNEL(name) _##name = createSetCopyKernel(#name);
#define CREATE_COPYP_KERNEL(name) _##name = createCopypKernel(#name);

#define DEFINE_FORWARD(u) void forward##u(const int lm) { ek_fb(_forward##u, lm - LVSIZE, u / 4 * CHUNK##u##m, 4 * VSIZE); }
#define DEFINE_BACKWARD(u) void backward##u(const int lm) { ek_fb(_backward##u, lm - LVSIZE, u / 4 * CHUNK##u##m, 4 * VSIZE); }
#define DEFINE_FORWARD0(u) void forward##u##_0() { ek(_forward##u##_0, u / 4 * CHUNK##u##m, 4 * VSIZE); }

#define DEFINE_SQUARE(u) void square##u() { ek(_square##u, std::min(_n / (4 * VSIZE), size_t(u / (4 * VSIZE) * BLK##u##m)), 4 * VSIZE); }
#define DEFINE_FWDP(u) void fwd##u##p() { ek(_fwd##u##p, std::min(_n / (4 * VSIZE), size_t(u / (4 * VSIZE) * BLK##u##m)), 4 * VSIZE); }
#define DEFINE_MUL(u) void mul##u() { ek(_mul##u, std::min(_n / (4 * VSIZE), size_t(u / (4 * VSIZE) * BLK##u##m)), 4 * VSIZE); }

#define DEFINE_FORWARDP(u) \
	void forward##u##p(const int lm) { setTransformArgs(_forward##u, false); forward##u(lm); setTransformArgs(_forward##u);	}
#define DEFINE_FORWARDP0(u) \
	void forward##u##p_0() { setTransformArgs(_forward##u##_0, false); forward##u##_0(); setTransformArgs(_forward##u##_0);	}


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
	cl_kernel _square2x2 = nullptr, _square4 = nullptr, _square8 = nullptr;
	cl_kernel _fwd4p = nullptr, _fwd8p = nullptr;
	cl_kernel _mul2x2 = nullptr, _mul4 = nullptr, _mul8 = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward64_0 = nullptr;
	cl_kernel _forward256 = nullptr, _backward256 = nullptr, _forward256_0 = nullptr;
	cl_kernel _forward1024 = nullptr, _backward1024 = nullptr, _forward1024_0 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr, _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr;
	cl_kernel _fwd32p = nullptr, _fwd64p = nullptr, _fwd128p = nullptr, _fwd256p = nullptr, _fwd512p = nullptr, _fwd1024p = nullptr, _fwd2048p = nullptr;
	cl_kernel _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr, _mul256 = nullptr, _mul512 = nullptr, _mul1024 = nullptr, _mul2048 = nullptr;
	cl_kernel _normalize1 = nullptr, _normalize2 = nullptr, _mulscalar = nullptr;
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
			_w = _createBuffer(CL_MEM_READ_ONLY, RNS_SIZE * n * sizeof(ZP));
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

	cl_kernel createSetCopyKernel(const char * const kernelName)
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
		const int ln = _ln;

		CREATE_TRANSFORM_KERNEL(forward4);
		CREATE_TRANSFORM_KERNEL(backward4);
		CREATE_TRANSFORM_KERNEL(forward4_0);

		CREATE_TRANSFORM_KERNEL(square2x2);
		CREATE_TRANSFORM_KERNEL(square4);
		CREATE_TRANSFORM_KERNEL(square8);

		CREATE_TRANSFORM_KERNELP(fwd4p);
		CREATE_TRANSFORM_KERNELP(fwd8p);

		CREATE_MUL_KERNEL(mul2x2);
		CREATE_MUL_KERNEL(mul4);
		CREATE_MUL_KERNEL(mul8);

#if !defined(CHECK_RADIX4_FUNCTIONS)
		CREATE_TRANSFORM_KERNEL(forward64);
		CREATE_TRANSFORM_KERNEL(backward64);
		CREATE_TRANSFORM_KERNEL(forward64_0);
		CREATE_TRANSFORM_KERNEL(forward256);
		CREATE_TRANSFORM_KERNEL(backward256);
		CREATE_TRANSFORM_KERNEL(forward256_0);
		CREATE_TRANSFORM_KERNEL(forward1024);
		CREATE_TRANSFORM_KERNEL(backward1024);
		CREATE_TRANSFORM_KERNEL(forward1024_0);

		CREATE_TRANSFORM_KERNEL(square32);
		CREATE_TRANSFORM_KERNEL(square64);
		CREATE_TRANSFORM_KERNEL(square128);
		CREATE_TRANSFORM_KERNEL(square256);
		CREATE_TRANSFORM_KERNEL(square512);
		CREATE_TRANSFORM_KERNEL(square1024);
		CREATE_TRANSFORM_KERNEL(square2048);

		CREATE_TRANSFORM_KERNELP(fwd32p);
		CREATE_TRANSFORM_KERNELP(fwd64p);
		CREATE_TRANSFORM_KERNELP(fwd128p);
		CREATE_TRANSFORM_KERNELP(fwd256p);
		CREATE_TRANSFORM_KERNELP(fwd512p);
		CREATE_TRANSFORM_KERNELP(fwd1024p);
		CREATE_TRANSFORM_KERNELP(fwd2048p);

		CREATE_MUL_KERNEL(mul32);
		CREATE_MUL_KERNEL(mul64);
		CREATE_MUL_KERNEL(mul128);
		CREATE_MUL_KERNEL(mul256);
		CREATE_MUL_KERNEL(mul512);
		CREATE_MUL_KERNEL(mul1024);
		CREATE_MUL_KERNEL(mul2048);
#endif
		const cl_uint b_ui = static_cast<cl_uint>(b);
		const cl_int b_s = static_cast<cl_int>(31 - __builtin_clz(b) - 1);
		const cl_uint b_inv = static_cast<cl_uint>((static_cast<uint64_t>(1) << (b_s + 32)) / b);
		CREATE_NORMALIZE_KERNEL(normalize1, b_ui, b_inv, b_s);
		CREATE_NORMALIZE_KERNEL(normalize2, b_ui, b_inv, b_s);
		CREATE_NORMALIZE_KERNEL(mulscalar, b_ui, b_inv, b_s);

		CREATE_SETCOPY_KERNEL(set);
		CREATE_SETCOPY_KERNEL(copy);
		CREATE_COPYP_KERNEL(copyp);

		_pSplit = new splitter(size_t(ln), CHUNK256m, CHUNK1024m, sizeof(ZP), 11, getLocalMemSize(), getMaxWorkGroupSize());
	}

	void releaseKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		delete _pSplit;

		_releaseKernel(_forward4); _releaseKernel(_backward4); _releaseKernel(_forward4_0);
		_releaseKernel(_square2x2); _releaseKernel(_square4); _releaseKernel(_square8);
		_releaseKernel(_fwd4p); _releaseKernel(_fwd8p);
		_releaseKernel(_mul2x2); _releaseKernel(_mul4); _releaseKernel(_mul8);

		_releaseKernel(_forward64); _releaseKernel(_backward64); _releaseKernel(_forward64_0);
		_releaseKernel(_forward256); _releaseKernel(_backward256); _releaseKernel(_forward256_0);
		_releaseKernel(_forward1024); _releaseKernel(_backward1024); _releaseKernel(_forward1024_0);
		 
		_releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128); _releaseKernel(_square256);
		_releaseKernel(_square512); _releaseKernel(_square1024); _releaseKernel(_square2048);
		_releaseKernel(_fwd32p); _releaseKernel(_fwd64p); _releaseKernel(_fwd128p); _releaseKernel(_fwd256p);
		_releaseKernel(_fwd512p); _releaseKernel(_fwd1024p); _releaseKernel(_fwd2048p);
		_releaseKernel(_mul32); _releaseKernel(_mul64); _releaseKernel(_mul128); _releaseKernel(_mul256);
		_releaseKernel(_mul512); _releaseKernel(_mul1024); _releaseKernel(_mul2048);

		_releaseKernel(_normalize1); _releaseKernel(_normalize2); _releaseKernel(_mulscalar);

		_releaseKernel(_set); _releaseKernel(_copy); _releaseKernel(_copyp);
	}

///////////////////////////////

	void readMemory_z(ZP * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP)); }
	void writeMemory_z(const ZP * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, RNS_SIZE * _n * count * sizeof(ZP)); }
	void writeMemory_w(const ZP * const wPtr, const size_t offset) { _writeBuffer(_w, wPtr, _n * sizeof(ZP), offset * _n * sizeof(ZP)); }

///////////////////////////////

private:
	void ek(cl_kernel & kernel, const size_t localWorkSize, const size_t step)
	{
		const size_t n_s = _n / step;
		_executeKernel(kernel, RNS_SIZE * n_s, localWorkSize);
	}

	void ek_fb(cl_kernel & kernel, const int lm, const size_t localWorkSize, const size_t step)
	{
		const size_t n_s = _n / step;
		const cl_int ilm = static_cast<cl_int>(lm);
		const cl_uint is = static_cast<cl_uint>(n_s >> lm);
		_setKernelArg(kernel, 2, sizeof(cl_int), &ilm);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &is);
		_executeKernel(kernel, RNS_SIZE * n_s, localWorkSize);
	}

	void forward4(const int lm) { ek_fb(_forward4, lm - LVSIZE, 0, 4 * VSIZE); }
	void backward4(const int lm) { ek_fb(_backward4, lm - LVSIZE, 0, 4 * VSIZE); }
	void forward4_0() { ek(_forward4_0, 0, 4 * VSIZE); }

	void square4() { ek(_square4, 0, 4 * VSIZE); }
	void fwd4p() { ek(_fwd4p, 0, 4 * VSIZE); }
	void mul4() { ek(_mul4, 0, 4 * VSIZE); }
	
#if VSIZE == 1
	void square8() { forward4(1); ek(_square2x2, 0, 4); backward4(1); }
	void fwd8p() { forward4p(1); }
	void mul8() { forward4(1); ek(_mul2x2, 0, 4); backward4(1); }
#else
	void square8() { ek(_square8, 0, 4 * VSIZE); }
	void fwd8p() { ek(_fwd8p, 0, 4 * VSIZE); }
	void mul8() { ek(_mul8, 0, 4 * VSIZE); }
#endif

	DEFINE_FORWARD(64);
	DEFINE_BACKWARD(64);
	DEFINE_FORWARD0(64);
	DEFINE_FORWARD(256);
	DEFINE_BACKWARD(256);
	DEFINE_FORWARD0(256);
	DEFINE_FORWARD(1024);
	DEFINE_BACKWARD(1024);
	DEFINE_FORWARD0(1024);

	DEFINE_SQUARE(32);
	DEFINE_SQUARE(64);
	DEFINE_SQUARE(128);
	DEFINE_SQUARE(256);
	DEFINE_SQUARE(512);
	void square1024() { ek(_square1024, 1024 / (4 * VSIZE), 4 * VSIZE); }
	void square2048() { ek(_square2048, 2048 / (4 * VSIZE), 4 * VSIZE); }

	DEFINE_FWDP(32);
	DEFINE_FWDP(64);
	DEFINE_FWDP(128);
	DEFINE_FWDP(256);
	DEFINE_FWDP(512);
	void fwd1024p() { ek(_fwd1024p, 1024 / (4 * VSIZE), 4 * VSIZE); }
	void fwd2048p() { ek(_fwd2048p, 2048 / (4 * VSIZE), 4 * VSIZE); }

	DEFINE_MUL(32);
	DEFINE_MUL(64);
	DEFINE_MUL(128);
	DEFINE_MUL(256);
	DEFINE_MUL(512);
	void mul1024() { ek(_mul1024, 1024 / (4 * VSIZE), 4 * VSIZE); }
	void mul2048() { ek(_mul2048, 2048 / (4 * VSIZE), 4 * VSIZE); }

	void setTransformArgs(cl_kernel & kernel, const bool isMultiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
	}

	DEFINE_FORWARDP(4);
	DEFINE_FORWARDP(64);
	DEFINE_FORWARDP(256);
	DEFINE_FORWARDP(1024);

	DEFINE_FORWARDP0(4);
	DEFINE_FORWARDP0(64);
	DEFINE_FORWARDP0(256);
	DEFINE_FORWARDP0(1024);

private:
	void _mul(const size_t sIndex, const bool isSquare)
	{
#if defined(CHECK_FUNC_1)
		if (_ln == 11) { forward256_0(); if (isSquare) square8(); else mul8(); backward256(11 - 8); return; }
		if (_ln == 12) { forward4_0(); forward256(12 - 10); if (isSquare) square4(); else mul4(); backward256(12 - 10); backward4(12 - 2); return; }
		if (_ln == 13) { forward4_0(); forward64(13 - 8); if (isSquare) { square32(); } else mul32(); backward64(13 - 8); backward4(13 - 2); return; }
#endif
#if defined(CHECK_FUNC_2)
		if (_ln == 11) { forward4_0(); forward4(11 - 4); if (isSquare) square128(); else mul128(); backward4(11 - 4); backward4(11 - 2); return; }
		if (_ln == 12) { forward64_0(); if (isSquare) square64(); else mul64(); backward64(12 - 6); return; }
		if (_ln == 13) { forward1024_0(); if (isSquare) square8(); else mul8(); backward1024(13 - 10); return; }
#endif
#if defined(CHECK_FUNC_3)
		if (_ln == 12) { forward4_0(); forward4(12 - 4); if (isSquare) square256(); else mul256(); backward4(12 - 4); backward4(12 - 2); return; }
		if (_ln == 13) { forward4_0(); forward4(13 - 4); if (isSquare) square512(); else mul512(); backward4(13 - 4); backward4(13 - 2); return; }
#endif
#if defined(CHECK_FUNC_4)
		if (_ln == 12) { forward4_0(); if (isSquare) square1024(); else mul1024(); backward4(12 - 2); return; }
		if (_ln == 13) { forward4_0(); if (isSquare) square2048(); else mul2048(); backward4(13 - 2); return; }
		if (_ln == 14) { forward4_0(); forward1024(14 - 12); if (isSquare) square4(); else mul4(); backward1024(14 - 12); backward4(14 - 2); return; }
#endif

		int lm = _ln;

#if defined(CHECK_RADIX4_FUNCTIONS)
		lm -= 2; forward4_0();
		while (lm > 3) { lm -= 2; forward4(lm); }
		if (isSquare) { if (lm == 3) square8(); else square4(); } else { if (lm == 3) mul8(); else mul4(); }
		while (lm < _ln) { backward4(lm); lm += 2; }
		return;
#endif
		const splitter * const pSplit = _pSplit;
		const size_t s = pSplit->getPartSize(sIndex);

		const uint32_t k0 = pSplit->getPart(sIndex, 0);
		lm -= int(k0);
		if (k0 == 10) forward1024_0();
		else if (k0 == 8) forward256_0();
		else forward64_0();	// k0 = 6

		for (size_t i = 2; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			lm -= int(k);
			if (k == 10) forward1024(lm);
			else if (k == 8) forward256(lm);
			else forward64(lm); // k = 6
		}

		// lm = pSplit->getPart(sIndex, s - 1);
		if (isSquare)
		{
			if (lm == 11) square2048();
			else if (lm == 10) square1024();
			else if (lm == 9) square512();
			else if (lm == 8) square256();
			else if (lm == 7) square128();
			else if (lm == 6) square64();
			else if (lm == 5) square32();
		}
		else
		{
			if (lm == 11) mul2048();
			else if (lm == 10) mul1024();
			else if (lm == 9) mul512();
			else if (lm == 8) mul256();
			else if (lm == 7) mul128();
			else if (lm == 6) mul64();
			else if (lm == 5) mul32();
		}

		for (size_t i = s - 1; i > 0; --i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10) backward1024(lm);
			else if (k == 8) backward256(lm);
			else backward64(lm);	// k = 6
			lm += int(k);
		}
	}

public:
	void square()
	{
#if defined(CHECK_ALL_FUNCTIONS)
		_mul(size_t(rand()) % _pSplit->getSize(), true);
#else
		_mul(_splitIndex, true);
#endif
		if (_first)
		{
			info();
			_first = false;
		} 
	}

	void mul()
	{
		_mul(_splitIndex, false);
	}

	void initMultiplicand(const size_t src)
	{
		const cl_uint isrc = static_cast<cl_uint>(src * RNS_SIZE * _n);
		_setKernelArg(_copyp, 2, sizeof(cl_uint), &isrc);
		_executeKernel(_copyp, RNS_SIZE * _n);

#if defined(CHECK_FUNC_1)
		if (_ln == 11) { forward256p_0(); fwd8p(); return; }
		if (_ln == 12) { forward4p_0(); forward256p(12 - 10); fwd4p(); return; }
		if (_ln == 13) { forward4p_0(); forward64p(13 - 8); fwd32p(); return; }
#endif
#if defined(CHECK_FUNC_2)
		if (_ln == 11) { forward4p_0(); forward4p(11 - 4); fwd128p(); return; }
		if (_ln == 12) { forward64p_0(); fwd64p(); return; }
		if (_ln == 13) { forward1024p_0(); fwd8p(); return; }
#endif
#if defined(CHECK_FUNC_3)
		if (_ln == 12) { forward4p_0(); forward4p(12 - 4); fwd256p(); return; }
		if (_ln == 13) { forward4p_0(); forward4p(13 - 4); fwd512p(); return; }
#endif
#if defined(CHECK_FUNC_4)
		if (_ln == 12) { forward4p_0(); fwd1024p(); return; }
		if (_ln == 13) { forward4p_0(); fwd2048p(); return; }
		if (_ln == 14) { forward4p_0(); forward1024p(14 - 12); fwd4p(); return; }
#endif

		const splitter * const pSplit = _pSplit;
#if defined(CHECK_ALL_FUNCTIONS)
		_splitIndex = size_t(rand()) % pSplit->getSize();
#endif

		int lm = _ln;

#if defined(CHECK_RADIX4_FUNCTIONS)
		lm -= 2; forward4p_0();
		while (lm > 3) { lm -= 2; forward4p(lm); }
		if (lm == 3) fwd8p(); else fwd4p();
		return;
#endif

		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		const uint32_t k0 = pSplit->getPart(sIndex, 0);
		lm -= int(k0);
		if (k0 == 10) forward1024p_0();
		else if (k0 == 8) forward256p_0();
		else forward64p_0();	// k0 = 6

		for (size_t i = 2; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			lm -= int(k);
			if (k == 10) forward1024p(lm);
			else if (k == 8) forward256p(lm);
			else forward64p(lm);	// k = 6
		}

		// lm = pSplit->getPart(sIndex, s - 1);
		if (lm == 11) fwd2048p();
		else if (lm == 10) fwd1024p();
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
			_mul(sIndex, true);
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
			if (RNS_SIZE >= 3) Z[2 * n + i] = ZP3().set_int(static_cast<int32>((P1S - 1) * cos(i + 0.47)));
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
		for (size_t sIndex = 0, ns = _pSplit->getSize(); sIndex < ns; ++sIndex)
		{
			int lm = _ln;
			const size_t s = _pSplit->getPartSize(sIndex);
			for (size_t i = 1; i < s; ++i)
			{
				const uint32_t k = _pSplit->getPart(sIndex, i - 1);
				lm -= int(k);
				ss << " " << k;
				if (i != 1) ss << "(" << lm << ")"; else ss << "_0";
			}
			ss << " s" << lm;

			if (sIndex == _splitIndex) ss << " *";
			ss << ",";
		}

		ss << " blk = " << _baseModBlk << ", wsize1 = " << _naLocalWS << ", wsize2 = " << _nbLocalWS << "." << std::endl;
		pio::display(ss.str());
	}
};


template<size_t RNS_SIZE>
class transformGPUs : public transform
{
private:
	const size_t _mem_size, _cache_size;
	const size_t _num_regs;
	ZP * const _z;
	engines<RNS_SIZE> * _pEngine = nullptr;

public:
	transformGPUs(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
				 const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
		: transform(size_t(1) << n, n, b, (RNS_SIZE == 2) ? EKind::NTT2s : EKind::NTT3s),
#if defined(USE_WI)
		_mem_size(RNS_SIZE * (size_t(1) << n) * (num_regs + 2) * sizeof(ZP) + (size_t(1) << n) / 4 * sizeof(cl_long)),
		_cache_size(RNS_SIZE * (size_t(1) << n) * 2 * sizeof(ZP)),
#else
		_mem_size(RNS_SIZE * (size_t(1) << n) * (2 * num_regs + 3) / 2 * sizeof(ZP) + (size_t(1) << n) / 4 * sizeof(cl_long)),
		_cache_size(RNS_SIZE * (size_t(1) << n) * 3 / 2 * sizeof(ZP)),
#endif
		_num_regs(num_regs), _z(new ZP[RNS_SIZE * (size_t(1) << n) * num_regs])
	{
		const size_t size = getSize();

		const bool is_boinc_platform = isBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pEngine = new engines<RNS_SIZE>(eng_platform, is_boinc_platform ? 0 : device, static_cast<int>(n), isBoinc, num_regs, verbose);

		std::ostringstream src;

		src << "#define N_SZ\t" << (1u << n) << "u" << std::endl;
		src << "#define LN_SZ\t" << n << std::endl;
		src << "#define RNS_SZ\t" << RNS_SIZE << std::endl;
		src << "#define VSIZE\t" << VSIZE << std::endl;
		src << "#define LVSIZE\t" << LVSIZE << std::endl;

		src << "#define NORM1\t" << ZP1::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM2\t" << ZP2::norm(uint32(size / 2)).get() << "u" << std::endl;
		src << "#define NORM3\t" << ZP3::norm(uint32(size / 2)).get() << "u" << std::endl;

		src << "#define W_SHFT\t" << size << "u" << std::endl;
		src << "#define WI_SHFT\t" << size / 2 << "u" << std::endl;
#if defined(USE_WI)
		src << "#define USE_WI\t" << 1 << std::endl;
#endif
		src << "#define BLK32\t" << BLK32m << std::endl;
		src << "#define BLK64\t" << BLK64m << std::endl;
		src << "#define BLK128\t" << BLK128m << std::endl;
		src << "#define BLK256\t" << BLK256m << std::endl;
		src << "#define BLK512\t" << BLK512m << std::endl;

		src << "#define CHUNK64\t" << CHUNK64m << std::endl;
		src << "#define CHUNK256\t" << CHUNK256m << std::endl;
		src << "#define CHUNK1024\t" << CHUNK1024m << std::endl;

#if defined(CHECK_RADIX4_FUNCTIONS)
		src << "#define SHORT_VER\t" << 1 << std::endl;
#endif
		src << "#define MAX_WG_SZ\t" << _pEngine->getMaxWorkGroupSize() << std::endl << std::endl;

		if (isBoinc || !_pEngine->readOpenCL("ocl/kernels.cl", "src/ocl/kernels.h", "src_ocl_kernels", src)) src << src_ocl_kernels;

		_pEngine->loadProgram(src.str());
		_pEngine->allocMemory();
		_pEngine->createKernels(b);

		ZP1 * const wr1 = new ZP1[size];
		ZP2 * const wr2 = new ZP2[size];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const ZP1 r_s1 = ZP1::primroot_n(4 * s);
			const ZP2 r_s2 = ZP2::primroot_n(4 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const size_t sj = s + j, sji = s + (s - j - 1);
				const size_t e = bitRev(j, 2 * s) + 1;
				wr1[size / 2 + sji] = wr1[sj] = r_s1.pow(e);
				wr2[size / 2 + sji] = wr2[sj] = r_s2.pow(e);
			}
		}
		_pEngine->writeMemory_w(wr1, 0);
		_pEngine->writeMemory_w(wr2, 1);
		delete[] wr1;
		delete[] wr2;

		if (RNS_SIZE == 3)
		{
			ZP3 * const wr3 = new ZP3[size];
			for (size_t s = 1; s < size / 2; s *= 2)
			{
				const ZP3 r_s3 = ZP3::primroot_n(4 * s);
				for (size_t j = 0; j < s; ++j)
				{
					wr3[size / 2 + s + (s - j - 1)] = wr3[s + j] = r_s3.pow(bitRev(j, 2 * s) + 1);
				}
			}
			_pEngine->writeMemory_w(wr3, 2);
			delete[] wr3;
		}

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
	size_t getCacheSize() const override { return _cache_size; }

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
			if (RNS_SIZE >= 3) z[2 * size + i] = ZP3().set_int(zi[i]);
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
