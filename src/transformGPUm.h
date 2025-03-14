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

#include "ocl/kernel2m.h"
#include "ocl/kernel3m.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;

#define	M31		((uint32(1) << 31) - 1)
#define	P1M		(127 * (uint32(1) << 24) + 1)
#define	P2M		(63 * (uint32(1) << 25) + 1)

// GF((2^31 - 1)^2)
class GF31
{
private:
	static const uint32 _p = M31;
	static const uint32 _h_order = uint32(1) << 31;
	static const uint32 _h_0 = 105066u, _h_1 = 333718u;	// a primitive root of order 2^31 which is a root of (0, 1)
	cl_uint2 _n;

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= _p) ? _p : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? _p : 0); }

	static uint32 _mul(const uint32 a, const uint32 b)
	{
		const uint64 t = a * uint64(b);
		const uint32 lo = uint32(t) & _p, hi = uint32(t >> 31);
		return _add(hi, lo);
	}

	static int32 _get_int(const uint32 a) { return (a >= _p / 2) ? int32(a - _p) : int32(a); }
	static uint32 _set_int(const int32 a) { return (a < 0) ? (uint32(a) + _p) : uint32(a); }

public:
	GF31() {}
	explicit GF31(const uint32 n0, const uint32 n1) { _n.s[0] = n0; _n.s[1] = n1; }

	void get_int(int32 & i0, int32 & i1) const { i0 = _get_int(_n.s[0]); i1 = _get_int(_n.s[1]); }
	GF31 & set_int(const int32 i0, const int32 i1) { _n.s[0] = _set_int(i0); _n.s[1] = _set_int(i1); return *this; }

	GF31 mul(const GF31 & rhs) const
	{
		return GF31(_sub(_mul(_n.s[0], rhs._n.s[0]), _mul(_n.s[1], rhs._n.s[1])),
					_add(_mul(_n.s[1], rhs._n.s[0]), _mul(_n.s[0], rhs._n.s[1])));
	}
	GF31 sqr() const
	{
		const uint32 t = _mul(_n.s[0], _n.s[1]);
		return GF31(_sub(_mul(_n.s[0], _n.s[0]), _mul(_n.s[1], _n.s[1])), _add(t, t));
	}

	GF31 pow(const size_t e) const
	{
		if (e == 0) return GF31(1, 0);
		GF31 r = GF31(1, 0), y = *this;
		for (size_t i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF31 primroot_n(const uint32 n) { return GF31(_h_0, _h_1).pow(_h_order / n); }
};

class ZP1
{
private:
	static const uint32 _p = P1M;
	static const uint32 _q = 2164260865u;	// p * q = 1 (mod 2^32)
	static const uint32 _r2 = 402124772u;	// (2^32)^2 mod p
	static const uint32 _h = 167772150u;	// Montgomery form of the primitive root 5
	static const uint32 _i = 66976762u; 	// Montgomery form of 5^{(p + 1)/4} = 16711679
	cl_uint2 _n;

	static uint32 _add(const uint32 a, const uint32 b) { const uint32 t = a + b; return t - ((t >= _p) ? _p : 0); }
	static uint32 _sub(const uint32 a, const uint32 b) { const uint32 t = a - b; return t + ((int32(t) < 0) ? _p : 0); }

	static uint32 _mul(const uint32 lhs, const uint32 rhs)
	{
		const uint64 t = lhs * uint64(rhs);
		const uint32 lo = uint32(t), hi = uint32(t >> 32);
		const uint32 mp = uint32(((lo * _q) * uint64(_p)) >> 32);
		return _sub(hi, mp);
	}

	static int32 _get_int(const uint32 a) { return (a >= _p / 2) ? int32(a - _p) : int32(a); }
	static uint32 _set_int(const int32 a) { return (a < 0) ? (uint32(a) + _p) : uint32(a); }

	ZP1 pow(const size_t e) const
	{
		static const uint32 one = -_p * 2u;	// Montgomery form of 1 is 2^32 (mod p)
		if (e == 0) return ZP1(one, 0);
		ZP1 r = ZP1(one, 0), y = *this;
		for (size_t i = e; i != 1; i /= 2)
		{
			if (i % 2 != 0) r._n.s[0] = _mul(r._n.s[0], y._n.s[0]);
			y._n.s[0] = _mul(y._n.s[0], y._n.s[0]);
		}
		r._n.s[0] = _mul(r._n.s[0], y._n.s[0]);
		return r;
	}

public:
	ZP1() {}
	explicit ZP1(const uint32 n0, const uint32 n1) { _n.s[0] = n0; _n.s[1] = n1; }

	ZP1 & set_int(const int32 i0, const int32 i1) { _n.s[0] = _set_int(i0); _n.s[1] = _set_int(i1); return *this; }

	ZP1 muli() const { return ZP1(_mul(_n.s[0], _i), _mul(_n.s[1], _i)); }
	ZP1 sqr() const { return ZP1(_mul(_n.s[0], _n.s[0]), _mul(_n.s[1], _n.s[1])); }

	// Conversion into / out of Montgomery form
	// ZP1 toMonty() const { return ZP1(_mul(_n.s[0], _r2), _mul(_n.s[1], _r2)); }
	// ZP1 fromMonty() const { return ZP1(_mul(_n.s[0], 1), _mul(_n.s[1], 1)); }

	ZP1 pow_mul_sqr(const size_t e) const { ZP1 r = pow(e); r._n.s[1] = _mul(r._n.s[0], _n.s[1]); return r; }

	static const ZP1 primroot_n(const uint32 n) { ZP1 r = ZP1(_h, 0).pow((_p - 1) / n); r._n.s[1] = _mul(r._n.s[0], r._n.s[0]); return r; }
	static uint32 norm(const uint32 n) { return _p - (_p - 1) / n; }
};

class ZP2
{
private:
	static const uint32 _p = P2M;
	cl_uint2 _n;

public:
	ZP2() {}

	ZP2 & set_int(const int32, const int32) { return *this; }
};

// Warning: DECLARE_VAR_32/64/128/256 in kernerl.cl must be modified if BLKxx = 1 or != 1.

#define BLK32m		8
#define BLK64m		4
#define BLK128m		2
#define BLK256m		1

#define CHUNK64m	4
#define CHUNK256m	2
#define CHUNK1024m	1

template<size_t M_SIZE>
class engineg : public device
{
private:
	const size_t _n;
	const int _ln;
	const bool _isBoinc;
	const size_t _num_regs;
	cl_mem _z = nullptr, _zp = nullptr, _w = nullptr, _c = nullptr;
	cl_kernel _forward4 = nullptr, _backward4 = nullptr, _forward4_0 = nullptr, _backward4_0 = nullptr, _square22 = nullptr, _square4 = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward256 = nullptr, _backward256 = nullptr, _forward1024 = nullptr, _backward1024 = nullptr;
	cl_kernel _forward64_0 = nullptr, _backward64_0 = nullptr, _forward256_0 = nullptr, _backward256_0 = nullptr, _forward1024_0 = nullptr, _backward1024_0 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr, _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr;
	cl_kernel _normalize1 = nullptr, _normalize2 = nullptr, _mulscalar = nullptr;
	cl_kernel _fwd32p = nullptr, _fwd64p = nullptr, _fwd128p = nullptr, _fwd256p = nullptr, _fwd512p = nullptr, _fwd1024p = nullptr, _fwd2048p = nullptr;
	cl_kernel _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr, _mul256 = nullptr, _mul512 = nullptr, _mul1024 = nullptr, _mul2048 = nullptr;
	cl_kernel _set = nullptr, _copy = nullptr, _copyp = nullptr;
	splitter * _pSplit = nullptr;
	size_t _naLocalWS = 32, _nbLocalWS = 32, _baseModBlk = 16, _splitIndex = 0;
	bool _first = true;

public:
	engineg(const platform & platform, const size_t d, const int ln, const bool isBoinc, const size_t num_regs, const bool verbose)
		: device(platform, d, verbose), _n(size_t(1) << ln), _ln(ln), _isBoinc(isBoinc), _num_regs(num_regs) {}
	virtual ~engineg() {}

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
			_z = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * M_SIZE * n * _num_regs);
			_zp = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint2) * M_SIZE * n);
			_w = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_uint2) * M_SIZE * 3 * n / 2);
			_c = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_long2) * n / 4);
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
		_backward4_0 = createTransformKernel("backward4_0");

		_square22 = createTransformKernel("square22");
		_square4 = createTransformKernel("square4");

		_forward64 = createTransformKernel("forward64");
		_backward64 = createTransformKernel("backward64");
		_forward256 = createTransformKernel("forward256");
		_backward256 = createTransformKernel("backward256");
		_forward1024 = createTransformKernel("forward1024");
		_backward1024 = createTransformKernel("backward1024");

		_forward64_0 = createTransformKernel("forward64_0");
		_backward64_0 = createTransformKernel("backward64_0");
		// _forward256_0 = createTransformKernel("forward256_0");
		// _backward256_0 = createTransformKernel("backward256_0");
		// _forward1024_0 = createTransformKernel("forward1024_0");
		// _backward1024_0 = createTransformKernel("backward1024_0");

		_square32 = createTransformKernel("square32");
		_square64 = createTransformKernel("square64");
		_square128 = createTransformKernel("square128");
		_square256 = createTransformKernel("square256");
		_square512 = createTransformKernel("square512");
		_square1024 = createTransformKernel("square1024");
		_square2048 = createTransformKernel("square2048");

		const cl_int b_s = static_cast<cl_int>(31 - __builtin_clz(b) - 1);
		const cl_uint b_inv = static_cast<cl_uint>((static_cast<uint64_t>(1) << (b_s + 32)) / b);
		_normalize1 = createNormalizeKernel("normalize1", static_cast<cl_uint>(b), b_inv, b_s);
		_normalize2 = createNormalizeKernel("normalize2", static_cast<cl_uint>(b), b_inv, b_s);
		_mulscalar = createNormalizeKernel("mulscalar", static_cast<cl_uint>(b), b_inv, b_s);

		_fwd32p = createTransformKernel("fwd32p", false);
		_fwd64p = createTransformKernel("fwd64p", false);
		_fwd128p = createTransformKernel("fwd128p", false);
		_fwd256p = createTransformKernel("fwd256p", false);
		_fwd512p = createTransformKernel("fwd512p", false);
		_fwd1024p = createTransformKernel("fwd1024p", false);
		_fwd2048p = createTransformKernel("fwd2048p", false);

		_mul32 = createMulKernel("mul32");
		_mul64 = createMulKernel("mul64");
		_mul128 = createMulKernel("mul128");
		_mul256 = createMulKernel("mul256");
		_mul512 = createMulKernel("mul512");
		_mul1024 = createMulKernel("mul1024");
		_mul2048 = createMulKernel("mul2048");

		_set = createSetKernel("set");
		_copy = createCopyKernel("copy");
		_copyp = createCopypKernel("copyp");

		_pSplit = new splitter(size_t(_ln), CHUNK256m, CHUNK1024m, sizeof(GF31), 11, getLocalMemSize(), getMaxWorkGroupSize());

		std::ostringstream ss; ss << std::endl << "split:";
		for (size_t i = 0, ns = _pSplit->getSize(); i < ns; ++i)
		{
			for (size_t j = 0, nps = _pSplit->getPartSize(i); j < nps; ++j) ss << " " << _pSplit->getPart(i, j);
			ss << ",";
		}
		ss << std::endl;
		pio::display(ss.str());
	}

	void releaseKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		delete _pSplit;

		_releaseKernel(_forward4); _releaseKernel(_backward4);
		_releaseKernel(_forward4_0); _releaseKernel(_backward4_0);
		_releaseKernel(_square22); _releaseKernel(_square4);
		_releaseKernel(_forward64); _releaseKernel(_backward64);
		_releaseKernel(_forward256); _releaseKernel(_backward256);
		_releaseKernel(_forward1024); _releaseKernel(_backward1024);
		_releaseKernel(_forward64_0); _releaseKernel(_backward64_0);
		_releaseKernel(_forward256_0); _releaseKernel(_backward256_0);
		_releaseKernel(_forward1024_0); _releaseKernel(_backward1024_0);
		_releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128); _releaseKernel(_square256);
		_releaseKernel(_square512); _releaseKernel(_square1024); _releaseKernel(_square2048);
		_releaseKernel(_normalize1); _releaseKernel(_normalize2); _releaseKernel(_mulscalar);
		_releaseKernel(_fwd32p); _releaseKernel(_fwd64p); _releaseKernel(_fwd128p); _releaseKernel(_fwd256p);
		_releaseKernel(_fwd512p); _releaseKernel(_fwd1024p); _releaseKernel(_fwd2048p);
		_releaseKernel(_mul32); _releaseKernel(_mul64); _releaseKernel(_mul128); _releaseKernel(_mul256);
		_releaseKernel(_mul512); _releaseKernel(_mul1024); _releaseKernel(_mul2048);
		_releaseKernel(_set); _releaseKernel(_copy); _releaseKernel(_copyp);
	}

///////////////////////////////

	void readMemory_z31(GF31 * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, sizeof(GF31) * _n * count); }
	void readMemory_z1(ZP1 * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, sizeof(ZP1) * _n * count, sizeof(GF31) * _n * _num_regs); }
	void readMemory_z2(ZP2 * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, sizeof(ZP2) * _n * count, (sizeof(GF31) + sizeof(ZP1)) * _n * _num_regs); }

	void writeMemory_z31(const GF31 * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, sizeof(GF31) * _n * count); }
	void writeMemory_z1(const ZP1 * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, sizeof(ZP1) * _n * count, sizeof(GF31) * _n * _num_regs); }
	void writeMemory_z2(const ZP2 * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, sizeof(ZP2) * _n * count, (sizeof(GF31) + sizeof(ZP1)) * _n * _num_regs); }

	void writeMemory_w31(const GF31 * const wPtr) { _writeBuffer(_w, wPtr, sizeof(GF31) * 3 * _n / 2); }
	void writeMemory_w1(const ZP1 * const wPtr) { _writeBuffer(_w, wPtr, sizeof(ZP1) * 3 * _n / 2, sizeof(GF31) * 3 * _n / 2); }
	void writeMemory_w2(const ZP2 * const wPtr) { _writeBuffer(_w, wPtr, sizeof(ZP2) * 3 * _n / 2, (sizeof(GF31) + sizeof(ZP1)) * 3 * _n / 2); }

///////////////////////////////

private:
	void fb(cl_kernel & kernel, const int lm, const size_t localWorkSize)
	{
		const size_t n_4 = _n / 4;
		const cl_int ilm = static_cast<cl_int>(lm);
		const cl_uint is = static_cast<cl_uint>(n_4 >> lm);
		_setKernelArg(kernel, 2, sizeof(cl_int), &ilm);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &is);
		_executeKernel(kernel, n_4, localWorkSize);
	}

	void forward4(const int lm) { fb(_forward4, lm, 0); }
	void backward4(const int lm) { fb(_backward4, lm, 0); }
	void forward4_0() { const size_t n_4 = _n / 4; _executeKernel(_forward4_0, n_4, 0); }
	void backward4_0() { const size_t n_4 = _n / 4; _executeKernel(_backward4_0, n_4, 0); }
	void square22() { const size_t n_4 = _n / 4; _executeKernel(_square22, n_4, 0); }
	void square4() { const size_t n_4 = _n / 4; _executeKernel(_square4, n_4, 0); }

	void forward64(const int lm) { fb(_forward64, lm, 64 / 4 * CHUNK64m); }
	void backward64(const int lm) { fb(_backward64, lm, 64 / 4 * CHUNK64m); }
	void forward256(const int lm) { fb(_forward256, lm, 256 / 4 * CHUNK256m); }
	void backward256(const int lm) { fb(_backward256, lm, 256 / 4 * CHUNK256m); }
	void forward1024(const int lm) { fb(_forward1024, lm, 1024 / 4 * CHUNK1024m); }
	void backward1024(const int lm) { fb(_backward1024, lm, 1024 / 4 * CHUNK1024m); }

	void forward64_0(const int lm) { fb(_forward64_0, lm, 64 / 4 * CHUNK64m); }
	void backward64_0(const int lm) { fb(_backward64_0, lm, 64 / 4 * CHUNK64m); }
	void forward256_0(const int lm) { fb(_forward256_0, lm, 256 / 4 * CHUNK256m); }
	void backward256_0(const int lm) { fb(_backward256_0, lm, 256 / 4 * CHUNK256m); }
	void forward1024_0(const int lm) { fb(_forward1024_0, lm, 1024 / 4 * CHUNK1024m); }
	void backward1024_0(const int lm) { fb(_backward1024_0, lm, 1024 / 4 * CHUNK1024m); }

	void square32() { const size_t n_4 = _n / 4; _executeKernel(_square32, n_4, std::min(n_4, size_t(32 / 4 * BLK32m))); }
	void square64() { const size_t n_4 = _n / 4; _executeKernel(_square64, n_4, std::min(n_4, size_t(64 / 4 * BLK64m))); }
	void square128() { const size_t n_4 = _n / 4; _executeKernel(_square128, n_4, std::min(n_4, size_t(128 / 4 * BLK128m))); }
	void square256() { const size_t n_4 = _n / 4; _executeKernel(_square256, n_4, std::min(n_4, size_t(256 / 4 * BLK256m))); }
	void square512() { const size_t n_4 = _n / 4; _executeKernel(_square512, n_4, 512 / 4); }
	void square1024() { const size_t n_4 = _n / 4; _executeKernel(_square1024, n_4, 1024 / 4); }
	void square2048() { const size_t n_4 = _n / 4; _executeKernel(_square2048, n_4, 2048 / 4); }

	void fwd32p() { const size_t n_4 = _n / 4; _executeKernel(_fwd32p, n_4, std::min(n_4, size_t(32 / 4 * BLK32m))); }
	void fwd64p() { const size_t n_4 = _n / 4; _executeKernel(_fwd64p, n_4, std::min(n_4, size_t(64 / 4 * BLK64m))); }
	void fwd128p() { const size_t n_4 = _n / 4; _executeKernel(_fwd128p, n_4, std::min(n_4, size_t(128 / 4 * BLK128m))); }
	void fwd256p() { const size_t n_4 = _n / 4; _executeKernel(_fwd256p, n_4, std::min(n_4, size_t(256 / 4 * BLK256m))); }
	void fwd512p() { const size_t n_4 = _n / 4; _executeKernel(_fwd512p, n_4, 512 / 4); }
	void fwd1024p() { const size_t n_4 = _n / 4; _executeKernel(_fwd1024p, n_4, 1024 / 4); }
	void fwd2048p() { const size_t n_4 = _n / 4; _executeKernel(_fwd2048p, n_4, 2048 / 4); }

	void mul32() { const size_t n_4 = _n / 4; _executeKernel(_mul32, n_4, std::min(n_4, size_t(32 / 4 * BLK32m))); }
	void mul64() { const size_t n_4 = _n / 4; _executeKernel(_mul64, n_4, std::min(n_4, size_t(64 / 4 * BLK64m))); }
	void mul128() { const size_t n_4 = _n / 4; _executeKernel(_mul128, n_4, std::min(n_4, size_t(128 / 4 * BLK128m))); }
	void mul256() { const size_t n_4 = _n / 4; _executeKernel(_mul256, n_4, std::min(n_4, size_t(256 / 4 * BLK256m))); }
	void mul512() { const size_t n_4 = _n / 4; _executeKernel(_mul512, n_4, 512 / 4); }
	void mul1024() { const size_t n_4 = _n / 4; _executeKernel(_mul1024, n_4, 1024 / 4); }
	void mul2048() { const size_t n_4 = _n / 4; _executeKernel(_mul2048, n_4, 2048 / 4); }

	void setTransformArgs(cl_kernel & kernel, const bool isMultiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
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

private:
	void _square(const size_t sIndex, const bool verbose)
	{
		const splitter * const pSplit = _pSplit;

		const size_t s = pSplit->getPartSize(sIndex);

		int lm = _ln;

		for (size_t i = 1; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10)
			{
				lm -= 10;
				if (i != 1) forward1024(lm); else forward1024_0(lm);
				if (verbose) std::cout << "forward1024 (" << lm << ") ";
			}
			else if (k == 8)
			{
				lm -= 8;
				if (i != 1) forward256(lm); else forward256_0(lm);
				if (verbose) std::cout << "forward256 (" << lm << ") ";
			}
			else // if (k == 6)
			{
				lm -= 6;
				if (i != 1) forward64(lm); else forward64_0(lm);
				if (verbose) std::cout << "forward64 (" << lm << ") ";
			}
		}

		// lm = split.GetPart(sIndex, s - 1);
		if (lm == 11) square2048();
		else if (lm == 10) square1024();
		else if (lm == 9) square512();
		else if (lm == 8) square256();
		else if (lm == 7) square128();
		else if (lm == 6) square64();
		else if (lm == 5) square32();
		if (verbose) std::cout << "square" << (1u << lm) << " ";

		for (size_t i = s - 1; i > 0; --i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10)
			{
				if (i != 1) backward1024(lm); else backward1024_0(lm);
				if (verbose) std::cout << "backward1024 (" << lm << ") ";
				lm += 10;
			}
			else if (k == 8)
			{
				if (i != 1) backward256(lm); else backward256_0(lm);
				if (verbose) std::cout << "backward256 (" << lm << ") ";
				lm += 8;
			}
			else // if (k == 6)
			{
				if (i != 1) backward64(lm); else backward64_0(lm);
				if (verbose) std::cout << "backward64 (" << lm << ") ";
				lm += 6;
			}
		}

		if (verbose) std::cout << std::endl;
	}

public:
	void square()
	{
		_square(_splitIndex, _first);
		if (_first) _first = false;
	}

private:
	void squareTune(const size_t count, const size_t sIndex, const GF31 * const Z31, ZP1 * const Z1, ZP2 * const Z2)
	{
		for (size_t j = 0; j != count; ++j)
		{
			writeMemory_z31(Z31); writeMemory_z1(Z1);
			if (M_SIZE == 3) writeMemory_z2(Z2);
			_square(sIndex, false);
		}
	}

public:
	void initMultiplicand(const size_t src)
	{
		const cl_uint isrc = static_cast<cl_uint>(src * _n);
		_setKernelArg(_copyp, 2, sizeof(cl_uint), &isrc);
		for (cl_uint i = 0; i < M_SIZE; ++i)
		{
			const cl_uint offset = _n * _num_regs * i;
			_setKernelArg(_copyp, 3, sizeof(cl_uint), &offset);
			_executeKernel(_copyp, _n);
		}

		const splitter * const pSplit = _pSplit;

		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		int lm = _ln;

		for (size_t i = 1; i < s; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i - 1);
			if (k == 10)
			{
				lm -= 10;
				forward1024p(lm);
			}
			else if (k == 8)
			{
				lm -= 8;
				forward256p(lm);
			}
			else // if (k == 6)
			{
				lm -= 6;
				forward64p(lm);
			}
		}

		// lm = split.GetPart(sIndex, s - 1);
		if (lm == 11) fwd2048p();
		else if (lm == 10) fwd1024p();
		else if (lm == 9) fwd512p();
		else if (lm == 8) fwd256p();
		else if (lm == 7) fwd128p();
		else if (lm == 6) fwd64p();
		else if (lm == 5) fwd32p();
	}

	void mul()
	{
		const splitter * const pSplit = _pSplit;

		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		int lm = _ln;

		for (size_t i = 0; i < s - 1; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i);
			if (k == 10)
			{
				lm -= 10;
				forward1024(lm);
			}
			else if (k == 8)
			{
				lm -= 8;
				forward256(lm);
			}
			else // if (k == 6)
			{
				lm -= 6;
				forward64(lm);
			}
		}

		// lm = split.GetPart(sIndex, s - 1);
		if (lm == 11) mul2048();
		else if (lm == 10) mul1024();
		else if (lm == 9) mul512();
		else if (lm == 8) mul256();
		else if (lm == 7) mul128();
		else if (lm == 6) mul64();
		else if (lm == 5) mul32();

		for (size_t i = 0; i < s - 1; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, s - 2 - i);
			if (k == 10)
			{
				backward1024(lm);
				lm += 10;
			}
			else if (k == 8)
			{
				backward256(lm);
				lm += 8;
			}
			else // if (k == 6)
			{
				backward64(lm);
				lm += 6;
			}
		}
	}

	void set(const uint32_t a)
	{
		const cl_uint ia = static_cast<cl_uint>(a);
		_setKernelArg(_set, 1, sizeof(cl_uint), &ia);

		for (cl_uint i = 0; i < M_SIZE; ++i)
		{
			const cl_uint offset = _n * _num_regs * i;
			_setKernelArg(_set, 2, sizeof(cl_uint), &offset);
			_executeKernel(_set, _n);
		}
	}

	void copy(const size_t dst, const size_t src)
	{
		const cl_uint idst = static_cast<cl_uint>(dst * _n), isrc = static_cast<cl_uint>(src * _n);
		_setKernelArg(_copy, 1, sizeof(cl_uint), &idst);
		_setKernelArg(_copy, 2, sizeof(cl_uint), &isrc);

		for (cl_uint i = 0; i < M_SIZE; ++i)
		{
			const cl_uint offset = _n * _num_regs * i;
			_setKernelArg(_copy, 3, sizeof(cl_uint), &offset);
			_executeKernel(_copy, _n);
		}
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
	void baseModTune(const size_t count, const size_t blk, const size_t n3aLocalWS, const size_t n3bLocalWS,
		const GF31 * const Z31, const ZP1 * const Z1, const ZP2 * const Z2)
	{
		const cl_uint cblk = static_cast<cl_uint>(blk);
		const cl_int sblk = static_cast<cl_int>(blk);
		const size_t size = _n / blk;

		for (size_t i = 0; i != count; ++i)
		{
			writeMemory_z31(Z31); writeMemory_z1(Z1);
			if (M_SIZE == 3) writeMemory_z2(Z2);

			_setKernelArg(_normalize1, 5, sizeof(cl_int), &sblk);
			_executeKernel(_normalize1, size, std::min(size, n3aLocalWS));

			_setKernelArg(_normalize2, 5, sizeof(cl_uint), &cblk);
			_executeKernel(_normalize2, size, std::min(size, n3bLocalWS));
		}
	}

public:
	void tune(const uint32_t base)
	{
		const size_t n = _n;

		GF31 * const Z31 = new GF31[n];
		for (size_t i = 0; i != n; ++i)
		{
			const double id = static_cast<double>(i);
			const int32 va31 = static_cast<int32>((M31 - 1) * cos(id)), vb31 = static_cast<int32>((M31 - 1) * cos(id + 0.5));
			Z31[i].set_int(va31, vb31);
		}

		ZP1 * const Z1 = new ZP1[n];
		for (size_t i = 0; i != n; ++i)
		{
			const double id = static_cast<double>(i);
			const int32 va1 = static_cast<int32>((P1M - 1) * cos(id + 0.25)), vb1 = static_cast<int32>((P1M - 1) * cos(id + 0.75));
			Z1[i].set_int(va1, vb1);
		}

		ZP2 * const Z2 = (M_SIZE == 3) ? new ZP2[n] : nullptr;
		if (M_SIZE == 3)
		{
			for (size_t i = 0; i != n; ++i)
			{
				const double id = static_cast<double>(i);
				const int32 va2 = static_cast<int32>((P2M - 1) * sin(id + 0.33)), vb2 = static_cast<cl_int>((P2M - 1) * sin(id + 0.66));
				Z2[i].set_int(va2, vb2);
			}
		}

		setProfiling(true);

		resetProfiles();
		baseModTune(1, 16, 0, 0, Z31, Z1, Z2);
		const cl_ulong time = getProfileTime();
		if (time == 0) { delete[] Z31; delete[] Z1; if (M_SIZE == 3) delete[] Z2; setProfiling(false); return; }
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
			baseModTune(count, b, 0, 0, Z31, Z1, Z2);
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
					baseModTune(count, b, sa, sb, Z31, Z1, Z2);
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
// #if defined(ocl_debug)
		{
			std::ostringstream ss; ss << "baseModBlk = " << _baseModBlk << ", WorkgroupSize1 = " << _naLocalWS << ", WorkgroupSize2 = " << _nbLocalWS << "." << std::endl;
			pio::display(ss.str());
		}
// #endif

		const splitter * const pSplit = _pSplit;
		const size_t ns = pSplit->getSize();
		if (ns > 1)
		{
			cl_ulong minT = cl_ulong(-1);
			for (size_t i = 0; i < ns; ++i)
			{
				resetProfiles();
				squareTune(2, i, Z31, Z1, Z2);
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

		delete[] Z31; delete[] Z1;
		if (M_SIZE == 3) delete[] Z2;

		setProfiling(false);
	}
};

template<size_t M_SIZE>
class transformGPUm : public transform
{
private:
	const size_t _mem_size;
	const size_t _num_regs;
	GF31 * const _z31;
	ZP1 * const _z1;
	ZP2 * const _z2;
	engineg<M_SIZE> * _pEngine = nullptr;

public:
	transformGPUm(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
				 const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
		: transform(size_t(1) << (n - 1), n, b, (M_SIZE == 2) ? EKind::NTT2m : EKind::NTT3m),
		_mem_size((size_t(1) << (n - 1)) * num_regs * (sizeof(GF31) + sizeof(ZP1) + ((M_SIZE == 3) ? sizeof(ZP2) : 0))), _num_regs(num_regs),
		_z31(new GF31[(size_t(1) << (n - 1)) * num_regs]), _z1(new ZP1[(size_t(1) << (n - 1)) * num_regs]),
		_z2((M_SIZE == 3) ? new ZP2[(size_t(1) << (n - 1)) * num_regs] : nullptr)
	{
		const size_t size = getSize();

		const bool is_boinc_platform = isBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pEngine = new engineg<M_SIZE>(eng_platform, is_boinc_platform ? 0 : device, static_cast<int>(n - 1), isBoinc, num_regs, verbose);

		std::ostringstream src;

		src << "#define\tNSIZE\t" << (1u << (n - 1)) << std::endl;
		src << "#define\tSNORM31\t" << 31 - n + 2 << std::endl;
		src << "#define\tNORM1\t" << ZP1::norm(uint32(size)) << "u" << std::endl;

		for (size_t i = 1; i < M_SIZE; ++i)
		{
			src << "#define\tZOFFSET_" << i << "\t" << i * size * _num_regs << std::endl;
			src << "#define\tWOFFSET_" << i << "\t" << i * 3 * size / 2 << std::endl;
		}

		src << "#define\tBLK32\t" << BLK32m << std::endl;
		src << "#define\tBLK64\t" << BLK64m << std::endl;
		src << "#define\tBLK128\t" << BLK128m << std::endl;
		src << "#define\tBLK256\t" << BLK256m << std::endl << std::endl;

		src << "#define\tCHUNK64\t" << CHUNK64m << std::endl;
		src << "#define\tCHUNK256\t" << CHUNK256m << std::endl;
		src << "#define\tCHUNK1024\t" << CHUNK1024m << std::endl << std::endl;

		src << "#define\tMAX_WORK_GROUP_SIZE\t" << _pEngine->getMaxWorkGroupSize() << std::endl << std::endl;

		if (M_SIZE == 2)
		{
			if (isBoinc || !_pEngine->readOpenCL("ocl/kernel2m.cl", "src/ocl/kernel2m.h", "src_ocl_kernel2m", src)) src << src_ocl_kernel2m;
		}
		else	// M_SIZE == 3
		{
			if (isBoinc || !_pEngine->readOpenCL("ocl/kernel3m.cl", "src/ocl/kernel3m.h", "src_ocl_kernel3m", src)) src << src_ocl_kernel3m;
		}

		_pEngine->loadProgram(src.str());
		_pEngine->allocMemory();
		_pEngine->createKernels(b);

		GF31 * const wr31 = new GF31[3 * size / 2];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const GF31 r_s = GF31::primroot_n(16 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const GF31 w2 = r_s.pow(bitRev(j, 4 * s) + 1), w1 = w2.sqr(), w3 = w1.mul(w2);
				wr31[s + j] = w1; wr31[size / 2 + s + j] = w2; wr31[size + s + j] = w3;
			}
		}
		_pEngine->writeMemory_w31(wr31);
		delete[] wr31;

		ZP1 * const wr1 = new ZP1[3 * size / 2];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const ZP1 r_s = ZP1::primroot_n(16 * s);
			for (size_t j = 0; j < s; ++j)
			{
				const ZP1 w20 = r_s.pow_mul_sqr(bitRev(j, 4 * s) + 1), w1 = w20.sqr(), w21 = w20.muli();
				wr1[s + j] = w1; wr1[size / 2 + s + j] = w20; wr1[size + s + j] = w21;
			}
		}
		_pEngine->writeMemory_w1(wr1);
		delete[] wr1;

		if (M_SIZE == 3)
		{
			ZP2 * const wr2 = new ZP2[3 * size / 2];
			_pEngine->writeMemory_w2(wr2);
			delete[] wr2;
		}

		_pEngine->tune(b);
	}

	virtual ~transformGPUm()
	{
		_pEngine->releaseKernels();
		_pEngine->releaseMemory();
		_pEngine->clearProgram();
		delete _pEngine;

		delete[] _z31;
		delete[] _z1;
		if (M_SIZE == 3) delete[] _z2;
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return 0; }

protected:
	void getZi(int32_t * const zi) const override
	{
		_pEngine->readMemory_z31(_z31);

		const size_t size = getSize();

		const GF31 * const z31 = _z31;
		for (size_t i = 0; i < size; ++i)
		{
			int32_t a, b; z31[i].get_int(a, b);
			zi[i + 0 * size] = a; zi[i + 1 * size] = b;
		}
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size = getSize();

		GF31 * const z31 = _z31;
		for (size_t i = 0; i < size; ++i) z31[i].set_int(zi[i + 0 * size], zi[i + 1 * size]);
		_pEngine->writeMemory_z31(z31);

		ZP1 * const z1 = _z1;
		for (size_t i = 0; i < size; ++i) z1[i].set_int(zi[i + 0 * size], zi[i + 1 * size]);
		_pEngine->writeMemory_z1(z1);

		if (M_SIZE == 3)
		{
			ZP2 * const z2 = _z2;
			for (size_t i = 0; i < size; ++i) z2[i].set_int(zi[i + 0 * size], zi[i + 1 * size]);
			_pEngine->writeMemory_z2(z2);
		}
	}

public:
	bool readContext(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(getKind())) return false;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		if (!cFile.read(reinterpret_cast<char *>(_z31), sizeof(GF31) * size * num_regs)) return false;
		_pEngine->writeMemory_z31(_z31, num_regs);

		if (!cFile.read(reinterpret_cast<char *>(_z1), sizeof(ZP1) * size * num_regs)) return false;
		_pEngine->writeMemory_z1(_z1, num_regs);

		if (M_SIZE == 3)
		{
			if (!cFile.read(reinterpret_cast<char *>(_z2), sizeof(ZP2) * size * num_regs)) return false;
			_pEngine->writeMemory_z2(_z2, num_regs);
		}

		return true;
	}

	void saveContext(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		_pEngine->readMemory_z31(_z31, num_regs);
		if (!cFile.write(reinterpret_cast<const char *>(_z31), sizeof(GF31) * size * num_regs)) return;

		_pEngine->readMemory_z1(_z1, num_regs);
		if (!cFile.write(reinterpret_cast<const char *>(_z1), sizeof(ZP1) * size * num_regs)) return;

		if (M_SIZE == 3)
		{
			_pEngine->readMemory_z2(_z2, num_regs);
			if (!cFile.write(reinterpret_cast<const char *>(_z2), sizeof(ZP2) * size * num_regs)) return;
		}
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
};
