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

#include "ocl/kernel2.h"
#include "ocl/kernel3.h"
#include "ocl/kernel1g.h"
#include "ocl/kernel2g.h"

// Modulo 2^61 - 1
class Z61
{
private:
	static const cl_ulong _p = (cl_ulong(1) << 61) - 1;
	cl_ulong _n;	// 0 <= n <= p

	static cl_ulong _add(const cl_ulong a, const cl_ulong b)
	{
		const cl_ulong t = a + b;
		const cl_ulong c = (cl_uint(t >> 32) > cl_uint(_p >> 32)) ? _p : 0;	// t > p ?
		return t - c;
	}

	static cl_ulong _sub(const cl_ulong a, const cl_ulong b)
	{
		const cl_ulong t = a - b;
		const cl_ulong c = (cl_int(t >> 32) < 0) ? _p : 0;	// t < 0 ?
		return t + c;
	}

	static cl_ulong _mul(const cl_ulong a, const cl_ulong b)
	{
		const __uint128_t t = a * __uint128_t(b);
		const cl_ulong lo = cl_ulong(t) & _p, hi = cl_ulong(t >> 61);
		return _add(hi, lo);
	}

public:
	Z61() {}
	explicit Z61(const cl_ulong n) : _n(n) {}

	cl_long get_int() const { return (_n >= _p / 2) ? cl_long(_n - _p) : cl_long(_n); }	// if n = p then return 0
	Z61 & set_int(const cl_long i) { _n = (i < 0) ? cl_ulong(i) + _p : cl_ulong(i); return *this; }

	// Z61 neg() const { return Z61((_n == 0) ? 0 : _p - _n); }

	Z61 add(const Z61 & rhs) const { return Z61(_add(_n, rhs._n)); }
	Z61 sub(const Z61 & rhs) const { return Z61(_sub(_n, rhs._n)); }
	Z61 mul(const Z61 & rhs) const { return Z61(_mul(_n, rhs._n)); }

	static Z61 inv(const cl_uint n) { return Z61((_p + 1) / n); }
};

// GF((2^61 - 1)^2)
class GF61
{
private:
	Z61 _a, _b;
	// a primitive root of order 2^62 which is a root of (0, 1).
	static const cl_ulong _h_order = cl_ulong(1) << 62;
	static const cl_ulong _h_a = 2147483648ull, _h_b = 1272521237944691271ull;

public:
	GF61() {}
	explicit GF61(const Z61 & a, const Z61 & b) : _a(a), _b(b) {}

	const Z61 & a() const { return _a; }
	const Z61 & b() const { return _b; }

	// GF61 conj() const { return GF61(_a, -_b); }

	GF61 add(const GF61 & rhs) const { return GF61(_a.add(rhs._a), _b.add(rhs._b)); }
	GF61 sub(const GF61 & rhs) const { return GF61(_a.sub(rhs._a), _b.sub(rhs._b)); }
	GF61 muls(const Z61 & rhs) const { return GF61(_a.mul(rhs), _b.mul(rhs)); }

	GF61 sqr() const { const Z61 t = _a.mul(_b); return GF61(_a.mul(_a).sub(_b.mul(_b)), t.add(t)); }
	GF61 mul(const GF61 & rhs) const { return GF61(_a.mul(rhs._a).sub(_b.mul(rhs._b)), _b.mul(rhs._a).add(_a.mul(rhs._b))); }
	GF61 mulconj(const GF61 & rhs) const { return GF61(_a.mul(rhs._a).add(_b.mul(rhs._b)), _b.mul(rhs._a).sub(_a.mul(rhs._b))); }

	// GF61 muli() const { return GF61(-_b, _a); }
	GF61 addi(const GF61 & rhs) const { return GF61(_a.sub(rhs._b), _b.add(rhs._a)); }
	GF61 subi(const GF61 & rhs) const { return GF61(_a.add(rhs._b), _b.sub(rhs._a)); }

	GF61 pow(const cl_ulong e) const
	{
		if (e == 0) return GF61(Z61(1), Z61(0));
		GF61 r = GF61(Z61(1), Z61(0)), y = *this;
		for (cl_ulong i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF61 primroot_n(const cl_uint n) { return GF61(Z61(_h_a), Z61(_h_b)).pow(_h_order / n); }
};

// Modulo 2^31 - 1
class Z31
{
private:
	static const cl_uint _p = (cl_uint(1) << 31) - 1;
	cl_uint _n;	// 0 <= n < p

	static cl_uint _add(const cl_uint a, const cl_uint b)
	{
		const cl_uint t = a + b;
		return t - ((t >= _p) ? _p : 0);
	}

	static cl_uint _sub(const cl_uint a, const cl_uint b)
	{
		const cl_uint t = a - b;
		return t + ((cl_int(t) < 0) ? _p : 0);
	}

	static cl_uint _mul(const cl_uint a, const cl_uint b)
	{
		const cl_ulong t = a * cl_ulong(b);
		const cl_uint lo = cl_uint(t) & _p, hi = cl_uint(t >> 31);
		return _add(hi, lo);
	}

public:
	Z31() {}
	explicit Z31(const cl_uint n) : _n(n) {}

	cl_int get_int() const { return (_n >= _p / 2) ? cl_int(_n - _p) : cl_int(_n); }
	Z31 & set_int(const cl_int i) { _n = (i < 0) ? cl_uint(i) + _p : cl_uint(i); return *this; }

	// Z31 neg() const { return Z31((_n == 0) ? 0 : _p - _n); }

	Z31 add(const Z31 & rhs) const { return Z31(_add(_n, rhs._n)); }
	Z31 sub(const Z31 & rhs) const { return Z31(_sub(_n, rhs._n)); }
	Z31 mul(const Z31 & rhs) const { return Z31(_mul(_n, rhs._n)); }

	static Z31 inv(const cl_uint n) { return Z31((_p + 1) / n); }
};

// GF((2^31 - 1)^2)
class GF31
{
private:
	Z31 _a, _b;
	// a primitive root of order 2^31 which is a root of (0, 1).
	static const cl_uint _h_order = cl_uint(1) << 31;
	static const cl_uint _h_a = 105066u, _h_b = 333718u;

public:
	GF31() {}
	explicit GF31(const Z31 & a, const Z31 & b) : _a(a), _b(b) {}

	const Z31 & a() const { return _a; }
	const Z31 & b() const { return _b; }

	// GF31 conj() const { return GF31(_a, -_b); }

	GF31 add(const GF31 & rhs) const { return GF31(_a.add(rhs._a), _b.add(rhs._b)); }
	GF31 sub(const GF31 & rhs) const { return GF31(_a.sub(rhs._a), _b.sub(rhs._b)); }
	GF31 muls(const Z31 & rhs) const { return GF31(_a.mul(rhs), _b.mul(rhs)); }

	GF31 sqr() const { const Z31 t = _a.mul(_b); return GF31(_a.mul(_a).sub(_b.mul(_b)), t.add(t)); }
	GF31 mul(const GF31 & rhs) const { return GF31(_a.mul(rhs._a).sub(_b.mul(rhs._b)), _b.mul(rhs._a).add(_a.mul(rhs._b))); }
	GF31 mulconj(const GF31 & rhs) const { return GF31(_a.mul(rhs._a).add(_b.mul(rhs._b)), _b.mul(rhs._a).sub(_a.mul(rhs._b))); }

	// GF31 muli() const { return GF31(-_b, _a); }
	GF31 addi(const GF31 & rhs) const { return GF31(_a.sub(rhs._b), _b.add(rhs._a)); }
	GF31 subi(const GF31 & rhs) const { return GF31(_a.add(rhs._b), _b.sub(rhs._a)); }

	GF31 pow(const cl_ulong e) const
	{
		if (e == 0) return GF31(Z31(1), Z31(0));
		GF31 r = GF31(Z31(1), Z31(0)), y = *this;
		for (cl_ulong i = e; i != 1; i /= 2) { if (i % 2 != 0) r = r.mul(y); y = y.sqr(); }
		return r.mul(y);
	}

	static const GF31 primroot_n(const cl_uint n) { return GF31(Z31(_h_a), Z31(_h_b)).pow(_h_order / n); }
};

// Warning: DECLARE_VAR_32/64/128/256 in kernerl.cl must be modified if BLKxx = 1 or != 1.

#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1

#define CHUNK64		8
#define CHUNK256	4
#define CHUNK1024	2

template<size_t GF_SIZE>
class engineg : public device
{
private:
	const size_t _n_2;
	const int _ln_m1;
	const bool _isBoinc;
	cl_mem _z = nullptr, _zp = nullptr, _w = nullptr;
	cl_mem _ze = nullptr, _zpe = nullptr, _we = nullptr;
	cl_mem _c = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward256 = nullptr, _backward256 = nullptr, _forward1024 = nullptr, _backward1024 = nullptr;
	cl_kernel _forward64_0 = nullptr, _forward256_0 = nullptr, _forward1024_0 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr, _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr;
	cl_kernel _normalize1 = nullptr, _normalize2 = nullptr, _mul1 = nullptr;
	cl_kernel _fwd32p = nullptr, _fwd64p = nullptr, _fwd128p = nullptr, _fwd256p = nullptr, _fwd512p = nullptr, _fwd1024p = nullptr, _fwd2048p = nullptr;
	cl_kernel _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr, _mul256 = nullptr, _mul512 = nullptr, _mul1024 = nullptr, _mul2048 = nullptr;
	cl_kernel _set = nullptr, _copy = nullptr, _copyp = nullptr;
	splitter * _pSplit = nullptr;
	size_t _naLocalWS = 32, _nbLocalWS = 32, _baseModBlk = 16, _splitIndex = 0;
	// bool _first = true;

public:
	engineg(const platform & platform, const size_t d, const int ln, const bool isBoinc, const bool verbose)
		: device(platform, d, verbose), _n_2(size_t(1) << (ln - 1)), _ln_m1(ln - 1), _isBoinc(isBoinc) {}
	virtual ~engineg() {}

///////////////////////////////

public:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::ostringstream & src) const
	{
		if (_isBoinc) return false;

		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2022, Yves Gallot" << std::endl << std::endl;
		hFile << "genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "#include <cstdint>" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

public:
	void allocMemory(const size_t num_regs)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t n_2 = _n_2;
		if (n_2 != 0)
		{
			_z = _createBuffer(CL_MEM_READ_WRITE, sizeof(GF61) * n_2 * num_regs);
			_zp = _createBuffer(CL_MEM_READ_WRITE, sizeof(GF61) * n_2);
			_w = _createBuffer(CL_MEM_READ_ONLY, sizeof(GF61) * n_2 / 2);
			if (GF_SIZE == 2)
			{
				_ze = _createBuffer(CL_MEM_READ_WRITE, sizeof(GF31) * n_2 * num_regs);
				_zpe = _createBuffer(CL_MEM_READ_WRITE, sizeof(GF31) * n_2);
				_we = _createBuffer(CL_MEM_READ_ONLY, sizeof(GF31) * n_2 / 2);
			}
			_c = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_long) * n_2 / 4);
		}
	}

	void releaseMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_n_2 != 0)
		{
			_releaseBuffer(_z);
			_releaseBuffer(_zp); 
			_releaseBuffer(_w);  
			if (GF_SIZE == 2)
			{
				_releaseBuffer(_ze);
				_releaseBuffer(_zpe);
				_releaseBuffer(_we);
			}
			_releaseBuffer(_c);
		}
	}

///////////////////////////////

private:
	cl_kernel createTransformKernel(const char * const kernelName, const bool isMultiplier = true)
	{
		cl_kernel kernel = _createKernel(kernelName);
		cl_uint index = 0;
		_setKernelArg(kernel, index++, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), isMultiplier ? &_ze : &_zpe);
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_w);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_we);
		return kernel;
	}

	cl_kernel createNormalizeKernel(const char * const kernelName, const cl_uint b, const cl_uint b_inv, const cl_int b_s)
	{
		cl_kernel kernel = _createKernel(kernelName);
		cl_uint index = 0;
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_z);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_ze);
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_c);
		_setKernelArg(kernel, index++, sizeof(cl_uint), &b);
		_setKernelArg(kernel, index++, sizeof(cl_uint), &b_inv);
		_setKernelArg(kernel, index++, sizeof(cl_int), &b_s);
		return kernel;
	}

	cl_kernel createMulKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		cl_uint index = 0;
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_z);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_ze);
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_zp);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_zpe);
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_w);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_we);
		return kernel;
	}

	cl_kernel createSetKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		cl_uint index = 0;
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_z);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_ze);
		return kernel;
	}

	cl_kernel createCopyKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		cl_uint index = 0;
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_z);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_ze);
		return kernel;
	}

	cl_kernel createCopypKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		cl_uint index = 0;
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_zp);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_zpe);
		_setKernelArg(kernel, index++, sizeof(cl_mem), &_z);
		if (GF_SIZE == 2) _setKernelArg(kernel, index++, sizeof(cl_mem), &_ze);
		return kernel;
	}

public:
	void createKernels(const uint32_t b)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_forward64 = createTransformKernel("forward64");
		_backward64 = createTransformKernel("backward64");
		_forward256 = createTransformKernel("forward256");
		_backward256 = createTransformKernel("backward256");
		_forward1024 = createTransformKernel("forward1024");
		_backward1024 = createTransformKernel("backward1024");

		_forward64_0 = createTransformKernel("forward64_0");
		_forward256_0 = createTransformKernel("forward256_0");
		_forward1024_0 = createTransformKernel("forward1024_0");

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
		_mul1 = createNormalizeKernel("mul1", static_cast<cl_uint>(b), b_inv, b_s);

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

		_pSplit = new splitter(size_t(_ln_m1), CHUNK256, CHUNK1024, sizeof(GF61) + ((GF_SIZE == 2) ? sizeof(GF31) : 0), 11, getLocalMemSize(), getMaxWorkGroupSize());
	}

	void releaseKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		delete _pSplit;

		_releaseKernel(_forward64); _releaseKernel(_backward64);
		_releaseKernel(_forward256); _releaseKernel(_backward256);
		_releaseKernel(_forward1024); _releaseKernel(_backward1024);
		_releaseKernel(_forward64_0); _releaseKernel(_forward256_0); _releaseKernel(_forward1024_0);
		_releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128); _releaseKernel(_square256);
		_releaseKernel(_square512); _releaseKernel(_square1024); _releaseKernel(_square2048);
		_releaseKernel(_normalize1); _releaseKernel(_normalize2); _releaseKernel(_mul1);
		_releaseKernel(_fwd32p); _releaseKernel(_fwd64p); _releaseKernel(_fwd128p); _releaseKernel(_fwd256p);
		_releaseKernel(_fwd512p); _releaseKernel(_fwd1024p); _releaseKernel(_fwd2048p);
		_releaseKernel(_mul32); _releaseKernel(_mul64); _releaseKernel(_mul128); _releaseKernel(_mul256);
		_releaseKernel(_mul512); _releaseKernel(_mul1024); _releaseKernel(_mul2048);
		_releaseKernel(_set); _releaseKernel(_copy); _releaseKernel(_copyp);
	}

///////////////////////////////

	void readMemory_z(GF61 * const zPtr, const size_t count = 1) { _readBuffer(_z, zPtr, sizeof(GF61) * _n_2 * count); }
	void readMemory_ze(GF31  * const zPtre, const size_t count = 1) { _readBuffer(_ze, zPtre, sizeof(GF31) * _n_2 * count); }

	void writeMemory_z(const GF61 * const zPtr, const size_t count = 1) { _writeBuffer(_z, zPtr, sizeof(GF61) * _n_2 * count); }
	void writeMemory_ze(const GF31 * const zPtre, const size_t count = 1) { _writeBuffer(_ze, zPtre, sizeof(GF31) * _n_2 * count); }

	void writeMemory_w(const GF61 * const wPtr) { _writeBuffer(_w, wPtr, sizeof(GF61) * _n_2 / 2); }
	void writeMemory_we(const GF31 * const wPtre) { _writeBuffer(_we, wPtre, sizeof(GF31) * _n_2 / 2); }

///////////////////////////////

private:
	void fb(cl_kernel & kernel, const int lm, const size_t localWorkSize)
	{
		const size_t n_4 = _n_2 / 4;
		const cl_int ilm = static_cast<cl_int>(lm);
		const cl_uint is = static_cast<cl_uint>(n_4 >> lm);
		cl_uint index = (GF_SIZE == 2) ? 4 : 2;
		_setKernelArg(kernel, index++, sizeof(cl_int), &ilm);
		_setKernelArg(kernel, index++, sizeof(cl_uint), &is);
		_executeKernel(kernel, n_4, localWorkSize);
	}

	void forward64(const int lm) { fb(_forward64, lm, 64 / 4 * CHUNK64); }
	void backward64(const int lm) { fb(_backward64, lm, 64 / 4 * CHUNK64); }
	void forward256(const int lm) { fb(_forward256, lm, 256 / 4 * CHUNK256); }
	void backward256(const int lm) { fb(_backward256, lm, 256 / 4 * CHUNK256); }
	void forward1024(const int lm) { fb(_forward1024, lm, 1024 / 4 * CHUNK1024); }
	void backward1024(const int lm) { fb(_backward1024, lm, 1024 / 4 * CHUNK1024); }

	void forward64_0() { const size_t n_4 = _n_2 / 4; _executeKernel(_forward64_0, n_4, 64 / 4 * CHUNK64); }
	void forward256_0() { const size_t n_4 = _n_2 / 4; _executeKernel(_forward256_0, n_4, 256 / 4 * CHUNK256); }
	void forward1024_0() { const size_t n_4 = _n_2 / 4; _executeKernel(_forward1024_0, n_4, 1024 / 4 * CHUNK1024); }
	
	void square32() { const size_t n_4 = _n_2 / 4; _executeKernel(_square32, n_4, std::min(n_4, size_t(32 / 4 * BLK32))); }
	void square64() { const size_t n_4 = _n_2 / 4; _executeKernel(_square64, n_4, std::min(n_4, size_t(64 / 4 * BLK64))); }
	void square128() { const size_t n_4 = _n_2 / 4; _executeKernel(_square128, n_4, std::min(n_4, size_t(128 / 4 * BLK128))); }
	void square256() { const size_t n_4 = _n_2 / 4; _executeKernel(_square256, n_4, std::min(n_4, size_t(256 / 4 * BLK256))); }
	void square512() { const size_t n_4 = _n_2 / 4; _executeKernel(_square512, n_4, 512 / 4); }
	void square1024() { const size_t n_4 = _n_2 / 4; _executeKernel(_square1024, n_4, 1024 / 4); }
	void square2048() { const size_t n_4 = _n_2 / 4; _executeKernel(_square2048, n_4, 2048 / 4); }

	void fwd32p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd32p, n_4, std::min(n_4, size_t(32 / 4 * BLK32))); }
	void fwd64p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd64p, n_4, std::min(n_4, size_t(64 / 4 * BLK64))); }
	void fwd128p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd128p, n_4, std::min(n_4, size_t(128 / 4 * BLK128))); }
	void fwd256p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd256p, n_4, std::min(n_4, size_t(256 / 4 * BLK256))); }
	void fwd512p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd512p, n_4, 512 / 4); }
	void fwd1024p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd1024p, n_4, 1024 / 4); }
	void fwd2048p() { const size_t n_4 = _n_2 / 4; _executeKernel(_fwd2048p, n_4, 2048 / 4); }

	void mul32() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul32, n_4, std::min(n_4, size_t(32 / 4 * BLK32))); }
	void mul64() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul64, n_4, std::min(n_4, size_t(64 / 4 * BLK64))); }
	void mul128() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul128, n_4, std::min(n_4, size_t(128 / 4 * BLK128))); }
	void mul256() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul256, n_4, std::min(n_4, size_t(256 / 4 * BLK256))); }
	void mul512() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul512, n_4, 512 / 4); }
	void mul1024() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul1024, n_4, 1024 / 4); }
	void mul2048() { const size_t n_4 = _n_2 / 4; _executeKernel(_mul2048, n_4, 2048 / 4); }

	void setTransformArgs(cl_kernel & kernel, const bool isMultiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
		if (GF_SIZE == 2) _setKernelArg(kernel, 1, sizeof(cl_mem), isMultiplier ? &_ze : &_zpe);
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

public:
	void square()
	{
		const splitter * const pSplit = _pSplit;

		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		int lm = _ln_m1;

		for (size_t i = 0; i < s - 1; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i);
			if (k == 10)
			{
				lm -= 10;
				if (i == 0) forward1024_0(); else forward1024(lm);
				// if (_first) std::cout << "forward1024 (" << lm << ") ";
			}
			else if (k == 8)
			{
				lm -= 8;
				if (i == 0) forward256_0(); else forward256(lm);
				// if (_first) std::cout << "forward256 (" << lm << ") ";
			}
			else // if (k == 6)
			{
				lm -= 6;
				if (i == 0) forward64_0(); else forward64(lm);
				// if (_first) std::cout << "forward64 (" << lm << ") ";
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
		// if (_first) std::cout << "square" << (1u << lm) << " ";

		for (size_t i = 0; i < s - 1; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, s - 2 - i);
			if (k == 10)
			{
				backward1024(lm);
				// if (_first) std::cout << "backward1024 (" << lm << ") ";
				lm += 10;
			}
			else if (k == 8)
			{
				backward256(lm);
				// if (_first) std::cout << "backward256 (" << lm << ") ";
				lm += 8;
			}
			else // if (k == 6)
			{
				backward64(lm);
				// if (_first) std::cout << "backward64 (" << lm << ") ";
				lm += 6;
			}
		}

		// if (_first) { _first = false; std::cout << std::endl; }
	}

private:
	void squareTune(const size_t count, const size_t sIndex, const GF61 * const Z, GF31 * const Ze)
	{
		const splitter * const pSplit = _pSplit;

		for (size_t j = 0; j != count; ++j)
		{
			writeMemory_z(Z);
			if (GF_SIZE == 2) writeMemory_ze(Ze);

			const size_t s = pSplit->getPartSize(sIndex);

			int lm = _ln_m1;

			for (size_t i = 0; i < s - 1; ++i)
			{
				const uint32_t k = pSplit->getPart(sIndex, i);
				if (k == 10)
				{
					lm -= 10;
					if (i == 0) forward1024_0(); else forward1024(lm);
				}
				else if (k == 8)
				{
					lm -= 8;
					if (i == 0) forward256_0(); else forward256(lm);
				}
				else // if (k == 6)
				{
					lm -= 6;
					if (i == 0) forward64_0(); else forward64(lm);
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
	}

public:
	void initMultiplicand(const size_t src)
	{
		const cl_uint isrc = static_cast<cl_uint>(src * _n_2);
		_setKernelArg(_copyp, (GF_SIZE == 2) ? 4 : 2, sizeof(cl_uint), &isrc);
		_executeKernel(_copyp, _n_2);

		const splitter * const pSplit = _pSplit;

		const size_t sIndex = _splitIndex;
		const size_t s = pSplit->getPartSize(sIndex);

		int lm = _ln_m1;

		for (size_t i = 0; i < s - 1; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i);
			if (k == 10)
			{
				lm -= 10;
				if (i == 0) forward1024p_0(); else forward1024p(lm);
			}
			else if (k == 8)
			{
				lm -= 8;
				if (i == 0) forward256p_0(); else forward256p(lm);
			}
			else // if (k == 6)
			{
				lm -= 6;
				if (i == 0) forward64p_0(); else forward64p(lm);
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

		int lm = _ln_m1;

		for (size_t i = 0; i < s - 1; ++i)
		{
			const uint32_t k = pSplit->getPart(sIndex, i);
			if (k == 10)
			{
				lm -= 10;
				if (i == 0) forward1024_0(); else forward1024(lm);
			}
			else if (k == 8)
			{
				lm -= 8;
				if (i == 0) forward256_0(); else forward256(lm);
			}
			else // if (k == 6)
			{
				lm -= 6;
				if (i == 0) forward64_0(); else forward64(lm);
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

	void set(const int32_t a)
	{
		const cl_int ia = static_cast<cl_int>(a);
		cl_uint index = (GF_SIZE == 2) ? 2 : 1;
		_setKernelArg(_set, index++, sizeof(cl_int), &ia);
		_executeKernel(_set, _n_2);
	}

	void copy(const size_t dst, const size_t src)
	{
		const cl_uint idst = static_cast<cl_uint>(dst * _n_2), isrc = static_cast<cl_uint>(src * _n_2);
		cl_uint index = (GF_SIZE == 2) ? 2 : 1;
		_setKernelArg(_copy, index++, sizeof(cl_uint), &idst);
		_setKernelArg(_copy, index++, sizeof(cl_uint), &isrc);
		_executeKernel(_copy, _n_2);
	}

public:
	void baseMod(const bool dup)
	{
		const cl_uint blk = static_cast<cl_uint>(_baseModBlk);
		const cl_int sblk = dup ? -static_cast<cl_int>(blk) : static_cast<cl_int>(blk);
		const size_t size = _n_2 / blk;

		_setKernelArg(_normalize1, (GF_SIZE == 2) ? 6 : 5, sizeof(cl_int), &sblk);
		_executeKernel(_normalize1, size, std::min(size, _naLocalWS));

		_setKernelArg(_normalize2, (GF_SIZE == 2) ? 6 : 5, sizeof(cl_uint), &blk);
		_executeKernel(_normalize2, size, std::min(size, _nbLocalWS));
	}

public:
	void baseModMul(const int a)
	{
		baseMod(false);

		const cl_uint blk = static_cast<cl_uint>(_baseModBlk);
		const size_t size = _n_2 / blk;
		const cl_int ia = static_cast<cl_int>(a);

		cl_uint index1 = (GF_SIZE == 2) ? 6 : 5;
		_setKernelArg(_mul1, index1++, sizeof(cl_int), &blk);
		_setKernelArg(_mul1, index1++, sizeof(cl_int), &ia);
		_executeKernel(_mul1, size, std::min(size, _naLocalWS));

		cl_uint index2 = (GF_SIZE == 2) ? 6 : 5;
		_setKernelArg(_normalize2, index2++, sizeof(cl_uint), &blk);
		_executeKernel(_normalize2, size, std::min(size, _nbLocalWS));
	}

private:
	void baseModTune(const size_t count, const size_t blk, const size_t n3aLocalWS, const size_t n3bLocalWS, const GF61 * const Z, const GF31 * const Ze)
	{
		const cl_uint cblk = static_cast<cl_uint>(blk);
		const cl_int sblk = static_cast<cl_int>(blk);
		const size_t size = _n_2 / blk;

		for (size_t i = 0; i != count; ++i)
		{
			writeMemory_z(Z);
			if (GF_SIZE == 2) writeMemory_ze(Ze);

			_setKernelArg(_normalize1, (GF_SIZE == 2) ? 6 : 5, sizeof(cl_int), &sblk);
			_executeKernel(_normalize1, size, std::min(size, n3aLocalWS));

			_setKernelArg(_normalize2, (GF_SIZE == 2) ? 6 : 5, sizeof(cl_uint), &cblk);
			_executeKernel(_normalize2, size, std::min(size, n3bLocalWS));
		}
	}

public:
	void tune(const uint32_t base)
	{
		const size_t n_2 = _n_2;

		GF61 * const Z = new GF61[n_2];
		GF31 * const Ze = (GF_SIZE == 2) ? new GF31[n_2] : nullptr;
		const double v61_max = pow(2.0, 60), v31_max = pow(2.0, 30);
		for (size_t i = 0; i != n_2; ++i)
		{
			const cl_long va61 = static_cast<int>(v61_max * cos(static_cast<double>(i))), vb61 = static_cast<int>(v61_max * cos(static_cast<double>(i) + 0.5));
			const Z61 a61 = Z61().set_int(va61), b61 = Z61().set_int(vb61);
			Z[i] = GF61(a61, b61);
			if (GF_SIZE == 2)
			{
				const cl_long va31 = static_cast<int>(v31_max * cos(static_cast<double>(i))), vb31 = static_cast<int>(v31_max * cos(static_cast<double>(i) + 0.5));
				const Z31 a31 = Z31().set_int(va31), b31 = Z31().set_int(vb31);
				Ze[i] = GF31(a31, b31);
			}
		}

		setProfiling(true);

		resetProfiles();
		baseModTune(1, 16, 0, 0, Z, Ze);
		const cl_ulong time = getProfileTime();
		if (time == 0) { delete[] Z; if (GF_SIZE == 2) delete[] Ze; setProfiling(false); return; }
		const size_t count = std::min(std::max(size_t(100 * getTimerResolution() / time), size_t(2)), size_t(100));

		cl_ulong minT = cl_ulong(-1);

		size_t bMin = 4;
		while (bMin < log(n_2 * static_cast<double>(base + 2)) / log(static_cast<double>(base))) bMin *= 2;

		const double maxSqr = n_2 * (base * static_cast<double>(base));
		for (size_t b = bMin; b <= 64; b *= 2)
		{
			// Check convergence
			if (log(maxSqr) >= base * log(static_cast<double>(b))) continue;

			resetProfiles();
			baseModTune(count, b, 0, 0, Z, Ze);
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
					baseModTune(count, b, sa, sb, Z, Ze);
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
			std::ostringstream ss; ss << "n3aLocalWS = " << _naLocalWS << ", n3bLocalWS = " << _nbLocalWS << ", baseModBlk = " << _baseModBlk << "." << std::endl;
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
				squareTune(2, i, Z, Ze);
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
		if (GF_SIZE == 2) delete[] Ze;

		setProfiling(false);
	}
};

template<size_t GF_SIZE>
class transformGPUg : public transform
{
private:
	const size_t _mem_size;
	const size_t _num_regs;
	GF61 * const _z;
	GF31 * const _ze;
	engineg<GF_SIZE> * _pEngine = nullptr;

public:
	transformGPUg(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
				 const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
		: transform(size_t(1) << (n - 1), n, b, (GF_SIZE == 2) ? EKind::NTT3g : EKind::NTT2g),
		_mem_size((size_t(1) << (n - 1)) * num_regs * (sizeof(GF61) + ((GF_SIZE == 2) ? sizeof(GF31) : 0))), _num_regs(num_regs),
		_z(new GF61[(size_t(1) << (n - 1)) * num_regs]), _ze((GF_SIZE == 2) ? new GF31[(size_t(1) << (n - 1)) * num_regs] : nullptr)
	{
		const size_t size = getSize();

		const bool is_boinc_platform = isBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pEngine = new engineg<GF_SIZE>(eng_platform, is_boinc_platform ? 0 : device, static_cast<int>(n), isBoinc, verbose);

		std::ostringstream src;

		src << "#define\tLNSIZE\t" << n << std::endl;
		if (_pEngine->isIntel())	// Fix Intel compiler issue
		{
			src << "#define\tNSIZE_4\t((sz_t)get_global_size(0))" << std::endl;
		}
		else
		{
			src << "#define\tNSIZE_4\t" << (1u << (n - 2)) << "u" << std::endl;
		}

		src << "#define\tM61\t" << P1_32 << "u" << std::endl;

		src << "#define\tBLK32\t" << BLK32 << std::endl;
		src << "#define\tBLK64\t" << BLK64 << std::endl;
		src << "#define\tBLK128\t" << BLK128 << std::endl;
		src << "#define\tBLK256\t" << BLK256 << std::endl << std::endl;

		src << "#define\tCHUNK64\t" << CHUNK64 << std::endl;
		src << "#define\tCHUNK256\t" << CHUNK256 << std::endl;
		src << "#define\tCHUNK1024\t" << CHUNK1024 << std::endl << std::endl;

		src << "#define\tMAX_WORK_GROUP_SIZE\t" << _pEngine->getMaxWorkGroupSize() << std::endl << std::endl;

		if (GF_SIZE == 1)
		{
			if (!_pEngine->readOpenCL("ocl/kernel1g.cl", "src/ocl/kernel1g.h", "src_ocl_kernel1g", src)) src << src_ocl_kernel1g;
		}
		else	// GF_SIZE == 2
		{
			if (!_pEngine->readOpenCL("ocl/kernel2g.cl", "src/ocl/kernel2g.h", "src_ocl_kernel2g", src)) src << src_ocl_kernel2g;
		}

		_pEngine->loadProgram(src.str());
		_pEngine->allocMemory(num_regs);
		_pEngine->createKernels(b);

		GF61 * const wr = new GF61[size / 2];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const size_t m = 4 * s;
			const GF61 r_s = GF61::primroot_n(cl_uint(2 * m));
			for (size_t j = 0; j < s; ++j)
			{
				wr[s + j] = r_s.pow(bitRev(j, m) + 1);
			}
		}
		_pEngine->writeMemory_w(wr);
		delete[] wr;

		if (GF_SIZE == 2)
		{
			GF31 * const wre = new GF31[size / 2];
			for (size_t s = 1; s < size / 2; s *= 2)
			{
				const size_t m = 4 * s;
				const GF31 r_s = GF31::primroot_n(cl_uint(2 * m));
				for (size_t j = 0; j < s; ++j)
				{
					wre[s + j] = r_s.pow(bitRev(j, m) + 1);
				}
			}
			_pEngine->writeMemory_we(wre);
			delete[] wre;
		}

		_pEngine->tune(b);
	}

	virtual ~transformGPUg()
	{
		_pEngine->releaseKernels();
		_pEngine->releaseMemory();
		_pEngine->clearProgram();
		delete _pEngine;

		delete[] _z;
		if (GF_SIZE == 2) delete[] _ze;
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return 0; }

protected:
	void getZi(int32_t * const zi) const override
	{
		_pEngine->readMemory_z(_z);

		const size_t size = getSize();

		GF61 * const z = _z;
		for (size_t i = 0; i < size; ++i)
		{
			zi[i + 0 * size] = z[i].a().get_int();
			zi[i + 1 * size] = z[i].b().get_int();
		}
	}

	void setZi(const int32_t * const zi) override
	{
		const size_t size = getSize();

		GF61 * const z = _z;
		for (size_t i = 0; i < size; ++i) z[i] = GF61(Z61().set_int(zi[i + 0 * size]), Z61().set_int(zi[i + 1 * size]));
		_pEngine->writeMemory_z(z);

		if (GF_SIZE == 2)
		{
			GF31 * const ze = _ze;
			for (size_t i = 0; i < size; ++i) ze[i] = GF31(Z31().set_int(zi[i + 0 * size]), Z31().set_int(zi[i + 1 * size]));
			_pEngine->writeMemory_ze(_ze);
		}
	}

public:
	bool readContext(file & cFile, const size_t nregs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(getKind())) return false;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		if (!cFile.read(reinterpret_cast<char *>(_z), sizeof(GF61) * size * num_regs)) return false;
		_pEngine->writeMemory_z(_z, num_regs);

		if (GF_SIZE == 2)
		{
			if (!cFile.read(reinterpret_cast<char *>(_ze), sizeof(GF31) * size * num_regs)) return false;
			_pEngine->writeMemory_ze(_ze, num_regs);
		}

		return true;
	}

	void saveContext(file & cFile, const size_t nregs) const override
	{
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		const size_t size = getSize(), num_regs = (nregs != 0) ? nregs : _num_regs;

		_pEngine->readMemory_z(_z, num_regs);
		if (!cFile.write(reinterpret_cast<const char *>(_z), sizeof(GF61) * size * num_regs)) return;

		if (GF_SIZE == 2)
		{
			_pEngine->readMemory_ze(_ze, num_regs);
			if (!cFile.write(reinterpret_cast<const char *>(_ze), sizeof(GF31) * size * num_regs)) return;
		}
	}

	void set(const int32_t a) override
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
