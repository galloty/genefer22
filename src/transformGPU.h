/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <fstream>

#include "ocl.h"
#include "transform.h"

#include "ocl/kernel.h"

template <cl_uint p, cl_uint prRoot>
class Zp
{
private:
	cl_uint _n;

private:
	explicit Zp(const cl_uint n) : _n(n) {}

	Zp _mulMod(const Zp & rhs) const { return Zp((cl_uint)((_n * (cl_ulong)rhs._n) % p)); }

public:
	Zp() {}
	explicit Zp(const int32_t i) : _n(i + ((i < 0) ? p : 0)) {}

	cl_uint get() const { return _n; }
	int32_t getInt() const { return (_n > p / 2) ? int32_t(_n - p) : int32_t(_n); }

	void set(const cl_uint n) { _n = n; }

	Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

	Zp & operator*=(const Zp & rhs) { *this = _mulMod(rhs); return *this; }
	Zp operator*(const Zp & rhs) const { return _mulMod(rhs); }

	Zp pow(const uint32_t e) const
	{
		if (e == 0) return Zp(1);

		Zp r = Zp(1), y = *this;
		for (uint32_t i = e; i != 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;

		return r;
	}

	static const Zp prRoot_n(const uint32_t n) { return Zp(prRoot).pow((p - 1) / n); }
};

#define P1		4253024257u		// 507 * 2^23 + 1
#define P2		4194304001u		// 125 * 2^25 + 1
#define P3		4076863489u		// 243 * 2^24 + 1

typedef Zp<P1, 5> Zp1;
typedef Zp<P2, 3> Zp2;
typedef Zp<P3, 7> Zp3;

class RNS
{
private:
	cl_uint2 r;	// Zp1, Zp2

private:
	explicit RNS(const Zp1 & r1, const Zp2 & r2) { r.s[0] = r1.get(); r.s[1] = r2.get(); }

public:
	RNS() {}
	explicit RNS(const int32_t i) { r.s[0] = Zp1(i).get(); r.s[1] = Zp2(i).get(); }

	Zp1 r1() const { Zp1 r1; r1.set(r.s[0]); return r1; }
	Zp2 r2() const { Zp2 r2; r2.set(r.s[1]); return r2; }
	void set(const cl_uint r1, const cl_uint r2) { r.s[0] = r1; r.s[1] = r2; }

	RNS operator-() const { return RNS(-r1(), -r2()); }

	RNS operator*(const RNS & rhs) const { return RNS(r1() * rhs.r1(), r2() * rhs.r2()); }

	RNS pow(const uint32_t e) const { return RNS(r1().pow(e), r2().pow(e)); }

	static const RNS prRoot_n(const uint32_t n) { return RNS(Zp1::prRoot_n(n), Zp2::prRoot_n(n)); }
};

class RNSe
{
private:
	cl_uint r;	// Zp3

private:
	explicit RNSe(const Zp3 & r3) { r = r3.get(); }

public:
	RNSe() {}
	explicit RNSe(const int32_t i) { r = Zp3(i).get(); }

	Zp3 r3() const { Zp3 _r3; _r3.set(r); return _r3; }
	void set(const cl_uint r3) { r = r3; }

	RNSe operator-() const { return RNSe(-r3()); }

	RNSe operator*(const RNSe & rhs) const { return RNSe(r3() * rhs.r3()); }

	RNSe pow(const uint32_t e) const { return RNSe(r3().pow(e)); }

	static const RNSe prRoot_n(const uint32_t n) { return RNSe(Zp3::prRoot_n(n)); }
};

typedef RNS		RNS_W;
typedef RNSe	RNS_We;

// Warning: DECLARE_VAR_32/64/128/256 in kernerl.cl must be modified if BLKxx = 1 or != 1.

#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1

#define CHUNK64		8
#define CHUNK256	4
#define CHUNK1024	2

class engine : public device
{
private:
	const size_t _n;
	const int _ln;
	cl_mem _z = nullptr, _ze = nullptr, _zp = nullptr, _zpe = nullptr, _w = nullptr, _we = nullptr, _c = nullptr;
	cl_kernel _forward64 = nullptr, _backward64 = nullptr, _forward256 = nullptr, _backward256 = nullptr, _forward1024 = nullptr, _backward1024 = nullptr;
	cl_kernel _square32 = nullptr, _square64 = nullptr, _square128 = nullptr, _square256 = nullptr, _square512 = nullptr, _square1024 = nullptr, _square2048 = nullptr;
	cl_kernel _normalize3a = nullptr, _normalize3b = nullptr;
	cl_kernel _fwd32p = nullptr, _fwd64p = nullptr, _fwd128p = nullptr, _fwd256p = nullptr, _fwd512p = nullptr, _fwd1024p = nullptr, _fwd2048p = nullptr;
	cl_kernel _mul32 = nullptr, _mul64 = nullptr, _mul128 = nullptr, _mul256 = nullptr, _mul512 = nullptr, _mul1024 = nullptr, _mul2048 = nullptr;
	splitter * _pSplit = nullptr;
	size_t _n3aLocalWS = 32, _n3bLocalWS = 32, _baseModBlk = 16, _splitIndex = 0;

public:
	engine(const platform & platform, const size_t d, const int ln, const bool verbose) : device(platform, d, verbose), _n(size_t(1) << ln), _ln(ln) {}
	virtual ~engine() {}

///////////////////////////////

public:
	static bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::ostringstream & src)
	{
		// if (_isBoinc) return false;

		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;
		
		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2022, Yves Gallot" << std::endl << std::endl;
		hFile << "genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
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
	void allocMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t n = _n;
		if (n != 0)
		{
			_z = _createBuffer(CL_MEM_READ_WRITE, sizeof(RNS) * n);
			_ze = _createBuffer(CL_MEM_READ_WRITE, sizeof(RNSe) * n);
			_zp = _createBuffer(CL_MEM_READ_WRITE, sizeof(RNS) * n);
			_zpe = _createBuffer(CL_MEM_READ_WRITE, sizeof(RNSe) * n);
			_w = _createBuffer(CL_MEM_READ_ONLY, sizeof(RNS_W) * 2 * n);
			_we = _createBuffer(CL_MEM_READ_ONLY, sizeof(RNS_We) * 2 * n);
			_c = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_long) * n / 4);
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
			_releaseBuffer(_z); _releaseBuffer(_ze);
			_releaseBuffer(_zp); _releaseBuffer(_zpe);
			_releaseBuffer(_w); _releaseBuffer(_we);
			_releaseBuffer(_c);
		}
	}

///////////////////////////////

private:
	cl_kernel createTransformKernel(const char * const kernelName, const bool isMultiplier = true)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), isMultiplier ? &_ze : &_zpe);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_w);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_we);
		return kernel;
	}

	cl_kernel createNormalizeKernel(const char * const kernelName, const cl_uint b, const cl_uint b_inv, const cl_int b_s)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_ze);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_c);
		_setKernelArg(kernel, 3, sizeof(cl_uint), &b);
		_setKernelArg(kernel, 4, sizeof(cl_uint), &b_inv);
		_setKernelArg(kernel, 5, sizeof(cl_int), &b_s);
		return kernel;
	}

	cl_kernel createMulKernel(const char * const kernelName)
	{
		cl_kernel kernel = _createKernel(kernelName);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_z);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_ze);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_zp);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_zpe);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_w);
		_setKernelArg(kernel, 5, sizeof(cl_mem), &_we);
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

		_square32 = createTransformKernel("square32");
		_square64 = createTransformKernel("square64");
		_square128 = createTransformKernel("square128");
		_square256 = createTransformKernel("square256");
		_square512 = createTransformKernel("square512");
		_square1024 = createTransformKernel("square1024");
		_square2048 = createTransformKernel("square2048");

		const cl_int b_s = cl_int(31 - __builtin_clz(b) - 1);
		const cl_uint b_inv = cl_uint((uint64_t(1) << (b_s + 32)) / b);
		_normalize3a = createNormalizeKernel("normalize3a", cl_uint(b), b_inv, b_s);
		_normalize3b = createNormalizeKernel("normalize3b", cl_uint(b), b_inv, b_s);

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

		_pSplit = new splitter(_ln, CHUNK256, CHUNK1024, sizeof(RNS) + sizeof(RNSe), 11, getLocalMemSize(), getMaxWorkGroupSize());
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
		_releaseKernel(_square32); _releaseKernel(_square64); _releaseKernel(_square128); _releaseKernel(_square256);
		_releaseKernel(_square512); _releaseKernel(_square1024); _releaseKernel(_square2048);
		_releaseKernel(_normalize3a); _releaseKernel(_normalize3b);
		_releaseKernel(_fwd32p); _releaseKernel(_fwd64p); _releaseKernel(_fwd128p); _releaseKernel(_fwd256p);
		_releaseKernel(_fwd512p); _releaseKernel(_fwd1024p); _releaseKernel(_fwd2048p);
		_releaseKernel(_mul32); _releaseKernel(_mul64); _releaseKernel(_mul128); _releaseKernel(_mul256);
		_releaseKernel(_mul512); _releaseKernel(_mul1024); _releaseKernel(_mul2048);
	}

///////////////////////////////

	void readMemory_z(RNS * const zPtr, RNSe  * const zPtre)
	{
		const size_t n = _n;
		_readBuffer(_z, zPtr, sizeof(RNS) * n);
		_readBuffer(_ze, zPtre, sizeof(RNSe) * n);
	}

	void writeMemory_z(const RNS * const zPtr, const RNSe * const zPtre)
	{
		const size_t n = _n;
		_writeBuffer(_z, zPtr, sizeof(RNS) * n);
		_writeBuffer(_ze, zPtre, sizeof(RNSe) * n);
	}

	void writeMemory_zp(const RNS * const zPtr, const RNSe * const zPtre)
	{
		const size_t n = _n;
		_writeBuffer(_zp, zPtr, sizeof(RNS) * n);
		_writeBuffer(_zpe, zPtre, sizeof(RNSe) * n);
	}

	void writeMemory_w(const RNS_W * const wPtr, const RNS_We * const wPtre)
	{
		const size_t n = _n;
		_writeBuffer(_w, wPtr, sizeof(RNS_W) * 2 * n);
		_writeBuffer(_we, wPtre, sizeof(RNS_We) * 2 * n);
	}

///////////////////////////////

private:
	void fb(cl_kernel & kernel, const int lm, const size_t localWorkSize)
	{
		const size_t n_4 = _n / 4;
		const cl_int ilm = cl_int(lm), is = cl_uint(n_4 >> lm);
		_setKernelArg(kernel, 4, sizeof(cl_int), &ilm);
		_setKernelArg(kernel, 5, sizeof(cl_uint), &is);
		_executeKernel(kernel, n_4, localWorkSize);
	}

	void forward64(const int lm) { fb(_forward64, lm, 64 / 4 * CHUNK64); }
	void backward64(const int lm) { fb(_backward64, lm, 64 / 4 * CHUNK64); }
	void forward256(const int lm) { fb(_forward256, lm, 256 / 4 * CHUNK256); }
	void backward256(const int lm) { fb(_backward256, lm, 256 / 4 * CHUNK256); }
	void forward1024(const int lm) { fb(_forward1024, lm, 1024 / 4 * CHUNK1024); }
	void backward1024(const int lm) { fb(_backward1024, lm, 1024 / 4 * CHUNK1024); }

	void square32() { const size_t n_4 = _n / 4; _executeKernel(_square32, n_4, std::min(n_4, size_t(32 / 4 * BLK32))); }
	void square64() { const size_t n_4 = _n / 4; _executeKernel(_square64, n_4, std::min(n_4, size_t(64 / 4 * BLK64))); }
	void square128() { const size_t n_4 = _n / 4; _executeKernel(_square128, n_4, std::min(n_4, size_t(128 / 4 * BLK128))); }
	void square256() { const size_t n_4 = _n / 4; _executeKernel(_square256, n_4, std::min(n_4, size_t(256 / 4 * BLK256))); }
	void square512() { const size_t n_4 = _n / 4; _executeKernel(_square512, n_4, 512 / 4); }
	void square1024() { const size_t n_4 = _n / 4; _executeKernel(_square1024, n_4, 1024 / 4); }
	void square2048() { const size_t n_4 = _n / 4; _executeKernel(_square2048, n_4, 2048 / 4); }

	void fwd32p() { const size_t n_4 = _n / 4; _executeKernel(_fwd32p, n_4, std::min(n_4, size_t(32 / 4 * BLK32))); }
	void fwd64p() { const size_t n_4 = _n / 4; _executeKernel(_fwd64p, n_4, std::min(n_4, size_t(64 / 4 * BLK64))); }
	void fwd128p() { const size_t n_4 = _n / 4; _executeKernel(_fwd128p, n_4, std::min(n_4, size_t(128 / 4 * BLK128))); }
	void fwd256p() { const size_t n_4 = _n / 4; _executeKernel(_fwd256p, n_4, std::min(n_4, size_t(256 / 4 * BLK256))); }
	void fwd512p() { const size_t n_4 = _n / 4; _executeKernel(_fwd512p, n_4, 512 / 4); }
	void fwd1024p() { const size_t n_4 = _n / 4; _executeKernel(_fwd1024p, n_4, 1024 / 4); }
	void fwd2048p() { const size_t n_4 = _n / 4; _executeKernel(_fwd2048p, n_4, 2048 / 4); }

	void mul32() { const size_t n_4 = _n / 4; _executeKernel(_mul32, n_4, std::min(n_4, size_t(32 / 4 * BLK32))); }
	void mul64() { const size_t n_4 = _n / 4; _executeKernel(_mul64, n_4, std::min(n_4, size_t(64 / 4 * BLK64))); }
	void mul128() { const size_t n_4 = _n / 4; _executeKernel(_mul128, n_4, std::min(n_4, size_t(128 / 4 * BLK128))); }
	void mul256() { const size_t n_4 = _n / 4; _executeKernel(_mul256, n_4, std::min(n_4, size_t(256 / 4 * BLK256))); }
	void mul512() { const size_t n_4 = _n / 4; _executeKernel(_mul512, n_4, 512 / 4); }
	void mul1024() { const size_t n_4 = _n / 4; _executeKernel(_mul1024, n_4, 1024 / 4); }
	void mul2048() { const size_t n_4 = _n / 4; _executeKernel(_mul2048, n_4, 2048 / 4); }

	void setTransformArgs(cl_kernel & kernel, const bool isMultiplier = true)
	{
		_setKernelArg(kernel, 0, sizeof(cl_mem), isMultiplier ? &_z : &_zp);
		_setKernelArg(kernel, 1, sizeof(cl_mem), isMultiplier ? &_ze : &_zpe);
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

public:
	void square()
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

private:
	void squareTune(const size_t count, const size_t sIndex, const RNS * const Z, RNSe * const Ze)
	{
		const splitter * const pSplit = _pSplit;

		for (size_t j = 0; j != count; ++j)
		{
			writeMemory_z(Z, Ze);

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
	void initMultiplicand()
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

public:
	void baseMod(const bool dup)
	{
		const cl_uint blk = cl_uint(_baseModBlk);
		const cl_int sblk = dup ? -cl_int(blk) : cl_int(blk);
		const size_t size = _n / blk;
		const cl_int ln = cl_int(_ln);

		_setKernelArg(_normalize3a, 6, sizeof(cl_int), &sblk);
		_setKernelArg(_normalize3a, 7, sizeof(cl_int), &ln);
		_setKernelArg(_normalize3b, 6, sizeof(cl_uint), &blk);
		_executeKernel(_normalize3a, size, std::min(size, _n3aLocalWS));
		_executeKernel(_normalize3b, size, std::min(size, _n3bLocalWS));
	}

private:
	void baseModTune(const size_t count, const size_t blk, const size_t n3aLocalWS, const size_t n3bLocalWS, const RNS * const Z, const RNSe * const Ze)
	{
		const cl_uint cblk = cl_uint(blk);
		const cl_int sblk = cl_int(blk);
		const size_t size = _n / blk;
		const cl_int ln = cl_int(_ln);

		for (size_t i = 0; i != count; ++i)
		{
			writeMemory_z(Z, Ze);

			_setKernelArg(_normalize3a, 6, sizeof(cl_int), &sblk);
			_setKernelArg(_normalize3a, 7, sizeof(cl_int), &ln);
			_setKernelArg(_normalize3b, 6, sizeof(cl_uint), &cblk);
			_executeKernel(_normalize3a, size, std::min(size, n3aLocalWS));
			_executeKernel(_normalize3b, size, std::min(size, n3bLocalWS));
		}
	}

public:
	void tune(const uint32_t base)
	{
		const size_t n = _n;

		RNS * const Z = new RNS[n];
		RNSe * const Ze = new RNSe[n];
		const double maxSqr = n * (base * double(base));
		for (size_t i = 0; i != n; ++i)
		{
			const int v = int(maxSqr * cos(double(i)));
			Z[i] = RNS(v); Ze[i] = RNSe(v);
		}

		setProfiling(true);

		resetProfiles();
		baseModTune(1, 16, 0, 0, Z, Ze);
		const cl_ulong time = getProfileTime();
		if (time == 0) { delete[] Z; delete[] Ze; setProfiling(false); return; }
		const size_t count = std::min(std::max(size_t(100 * getTimerResolution() / time), size_t(2)), size_t(100));

		cl_ulong minT = cl_ulong(-1);

		size_t bMin = 4;
		while (bMin < log(n * double(base + 2)) / log(double(base))) bMin *= 2;

		for (size_t b = bMin; b <= 64; b *= 2)
		{
			// Check convergence
			if (log(maxSqr) >= base * log(double(b))) continue;

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
				_n3aLocalWS = minsa;
				_n3bLocalWS = minsb;
				_baseModBlk = b;
			}
		}
#if defined(ocl_debug)
		{
			std::ostringstream ss; ss << "n3aLocalWS = " << _n3aLocalWS << ", n3bLocalWS = " << _n3bLocalWS << ", baseModBlk = " << _baseModBlk << "." << std::endl;
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
		delete[] Ze;

		setProfiling(false);
	}
};

class transformGPU : public transform
{
private:
	const size_t _mem_size;
	const size_t _num_regs;
	RNS * const _z;
	RNSe * const _ze;
	engine * _pEngine = nullptr;

	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

public:
	transformGPU(const uint32_t b, const uint32_t n, const bool bBoinc, const size_t device, const size_t num_regs,
				 const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose) : transform(1 << n, b, bBoinc),
		_mem_size((1 << n) * num_regs * (sizeof(RNS) + sizeof(RNSe))), _num_regs(num_regs),
		_z(new RNS[(1 << n) * num_regs]), _ze(new RNSe[(1 << n) * num_regs])
	{
		const size_t size = getSize();

		const bool is_boinc_platform = bBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const platform eng_platform = is_boinc_platform ? platform(boinc_platform_id, boinc_device_id) : platform();

		_pEngine = new engine(eng_platform, is_boinc_platform ? 0 : device, n, verbose);

		std::ostringstream src;

		src << "#define\tP1\t" << P1 << "u" << std::endl;
		src << "#define\tP2\t" << P2 << "u" << std::endl;
		src << "#define\tP3\t" << P3 << "u" << std::endl;
		src << "#define\tP1_INV\t" << uint64_t(-1) / P1 - (uint64_t(1) << 32) << "u" << std::endl;
		src << "#define\tP2_INV\t" << uint64_t(-1) / P2 - (uint64_t(1) << 32) << "u" << std::endl;
		src << "#define\tP3_INV\t" << uint64_t(-1) / P3 - (uint64_t(1) << 32) << "u" << std::endl;
		src << "#define\tInvP2_P1\t1822724754u" << std::endl;		// 1 / P2 mod P1
		src << "#define\tInvP3_P1\t607574918u" << std::endl;		// 1 / P3 mod P1
		src << "#define\tInvP3_P2\t2995931465u" << std::endl;		// 1 / P3 mod P2
		src << "#define\tP1P2\t(P1 * (ulong)P2)" << std::endl;
		src << "#define\tP2P3\t(P2 * (ulong)P3)" << std::endl;
		src << "#define\tP1P2P3l\t15383592652180029441ul" << std::endl;
		src << "#define\tP1P2P3h\t3942432002u" << std::endl;
		src << "#define\tP1P2P3_2l\t7691796326090014720ul" << std::endl;
		src << "#define\tP1P2P3_2h\t1971216001u" << std::endl << std::endl;

		src << "#define\tP1\t" << P1 << "u" << std::endl;
		src << "#define\tP1\t" << P1 << "u" << std::endl;
		src << "#define\tP1\t" << P1 << "u" << std::endl;
		src << "#define\tP1\t" << P1 << "u" << std::endl;

		src << "#define\tBLK32\t" << BLK32 << std::endl;
		src << "#define\tBLK64\t" << BLK64 << std::endl;
		src << "#define\tBLK128\t" << BLK128 << std::endl;
		src << "#define\tBLK256\t" << BLK256 << std::endl << std::endl;

		src << "#define\tCHUNK64\t" << CHUNK64 << std::endl;
		src << "#define\tCHUNK256\t" << CHUNK256 << std::endl;
		src << "#define\tCHUNK1024\t" << CHUNK1024 << std::endl << std::endl;

		if (!engine::readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_pEngine->loadProgram(src.str());
		_pEngine->allocMemory();
		_pEngine->createKernels(b);

		RNS_W * const wr = new RNS_W[2 * size];
		RNS_W * const wri = &wr[size];
		RNS_We * const wre = new RNS_We[2 * size];
		RNS_We * const wrie = &wre[size];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const size_t m = 4 * s;
			const RNS prRoot_m = RNS::prRoot_n(uint32_t(m));
			const RNSe prRoot_me = RNSe::prRoot_n(uint32_t(m));
			for (size_t i = 0; i < s; ++i)
			{
				const size_t e = bitRev(i, 2 * s) + 1;
				const RNS wrsi = prRoot_m.pow(uint32_t(e));
				wr[s + i] = wrsi; wri[s + s - i - 1] = -wrsi;
				const RNSe wrsie = prRoot_me.pow(uint32_t(e));
				wre[s + i] = wrsie; wrie[s + s - i - 1] = -wrsie;
			}
		}

		const size_t m = 4 * size / 2;
		const RNS prRoot_m = RNS::prRoot_n(uint32_t(m));
		const RNSe prRoot_me = RNSe::prRoot_n(uint32_t(m));
		for (size_t i = 0; i != size / 4; ++i)
		{
			const size_t e = bitRev(2 * i, 2 * size / 2) + 1;
			wr[size / 2 + i] = prRoot_m.pow(uint32_t(e));
			wri[size / 2 + i] = prRoot_m.pow(uint32_t(m - e));
			wre[size / 2 + i] = prRoot_me.pow(uint32_t(e));
			wrie[size / 2 + i] = prRoot_me.pow(uint32_t(m - e));
		}

		_pEngine->writeMemory_w(wr, wre);
		_pEngine->tune(b);

		delete[] wr;
		delete[] wre;
	}

	virtual ~transformGPU()
	{
		_pEngine->releaseKernels();
		_pEngine->releaseMemory();
		_pEngine->clearProgram();
		delete _pEngine;

		delete[] _z;
		delete[] _ze;
	}

	size_t getMemSize() const override { return _mem_size; }

protected:
	void getZi(int32_t * const zi) const override
	{
		RNS * const z = _z;
		_pEngine->readMemory_z(z, _ze);

		for (size_t i = 0, size = getSize(); i < size; ++i) zi[i] = z[i].r1().getInt();
	}

	void setZi(const int32_t * const zi) override
	{
		RNS * const z = _z;
		RNSe * const ze = _ze;
		for (size_t i = 0, size = getSize(); i < size; ++i) { z[i] = RNS(zi[i]); ze[i] = RNSe(zi[i]); }

		_pEngine->writeMemory_z(z, ze);
	}

public:
	void readContext(file & cFile, const size_t num_regs) override
	{
		RNS * const z = _z;
		RNSe * const ze = _ze;

		const size_t size = getSize();
		cFile.read(reinterpret_cast<char *>(z), sizeof(RNS) * size * num_regs);
		cFile.read(reinterpret_cast<char *>(ze), sizeof(RNSe) * size * num_regs);

		_pEngine->writeMemory_z(z, ze);
	}

	void saveContext(file & cFile, const size_t num_regs) const override
	{
		RNS * const z = _z;
		RNSe * const ze = _ze;

		_pEngine->readMemory_z(z, ze);

		const size_t size = getSize();
		cFile.write(reinterpret_cast<const char *>(z), sizeof(RNS) * size * num_regs);
		cFile.write(reinterpret_cast<const char *>(ze), sizeof(RNSe) * size * num_regs);
	}

	void set(const int32_t a) override
	{
		RNS * const z = _z;
		RNSe * const ze = _ze;
		z[0] = RNS(a); ze[0] = RNSe(a);
		for (size_t i = 1, size = getSize(); i < size; ++i) { z[i] = RNS(0); ze[i] = RNSe(0); }

		_pEngine->writeMemory_z(z, ze);
	}

	void squareDup(const bool dup) override
	{
		_pEngine->square();
		_pEngine->baseMod(dup);
	}

	void initMultiplicand(const size_t src) override
	{
		const size_t size = getSize();
		_pEngine->writeMemory_zp(&_z[src * size], &_ze[src * size]);
		_pEngine->initMultiplicand();
	}

	void mul() override
	{
		_pEngine->mul();
		_pEngine->baseMod(false);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		RNS * const z = _z;
		RNSe * const ze = _ze;

		if (src == 0) _pEngine->readMemory_z(z, ze);

		for (size_t i = 0, size = getSize(); i < size; ++i) { z[dst * size + i] = z[src * size + i]; ze[dst * size + i] = ze[src * size + i]; }

		if (dst == 0) _pEngine->writeMemory_z(z, ze);
	}
};
