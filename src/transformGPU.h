/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>
#include <fstream>

#include "transform.h"
#include "ocl.h"

#include "ocl/kernel.h"

// Test Forward/Backward 64/256/1024 code
//#define CHECK64
//#define CHECK256
//#define CHECK1024

#define	P1_i31			2130706433u		// 127 * 2^24 + 1
#define	P2_i31			2013265921u		// 15 * 2^27 + 1
#define	P3_i31			1711276033u		// 51 * 2^25 + 1
#define	InvP2_P1_i31	 913159918u		// 1 / P2 mod P1
#define	InvP3_P2_i31	 671088647u		// 1 / P3 mod P2
#define	InvP2P3_P1_i31	1059265576u		// 1 / (P2*P3) mod P1

#define	P1P2_i31	(P1_i31 * (cl_ulong)P2_i31)
#define	P2P3_i31	(P2_i31 * (cl_ulong)P3_i31)

#define	P1P2P3hi_i31	397946880u				// P1 * P2 * P3 / 2^64
#define	P1P2P3lo_i31	11381159214173913089ull	// P1 * P2 * P3 mod 2^64

template <cl_uint p, cl_uint prRoot>
class Zp
{
private:
	cl_uint n;

private:
	explicit Zp(const cl_uint l) :n(l) {}

	Zp MulMod(const Zp & rhs) const
	{
		return Zp((cl_uint)((n * (cl_ulong)rhs.n) % p));
	}

public:
	Zp() {}
	explicit Zp(const int i) :n(i + ((i < 0) ? p : 0)) {}
	explicit Zp(const int64_t i) :n((cl_uint)(i + ((i < 0) ? p : 0))) {}

	cl_uint Get() const { return n; }
	void Set(const cl_uint _n) { n = _n; }

	Zp operator-() const { return Zp((n != 0) ? p - n : 0); }

	Zp & operator*=(const Zp & rhs) { *this = MulMod(rhs); return *this; }
	Zp operator*(const Zp & rhs) const { return MulMod(rhs); }

	Zp Pow(const unsigned int e) const
	{
		if (e == 0) return Zp(1);

		Zp r = Zp(1), y = *this;
		for (unsigned int i = e; i != 1; i /= 2)
		{
			if (i % 2 != 0) r *= y;
			y *= y;
		}
		r *= y;

		return r;
	}

	static const Zp PrRoot_n(const unsigned int n) { return Zp(prRoot).Pow((p - 1) / n); }
};

typedef Zp<P1_i31, 3> Zp1;
typedef Zp<P2_i31, 31> Zp2;
typedef Zp<P3_i31, 29> Zp3;

class RNS
{
private:
	cl_uint2 r;	// Zp1, Zp2

private:
	explicit RNS(const Zp1 & r1, const Zp2 & r2) { r.s[0] = r1.Get(); r.s[1] = r2.Get(); }
	Zp1 r1() const { Zp1 r1; r1.Set(r.s[0]); return r1; }
	Zp2 r2() const { Zp2 r2; r2.Set(r.s[1]); return r2; }

public:
	RNS() {}
	explicit RNS(const int i) { r.s[0] = Zp1(i).Get(); r.s[1] = Zp2(i).Get(); }
	explicit RNS(const int64_t i) { r.s[0] = Zp1(i).Get(); r.s[1] = Zp2(i).Get(); }

	cl_uint Get1() const { return r.s[0]; }
	cl_uint Get2() const { return r.s[1]; }
	void Set(const cl_uint r1, const cl_uint r2) { r.s[0] = r1; r.s[1] = r2; }

	RNS operator-() const { return RNS(-r1(), -r2()); }

	RNS operator*(const RNS & rhs) const { return RNS(r1() * rhs.r1(), r2() * rhs.r2()); }

	RNS Pow(const unsigned int e) const { return RNS(r1().Pow(e), r2().Pow(e)); }

	static const RNS PrRoot_n(const unsigned int n) { return RNS(Zp1::PrRoot_n(n), Zp2::PrRoot_n(n)); }
};

class RNSe
{
private:
	cl_uint r;	// Zp3

private:
	explicit RNSe(const Zp3 & r3) { r = r3.Get(); }
	Zp3 r3() const { Zp3 _r3; _r3.Set(r); return _r3; }

public:
	RNSe() {}
	explicit RNSe(const int i) { r = Zp3(i).Get(); }
	explicit RNSe(const int64_t i) { r = Zp3(i).Get(); }

	cl_uint Get3() const { return r; }
	void Set(const cl_uint r3) { r = r3; }

	RNSe operator-() const { return RNSe(-r3()); }

	RNSe operator*(const RNSe & rhs) const { return RNSe(r3() * rhs.r3()); }

	RNSe Pow(const unsigned int e) const { return RNSe(r3().Pow(e)); }

	static const RNSe PrRoot_n(const unsigned int n) { return RNSe(Zp3::PrRoot_n(n)); }
};

class RNS_W
{
private:
	cl_uint4 r;	// r1 in Zp1, r2 in Zp2, (r1 * 2^32) / p1, (r2 * 2^32) / p2)

public:
	RNS_W & operator=(const RNS & rhs)
	{
		const cl_uint r1 = rhs.Get1(), r2 = rhs.Get2();
		r.s[0] = r1; r.s[1] = r2;
		r.s[2] = (cl_uint)(((cl_ulong)r1 << 32) / P1_i31);
		r.s[3] = (cl_uint)(((cl_ulong)r2 << 32) / P2_i31);
		return *this;
	}

	RNS Get() const { RNS r12; r12.Set(r.s[0], r.s[1]); return r12; }
};

class RNS_We
{
private:
	cl_uint2 r;	// r3 in Zp3, (r3 * 2^32) / p3

public:
	RNS_We & operator=(const RNSe & rhs)
	{
		const cl_uint r3 = rhs.Get3();
		r.s[0] = r3;
		r.s[1] = (cl_uint)(((cl_ulong)r3 << 32) / P3_i31);
		return *this;
	}

	RNSe Get() const { RNSe r3; r3.Set(r.s[0]); return r3; }
};

// Garner Algorithm

struct uint96
{
	cl_ulong lo;
	cl_uint hi;

	uint96() {}
	explicit uint96(const cl_uint h, const cl_ulong l) :lo(l), hi(h) {}

	static uint96 Mul(const cl_uint lhs, const cl_ulong rhs)
	{
		const cl_uint bl = (cl_uint)rhs, bh = (cl_uint)(rhs >> 32);
		const cl_ulong abl = lhs * (cl_ulong)bl, abh = lhs * (cl_ulong)bh;
		const cl_uint lo = (cl_uint)abl;
		const cl_ulong hi = abh + (abl >> 32);
		uint96 r; r.lo = (hi << 32) | lo; r.hi = (cl_uint)(hi >> 32);
		return r;
	}

	uint96 operator+(const cl_ulong rhs) const
	{
		uint96 r; r.lo = lo + rhs; r.hi = hi + ((r.lo < lo) ? 1u : 0u);
		return r;
	}

	uint96 & operator-=(const uint96 & rhs)
	{
		const cl_uint c = (this->lo < rhs.lo) ? 1u : 0u;
		this->lo -= rhs.lo; this->hi -= rhs.hi + c;
		return *this;
	}
};

static int64_t GetInt(const cl_uint r1, const cl_uint r2, const cl_uint r3)
{
	cl_uint d2 = r2 - r3; if (r2 < r3) d2 += P2_i31;		// mod P2
	const cl_uint u2 = (cl_uint)((d2 * (cl_ulong)InvP3_P2_i31) % P2_i31);
	const cl_ulong n2 = r3 + u2 * (cl_ulong)P3_i31;

	const cl_uint n1 = (cl_uint)(n2 % P1_i31);
	cl_uint d1 = r1 - n1; if (r1 < n1) d1 += P1_i31;		// mod P1
	const cl_uint u1 = (cl_uint)((d1 * (cl_ulong)InvP2P3_P1_i31) % P1_i31);
	uint96 n = uint96::Mul(u1, P2P3_i31) + n2;

	if (n.hi > P1P2P3hi_i31 / 2) n -= uint96(P1P2P3hi_i31, P1P2P3lo_i31);

	return (int64_t)n.lo;
}

#define BLK32_i31a		8
#define BLK64_i31a		4
#define BLK128_i31a		2
#define BLK256_i31a		1

#define CHUNK64_i31a	8
#define CHUNK256_i31a	4
#define CHUNK1024_i31a	2

class Program_i31a : public HostProgram
{
private:
	const unsigned int base;
	cl_kernel forward64, backward64, forward256, backward256, forward1024, backward1024;
	cl_kernel square32, square64, square128, square256, square512, square1024, square2048;
	cl_kernel baseMod0, baseMod1;
	cl_mem z, ze, w, we, c, bErr;
	size_t baseMod0LocalWorkSize, baseMod1LocalWorkSize, baseModBlk, splitIndex;
	cl_uint * const ePtr;
	Splitter * pSplit;

private:
	cl_int SetArgs(cl_kernel kernel)
	{
		cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &z);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &ze);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &w);
		err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &we);
		return err;
	}

private:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src) const
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
	Program_i31a(const size_t size, const unsigned int lgSize, const unsigned int b, const bool verbose = false) :
		HostProgram(size, lgSize, verbose),
		base(b),
		z(nullptr), ze(nullptr), w(nullptr), we(nullptr), c(nullptr), bErr(nullptr),
		baseMod0LocalWorkSize(32), baseMod1LocalWorkSize(32), baseModBlk(16), splitIndex(0),
		ePtr(new cl_uint[size / 4]), pSplit(nullptr)
	{
		std::stringstream src;
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		cl_int err = BuildProgram(src.str().c_str());
		if (err != CL_SUCCESS) return;

		pSplit = new Splitter(lgSize, CHUNK256_i31a, CHUNK1024_i31a, sizeof(RNS) + sizeof(RNSe), 11, GetLocalMemSize(), GetMaxWorkGroupSize());
		if (pSplit->GetSize() == 0) return;

		forward64 = CreateKernel("Forward64"); if (forward64 == nullptr) return;
		backward64 = CreateKernel("Backward64"); if (backward64 == nullptr) return;
		forward256 = CreateKernel("Forward256"); if (forward256 == nullptr) return;
		backward256 = CreateKernel("Backward256"); if (backward256 == nullptr) return;
		forward1024 = CreateKernel("Forward1024"); if (forward1024 == nullptr) return;
		backward1024 = CreateKernel("Backward1024"); if (backward1024 == nullptr) return;
		square32 = CreateKernel("Square32"); if (square32 == nullptr) return;
		square64 = CreateKernel("Square64"); if (square64 == nullptr) return;
		square128 = CreateKernel("Square128"); if (square128 == nullptr) return;
		square256 = CreateKernel("Square256"); if (square256 == nullptr) return;
		square512 = CreateKernel("Square512"); if (square512 == nullptr) return;
		square1024 = CreateKernel("Square1024"); if (square1024 == nullptr) return;
		square2048 = CreateKernel("Square2048"); if (square2048 == nullptr) return;
		baseMod0 = CreateKernel("BaseMod0"); if (baseMod0 == nullptr) return;
		baseMod1 = CreateKernel("BaseMod1"); if (baseMod1 == nullptr) return;

		if (n > 0)
		{
			z = CreateBuffer(CL_MEM_READ_WRITE, sizeof(RNS) * n); if (z == nullptr) return;
			ze = CreateBuffer(CL_MEM_READ_WRITE, sizeof(RNSe) * n); if (ze == nullptr) return;
			w = CreateBuffer(CL_MEM_READ_ONLY, sizeof(RNS_W) * 2 * n); if (w == nullptr) return;
			we = CreateBuffer(CL_MEM_READ_ONLY, sizeof(RNS_We) * 2 * n); if (we == nullptr) return;
			c = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_long) * n / 4); if (c == nullptr) return;
			bErr = CreateBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * n / 4); if (bErr == nullptr) return;
		}

		err |= SetArgs(forward64);
		err |= SetArgs(backward64);
		err |= SetArgs(forward256);
		err |= SetArgs(backward256);
		err |= SetArgs(forward1024);
		err |= SetArgs(backward1024);
		err |= SetArgs(square32);
		err |= SetArgs(square64);
		err |= SetArgs(square128);
		err |= SetArgs(square256);
		err |= SetArgs(square512);
		err |= SetArgs(square1024);
		err |= SetArgs(square2048);

		const cl_int ibase = (cl_int)base;
		const cl_long recBase = (cl_long)(0xffffffffffffffffull / (unsigned int)base);
		const cl_uint iln = (cl_uint)ln;

		err |= clSetKernelArg(baseMod0, 0, sizeof(cl_mem), &z);
		err |= clSetKernelArg(baseMod0, 1, sizeof(cl_mem), &ze);
		err |= clSetKernelArg(baseMod0, 2, sizeof(cl_mem), &c);
		err |= clSetKernelArg(baseMod0, 3, sizeof(cl_mem), &bErr);
		err |= clSetKernelArg(baseMod0, 4, sizeof(cl_int), &ibase);
		err |= clSetKernelArg(baseMod0, 5, sizeof(cl_long), &recBase);
		err |= clSetKernelArg(baseMod0, 6, sizeof(cl_uint), &iln);

		err |= clSetKernelArg(baseMod1, 0, sizeof(cl_mem), &z);
		err |= clSetKernelArg(baseMod1, 1, sizeof(cl_mem), &ze);
		err |= clSetKernelArg(baseMod1, 2, sizeof(cl_mem), &c);
		err |= clSetKernelArg(baseMod1, 3, sizeof(cl_int), &ibase);
		err |= clSetKernelArg(baseMod1, 4, sizeof(cl_long), &recBase);

		if (err != CL_SUCCESS) { if (verbose) fprintf(stderr, "Error: cannot set args.\n"); error = err; return; }
	}

	~Program_i31a()
	{
		if (error == CL_SUCCESS)
		{
			if (z != nullptr) clReleaseMemObject(z);
			if (ze != nullptr) clReleaseMemObject(ze);
			if (w != nullptr) clReleaseMemObject(w);
			if (we != nullptr) clReleaseMemObject(we);
			if (c != nullptr) clReleaseMemObject(c);
			if (bErr != nullptr) clReleaseMemObject(bErr);

			clReleaseKernel(forward64);
			clReleaseKernel(backward64);
			clReleaseKernel(forward256);
			clReleaseKernel(backward256);
			clReleaseKernel(forward1024);
			clReleaseKernel(backward1024);
			clReleaseKernel(square32);
			clReleaseKernel(square64);
			clReleaseKernel(square128);
			clReleaseKernel(square256);
			clReleaseKernel(square512);
			clReleaseKernel(square1024);
			clReleaseKernel(square2048);
			clReleaseKernel(baseMod0);
			clReleaseKernel(baseMod1);

			ClearProgram();
		}

		if (pSplit != nullptr) delete pSplit;
		delete[] ePtr;
	}

public:
	void ReadZBuffer(RNS * const zPtr, RNSe  * const zPtre)
	{
		ReadBuffer(z, sizeof(RNS) * n, zPtr);
		ReadBuffer(ze, sizeof(RNSe) * n, zPtre);
	}

public:
	void WriteZBuffer(const RNS * const zPtr, const RNSe * const zPtre)
	{
		WriteBuffer(z, sizeof(RNS) * n, zPtr);
		WriteBuffer(ze, sizeof(RNSe) * n, zPtre);
	}

public:
	void WriteWBuffers(const RNS_W * const wPtr, const RNS_We * const wPtre)
	{
		WriteBuffer(w, sizeof(RNS_W) * 2 * n, wPtr);
		WriteBuffer(we, sizeof(RNS_We) * 2 * n, wPtre);
	}

public:
	cl_uint ReadErr()
	{
		const size_t size = n / baseModBlk;
		ReadBuffer(bErr, sizeof(cl_uint) * size, ePtr);
		cl_uint e = 0;
		for (size_t i = 0; i != size; ++i) e |= ePtr[i];
		return e;
	}

public:
	void ClearErr()
	{
		for (size_t i = 0; i != n / 4; ++i) ePtr[i] = 0;
		WriteBuffer(bErr, sizeof(cl_uint) * n / 4, ePtr);
	}

///////////////////////////////

private:
	void _Forward64(const unsigned int lm)
	{
		const cl_uint ilm = (cl_uint)lm, is = (cl_uint)((n / 4) >> lm);
		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(forward64, 4, sizeof(cl_uint), &ilm);
		err |= clSetKernelArg(forward64, 5, sizeof(cl_uint), &is);
		error |= err;
	}

private:
	void Forward64(const unsigned int lm)
	{
		_Forward64(lm);
		Execute(forward64, n / 4, 64 / 4 * CHUNK64_i31a);
	}

private:
	cl_ulong Forward64Tune(const unsigned int lm)
	{
		_Forward64(lm);
		return ExecuteProfiling(forward64, n / 4, 64 / 4 * CHUNK64_i31a);
	}

private:
	void _Backward64(const unsigned int lm)
	{
		const cl_uint ilm = (cl_uint)lm, is = (cl_uint)((n / 4) >> lm);
		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(backward64, 4, sizeof(cl_uint), &ilm);
		err |= clSetKernelArg(backward64, 5, sizeof(cl_uint), &is);
		error |= err;
	}

private:
	void Backward64(const unsigned int lm)
	{
		_Backward64(lm);
		Execute(backward64, n / 4, 64 / 4 * CHUNK64_i31a);
	}

private:
	cl_ulong Backward64Tune(const unsigned int lm)
	{
		_Backward64(lm);
		return ExecuteProfiling(backward64, n / 4, 64 / 4 * CHUNK64_i31a);
	}

///////////////////////////////

private:
	void _Forward256(const unsigned int lm)
	{
		const cl_uint ilm = (cl_uint)lm, is = (cl_uint)((n / 4) >> lm);
		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(forward256, 4, sizeof(cl_uint), &ilm);
		err |= clSetKernelArg(forward256, 5, sizeof(cl_uint), &is);
		error |= err;
	}

private:
	void Forward256(const unsigned int lm)
	{
		_Forward256(lm);
		Execute(forward256, n / 4, 256 / 4 * CHUNK256_i31a);
	}

private:
	cl_ulong Forward256Tune(const unsigned int lm)
	{
		_Forward256(lm);
		return ExecuteProfiling(forward256, n / 4, 256 / 4 * CHUNK256_i31a);
	}

private:
	void _Backward256(const unsigned int lm)
	{
		const cl_uint ilm = (cl_uint)lm, is = (cl_uint)((n / 4) >> lm);
		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(backward256, 4, sizeof(cl_uint), &ilm);
		err |= clSetKernelArg(backward256, 5, sizeof(cl_uint), &is);
		error |= err;
	}

private:
	void Backward256(const unsigned int lm)
	{
		_Backward256(lm);
		Execute(backward256, n / 4, 256 / 4 * CHUNK256_i31a);
	}

private:
	cl_ulong Backward256Tune(const unsigned int lm)
	{
		_Backward256(lm);
		return ExecuteProfiling(backward256, n / 4, 256 / 4 * CHUNK256_i31a);
	}

///////////////////////////////

private:
	void _Forward1024(const unsigned int lm)
	{
		const cl_uint ilm = (cl_uint)lm, is = (cl_uint)((n / 4) >> lm);
		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(forward1024, 4, sizeof(cl_uint), &ilm);
		err |= clSetKernelArg(forward1024, 5, sizeof(cl_uint), &is);
		error |= err;
	}

private:
	void Forward1024(const unsigned int lm)
	{
		_Forward1024(lm);
		Execute(forward1024, n / 4, 1024 / 4 * CHUNK1024_i31a);
	}

private:
	cl_ulong Forward1024Tune(const unsigned int lm)
	{
		_Forward1024(lm);
		return ExecuteProfiling(forward1024, n / 4, 1024 / 4 * CHUNK1024_i31a);
	}

private:
	void _Backward1024(const unsigned int lm)
	{
		const cl_uint ilm = (cl_uint)lm, is = (cl_uint)((n / 4) >> lm);
		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(backward1024, 4, sizeof(cl_uint), &ilm);
		err |= clSetKernelArg(backward1024, 5, sizeof(cl_uint), &is);
		error |= err;
	}

private:
	void Backward1024(const unsigned int lm)
	{
		_Backward1024(lm);
		Execute(backward1024, n / 4, 1024 / 4 * CHUNK1024_i31a);
	}

private:
	cl_ulong Backward1024Tune(const unsigned int lm)
	{
		_Backward1024(lm);
		return ExecuteProfiling(backward1024, n / 4, 1024 / 4 * CHUNK1024_i31a);
	}

///////////////////////////////

private:
	void Square32() { Execute(square32, n / 4, std::min(n / 4, (size_t)(32 / 4 * BLK32_i31a))); }
	void Square64() { Execute(square64, n / 4, std::min(n / 4, (size_t)(64 / 4 * BLK64_i31a))); }
	void Square128() { Execute(square128, n / 4, std::min(n / 4, (size_t)(128 / 4 * BLK128_i31a))); }
	void Square256() { Execute(square256, n / 4, std::min(n / 4, (size_t)(256 / 4 * BLK256_i31a))); }
	void Square512() { Execute(square512, n / 4, 512 / 4); }
	void Square1024() { Execute(square1024, n / 4, 1024 / 4); }
	void Square2048() { Execute(square2048, n / 4, 2048 / 4); }

private:
	cl_ulong Square32Tune() { return ExecuteProfiling(square32, n / 4, std::min(n / 4, (size_t)(32 / 4 * BLK32_i31a))); }
	cl_ulong Square64Tune() { return ExecuteProfiling(square64, n / 4, std::min(n / 4, (size_t)(64 / 4 * BLK64_i31a))); }
	cl_ulong Square128Tune() { return ExecuteProfiling(square128, n / 4, std::min(n / 4, (size_t)(128 / 4 * BLK128_i31a))); }
	cl_ulong Square256Tune() { return ExecuteProfiling(square256, n / 4, std::min(n / 4, (size_t)(256 / 4 * BLK256_i31a))); }
	cl_ulong Square512Tune() { return ExecuteProfiling(square512, n / 4, 512 / 4); }
	cl_ulong Square1024Tune() { return ExecuteProfiling(square1024, n / 4, 1024 / 4); }
	cl_ulong Square2048Tune() { return ExecuteProfiling(square2048, n / 4, 2048 / 4); }

public:
	void Square()
	{
		const Splitter * const pSplit = this->pSplit;

		const size_t sIndex = splitIndex;
		const size_t s = pSplit->GetPartSize(sIndex);

		unsigned int lm = ln;

		for (size_t i = 0; i < s - 1; ++i)
		{
			const unsigned int k = pSplit->GetPart(sIndex, i);
			if (k == 10)
			{
				lm -= 10;
				Forward1024(lm);
			}
			else if (k == 8)
			{
				lm -= 8;
				Forward256(lm);
			}
			else // if (k == 6)
			{
				lm -= 6;
				Forward64(lm);
			}
		}

		// lm = split.GetPart(sIndex, s - 1);
		if (lm == 11) Square2048();
		else if (lm == 10) Square1024();
		else if (lm == 9) Square512();
		else if (lm == 8) Square256();
		else if (lm == 7) Square128();
		else if (lm == 6) Square64();
		else if (lm == 5) Square32();

		for (size_t i = 0; i < s - 1; ++i)
		{
			const unsigned int k = pSplit->GetPart(sIndex, s - 2 - i);
			if (k == 10)
			{
				Backward1024(lm);
				lm += 10;
			}
			else if (k == 8)
			{
				Backward256(lm);
				lm += 8;
			}
			else // if (k == 6)
			{
				Backward64(lm);
				lm += 6;
			}
		}
	}

private:
	void SquareTune(const size_t count, const size_t sIndex, const RNS * const Z, RNSe * const Ze, cl_ulong & t)
	{
		const Splitter * const pSplit = this->pSplit;

		t = 0;

		for (size_t j = 0; j != count; ++j)
		{
			WriteZBuffer(Z, Ze);

			const size_t s = pSplit->GetPartSize(sIndex);

			unsigned int lm = ln;

			for (size_t i = 0; i < s - 1; ++i)
			{
				const unsigned int k = pSplit->GetPart(sIndex, i);
				if (k == 10)
				{
					lm -= 10;
#ifndef CHECK1024
					t +=
#endif
						Forward1024Tune(lm);
				}
				else if (k == 8)
				{
					lm -= 8;
#ifndef CHECK256
					t +=
#endif
						Forward256Tune(lm);
				}
				else // if (k == 6)
				{
					lm -= 6;
#ifndef CHECK64
					t +=
#endif
						Forward64Tune(lm);
				}
			}

			// lm = split.GetPart(sIndex, s - 1);
			if (lm == 11) t += Square2048Tune();
			else if (lm == 10) t += Square1024Tune();
			else if (lm == 9) t += Square512Tune();
			else if (lm == 8) t += Square256Tune();
			else if (lm == 7) t += Square128Tune();
			else if (lm == 6) t += Square64Tune();
			else if (lm == 5) t += Square32Tune();

			for (size_t i = 0; i < s - 1; ++i)
			{
				const unsigned int k = pSplit->GetPart(sIndex, s - 2 - i);
				if (k == 10)
				{
#ifndef CHECK1024
					t +=
#endif
						Backward1024Tune(lm);
					lm += 10;
				}
				else if (k == 8)
				{
#ifndef CHECK256
					t +=
#endif
						Backward256Tune(lm);
					lm += 8;
				}
				else // if (k == 6)
				{
#ifndef CHECK64
					t +=
#endif
						Backward64Tune(lm);
					lm += 6;
				}
			}
		}
	}

public:
	void BaseMod(const int g)
	{
		const cl_uint blk = (cl_uint)baseModBlk;
		const cl_int ig = (cl_int)g;
		const size_t size = (int)n / (int)baseModBlk;

		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(baseMod0, 7, sizeof(cl_uint), &blk);
		err |= clSetKernelArg(baseMod0, 8, sizeof(cl_int), &ig);
		err |= clSetKernelArg(baseMod1, 5, sizeof(cl_uint), &blk);
		error |= err; if (err != CL_SUCCESS) return;
		Execute(baseMod0, size, std::min(size, baseMod0LocalWorkSize));
		Execute(baseMod1, size, std::min(size, baseMod1LocalWorkSize));
	}

private:
	void BaseModTune(const size_t count, const size_t blk, const size_t localWorkSize, const RNS * const Z, const RNSe * const Ze, cl_ulong & t0, cl_ulong & t1)
	{
		t0 = t1 = 0;

		const cl_uint b = (cl_uint)blk;
		const cl_int ig = (cl_int)1;
		const size_t size = (int)n / (int)blk;

		for (size_t i = 0; i != count; ++i)
		{
			WriteZBuffer(Z, Ze);

			cl_int err = CL_SUCCESS;
			err |= clSetKernelArg(baseMod0, 7, sizeof(cl_uint), &b);
			err |= clSetKernelArg(baseMod0, 8, sizeof(cl_int), &ig);
			err |= clSetKernelArg(baseMod1, 5, sizeof(cl_uint), &b);
			error |= err; if (err != CL_SUCCESS) return;
			t0 += ExecuteProfiling(baseMod0, size, std::min(size, localWorkSize));
			t1 += ExecuteProfiling(baseMod1, size, std::min(size, localWorkSize));
		}
	}

public:
	void Tune()
	{
		if (!SelfTuning()) return;

		RNS * const Z = new RNS[n];
		RNSe * const Ze = new RNSe[n];
		const double maxSqr = n * (double)base * (double)base;
		for (size_t i = 0; i != n; ++i)
		{
			const int v = (int)(maxSqr * cos((double)i));
			Z[i] = RNS(v);
			Ze[i] = RNSe(v);
		}

		cl_ulong t0, t1;
		BaseModTune(1, 16, 0, Z, Ze, t0, t1);
		if (t0 + t1 == 0) { delete[] Z; delete[] Ze; return; }
		const size_t count = (size_t)(std::max(100 * GetTimerResolution() / (t0 + t1), (cl_ulong)2));
		if (count > 1000) { delete[] Z; delete[] Ze; return; }

		cl_ulong minT = (cl_ulong)(-1);

		size_t bMin = 4;
		while (bMin < log(n * (double)(base + 2)) / log((double)base)) bMin *= 2;

		for (size_t b = bMin; b <= 64; b *= 2)
		{
			// Check convergence
			if (log(maxSqr) >= base * log((double)b)) continue;

			cl_ulong minT0, minT1;
			BaseModTune(count, b, 0, Z, Ze, minT0, minT1);
			//printf("b = %d, s = %d: count = %d, t0 = %d, t1 = %d.\n", (int)b, (int)0, (int)count, (int)minT0, (int)minT1);

			size_t s0 = 0, s1 = 0;

			for (size_t s = 1; s <= 256; s *= 2)
			{
				cl_ulong t0, t1;
				BaseModTune(count, b, s, Z, Ze, t0, t1);
				//printf("b = %d, s = %d: count = %d, t0 = %d, t1 = %d.\n", (int)b, (int)s, (int)count, (int)t0, (int)t1);

				if (t0 < minT0)
				{
					minT0 = t0;
					s0 = s;
				}
				if (t1 < minT1)
				{
					minT1 = t1;
					s1 = s;
				}
			}

			if (minT0 + minT1 < minT)
			{
				minT = minT0 + minT1;
				baseMod0LocalWorkSize = s0;
				baseMod1LocalWorkSize = s1;
				baseModBlk = b;
			}
		}

		//printf("localWorkSize0 = %d, localWorkSize1 = %d, baseModBlk = %d.\n", (int)baseMod0LocalWorkSize, (int)baseMod1LocalWorkSize, (int)baseModBlk);

		const size_t ns = pSplit->GetSize();
		if (ns > 1)
		{
			minT = (cl_ulong)(-1);
			for (size_t i = 0; i < ns; ++i)
			{
				//printf("[%u]", (unsigned int)i);
				//for (size_t j = 0, nps = pSplit->GetPartSize(i); j < nps; ++j) printf(" %u", pSplit->GetPart(i, j));
				cl_ulong t;
				SquareTune(2, i, Z, Ze, t);
				//printf(": %u\n", (unsigned int)t);

				if (t < minT)
				{
					minT = t;
					splitIndex = i;
				}
			}

			//for (size_t j = 0, nps = pSplit->GetPartSize(splitIndex); j < nps; ++j) printf(" %u", pSplit->GetPart(splitIndex, j));
			//printf("\n");
		}

		delete[] Z;
		delete[] Ze;
	}
};

class transformGPU : public transform
{
private:
	RNS * const z;
	RNSe * const ze;
	Program_i31a * const pProgram;

	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

public:
	transformGPU(const uint32_t b, const uint32_t n, const size_t device, const size_t num_regs) : transform(1 << n, b),
		z(new RNS[(1 << n) * num_regs]), ze(new RNSe[(1 << n) * num_regs]), pProgram(new Program_i31a(1 << n, n, b))
	{
		const size_t size = getSize();

		RNS_W * const wr = new RNS_W[2 * size];
		RNS_We * const wre = new RNS_We[2 * size];
		RNS_W * const wri = &wr[size];
		RNS_We * const wrie = &wre[size];
		for (size_t s = 1; s < size / 2; s *= 2)
		{
			const size_t m = 4 * s;
			const RNS prRoot_m = RNS::PrRoot_n((unsigned int)m);
			const RNSe prRoot_me = RNSe::PrRoot_n((unsigned int)m);
			for (size_t i = 0; i < s; ++i)
			{
				const size_t e = bitRev(i, 2 * s) + 1;
				const RNS wrsi = prRoot_m.Pow((unsigned int)e);
				wr[s + i] = wrsi; wri[s + s - i - 1] = -wrsi;
				const RNSe wrsie = prRoot_me.Pow((unsigned int)e);
				wre[s + i] = wrsie; wrie[s + s - i - 1] = -wrsie;
			}
		}

		const size_t m = 4 * size / 2;
		const RNS prRoot_m = RNS::PrRoot_n((unsigned int)m);
		const RNSe prRoot_me = RNSe::PrRoot_n((unsigned int)m);
		for (size_t i = 0; i != size / 4; ++i)
		{
			const size_t e = bitRev(2 * i, 2 * size / 2) + 1;
			wr[size / 2 + i] = prRoot_m.Pow((unsigned int)e);
			wri[size / 2 + i] = prRoot_m.Pow((unsigned int)(m - e));
			wre[size / 2 + i] = prRoot_me.Pow((unsigned int)e);
			wrie[size / 2 + i] = prRoot_me.Pow((unsigned int)(m - e));
		}

		pProgram->IsOK();
		pProgram->WriteWBuffers(wr, wre);
		pProgram->ClearErr();
		pProgram->Tune();
		pProgram->ClearErr();

		delete[] wr;
		delete[] wre;
	}

	virtual ~transformGPU()
	{
		delete pProgram;

		delete[] z;
		delete[] ze;
	}

	size_t getMemSize() const override { return 0; }

protected:
	void getZi(int32_t * const zi) const override
	{
		pProgram->Sync();

		pProgram->ReadZBuffer(z, ze);

		for (size_t i = 0, size = getSize(); i < size; ++i) zi[i] = GetInt(z[i].Get1(), z[i].Get2(), ze[i].Get3());
	}

	void setZi(int32_t * const zi) override
	{
		pProgram->Sync();

		for (size_t i = 0, size = getSize(); i < size; ++i) { z[i] = RNS(zi[i]); ze[i] = RNSe(zi[i]); }

		pProgram->WriteZBuffer(z, ze);
	}

public:
	void set(const int32_t a) override
	{
		RNS * const z = this->z;
		RNSe * const ze = this->ze;
		z[0] = RNS(a); ze[0] = RNSe(a);
		for (size_t i = 1, size = getSize(); i < size; ++i) { z[i] = RNS(0); ze[i] = RNSe(0); }
		pProgram->WriteZBuffer(z, ze);
	}

	void squareDup(const bool dup) override
	{
		pProgram->Square();
		pProgram->BaseMod(dup ? 2 : 1);
		pProgram->IsOK();
	}

	void initMultiplicand(const size_t src) override
	{
	}

	void mul() override
	{
	}

	void copy(const size_t dst, const size_t src) const override
	{
		pProgram->Sync();

		if (src == 0) pProgram->ReadZBuffer(z, ze);

		for (size_t i = 0, size = getSize(); i < size; ++i) { z[dst * size + i] = z[src * size + i]; ze[dst * size + i] = ze[src * size + i]; }

		if (dst == 0) pProgram->WriteZBuffer(z, ze);
	}

	void setError(const double error) override {}
	double getError() const override { return 0.0; }
};
