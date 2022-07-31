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

	static Zp norm(const size_t n) { return -Zp(int32_t((p - 1) / n)); }
	static const Zp prRoot_n(const uint32_t n) { return Zp(prRoot).pow((p - 1) / n); }
};

typedef Zp<4253024257u, 5> Zp1;		// 507 * 2^23 + 1
typedef Zp<4194304001u, 3> Zp2;		// 125 * 2^25 + 1
typedef Zp<4076863489u, 7> Zp3;		// 243 * 2^24 + 1

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

#define BLK32		8
#define BLK64		4
#define BLK128		2
#define BLK256		1

#define CHUNK64		8
#define CHUNK256	4
#define CHUNK1024	2

class Program_i31a : public HostProgram
{
private:
	const unsigned int base;
	cl_kernel forward64, backward64, forward256, backward256, forward1024, backward1024;
	cl_kernel square32, square64, square128, square256, square512, square1024, square2048;
	cl_kernel normalize3a, normalize3b;
	cl_mem z, ze, w, we, c, bErr;
	size_t n3aLocalWS, n3bLocalWS, baseModBlk, splitIndex;
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
		n3aLocalWS(32), n3bLocalWS(32), baseModBlk(16), splitIndex(0),
		ePtr(new cl_uint[size / 4]), pSplit(nullptr)
	{
		std::stringstream src;
		src << "#define\tnorm1\t" << Zp1::norm(size / 2).get() << "u" << std::endl;
		src << "#define\tnorm2\t" << Zp2::norm(size / 2).get() << "u" << std::endl;
		src << "#define\tnorm3\t" << Zp3::norm(size / 2).get() << "u" << std::endl;
 		src << std::endl;
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		cl_int err = BuildProgram(src.str().c_str());
		if (err != CL_SUCCESS) return;

		pSplit = new Splitter(lgSize, CHUNK256, CHUNK1024, sizeof(RNS) + sizeof(RNSe), 11, GetLocalMemSize(), GetMaxWorkGroupSize());
		if (pSplit->GetSize() == 0) return;

		forward64 = CreateKernel("forward64"); if (forward64 == nullptr) return;
		backward64 = CreateKernel("backward64"); if (backward64 == nullptr) return;
		forward256 = CreateKernel("forward256"); if (forward256 == nullptr) return;
		backward256 = CreateKernel("backward256"); if (backward256 == nullptr) return;
		forward1024 = CreateKernel("forward1024"); if (forward1024 == nullptr) return;
		backward1024 = CreateKernel("backward1024"); if (backward1024 == nullptr) return;
		square32 = CreateKernel("square32"); if (square32 == nullptr) return;
		square64 = CreateKernel("square64"); if (square64 == nullptr) return;
		square128 = CreateKernel("square128"); if (square128 == nullptr) return;
		square256 = CreateKernel("square256"); if (square256 == nullptr) return;
		square512 = CreateKernel("square512"); if (square512 == nullptr) return;
		square1024 = CreateKernel("square1024"); if (square1024 == nullptr) return;
		square2048 = CreateKernel("square2048"); if (square2048 == nullptr) return;
		normalize3a = CreateKernel("normalize3a"); if (normalize3a == nullptr) return;
		normalize3b = CreateKernel("normalize3b"); if (normalize3b == nullptr) return;

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

		const cl_uint cb = cl_uint(b);
		const cl_int b_s = cl_int(30 - __builtin_clz(b));
		const cl_uint b_inv = cl_uint((uint64_t(1) << (b_s + 32)) / b);

		err |= clSetKernelArg(normalize3a, 0, sizeof(cl_mem), &z);
		err |= clSetKernelArg(normalize3a, 1, sizeof(cl_mem), &ze);
		err |= clSetKernelArg(normalize3a, 2, sizeof(cl_mem), &c);
		err |= clSetKernelArg(normalize3a, 3, sizeof(cl_uint), &cb);
		err |= clSetKernelArg(normalize3a, 4, sizeof(cl_uint), &b_inv);
		err |= clSetKernelArg(normalize3a, 5, sizeof(cl_int), &b_s);

		err |= clSetKernelArg(normalize3b, 0, sizeof(cl_mem), &z);
		err |= clSetKernelArg(normalize3b, 1, sizeof(cl_mem), &ze);
		err |= clSetKernelArg(normalize3b, 2, sizeof(cl_mem), &c);
		err |= clSetKernelArg(normalize3b, 3, sizeof(cl_uint), &cb);
		err |= clSetKernelArg(normalize3b, 4, sizeof(cl_uint), &b_inv);
		err |= clSetKernelArg(normalize3b, 5, sizeof(cl_int), &b_s);

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
			clReleaseKernel(normalize3a);
			clReleaseKernel(normalize3b);

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
		Execute(forward64, n / 4, 64 / 4 * CHUNK64);
	}

private:
	cl_ulong Forward64Tune(const unsigned int lm)
	{
		_Forward64(lm);
		return ExecuteProfiling(forward64, n / 4, 64 / 4 * CHUNK64);
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
		Execute(backward64, n / 4, 64 / 4 * CHUNK64);
	}

private:
	cl_ulong Backward64Tune(const unsigned int lm)
	{
		_Backward64(lm);
		return ExecuteProfiling(backward64, n / 4, 64 / 4 * CHUNK64);
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
		Execute(forward256, n / 4, 256 / 4 * CHUNK256);
	}

private:
	cl_ulong Forward256Tune(const unsigned int lm)
	{
		_Forward256(lm);
		return ExecuteProfiling(forward256, n / 4, 256 / 4 * CHUNK256);
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
		Execute(backward256, n / 4, 256 / 4 * CHUNK256);
	}

private:
	cl_ulong Backward256Tune(const unsigned int lm)
	{
		_Backward256(lm);
		return ExecuteProfiling(backward256, n / 4, 256 / 4 * CHUNK256);
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
		Execute(forward1024, n / 4, 1024 / 4 * CHUNK1024);
	}

private:
	cl_ulong Forward1024Tune(const unsigned int lm)
	{
		_Forward1024(lm);
		return ExecuteProfiling(forward1024, n / 4, 1024 / 4 * CHUNK1024);
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
		Execute(backward1024, n / 4, 1024 / 4 * CHUNK1024);
	}

private:
	cl_ulong Backward1024Tune(const unsigned int lm)
	{
		_Backward1024(lm);
		return ExecuteProfiling(backward1024, n / 4, 1024 / 4 * CHUNK1024);
	}

///////////////////////////////

private:
	void Square32() { Execute(square32, n / 4, std::min(n / 4, (size_t)(32 / 4 * BLK32))); }
	void Square64() { Execute(square64, n / 4, std::min(n / 4, (size_t)(64 / 4 * BLK64))); }
	void Square128() { Execute(square128, n / 4, std::min(n / 4, (size_t)(128 / 4 * BLK128))); }
	void Square256() { Execute(square256, n / 4, std::min(n / 4, (size_t)(256 / 4 * BLK256))); }
	void Square512() { Execute(square512, n / 4, 512 / 4); }
	void Square1024() { Execute(square1024, n / 4, 1024 / 4); }
	void Square2048() { Execute(square2048, n / 4, 2048 / 4); }

private:
	cl_ulong Square32Tune() { return ExecuteProfiling(square32, n / 4, std::min(n / 4, (size_t)(32 / 4 * BLK32))); }
	cl_ulong Square64Tune() { return ExecuteProfiling(square64, n / 4, std::min(n / 4, (size_t)(64 / 4 * BLK64))); }
	cl_ulong Square128Tune() { return ExecuteProfiling(square128, n / 4, std::min(n / 4, (size_t)(128 / 4 * BLK128))); }
	cl_ulong Square256Tune() { return ExecuteProfiling(square256, n / 4, std::min(n / 4, (size_t)(256 / 4 * BLK256))); }
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
	void BaseMod(const bool dup)
	{
		const cl_uint blk = cl_uint(baseModBlk);
		const cl_int sblk = dup ? -cl_int(blk) : cl_int(blk);
		const size_t size = int(n) / int(baseModBlk);

		cl_int err = CL_SUCCESS;
		err |= clSetKernelArg(normalize3a, 6, sizeof(cl_int), &sblk);
		err |= clSetKernelArg(normalize3b, 6, sizeof(cl_uint), &blk);
		error |= err; if (err != CL_SUCCESS) return;
		Execute(normalize3a, size, std::min(size, n3aLocalWS));
		Execute(normalize3b, size, std::min(size, n3bLocalWS));
	}

private:
	void BaseModTune(const size_t count, const size_t blk, const size_t localWorkSize, const RNS * const Z, const RNSe * const Ze, cl_ulong & t0, cl_ulong & t1)
	{
		t0 = t1 = 0;

		const cl_uint cblk = cl_uint(blk);
		const cl_int sblk = cl_int(blk);
		const size_t size = int(n) / int(blk);

		for (size_t i = 0; i != count; ++i)
		{
			WriteZBuffer(Z, Ze);

			cl_int err = CL_SUCCESS;
			err |= clSetKernelArg(normalize3a, 6, sizeof(cl_int), &sblk);
			err |= clSetKernelArg(normalize3b, 6, sizeof(cl_uint), &cblk);
			error |= err; if (err != CL_SUCCESS) return;
			t0 += ExecuteProfiling(normalize3a, size, std::min(size, localWorkSize));
			t1 += ExecuteProfiling(normalize3b, size, std::min(size, localWorkSize));
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
				n3aLocalWS = s0;
				n3bLocalWS = s1;
				baseModBlk = b;
			}
		}

		const size_t ns = pSplit->GetSize();
		if (ns > 1)
		{
			minT = (cl_ulong)(-1);
			for (size_t i = 0; i < ns; ++i)
			{
				cl_ulong t;
				SquareTune(2, i, Z, Ze, t);

				if (t < minT)
				{
					minT = t;
					splitIndex = i;
				}
			}
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
			const RNS prRoot_m = RNS::prRoot_n((unsigned int)m);
			const RNSe prRoot_me = RNSe::prRoot_n((unsigned int)m);
			for (size_t i = 0; i < s; ++i)
			{
				const size_t e = bitRev(i, 2 * s) + 1;
				const RNS wrsi = prRoot_m.pow((unsigned int)e);
				wr[s + i] = wrsi; wri[s + s - i - 1] = -wrsi;
				const RNSe wrsie = prRoot_me.pow((unsigned int)e);
				wre[s + i] = wrsie; wrie[s + s - i - 1] = -wrsie;
			}
		}

		const size_t m = 4 * size / 2;
		const RNS prRoot_m = RNS::prRoot_n((unsigned int)m);
		const RNSe prRoot_me = RNSe::prRoot_n((unsigned int)m);
		for (size_t i = 0; i != size / 4; ++i)
		{
			const size_t e = bitRev(2 * i, 2 * size / 2) + 1;
			wr[size / 2 + i] = prRoot_m.pow((unsigned int)e);
			wri[size / 2 + i] = prRoot_m.pow((unsigned int)(m - e));
			wre[size / 2 + i] = prRoot_me.pow((unsigned int)e);
			wrie[size / 2 + i] = prRoot_me.pow((unsigned int)(m - e));
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

		for (size_t i = 0, size = getSize(); i < size; ++i) zi[i] = z[i].r1().getInt();
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
		pProgram->BaseMod(dup);
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
