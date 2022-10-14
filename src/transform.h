/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <string>
#include <sstream>

#include "gint.h"
#include "file.h"

class transform
{
protected:
	enum class EKind { DTvec2, DTvec4, DTvec8, IBDTvec2, IBDTvec4, IBDTvec8, NTT2, NTT3, NTT3cpu }; 

private:
	const size_t _size;
	const uint32_t _n;
	const uint32_t _b;
	const EKind _kind;

protected:
	virtual void getZi(int32_t * const zi) const = 0;
	virtual void setZi(const int32_t * const zi) = 0;

public:
	virtual void set(const int32_t a) = 0;					// r_0 = a
	virtual void squareDup(const bool dup) = 0;				// r_0 = r_0^2 or 2*r_0^2
	virtual void initMultiplicand(const size_t src) = 0;	// r_m = transform(r_src)
	virtual void mul() = 0;									// r_0 *= r_m

	virtual void copy(const size_t dst, const size_t src) const = 0;	// r_dst = r_src

	virtual size_t getMemSize() const = 0;
	virtual size_t getCacheSize() const = 0;

	virtual bool readContext(file & cFile, const size_t num_regs) = 0;
	virtual void saveContext(file & cFile, const size_t num_regs) const = 0;

	virtual double getError() const { return 0; }

private:
#if defined(GPU)
	static transform * create_ocl(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose);
#elif defined(__aarch64__)
	static transform * create_neon(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError);
#else
	static transform * create_i32(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static transform * create_sse2(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError);
	static transform * create_sse4(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError);
	static transform * create_avx(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError);
	static transform * create_fma(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError);
	static transform * create_512(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError);
#endif

public:
	transform(const size_t size, const uint32_t n, const uint32_t b, const EKind kind) : _size(size), _n(n), _b(b), _kind(kind) {}
	virtual ~transform() {}

protected:
	size_t getSize() const { return _size; }
	uint32_t getB() const { return _b; }
	EKind getKind() const { return _kind; }

public:
#if defined(GPU)
	static transform * create_gpu(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								  const cl_platform_id boinc_platform_id, const cl_device_id boinc_device_id, const bool verbose)
	{
		transform * const pTransform = transform::create_ocl(b, n, isBoinc, device, num_regs, boinc_platform_id, boinc_device_id, verbose);
		if (pTransform == nullptr) throw std::runtime_error("OpenCL device not found");
		return pTransform;
	}
#else
	static transform * create_cpu(const uint32_t b, const uint32_t n, const size_t num_threads, const std::string & impl, const size_t num_regs,
								  const bool checkError, std::string & ttype)
	{
		transform * pTransform = nullptr;

#if defined(__aarch64__)
		pTransform = transform::create_neon(b, n, num_threads, num_regs, checkError);
		ttype = "neon";
#else
		if (__builtin_cpu_supports("avx512f") && (impl.empty() || (impl == "512")))
		{
			pTransform = transform::create_512(b, n, num_threads, num_regs, checkError);
			ttype = "512";
		}
		else if (__builtin_cpu_supports("fma") && (impl.empty() || (impl == "fma")))
		{
			pTransform = transform::create_fma(b, n, num_threads, num_regs, checkError);
			ttype = "fma";
		}
		else if (__builtin_cpu_supports("avx") && (impl.empty() || (impl == "avx")))
		{
			pTransform = transform::create_avx(b, n, num_threads, num_regs, checkError);
			ttype = "avx";
		}
		else if (__builtin_cpu_supports("sse4.1") && (impl.empty() || (impl == "sse4")))
		{
			pTransform = transform::create_sse4(b, n, num_threads, num_regs, checkError);
			ttype = "sse4";
		}
		else if (__builtin_cpu_supports("sse2") && (impl.empty() || (impl == "sse2")))
		{
			pTransform = transform::create_sse2(b, n, num_threads, num_regs, checkError);
			ttype = "sse2";
		}
		else if (__builtin_cpu_supports("avx2") && (impl.empty() || (impl == "i32")))
		{
			pTransform = transform::create_i32(b, n, num_threads, num_regs);
			ttype = "i32";
		}
		else
		{
			if (impl.empty()) throw std::runtime_error("processor must support sse2");
			std::ostringstream ss; ss << impl << " is not supported";
			throw std::runtime_error(ss.str());
		}
#endif

		return pTransform;
	}
#endif

	void mul(const size_t src)
	{
		initMultiplicand(src);
		mul();
	}

	void getInt(gint & g) const
	{
		if ((g.getSize() != (size_t(1) << _n)) || (g.getBase() != _b)) throw std::runtime_error("getInt");
		getZi(g.data());
		g.reset();
	}

	void setInt(gint & g)
	{
		if ((g.getSize() != (size_t(1) << _n)) || (g.getBase() != _b)) throw std::runtime_error("setInt");
		g.balance();
		setZi(g.data());
	}

	// void add1()
	// {
	// 	int32_t * const zi = new int32_t[size_t(1) << _n];
	// 	getZi(zi);
	// 	zi[0] += 1;
	// 	setZi(zi);
	// 	delete[] zi;
	// }
};
