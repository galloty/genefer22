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

class Transform
{
private:
	const size_t _size;
	const uint32_t _b;

protected:
	virtual void getZi(int32_t * const zi) const = 0;
	virtual void setZi(int32_t * const zi) = 0;

public:
	virtual void set(const int32_t a) = 0;					// r0 = a
	virtual void squareDup(const bool dup) = 0;				// r0 = r0^2 or 2*r0^2
	virtual void initMultiplicand(const size_t src) = 0;	// r1 = transform(r_src)
	virtual void mul() = 0;									// r0 *= r1

	virtual void copy(const size_t dst, const size_t src) const = 0;	// r_dst = r_src

	virtual void setError(const double error) = 0;
	virtual double getError() const = 0;

	virtual size_t getMemSize() const = 0;

private:
	static Transform * create_sse2(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_sse4(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_avx(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_fma(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_512(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);

public:
	Transform(const size_t size, const uint32_t b) : _size(size), _b(b) {}
	virtual ~Transform() {}

	static Transform * create(const uint32_t b, const uint32_t n, const size_t num_threads, const std::string & impl, const size_t num_regs, std::string & ttype)
	{
		Transform * transform = nullptr;

		if (__builtin_cpu_supports("avx512f") && (impl.empty() || (impl == "512")))
		{
			transform = Transform::create_512(b, n, num_threads, num_regs);
			ttype = "512";
		}
		else if (__builtin_cpu_supports("fma") && (impl.empty() || (impl == "fma")))
		{
			transform = Transform::create_fma(b, n, num_threads, num_regs);
			ttype = "fma";
		}
		else if (__builtin_cpu_supports("avx") && (impl.empty() || (impl == "avx")))
		{
			transform = Transform::create_avx(b, n, num_threads, num_regs);
			ttype = "avx";
		}
		else if (__builtin_cpu_supports("sse4.1") && (impl.empty() || (impl == "sse4")))
		{
			transform = Transform::create_sse4(b, n, num_threads, num_regs);
			ttype = "sse4";
		}
		else if (__builtin_cpu_supports("sse2") && (impl.empty() || (impl == "sse2")))
		{
			transform = Transform::create_sse2(b, n, num_threads, num_regs);
			ttype = "sse2";
		}
		else
		{
			if (impl.empty()) throw std::runtime_error("processor must support sse2");
			std::ostringstream ss; ss << impl << " is not supported";
			throw std::runtime_error(ss.str());
		}

		return transform;
	}

	void mul(const size_t src)
	{
		initMultiplicand(src);
		mul();
	}

	void getInt(gint & g) const
	{
		g.init(_size, _b);
		getZi(g.d());
	}

	void setInt(const gint & g)
	{
		g.balance();
		setZi(g.d());
	}
};
