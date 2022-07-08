/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

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

public:
	static Transform * create_sse2(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_sse4(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_avx(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_fma(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);
	static Transform * create_512(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs);

public:
	Transform(const size_t size, const uint32_t b) : _size(size), _b(b) {}
	virtual ~Transform() {}

	void mul(const size_t src)
	{
		initMultiplicand(src);
		mul();
	}

	gint * getInt() const
	{
		gint * const gptr = new gint(_size, _b);
		getZi(gptr->d());
		gptr->unbalance();
		return gptr;
	}

	void setInt(const gint * const gptr)
	{
		gptr->balance();
		setZi(gptr->d());
	}
};
