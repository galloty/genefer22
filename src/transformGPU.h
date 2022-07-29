/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

#include "transform.h"

class transformGPU : public transform
{
private:

public:
	transformGPU(const uint32_t b, const uint32_t n, const size_t device) : transform(1 << n, b)
	{
	}

	virtual ~transformGPU()
	{
	}

	size_t getMemSize() const override { return 0; }

protected:
	void getZi(int32_t * const zi) const override
	{
	}

	void setZi(int32_t * const zi) override
	{
	}

public:
	void set(const int32_t a) override
	{
	}

	void squareDup(const bool dup) override
	{
	}

	void initMultiplicand(const size_t src) override
	{
	}

	void mul() override
	{
	}

	void copy(const size_t dst, const size_t src) const override
	{
	}

	void setError(const double error) override {}
	double getError() const override { return 0.0; }
};
