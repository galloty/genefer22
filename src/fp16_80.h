/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

// Fixed point: 16-bit.80-bit
class fp16_80
{
private:
	static const size_t size = 3;
	uint32_t a[size];

public:
	explicit fp16_80() {}
	explicit fp16_80(const uint32_t n)
	{
		for (size_t k = 0; k < size - 1; ++k) a[k] = 0;
		a[size - 1] = n;
	}
	fp16_80(const fp16_80 & rhs) { for (size_t k = 0; k < size; ++k) a[k] = rhs.a[k]; }
	fp16_80 & operator=(const fp16_80 & rhs) { for (size_t k = 0; k < size; ++k) a[k] = rhs.a[k]; return *this; }

	bool operator==(const fp16_80 & rhs) const
	{
		for (size_t k = 0; k < size; ++k) if (a[k] != rhs.a[k]) return false;
		return true;
	}

	double hi() const { return std::ldexp(double(a[size - 1]), -16); }
	double lo() const { return std::ldexp(double((uint64_t(a[size - 2]) << 32) | a[size - 3]), -80); }

	uint32_t square_hi() const
	{
		uint64_t r[2 * size]; for (size_t k = 0; k < 2 * size; ++k) r[k] = 0;

		for (size_t j = 0; j < size; ++j)
		{
			for (size_t i = 0; i < size; ++i)
			{
				const uint64_t p = a[j] * uint64_t(a[i]);
				r[j + i] += uint32_t(p);
				r[j + i + 1] += uint32_t(p >> 32);
			}
		}

		uint64_t l = 0;
		for (size_t k = 0; k < 2 * size - 1; ++k) l = r[k] + (l >> 32);
		return uint32_t(r[2 * size - 1] + (l >> 32));
	}

	static fp16_80 hadd(const fp16_80 & x, const fp16_80 & y)
	{
		fp16_80 z;
		uint64_t l = 0;
		for (size_t k = 0; k < size; ++k)
		{
			l = x.a[k] + uint64_t(y.a[k]) + (l >> 32);
			z.a[k] = uint32_t(l);
		}
		for (size_t k = 0; k < size - 1; ++k)
		{
			z.a[k] = (z.a[k] >> 1) | (z.a[k + 1] << 31);
		}
		z.a[size - 1] >>= 1;
		return z;
	}

	static fp16_80 sqrt(const uint32_t x)
	{
		const uint32_t s = std::lround(std::sqrt(double(x)));
		fp16_80 a((s - 1) << 16), b((s + 1) << 16);

		for (size_t i = 0; i < 100; ++i)
		{
			const fp16_80 c = fp16_80::hadd(a, b);
			const uint32_t c2_high = c.square_hi();
			if (c2_high < x)
			{
				if (a == c) return c;
				a = c;
			}
			else
			{
				if (b == c) return c;
				b = c;
			}
		}

		return fp16_80::hadd(a, b);
	}
};
