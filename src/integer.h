/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

class integer
{
private:
	size_t len;
	size_t exp;
	uint32_t * a;

private:
	integer() : len(0), exp(0), a(nullptr) {}

private:
	static void clear(uint32_t * const a, const size_t len)
	{
		for (size_t i = 0; i != len; ++i) a[i] = 0;
	}

private:
	static void fill(uint32_t * const a, const uint32_t * const rhs_a, const size_t len)
	{
		for (size_t i = 0; i != len; ++i) a[i] = rhs_a[i];
	}

private:
	static void addArray(uint32_t * const a, const uint32_t * const b, const size_t len)
	{
		int64_t l = 0;
		for (size_t i = 0; i != len; ++i)
		{
			l += a[i] + (int64_t)b[i];
			a[i] = (uint32_t)l; l >>= 32;
		}
		for (size_t i = len; l != 0; ++i)
		{
			l += a[i]; a[i] = (uint32_t)l; l >>= 32;
		}
	}

private:
	static void subArray(uint32_t * const a, const uint32_t * const b, const size_t len)
	{
		int64_t l = 0;
		for (size_t i = 0; i != len; ++i)
		{
			l += a[i] - (int64_t)b[i];
			a[i] = (uint32_t)l; l >>= 32;
		}
		for (size_t i = len; l != 0; ++i)
		{
			l += a[i]; a[i] = (uint32_t)l; l >>= 32;
		}
	}

private:
	void norm()
	{
		size_t a_len = len;
		const uint32_t * const a_a = a;
		while ((a_len != 0) && (a_a[a_len - 1] == 0)) --a_len;
		len = a_len;
	}

private:
	void setArray(const size_t nlen, uint32_t * const na)
	{
		if (a != nullptr) delete[] a;
		len = nlen; a = na;
		norm();
	}

private:
	void grammarSchoolSquare()
	{
		const size_t a_len = len;
		if (a_len == 0) return;
		const uint32_t * const a_a = a;

		uint32_t * const na = new uint32_t[2 * a_len];

		uint64_t l = 0, m = a_a[0];
		for (size_t i = 0; i != a_len; ++i)
		{
			l += m * a_a[i];
			na[i] = (uint32_t)l; l >>= 32;
		}
		na[a_len] = (uint32_t)l;

		for (size_t j = 1; j < a_len; ++j)
		{
			uint64_t l = 0, m = a_a[j];
			for (size_t i = 0; i != a_len; ++i)
			{
				l += na[j + i] + m * a_a[i];
				na[j + i] = (uint32_t)l; l >>= 32;
			}
			na[j + a_len] = (uint32_t)l;
		}

		setArray(2 * a_len, na);
	}

private:
	void split(integer & hi, integer & lo, const size_t lo_len) const
	{
		uint32_t * const lo_a = new uint32_t[lo_len];
		fill(lo_a, a, lo_len);

		const size_t hi_len = len - lo_len;
		uint32_t * const hi_a = new uint32_t[hi_len];
		fill(hi_a, &a[lo_len], hi_len);

		lo.setArray(lo_len, lo_a);
		hi.setArray(hi_len, hi_a);
	}

private:
	void KaratsubaSquare()
	{
		const size_t m = len - len / 2, nlen = 2 * len;

		integer z2, z0; split(z2, z0, m);

		z0.squareMantissa();

		len = m + 1; a[m] = 0; // Set length such that it is >= hi.len
		addArray(a, z2.a, z2.len); norm();

		z2.squareMantissa();
		squareMantissa();

		subArray(a, z0.a, z0.len);
		subArray(a, z2.a, z2.len);
		norm();

		uint32_t * const na = new uint32_t[nlen];

		fill(na, z0.a, z0.len);
		clear(&na[z0.len], 2 * m - z0.len);
		fill(&na[2 * m], z2.a, z2.len);
		clear(&na[2 * m + z2.len], nlen - 2 * m - z2.len);

		addArray(&na[m], a, len);

		setArray(nlen, na);
	}

private:
	void squareMantissa()
	{
		if (len >= 64) KaratsubaSquare();
		else grammarSchoolSquare();
	}

private:
	void square()
	{
		exp *= 2;
		squareMantissa();
	}

public:
	integer(const uint32_t base, const size_t m) : len(1), exp(0), a(new uint32_t[2])
	{
		uint32_t b = base;
		while (b % 2 == 0) { b /= 2; ++exp; }

		a[0] = b;

		for (size_t j = m; j != 1; j /= 2) square();
	}

public:
	~integer()
	{
		if (a != nullptr) delete[] a;
	}

public:
	size_t bitSize() const
	{
		size_t l = exp + (len - 1) * 8 * sizeof(uint32_t) - 1;
		for (uint32_t x = a[len - 1]; x != 0; x /= 2) ++l;
		return l;
	}

public:
	char bit(const size_t i) const
	{
		if (i < exp) return 0;
		const size_t j = i - exp;
		return (a[j / (8 * sizeof(uint32_t))] >> (j % (8 * sizeof(uint32_t)))) & 1;
	}
};
