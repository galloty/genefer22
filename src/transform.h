/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

class Transform
{
private:
	const size_t _size;
	const uint32_t _b;

private:
	void unbalance(int32_t * const zi) const
	{
		const size_t size = _size;
		const int32_t base = int32_t(_b);

		int64_t f = 0;
		for (size_t i = 0; i != size; ++i)
		{
			f += zi[i];
			int32_t r = int32_t(f % base);
			if (r < 0) r += base;
			zi[i] = r;
			f -= r;
			f /= base;
		}

		while (f != 0)
		{
			f = -f;		// a[n] = -a[0]

			for (size_t i = 0; i != size; ++i)
			{
				f += zi[i];
				int32_t r = int32_t(f % base);
				if (r < 0) r += base;
				zi[i] = r;
				f -= r;
				f /= base;
				if (f == 0) break;
			}

			if (f == 1)
			{
				bool isMinusOne = true;
				for (size_t i = 0; i != size; ++i)
				{
					if (zi[i] != 0)
					{
						isMinusOne = false;
						break;
					}
				}
				if (isMinusOne)
				{
					// -1 cannot be unbalanced
					zi[0] = -1;
					break;
				}
			}
		}
	}

public:
	virtual void set(const int32_t a) = 0;
	virtual void getZi(int32_t * const zi) const = 0;
	virtual void setZi(int32_t * const zi) = 0;

	virtual void setError(const double error) = 0;
	virtual double getError() const = 0;

	virtual void squareDup(const bool dup) = 0;
	virtual void initMultiplicand() = 0;
	virtual void mul() = 0;

public:
	static Transform * create_sse2(const uint32_t b, const uint32_t n, const size_t num_threads);
	static Transform * create_sse4(const uint32_t b, const uint32_t n, const size_t num_threads);
	static Transform * create_avx(const uint32_t b, const uint32_t n, const size_t num_threads);
	static Transform * create_fma(const uint32_t b, const uint32_t n, const size_t num_threads);
	static Transform * create_512(const uint32_t b, const uint32_t n, const size_t num_threads);

public:
	Transform(const size_t size, const uint32_t b) : _size(size), _b(b) {}
	virtual ~Transform() {}

	bool isOne(uint64_t & residue) const
	{
		const size_t size = _size;

		int32_t * const zi = new int32_t[size];
		getZi(zi);

		unbalance(zi);

		bool isOne = (zi[0] == 1);
		if (isOne) for (size_t k = 1; k < size; ++k) isOne &= (zi[k] == 0);

		uint64_t res = 0;
		for (size_t i = 8; i != 0; --i) res = (res << 8) | (unsigned char)zi[size - i];
		residue = res;

		delete[] zi;

		return isOne;
	}	
};
