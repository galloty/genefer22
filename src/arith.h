/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <vector>

inline uint32_t powmod(const uint32_t a, const uint32_t e, const uint32_t m)
{
	uint64_t r = 1, t = a;

	for (uint32_t i = e; i != 1; i /= 2)
	{
		if (i % 2 != 0) r = (r * t) % m;
		t = (t * t) % m;
	}
	r = (r * t) % m;

	return uint32_t((r * t) % m);
}

inline int kronecker(const uint32_t x, const uint32_t y)
{
	static int tab[8] = { 0, 1, 0, -1, 0, -1 , 0, 1 };

	if (((x & 1) == 0) && ((y & 1) == 0)) return 0;

	uint32_t a = x, b = y;

	int v = 0; while ((b & 1) == 0) { b >>= 1; v = 1 - v; }

	int k; if ((v & 1) == 0) k = 1; else if (a == 0) k = 0; else k = tab[a & 7];

	while (true)
	{
		if (a == 0)
		{
			if (b > 1) return 0;
			else if (b == 1) return k;
		}

		v = 0; while ((a & 1) == 0) { a >>= 1; v = 1 - v; }
		if (((v & 1) != 0) && (b != 0) && (tab[b & 7] == -1)) k = -k;
		if ((a & b & 2) != 0) k = -k;

		const uint32_t t = b % a; b = a; a = t;
	}
}

inline std::vector<uint32_t> primeFactors(const uint32_t n)
{
	std::vector<uint32_t> p;
	uint32_t r = n;

	if (r % 2 == 0)
	{
		do { r /= 2; } while (r % 2 == 0);
		p.push_back(2);
	}

	for (uint32_t d = 3; r != 1; d += 2)
	{
		if (r % d == 0)
		{
			do { r /= d; } while (r % d == 0);
			p.push_back(d);
		}
	}

	return p;
}
