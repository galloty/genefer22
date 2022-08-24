/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <algorithm>

#include "file.h"
#include "pio.h"

class gint
{
private:
	const size_t _size;
	const uint32_t _base;
	int32_t * const _d;

	enum class EState { Unknown, Balanced, Unbalanced };
	EState _state;

private:
	static constexpr uint64_t rotl64(const uint64_t x, const uint8_t n) { return (x << n) | (x >> (-n & 63)); }

public:
	gint(const size_t size, const uint32_t base) : _size(size), _base(base), _d(new int32_t[size]), _state(EState::Unknown) {}
	virtual ~gint() { delete[] _d; }

	size_t getSize() const { return _size; }
	uint32_t getBase() const { return _base; }
	int32_t * data() const { return _d; }

	void reset() { _state = EState::Unknown; }

	void unbalance()
	{
		if (_state == EState::Unbalanced) return;
		_state = EState::Unbalanced;

		const size_t size = _size;
		const int32_t base = int32_t(_base);
		int32_t * const d = _d;

		int64_t f = 0;
		for (size_t i = 0; i < size; ++i)
		{
			f += d[i];
			int32_t r = int32_t(f % base);
			if (r < 0) r += base;
			d[i] = r;
			f -= r;
			f /= base;
		}

		while (f != 0)
		{
			f = -f;	// d[size] = -d[0]

			for (size_t i = 0; i < size; ++i)
			{
				f += d[i];
				int32_t r = int32_t(f % base);
				if (r < 0) r += base;
				d[i] = r;
				f -= r;
				f /= base;
				if (f == 0) break;
			}

			if (f == 1)
			{
				bool isMinusOne = true;
				for (size_t i = 0; i < size; ++i)
				{
					if (d[i] != 0)
					{
						isMinusOne = false;
						break;
					}
				}
				if (isMinusOne)
				{
					// -1 cannot be unbalanced
					d[0] = -1;
					break;
				}
			}
		}
	}

	void balance()
	{
		if (_state == EState::Balanced) return;
		_state = EState::Balanced;

		const size_t size = _size;
		const int32_t base = int32_t(_base);
		int32_t * const d = _d;

		int64_t f = 0;
		for (size_t i = 0; i < size; ++i)
		{
			f += d[i];
			int32_t r = int32_t(f % base);
			if (r > base / 2) r -= base;
			if (r <= -base / 2) r += base;
			d[i] = r;
			f -= r;
			f /= base;
		}

		while (f != 0)
		{
			f = -f;	// d[size] = -d[0]

			for (size_t i = 0; i < size; ++i)
			{
				f += d[i];
				int32_t r = int32_t(f % base);
				if (r > base / 2) r -= base;
				if (r <= -base / 2) r += base;
				d[i] = r;
				f -= r;
				f /= base;
				if (f == 0) break;
			}
		}
	}

	void read(file & cFile)
	{
		uint32_t size; cFile.read(reinterpret_cast<char *>(&size), sizeof(size));
		uint32_t base; cFile.read(reinterpret_cast<char *>(&base), sizeof(base));
		if ((size_t(size) != _size) || (base != _base)) cFile.error("bad file");
		cFile.read(reinterpret_cast<char *>(_d), _size * sizeof(int32_t));
		_state = EState::Unbalanced;
	}

	void write(file & cFile)
	{
		unbalance();
		const uint32_t size = uint32_t(_size);
		cFile.write(reinterpret_cast<const char *>(&size), sizeof(size));
		cFile.write(reinterpret_cast<const char *>(&_base), sizeof(_base));
		cFile.write(reinterpret_cast<const char *>(_d), _size * sizeof(int32_t));
	}

	bool isOne(uint64_t & res64, uint64_t & old64)
	{
		unbalance();
		const size_t size = _size;
		const uint32_t base = _base;
		const int32_t * const d = _d;

		bool bOne = (d[0] == 1);
		if (bOne) for (size_t i = 1; i < size; ++i) bOne &= (d[i] == 0);

		uint64_t r64 = 0, b = 1;
		for (size_t i = 0; i < size; ++i)
		{
			r64 += uint32_t(d[i]) * b;
			b *= base;
		}
		res64 = r64;

		uint64_t old = 0;
		for (size_t i = 8; i != 0; --i) old = (old << 8) | uint8_t(d[size - i]);
		old64 = old;

		return bOne;
	}

	uint64_t gethash64()
	{
		unbalance();
		const int32_t * const d = _d;
		uint64_t hash = 0;
		bool isZero = true;
		for (size_t i = 0, size = _size; i < size; ++i)
		{
			const uint32_t a_i = d[i];
			hash += a_i;
			hash ^= rotl64(a_i + 0xc39d8a0552b073e8ull, (17 * uint64_t(a_i) + 5) % 64);
			isZero &= (a_i == 0);
		}
		if (isZero) pio::error("value is zero", true);
		return hash;
	}

	uint32_t gethash32()
	{
		const uint64_t hash = gethash64();
		return std::max(uint32_t(2), uint32_t(hash) ^ uint32_t(hash >> 32));
	}
};
