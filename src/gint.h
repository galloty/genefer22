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

class gint
{
private:
	const size_t _size;
	const uint32_t _base;
	int32_t * const _d;

	enum class EState { Unknown, Balanced, Unbalanced };
	mutable EState _state;

public:
	gint(const size_t size, const uint32_t base) : _size(size), _base(base), _d(new int32_t[size]), _state(EState::Unknown) {}
	virtual ~gint() { delete[] _d; }

	size_t getSize() const { return _size; }
	uint32_t getBase() const { return _base; }
	int32_t * data() const { return _d; }

	void reset() { _state = EState::Unknown; }

	void read(file & cFile)
	{
		size_t size; cFile.read(reinterpret_cast<char *>(&size), sizeof(size));
		uint32_t base; cFile.read(reinterpret_cast<char *>(&base), sizeof(base));
		if ((size != _size) || (base != _base)) throw std::runtime_error("bad file");
		reset();
		cFile.read(reinterpret_cast<char *>(_d), _size * sizeof(int32_t));
	}

	void write(file & cFile) const
	{
		cFile.write(reinterpret_cast<const char *>(&_size), sizeof(_size));
		cFile.write(reinterpret_cast<const char *>(&_base), sizeof(_base));
		cFile.write(reinterpret_cast<const char *>(_d), _size * sizeof(int32_t));
	}

	void unbalance() const
	{
		if (_state == EState::Unbalanced) return;
		_state = EState::Unbalanced;

		const size_t size = _size;
		const int32_t base = int32_t(_base);
		int32_t * const d = _d;

		int64_t f = 0;
		for (size_t i = 0; i != size; ++i)
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

			for (size_t i = 0; i != size; ++i)
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
				for (size_t i = 0; i != size; ++i)
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

	void balance() const
	{
		if (_state == EState::Balanced) return;
		_state = EState::Balanced;

		const size_t size = _size;
		const int32_t base = int32_t(_base);
		int32_t * const d = _d;

		int64_t f = 0;
		for (size_t i = 0; i != size; ++i)
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

			for (size_t i = 0; i != size; ++i)
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

	bool isOne() const
	{
		unbalance();
		const int32_t * const d = _d;
		bool bOne = (d[0] == 1);
		if (bOne) for (size_t k = 1, size = _size; k < size; ++k) bOne &= (d[k] == 0);
		return bOne;
	}

	uint64_t gethash64() const
	{
		unbalance();
		const int32_t * const d = _d;
		uint64_t hash = 0;
		for (size_t i = 0, size = _size; i < size; ++i)
		{
			const uint64_t a_i = d[i];
			hash += a_i;
			hash ^= _rotl64(a_i + 0xc39d8a0552b073e8ull, (17 * a_i + 5) % 64);
		}
		return hash;
	}

	uint32_t gethash32() const
	{
		const uint64_t hash = gethash64();
		return std::max(uint32_t(2), uint32_t(hash) ^ uint32_t(hash >> 32));
	}
};
