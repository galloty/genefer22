/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <algorithm>

class gint
{
private:
	size_t _size;
	int32_t _base;
	int32_t * _d;

	enum class EState { Unknown, Balanced, Unbalanced };
	mutable EState _state;

public:
	gint() : _size(0), _base(0), _d(nullptr), _state(EState::Unknown) {}
	virtual ~gint() { clear(); }

	void init(const size_t size, const int32_t base)
	{
		_size = size;
		_base = base;
		_d = new int32_t[size];
	}

	void clear()
	{
		if (_d != nullptr) { delete[] _d; _d = nullptr; }
	}

	int32_t * d() const { return _d; }

	void unbalance() const
	{
		if (_state == EState::Unbalanced) return;
		_state = EState::Unbalanced;

		const size_t size = _size;
		const int32_t base = _base;
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
		const int32_t base = _base;
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
		const size_t size = _size;
		int32_t * const d = _d;

		unbalance();
		bool bOne = (d[0] == 1);
		if (bOne) for (size_t k = 1; k < size; ++k) bOne &= (d[k] == 0);
		return bOne;
	}

	uint64_t gethash64() const
	{
		const size_t size = _size;
		int32_t * const d = _d;

		unbalance();
		uint64_t hash = 0;
		for (size_t i = 0; i < size; ++i)
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
