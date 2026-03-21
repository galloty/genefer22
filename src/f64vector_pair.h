/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "f64vector.h"

namespace transformCPU_namespace
{

template<size_t N>
struct VcxPair { Vcx<N> l, h; };

template<size_t N>
class Vradix4Pair
{
	using Vc = Vcx<N>;
	using Vcp = VcxPair<N>;
	using Vr4 = Vradix4<N>;

private:
	Vr4 l, h;

public:
	finline explicit Vradix4Pair() {}

	finline void load_l(const Vcp * const mem, const size_t step)
	{
		for (size_t i = 0; i < 4; ++i) l[i] = mem[i * step].l;
	}

	finline void load_h(const Vcp * const mem, const size_t step)
	{
		for (size_t i = 0; i < 4; ++i) h[i] = mem[i * step].h;
	}

	finline void store_l(Vcp * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 4; ++i) mem[i * step].l = l[i];
	}

	finline void store_h(Vcp * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 4; ++i) mem[i * step].h = h[i];
	}

	finline void load_l(const Vcp * const mem)	// VSIZE = 8, 4_4
	{
		for (size_t i = 0; i < 4; ++i) l[i] = mem[(4 * i) / 8 + ((4 * i) % 8)].l;
	}

	finline void load_h(const Vcp * const mem)	// VSIZE = 8, 4_4
	{
		for (size_t i = 0; i < 4; ++i) h[i] = mem[(4 * i) / 8 + ((4 * i) % 8)].h;
	}

	finline void store_l(Vcp * const mem) const	// VSIZE = 8, 4_4
	{
		for (size_t i = 0; i < 4; ++i) mem[(4 * i) / 8 + ((4 * i) % 8)].l = l[i];
	}

	finline void store_h(Vcp * const mem) const	// VSIZE = 8, 4_4
	{
		for (size_t i = 0; i < 4; ++i) mem[(4 * i) / 8 + ((4 * i) % 8)].h = h[i];
	}

	finline static void forward4e(const size_t m, Vcp * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4Pair vrp; Vcp * const zi = &z[i];
			vrp.load_l(zi, m); vrp.l.forward4e(w0, w1); vrp.store_l(zi, m);
			vrp.load_h(zi, m); vrp.h.forward4e(w0, w1); vrp.store_h(zi, m);
		}
	}

	finline static void forward4o(const size_t m, Vcp * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4Pair vrp; Vcp * const zi = &z[i];
			vrp.load_l(zi, m); vrp.l.forward4o(w0, w2); vrp.store_l(zi, m);
			vrp.load_h(zi, m); vrp.h.forward4o(w0, w2); vrp.store_h(zi, m);
		}
	}

	finline static void backward4e(const size_t m, Vcp * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4Pair vrp; Vcp * const zi = &z[i];
			vrp.load_l(zi, m); vrp.l.backward4e(w0, w1); vrp.store_l(zi, m);
			vrp.load_h(zi, m); vrp.h.backward4e(w0, w1); vrp.store_h(zi, m);
		}
	}

	finline static void backward4o(const size_t m, Vcp * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t i = 0; i < m; ++i)
		{
			Vradix4Pair vrp; Vcp * const zi = &z[i];
			vrp.load_l(zi, m); vrp.l.backward4o(w0, w2); vrp.store_l(zi, m);
			vrp.load_h(zi, m); vrp.h.backward4o(w0, w2); vrp.store_h(zi, m);
		}
	}

	finline static void forward4e(const size_t mi, const size_t stepi, const size_t count, Vcp * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix4Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.forward4e(w0, w1); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.forward4e(w0, w1); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void forward4o(const size_t mi, const size_t stepi, const size_t count, Vcp * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix4Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.forward4o(w0, w2); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.forward4o(w0, w2); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void backward4e(const size_t mi, const size_t stepi, const size_t count, Vcp * const z, const Vc & w0, const Vc & w1)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix4Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.backward4e(w0, w1); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.backward4e(w0, w1); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void backward4o(const size_t mi, const size_t stepi, const size_t count, Vcp * const z, const Vc & w0, const Vc & w2)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix4Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.backward4o(w0, w2); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.backward4o(w0, w2); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void forward4_0(const size_t mi, const size_t stepi, const size_t count, Vcp * const z)
	{
		const Vc w0 =
#if defined(CYCLO)
			Vc::broadcast(cs2pi_1_24);
#else
			Vc::broadcast(cs2pi_1_16);
#endif
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix4Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.forward4_0(w0); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.forward4_0(w0); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void backward4_0(const size_t mi, const size_t stepi, const size_t count, Vcp * const z)
	{
		const Vc w0 =
#if defined(CYCLO)
			Vc::broadcast(cs2pi_1_24);
#else
			Vc::broadcast(cs2pi_1_16);
#endif
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix4Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.backward4_0(w0); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.backward4_0(w0); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void forward4e_4(Vcp * const z, const Vc & w0, const Vc & w1)	// VSIZE = 8
	{
		Vradix4Pair vrp;
		vrp.load_l(z); vrp.l.interleave(); vrp.l.forward4e(w0, w1); vrp.l.interleave(); vrp.store_l(z);
		vrp.load_h(z); vrp.h.interleave(); vrp.h.forward4e(w0, w1); vrp.h.interleave(); vrp.store_h(z);
	}

	finline static void forward4o_4(Vcp * const z, const Vc & w0, const Vc & w2)	// VSIZE = 8
	{
		Vradix4Pair vrp;
		vrp.load_l(z); vrp.l.interleave(); vrp.l.forward4o(w0, w2); vrp.l.interleave(); vrp.store_l(z);
		vrp.load_h(z); vrp.h.interleave(); vrp.h.forward4o(w0, w2); vrp.h.interleave(); vrp.store_h(z);
	}

	finline static void backward4e_4(Vcp * const z, const Vc & w0, const Vc & w1)	// VSIZE = 8
	{
		Vradix4Pair vrp;
		vrp.load_l(z); vrp.l.interleave(); vrp.l.backward4e(w0, w1); vrp.l.interleave(); vrp.store_l(z);
		vrp.load_h(z); vrp.h.interleave(); vrp.h.backward4e(w0, w1); vrp.h.interleave(); vrp.store_h(z);
	}

	finline static void backward4o_4(Vcp * const z, const Vc & w0, const Vc & w2)	// VSIZE = 8
	{
		Vradix4Pair vrp;
		vrp.load_l(z); vrp.l.interleave(); vrp.l.backward4o(w0, w2); vrp.l.interleave(); vrp.store_l(z);
		vrp.load_h(z); vrp.h.interleave(); vrp.h.backward4o(w0, w2); vrp.h.interleave(); vrp.store_h(z);
	}
};

template<size_t N>
class Vradix8Pair
{
	using Vc = Vcx<N>;
	using Vcp = VcxPair<N>;
	using Vr8 = Vradix8<N>;

private:
	Vr8 l, h;

public:
	finline void load_l(const Vcp * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i) l[i] = mem[i * step].l;
	}

	finline void load_h(const Vcp * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i) h[i] = mem[i * step].h;
	}

	finline void store_l(Vcp * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i * step].l = l[i];
	}

	finline void store_h(Vcp * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i * step].h = h[i];
	}

	finline static void forward8_0(const size_t mi, const size_t stepi, const size_t count, Vcp * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix8Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.forward8_0(); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.forward8_0(); vrp.store_h(zi, mi);
			}
		}
	}

	finline static void backward8_0(const size_t mi, const size_t stepi, const size_t count, Vcp * const z)
	{
		for (size_t j = 0; j < mi; j += stepi)
		{
			for (size_t i = 0; i < count; ++i)
			{
				Vradix8Pair vrp; Vcp * const zi = &z[j + i];
				vrp.load_l(zi, mi); vrp.l.backward8_0(); vrp.store_l(zi, mi);
				vrp.load_h(zi, mi); vrp.h.backward8_0(); vrp.store_h(zi, mi);
			}
		}
	}
};

}
