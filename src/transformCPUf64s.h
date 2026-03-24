/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include <gmp.h>
#include <omp.h>

#include "transform.h"
#include "f64vector_pair.h"

namespace transformCPU_namespace
{
template<size_t N>
class Vcx8s
{
	using Vc = Vcx<N>;
	using Vcp = VcxPair<N>;

public:
	static constexpr double split = 1 << 20, split_inv = 1.0 / split;

private:
	Vcp z[8];

private:
	Vcx8s() {}

public:
	finline explicit Vcx8s(const Vcp * const mem)
	{
		for (size_t i = 0; i < 8; ++i) z[i] = mem[i];
	}

	finline void store(Vcp * const mem) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i] = z[i];
	}

	finline explicit Vcx8s(const Vcp * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			z[i] = mem[(step * i_h + i_l) / N];
		}
	}

	finline void store(Vcp * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			mem[(step * i_h + i_l) / N] = z[i];
		}
	}

	finline void transpose_in()
	{
		Vc zl[8]; for (size_t i = 0; i < 8; ++i) zl[i] = z[i].l; Vc::transpose_in(zl); for (size_t i = 0; i < 8; ++i) z[i].l = zl[i];
		Vc zh[8]; for (size_t i = 0; i < 8; ++i) zh[i] = z[i].h; Vc::transpose_in(zh); for (size_t i = 0; i < 8; ++i) z[i].h = zh[i];
	}

	finline void transpose_out()
	{
		Vc zl[8]; for (size_t i = 0; i < 8; ++i) zl[i] = z[i].l; Vc::transpose_out(zl); for (size_t i = 0; i < 8; ++i) z[i].l = zl[i];
		Vc zh[8]; for (size_t i = 0; i < 8; ++i) zh[i] = z[i].h; Vc::transpose_out(zh); for (size_t i = 0; i < 8; ++i) z[i].h = zh[i];
	}

	finline void fwde(const Vc & w)
	{
		Vc::fwd2(z[0].l, z[2].l, w); Vc::fwd2(z[1].l, z[3].l, w);
		Vc::fwd2(z[0].h, z[2].h, w); Vc::fwd2(z[1].h, z[3].h, w);
	}

	finline void fwdo(const Vc & w)
	{
		Vc::fwd2i(z[4].l, z[6].l, w); Vc::fwd2a(z[5].l, z[7].l, w);
		Vc::fwd2i(z[4].h, z[6].h, w); Vc::fwd2a(z[5].h, z[7].h, w);
	}

	finline void bwde(const Vc & w)
	{
		Vc::bck2(z[0].l, z[2].l, w); Vc::bck2(z[1].l, z[3].l, w);
		Vc::bck2(z[0].h, z[2].h, w); Vc::bck2(z[1].h, z[3].h, w);
	}

	finline void bwdo(const Vc & w)
	{
		Vc::bck2a(z[4].l, z[6].l, w); Vc::bck2b(z[5].l, z[7].l, w);
		Vc::bck2a(z[4].h, z[6].h, w); Vc::bck2b(z[5].h, z[7].h, w);
	}

	finline void square4e(const Vc & w)
	{
		fwde(w);

		const Vc l0 = z[0].l, l1 = z[1].l, l2 = z[2].l, l3 = z[3].l;
		z[0].l = l0.sqr() + l1.sqr().mulW(w); z[1].l = (l0 + l0) * l1; z[2].l = l2.sqr() - l3.sqr().mulW(w); z[3].l = (l2 + l2) * l3;

		const Vc h0 = z[0].h, h1 = z[1].h, h2 = z[2].h, h3 = z[3].h;
		const Vc h2l0 = h0 + (l0 + l0), h2l1 = h1 + (l1 + l1), h2l2 = h2 + (l2 + l2), h2l3 = h3 + (l3 + l3);

		z[0].h = h0 * h2l0 + Vc(h1 * h2l1).mulW(w); z[1].h = h0 * h2l1 + h2l0 * h1;
		z[2].h = h2 * h2l2 - Vc(h3 * h2l3).mulW(w); z[3].h = h2 * h2l3 + h2l2 * h3;

		bwde(w);
	}

	finline void square4o(const Vc & w)
	{
		fwdo(w);

		const Vc l4 = z[4].l, l5 = z[5].l, l6 = z[6].l, l7 = z[7].l;
		z[4].l = l5.sqr().mulW(w).subi(l4.sqr()); z[5].l = (l4 + l4) * l5; z[6].l = l6.sqr().addi(l7.sqr().mulW(w)); z[7].l = (l6 + l6) * l7;

		const Vc h4 = z[4].h, h5 = z[5].h, h6 = z[6].h, h7 = z[7].h;
		const Vc h2l4 = h4 + (l4 + l4), h2l5 = h5 + (l5 + l5), h2l6 = h6 + (l6 + l6), h2l7 = h7 + (l7 + l7);

		z[4].h = Vc(h5 * h2l5).mulW(w).subi(h4 * h2l4); z[5].h = h4 * h2l5 + h2l4 * h5;
		z[6].h = Vc(h6 * h2l6).addi(Vc(h7 * h2l7).mulW(w)); z[7].h = h6 * h2l7 + h2l6 * h7;

		bwdo(w);
	}

	finline void mul4_forward(const Vc & w)
	{
		fwde(w);
		fwdo(w);
	}

	finline void mul4(const Vcx8s & rhs, const Vc & w)
	{
		fwde(w);

		const Vc l0 = z[0].l, l2 = z[2].l, l1 = z[1].l, l3 = z[3].l;
		const Vc lp0 = rhs.z[0].l, lp2 = rhs.z[2].l, lp1 = rhs.z[1].l, lp3 = rhs.z[3].l;
		z[0].l = l0 * lp0 + Vc(l1 * lp1).mulW(w); z[1].l = l0 * lp1 + lp0 * l1;
		z[2].l = l2 * lp2 - Vc(l3 * lp3).mulW(w); z[3].l = l2 * lp3 + lp2 * l3;

		const Vc h0 = z[0].h, h2 = z[2].h, h1 = z[1].h, h3 = z[3].h;
		const Vc hp0 = rhs.z[0].h, hp2 = rhs.z[2].h, hp1 = rhs.z[1].h, hp3 = rhs.z[3].h;
		const Vc lphp0 = lp0 + hp0, lphp2 = lp2 + hp2, lphp1 = lp1 + hp1, lphp3 = lp3 + hp3;

		z[0].h = h0 * lphp0 + l0 * hp0 + Vc(h1 * lphp1 + l1 * hp1).mulW(w);
		z[1].h = h0 * lphp1 + lphp0 * h1 + l0 * hp1 + hp0 * l1;
		z[2].h = h2 * lphp2 + l2 * hp2 - Vc(h3 * lphp3 + l3 * hp3).mulW(w);
		z[3].h = h2 * lphp3 + lphp2 * h3 + l2 * hp3 + hp2 * l3;

		bwde(w);

		fwdo(w);

		const Vc l4 = z[4].l, l6 = z[6].l, l5 = z[5].l, l7 = z[7].l;
		const Vc lp4 = rhs.z[4].l, lp6 = rhs.z[6].l, lp5 = rhs.z[5].l, lp7 = rhs.z[7].l;
		z[4].l = Vc(l5 * lp5).mulW(w).subi(l4 * lp4); z[5].l = l4 * lp5 + lp4 * l5;
		z[6].l = Vc(l6 * lp6).addi(Vc(l7 * lp7).mulW(w)); z[7].l = l6 * lp7 + lp6 * l7;

		const Vc h4 = z[4].h, h6 = z[6].h, h5 = z[5].h, h7 = z[7].h;
		const Vc hp4 = rhs.z[4].h, hp6 = rhs.z[6].h, hp5 = rhs.z[5].h, hp7 = rhs.z[7].h;
		const Vc lphp4 = lp4 + hp4, lphp6 = lp6 + hp6, lphp5 = lp5 + hp5, lphp7 = lp7 + hp7;

		z[4].h = Vc(h5 * lphp5 + l5 * hp5).mulW(w).subi(h4 * lphp4 + l4 * hp4);
		z[5].h = h4 * lphp5 + lphp4 * h5 + l4 * hp5 + hp4 * l5;
		z[6].h = Vc(h6 * lphp6 + l6 * hp6).addi(Vc(h7 * lphp7 + l7 * hp7).mulW(w));
		z[7].h = h6 * lphp7 + lphp6 * h7 + l6 * hp7 + hp6 * l7;

		bwdo(w);
	}

	finline void mul_carry(const Vcp & f_prev, Vcp & f_new, const double g, const double b, const double b_inv, const double t2_n)
	{
		Vc fl = f_prev.l, fh = f_prev.h;
		const Vd<N> vg = Vd<N>::broadcast(g), vb = Vd<N>::broadcast(b), vb_inv = Vd<N>::broadcast(b_inv);
		const Vd<N> vt2_n = Vd<N>::broadcast(t2_n), vt2_n_split_inv = Vd<N>::broadcast(t2_n * split_inv);
		const Vd<N> vsplit = Vd<N>::broadcast(split), vsplit_inv = Vd<N>::broadcast(split_inv);

		for (size_t i = 0; i < 8; ++i)
		{
			Vc & zli = z[i].l; Vc & zhi = z[i].h;
			const Vc ol = zli.mulS(vt2_n).round(), oh = zhi.mulS(vt2_n_split_inv).round();

			fl = fl.addmulS(ol, vg); fh = fh.addmulS(oh, vg);
			Vc fl_b = fl.mulS(vb_inv).round(), rl_b = fl.submulS(fl_b, vb);
			const Vc fh_b = fh.mulS(vb_inv).round(), rh_b = fh.submulS(fh_b, vb);
			fh = fh_b;

			rl_b = rl_b.addmulS(rh_b, vsplit);
			const Vc frl = rl_b.mulS(vb_inv).round(); rl_b = rl_b.submulS(frl, vb); fl_b += frl;
			fl = fl_b;

			const Vc h = rl_b.mulS(vsplit_inv).round().mulS(vsplit);
			zli = rl_b - h; zhi = h;
		}

		f_new.l = fl; f_new.h = fh;
	}

	finline void mul_carry(const Vcp & f_prev, Vcp & f_new, const double g, const double b, const double b_inv, const double t2_n, Vc & err)
	{
		Vc fl = f_prev.l, fh = f_prev.h;
		const Vd<N> vg = Vd<N>::broadcast(g), vb = Vd<N>::broadcast(b), vb_inv = Vd<N>::broadcast(b_inv);
		const Vd<N> vt2_n = Vd<N>::broadcast(t2_n), vt2_n_split_inv = Vd<N>::broadcast(t2_n * split_inv);
		const Vd<N> vsplit = Vd<N>::broadcast(split), vsplit_inv = Vd<N>::broadcast(split_inv);

		for (size_t i = 0; i < 8; ++i)
		{
			Vc & zli = z[i].l; Vc & zhi = z[i].h;
			const Vc ofl = zli.mulS(vt2_n), ofh = zhi.mulS(vt2_n_split_inv), ol = ofl.round(), oh = ofh.round();
			err.max(Vc(ofl - ol).abs()); err.max(Vc(ofh - oh).abs());

			fl = fl.addmulS(ol, vg); fh = fh.addmulS(oh, vg);
			Vc fl_b = fl.mulS(vb_inv).round(), rl_b = fl.submulS(fl_b, vb);
			const Vc fh_b = fh.mulS(vb_inv).round(), rh_b = fh.submulS(fh_b, vb);
			fh = fh_b;

			rl_b = rl_b.addmulS(rh_b, vsplit);
			const Vc frl = rl_b.mulS(vb_inv).round(); rl_b = rl_b.submulS(frl, vb); fl_b += frl;
			fl = fl_b;

			const Vc h = rl_b.mulS(vsplit_inv).round().mulS(vsplit);
			zli = rl_b - h; zhi = h;
		}

		f_new.l = fl; f_new.h = fh;
	}

	finline void carry(const Vc & fl_i, const Vc & fh_i, const double b, const double b_inv)
	{
		const Vd<N> vb = Vd<N>::broadcast(b), vb_inv = Vd<N>::broadcast(b_inv), v;
		const Vd<N> vsplit = Vd<N>::broadcast(split), vsplit_inv = Vd<N>::broadcast(split_inv);

		Vc f = fl_i.addmulS(fh_i, vsplit);

		for (size_t i = 0; i < 8 - 1; ++i)
		{
			Vc & zli = z[i].l; Vc & zhi = z[i].h;
			f += zli.round() + zhi.round();
			const Vc f_b = f.mulS(vb_inv).round();
			const Vc r_b = f.submulS(f_b, vb);
			f = f_b;
			const Vc h = r_b.mulS(vsplit_inv).round().mulS(vsplit);
			zli = r_b - h; zhi = h;
			if (f.isZero()) return;
		}

		Vc & zli = z[8 - 1].l; Vc & zhi = z[8 - 1].h;
		f += zli.round() + zhi.round();
		const Vc h = f.mulS(vsplit_inv).round().mulS(vsplit);
		zli = f - h; zhi = h;
	}
};

template<size_t N, size_t VSIZE>
class transformCPUf64s : public transform
{
	using Vc = Vcx<VSIZE>;
	using Vcp = VcxPair<VSIZE>;
	using Vr4 = Vradix4<VSIZE>;
	using Vr4p = Vradix4Pair<VSIZE>;
	using Vr8 = Vradix8<VSIZE>;
	using Vr8p = Vradix8Pair<VSIZE>;
	using Vc8s = Vcx8s<VSIZE>;

private:
	// Pass 1: n_io Complex (16 bytes), Pass 2/3: N / n_io Complex
	// n_io must be a power of 4, n_io >= 64, n >= 16 * n_io, n >= num_threads * n_io.
	static const size_t n_io = (N <= (1 << 11)) ? 64 : (N <= (1 << 13)) ? 256 : (N <= (1 << 17)) ? 1024 : 4096;
	static const size_t n_io_s = n_io / 4 / 2;
	static const size_t n_io_inv = N / n_io / VSIZE;
	static const size_t n_gap = (VSIZE <= 4) ? 64 : 16 * VSIZE;	// Cache line size is 64 bytes. Alignment is needed if VSIZE > 4.

	finline static constexpr size_t index(const size_t k) { const size_t j = k / n_io, i = k % n_io; return j * (n_io + n_gap / sizeof(Complex)) + i; }

	static const size_t wSize = N / 8 * sizeof(Complex);
	static const size_t wsSize = N / 8 * sizeof(Complex);
	static const size_t zSize = 2 * index(N) * sizeof(Complex);
	static const size_t fcSize = 2 * 64 * n_io_inv * sizeof(Vc);	// num_threads <= 64

	static const size_t wOffset = 0;
	static const size_t wsOffset = wOffset + wSize;
	static const size_t zOffset = wsOffset + wsSize;
	static const size_t fcOffset = zOffset + zSize;
	static const size_t zpOffset = fcOffset + fcSize;
	static const size_t zrOffset = zpOffset + zSize;

	const size_t _num_threads;
	const double _b, _b_inv;
	const size_t _mem_size, _cache_size;
	bool _checkError;
	double _error;
	char * const _mem;
	Vcp * const _z_copy;

private:
	finline static void forward_out(Vcp * const z, const Complex * const w122i)
	{
		static const size_t stepi = index(n_io) / VSIZE;

		size_t s = (N / 4) / n_io / 2; for (; s >= 4 * 2; s /= 4);

		if (s == 4) Vr8p::forward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, z);
		else        Vr4p::forward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, z);

		for (size_t mi = index((s == 4) ? N / 32 : N / 16) / VSIZE; mi >= stepi; mi /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
				Vr4p::forward4e(mi, stepi, 2 * 4 / VSIZE, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				Vr4p::forward4o(mi, stepi, 2 * 4 / VSIZE, &z[k + 1 * 4 * mi], w0, w2);
			}
		}
	}

	finline static void backward_out(Vcp * const z, const Complex * const w122i)
	{
		static const size_t stepi = index(n_io) / VSIZE;

		size_t s = (N / 4) / n_io / 2;
		for (size_t mi = stepi; s >= 2; mi *= 4, s /= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
				Vr4p::backward4e(mi, stepi, 2 * 4 / VSIZE, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				Vr4p::backward4o(mi, stepi, 2 * 4 / VSIZE, &z[k + 1 * 4 * mi], w0, w2);
			}
		}

		if (s == 1) Vr8p::backward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, z);
		else        Vr4p::backward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, z);
	}

	void pass1(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vcp * const z = (Vcp *)&_mem[zOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vcp * const zl = &z[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4p::forward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4p::forward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vcp * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4p::forward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4p::forward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vcp * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4p::forward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4p::forward4o_4(&zj[2], w0, w2);
				}
			}

			// square
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vcp * const zj = &zl[8 * j];
				Vc8s z8(zj);
				z8.transpose_in();
				z8.square4e(wsl[j]);
				z8.store(zj);
			}
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vcp * const zj = &zl[8 * j];
				Vc8s z8(zj);
				z8.square4o(wsl[j]);
				z8.transpose_out();
				z8.store(zj);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vcp * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4p::backward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4p::backward4o_4(&zj[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vcp * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4p::backward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4p::backward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4p::backward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4p::backward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}
		}
	}

	void pass1multiplicand(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vcp * const zp = (Vcp *)&_mem[zpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vcp * const zpl = &zp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4p::forward4e(n_io / 4 / VSIZE, zpl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4p::forward4o(n_io / 4 / VSIZE, zpl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vcp * const zpj = &zpl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4p::forward4e(m, &zpj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4p::forward4o(m, &zpj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vcp * const zpj = &zpl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4p::forward4e_4(&zpj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4p::forward4o_4(&zpj[2], w0, w2);
				}
			}

			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vcp * const zpj = &zpl[8 * j];
				Vc8s zp8(zpj);
				zp8.transpose_in();
				zp8.mul4_forward(wsl[j]);
				zp8.store(zpj);
			}
		}
	}

	void pass1mul(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vcp * const z = (Vcp *)&_mem[zOffset];
		const Vcp * const zp = (Vcp *)&_mem[zpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vcp * const zl = &z[index(n_io * l) / VSIZE];
			const Vcp * const zpl = &zp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4p::forward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4p::forward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vcp * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4p::forward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4p::forward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vcp * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4p::forward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4p::forward4o_4(&zj[2], w0, w2);
				}
			}

			// mul
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vcp * const zj = &zl[8 * j];
				const Vcp * const zpj = &zpl[8 * j];
				Vc8s z8(zj); z8.transpose_in();
				Vc8s zp8(zpj); z8.mul4(zp8, wsl[j]);
				z8.transpose_out();
				z8.store(zj);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vcp * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4p::backward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4p::backward4o_4(&zj[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vcp * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4p::backward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4p::backward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4p::backward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4p::backward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}
		}
	}

	double pass2_0(const size_t thread_id, const double g)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		Vcp * const z = (Vcp *)&_mem[zOffset];
		Vcp * const fc = (Vcp *)&_mem[fcOffset]; Vcp * const f = &fc[thread_id * n_io_inv];
		const double b = _b, b_inv = _b_inv;
		const bool checkError = _checkError;

		Vc err = Vc(0.0);

		const size_t num_threads = _num_threads;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			Vcp * const zl = &z[2 * 4 / VSIZE * lh];

			backward_out(zl, w122i);

			for (size_t j = 0; j < n_io_inv; ++j)
			{
				Vcp * const zj = &zl[index(n_io) * j];
				Vc8s z8(zj, index(n_io));
				z8.transpose_in();

				Vcp zero; zero.l = zero.h = Vc(0.0);
				const Vcp f_prev = (lh != l_min) ? f[j] : zero;
				if (!checkError) z8.mul_carry(f_prev, f[j], g, b, b_inv, 2.0 / N);
				else             z8.mul_carry(f_prev, f[j], g, b, b_inv, 2.0 / N, err);

				if (lh != l_min) z8.transpose_out();
				z8.store(zj, index(n_io));	// transposed if lh = l_min
			}

			if (lh != l_min) forward_out(zl, w122i);
		}

		return err.max();
	}

	void pass2_1(const size_t thread_id)
	{
		const size_t num_threads = _num_threads;
		const size_t thread_id_prev = ((thread_id != 0) ? thread_id : num_threads) - 1;
		const size_t lh = thread_id * n_io_s / num_threads;	// l_min of pass2

		Vcp * const z = (Vcp *)&_mem[zOffset]; Vcp * const zl = &z[2 * 4 / VSIZE * lh];
		const Vcp * const fc = (Vcp *)&_mem[fcOffset]; const Vcp * const f = &fc[thread_id_prev * n_io_inv];

		const double b = _b, b_inv = _b_inv;

		for (size_t j = 0; j < n_io_inv; ++j)
		{
			Vcp * const zj = &zl[index(n_io) * j];
			Vc8s z8(zj, index(n_io));	// transposed

			Vc fl_prev = f[j].l, fh_prev = f[j].h;
			if (thread_id == 0)
			{
				fl_prev.shift(f[((j == 0) ? n_io_inv : j) - 1].l, j == 0);
				fh_prev.shift(f[((j == 0) ? n_io_inv : j) - 1].h, j == 0);
			}
			z8.carry(fl_prev, fh_prev, b, b_inv);

			z8.transpose_out();
			z8.store(zj, index(n_io));
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		forward_out(zl, w122i);
	}

public:
	transformCPUf64s(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError)
		: transform(N, n, b, ((VSIZE == 2) ? EKind::SBDTvec2 : ((VSIZE == 4) ? EKind::SBDTvec4 : EKind::SBDTvec8))),
		_num_threads(num_threads),
		_b(b), _b_inv(1.0 / b),
		_mem_size(wSize + wsSize + zSize + fcSize + zSize + (num_regs - 1) * zSize + 2 * 1024 * 1024),
		_cache_size(wSize + wsSize + zSize + fcSize), _checkError(checkError), _error(0),
		_mem((char *)alignNew(_mem_size, 2 * 1024 * 1024)), _z_copy((Vcp *)alignNew(zSize, 1024))
	{
		Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t s = N / 16; s >= 4; s /= 4)
		{
			Complex * const w_s = &w122i[2 * s / 4];
			for (size_t j = 0; j < s / 2; ++j)
			{
				const size_t r = bitRev(j, 2 * s) + 1;
				w_s[3 * j + 0] = Complex::exp2iPi(r, 8 * s);
				w_s[3 * j + 1] = Complex::exp2iPi(r, 2 * 8 * s);
				w_s[3 * j + 2] = Complex::exp2iPi(r + 2 * s, 2 * 8 * s);
			}
		}

		Vc * const ws = (Vc *)&_mem[wsOffset];
		for (size_t j = 0; j < N / 8 / VSIZE; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				ws[j].set(i, Complex::exp2iPi(bitRev(VSIZE * j + i, 2 * (N / 4)) + 1, 8 * (N / 4)));
			}
		}
	}

	virtual ~transformCPUf64s()
	{
		alignDelete((void *)_mem);
		alignDelete((void *)_z_copy);
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return _cache_size; }

protected:
	void getZi(int32_t * const zi) const override
	{
		const Vcp * const z = (Vcp *)&_mem[zOffset];

		Vcp * const z_copy = _z_copy;
		for (size_t k = 0; k < index(N) / VSIZE; ++k) z_copy[k] = z[k];

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			backward_out(&z_copy[2 * 4 / VSIZE * lh], w122i);
		}

		const double n_io_N = static_cast<double>(n_io) / N;

		for (size_t k = 0; k < N; k += VSIZE)
		{
			const Vcp vcp = z_copy[index(k) / VSIZE];
			const Vc vc = vcp.l + vcp.h;
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const Complex zc = vc[i];
				zi[k + i + 0 * N] = std::lround(zc.real * n_io_N);
				zi[k + i + 1 * N] = std::lround(zc.imag * n_io_N);
			}
		}
	}

	void setZi(const int32_t * const zi) override
	{
		Vcp * const z = (Vcp *)&_mem[zOffset];

		const Vd<VSIZE> vsplit = Vd<VSIZE>::broadcast(Vc8s::split), vsplit_inv = Vd<VSIZE>::broadcast(Vc8s::split_inv);

		for (size_t k = 0; k < N; k += VSIZE)
		{
			Vc vc;
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const Complex zc(static_cast<double>(zi[k + i + 0 * N]), static_cast<double>(zi[k + i + 1 * N]));
				vc.set(i, zc);
			}
			const Vc h = vc.mulS(vsplit_inv).round().mulS(vsplit);
			z[index(k) / VSIZE].l = vc - h;
			z[index(k) / VSIZE].h = h;
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&z[2 * 4 / VSIZE * lh], w122i);
		}
	}

public:
	bool readContext(file & cFile, const size_t num_regs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(getKind())) return false;

		if (!cFile.read(reinterpret_cast<char *>(&_error), sizeof(_error))) return false;

		Vcp * const z = (Vcp *)&_mem[zOffset];
		if (!cFile.read(reinterpret_cast<char *>(z), zSize)) return false;
		if (num_regs > 1)
		{
			Vcp * const zr = (Vcp *)&_mem[zrOffset];
			if (!cFile.read(reinterpret_cast<char *>(zr), (num_regs - 1) * zSize)) return false;
		}

		return true;
	}

	void saveContext(file & cFile, const size_t num_regs) const override
	{
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		if (!cFile.write(reinterpret_cast<const char *>(&_error), sizeof(_error))) return;

		const Vcp * const z = (Vcp *)&_mem[zOffset];
		if (!cFile.write(reinterpret_cast<const char *>(z), zSize)) return;
		if (num_regs > 1)
		{
			const Vcp * const zr = (Vcp *)&_mem[zrOffset];
			if (!cFile.write(reinterpret_cast<const char *>(zr), (num_regs - 1) * zSize)) return;
		}
	}

	void set(const uint32_t a) override
	{
		Vcp * const z = (Vcp *)&_mem[zOffset];
		z[0].l = Vc(a); z[0].h = Vc(0.0);
		for (size_t k = 1; k < index(N) / VSIZE; ++k) { z[k].l = z[k].h = Vc(0.0); }

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&z[2 * 4 / VSIZE * lh], w122i);
		}
	}

	void squareDup(const bool dup) override
	{
		squareMul(dup ? 2 : 1);
	}

	void squareMul(const int32_t a) override
	{
		const size_t num_threads = _num_threads;
		double e[64];
		const double g = static_cast<double>(a);

		if (num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				pass1(thread_id);
#pragma omp barrier
				e[thread_id] = pass2_0(thread_id, g);
#pragma omp barrier
				pass2_1(thread_id);
			}
		}
		else
		{
			pass1(0);
			e[0] = pass2_0(0, g);
			pass2_1(0);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		_error = std::max(_error, err);
	}

	void initMultiplicand(const size_t src) override
	{
		const Vcp * const z_src = (Vcp *)&_mem[(src == 0) ? zOffset : zrOffset + (src - 1) * zSize];
		Vcp * const zp = (Vcp *)&_mem[zpOffset];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zp[k] = z_src[k];

		if (_num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());
				pass1multiplicand(thread_id);
			}
		}
		else
		{
			pass1multiplicand(0);
		}
	}

	void mul() override
	{
		const size_t num_threads = _num_threads;
		double e[64];

		if (num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				pass1mul(thread_id);
#pragma omp barrier
				e[thread_id] = pass2_0(thread_id, 1.0);
#pragma omp barrier
				pass2_1(thread_id);
			}
		}
		else
		{
			pass1mul(0);
			e[0] = pass2_0(0, 1.0);
			pass2_1(0);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		_error = std::max(_error, err);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const Vcp * const z_src = (Vcp *)&_mem[(src == 0) ? zOffset : zrOffset + (src - 1) * zSize];
		Vcp * const z_dst = (Vcp *)&_mem[(dst == 0) ? zOffset : zrOffset + (dst - 1) * zSize];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) z_dst[k] = z_src[k];
	}

	double getError() const override { return _error; }
};

template<size_t VSIZE>
inline transform * create_transformCPUf64s(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError)
{
	transform * pTransform = nullptr;
#if defined(DTRANSFORM) || defined(IBDTRANSFORM)
	(void)b; (void)n; (void)num_threads; (void)num_regs; (void)checkError;
#else
	if      (n == 12) pTransform = new transformCPUf64s<(1 << 11), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 13) pTransform = new transformCPUf64s<(1 << 12), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 14) pTransform = new transformCPUf64s<(1 << 13), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 15) pTransform = new transformCPUf64s<(1 << 14), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 16) pTransform = new transformCPUf64s<(1 << 15), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 17) pTransform = new transformCPUf64s<(1 << 16), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 18) pTransform = new transformCPUf64s<(1 << 17), VSIZE>(b, n, num_threads, num_regs, checkError);
#endif
#if defined(SBDTRANSFORM)
	if      (n == 19) pTransform = new transformCPUf64s<(1 << 18), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 20) pTransform = new transformCPUf64s<(1 << 19), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 21) pTransform = new transformCPUf64s<(1 << 20), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 22) pTransform = new transformCPUf64s<(1 << 21), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 23) pTransform = new transformCPUf64s<(1 << 22), VSIZE>(b, n, num_threads, num_regs, checkError);
#endif

	if (pTransform == nullptr) throw std::runtime_error("exponent is not supported");

	return pTransform;
}

}