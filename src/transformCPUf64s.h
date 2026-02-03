/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>
#include <cmath>

#include <omp.h>

#include "transform.h"
#include "f64vector.h"

namespace transformCPU_namespace
{
	static constexpr double split = 1 << 20, split_inv = 1.0 / split;

template<size_t N>
class Vcx8s
{
	using Vc = Vcx<N>;

private:
	Vc zl[8], zh[8];

private:
	Vcx8s() {}

public:
	finline explicit Vcx8s(const Vc * const mem_l, const Vc * const mem_h)
	{
		for (size_t i = 0; i < 8; ++i) zl[i] = mem_l[i];
		for (size_t i = 0; i < 8; ++i) zh[i] = mem_h[i];
	}

	finline void store(Vc * const mem_l, Vc * const mem_h) const
	{
		for (size_t i = 0; i < 8; ++i) mem_l[i] = zl[i];
		for (size_t i = 0; i < 8; ++i) mem_h[i] = zh[i];
	}

	finline explicit Vcx8s(const Vc * const mem_l, const Vc * const mem_h, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			zl[i] = mem_l[(step * i_h + i_l) / N];
			zh[i] = mem_h[(step * i_h + i_l) / N];
		}
	}

	finline void store(Vc * const mem_l, Vc * const mem_h, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			mem_l[(step * i_h + i_l) / N] = zl[i];
			mem_h[(step * i_h + i_l) / N] = zh[i];
		}
	}

	finline void transpose_in() { Vc::transpose_in(zl); Vc::transpose_in(zh); }
	finline void transpose_out() { Vc::transpose_out(zl); Vc::transpose_out(zh); }

	finline void fwde(const Vc & w)
	{
		const Vc l0 = zl[0], l2 = zl[2].mulW(w), l1 = zl[1], l3 = zl[3].mulW(w);
		zl[0] = l0 + l2; zl[2] = l0 - l2; zl[1] = l1 + l3; zl[3] = l1 - l3;

		const Vc h0 = zh[0], h2 = zh[2].mulW(w), h1 = zh[1], h3 = zh[3].mulW(w);
		zh[0] = h0 + h2; zh[2] = h0 - h2; zh[1] = h1 + h3; zh[3] = h1 - h3;
	}

	finline void fwdo(const Vc & w)
	{
		const Vc l4 = zl[4], l6 = zl[6].mulW(w), l5 = zl[5], l7 = zl[7].mulW(w);
		zl[4] = l4.addi(l6); zl[6] = l4.subi(l6); zl[5] = l5.addi(l7); zl[7] = l7.addi(l5);

		const Vc h4 = zh[4], h6 = zh[6].mulW(w), h5 = zh[5], h7 = zh[7].mulW(w);
		zh[4] = h4.addi(h6); zh[6] = h4.subi(h6); zh[5] = h5.addi(h7); zh[7] = h7.addi(h5);
	}

	finline void bwde(const Vc & w)
	{
		const Vc l0 = zl[0], l2 = zl[2], l1 = zl[1], l3 = zl[3];
		zl[0] = l0 + l2; zl[2] = Vc(l0 - l2).mulWconj(w); zl[1] = l1 + l3; zl[3] = Vc(l1 - l3).mulWconj(w);

		const Vc h0 = zh[0], h2 = zh[2], h1 = zh[1], h3 = zh[3];
		zh[0] = h0 + h2; zh[2] = Vc(h0 - h2).mulWconj(w); zh[1] = h1 + h3; zh[3] = Vc(h1 - h3).mulWconj(w);
	}

	finline void bwdo(const Vc & w)
	{
		const Vc l4 = zl[4], l6 = zl[6], l5 = zl[5], l7 = zl[7];
		zl[4] = l6.addi(l4); zl[6] = l4.addi(l6).mulWconj(w); zl[5] = l5.subi(l7); zl[7] = l7.subi(l5).mulWconj(w);

		const Vc h4 = zh[4], h6 = zh[6], h5 = zh[5], h7 = zh[7];
		zh[4] = h6.addi(h4); zh[6] = h4.addi(h6).mulWconj(w); zh[5] = h5.subi(h7); zh[7] = h7.subi(h5).mulWconj(w);
	}

	finline void square4e(const Vc & w)
	{
		fwde(w);

		const Vc l0 = zl[0], l1 = zl[1], l2 = zl[2], l3 = zl[3];
		zl[0] = l0.sqr() + l1.sqr().mulW(w); zl[1] = (l0 + l0) * l1; zl[2] = l2.sqr() - l3.sqr().mulW(w); zl[3] = (l2 + l2) * l3;

		const Vc h0 = zh[0], h1 = zh[1], h2 = zh[2], h3 = zh[3];
		const Vc h2l0 = h0 + (l0 + l0), h2l1 = h1 + (l1 + l1), h2l2 = h2 + (l2 + l2), h2l3 = h3 + (l3 + l3);

		zh[0] = h0 * h2l0 + Vc(h1 * h2l1).mulW(w); zh[1] = h0 * h2l1 + h2l0 * h1;
		zh[2] = h2 * h2l2 - Vc(h3 * h2l3).mulW(w); zh[3] = h2 * h2l3 + h2l2 * h3;

		bwde(w);
	}

	finline void square4o(const Vc & w)
	{
		fwdo(w);

		const Vc l4 = zl[4], l5 = zl[5], l6 = zl[6], l7 = zl[7];
		zl[4] = l5.sqr().mulW(w).subi(l4.sqr()); zl[5] = (l4 + l4) * l5; zl[6] = l6.sqr().addi(l7.sqr().mulW(w)); zl[7] = (l6 + l6) * l7;

		const Vc h4 = zh[4], h5 = zh[5], h6 = zh[6], h7 = zh[7];
		const Vc h2l4 = h4 + (l4 + l4), h2l5 = h5 + (l5 + l5), h2l6 = h6 + (l6 + l6), h2l7 = h7 + (l7 + l7);

		zh[4] = Vc(h5 * h2l5).mulW(w).subi(h4 * h2l4); zh[5] = h4 * h2l5 + h2l4 * h5;
		zh[6] = Vc(h6 * h2l6).addi(Vc(h7 * h2l7).mulW(w)); zh[7] = h6 * h2l7 + h2l6 * h7;

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

		const Vc l0 = zl[0], l2 = zl[2], l1 = zl[1], l3 = zl[3];
		const Vc lp0 = rhs.zl[0], lp2 = rhs.zl[2], lp1 = rhs.zl[1], lp3 = rhs.zl[3];
		zl[0] = l0 * lp0 + Vc(l1 * lp1).mulW(w); zl[1] = l0 * lp1 + lp0 * l1;
		zl[2] = l2 * lp2 - Vc(l3 * lp3).mulW(w); zl[3] = l2 * lp3 + lp2 * l3;

		const Vc h0 = zh[0], h2 = zh[2], h1 = zh[1], h3 = zh[3];
		const Vc hp0 = rhs.zh[0], hp2 = rhs.zh[2], hp1 = rhs.zh[1], hp3 = rhs.zh[3];
		const Vc lphp0 = lp0 + hp0, lphp2 = lp2 + hp2, lphp1 = lp1 + hp1, lphp3 = lp3 + hp3;

		zh[0] = h0 * lphp0 + l0 * hp0 + Vc(h1 * lphp1 + l1 * hp1).mulW(w);
		zh[1] = h0 * lphp1 + lphp0 * h1 + l0 * hp1 + hp0 * l1;
		zh[2] = h2 * lphp2 + l2 * hp2 - Vc(h3 * lphp3 + l3 * hp3).mulW(w);
		zh[3] = h2 * lphp3 + lphp2 * h3 + l2 * hp3 + hp2 * l3;

		bwde(w);

		fwdo(w);

		const Vc l4 = zl[4], l6 = zl[6], l5 = zl[5], l7 = zl[7];
		const Vc lp4 = rhs.zl[4], lp6 = rhs.zl[6], lp5 = rhs.zl[5], lp7 = rhs.zl[7];
		zl[4] = Vc(l5 * lp5).mulW(w).subi(l4 * lp4); zl[5] = l4 * lp5 + lp4 * l5;
		zl[6] = Vc(l6 * lp6).addi(Vc(l7 * lp7).mulW(w)); zl[7] = l6 * lp7 + lp6 * l7;

		const Vc h4 = zh[4], h6 = zh[6], h5 = zh[5], h7 = zh[7];
		const Vc hp4 = rhs.zh[4], hp6 = rhs.zh[6], hp5 = rhs.zh[5], hp7 = rhs.zh[7];
		const Vc lphp4 = lp4 + hp4, lphp6 = lp6 + hp6, lphp5 = lp5 + hp5, lphp7 = lp7 + hp7;

		zh[4] = Vc(h5 * lphp5 + l5 * hp5).mulW(w).subi(h4 * lphp4 + l4 * hp4);
		zh[5] = h4 * lphp5 + lphp4 * h5 + l4 * hp5 + hp4 * l5;
		zh[6] = Vc(h6 * lphp6 + l6 * hp6).addi(Vc(h7 * lphp7 + l7 * hp7).mulW(w));
		zh[7] = h6 * lphp7 + lphp6 * h7 + l6 * hp7 + hp6 * l7;

		bwdo(w);
	}

	finline void mul_carry(const Vc & fl_prev, const Vc & fh_prev, Vc & fl_new, Vc & fh_new, const double g, const double b, const double b_inv, const double t2_n)
	{
		Vc fl = fl_prev, fh = fh_prev;

		for (size_t i = 0; i < 8; ++i)
		{
			Vc & zli = zl[i]; Vc & zhi = zh[i];
			const Vc ol = Vc(zli * t2_n).round(), oh = Vc(zhi * (t2_n * split_inv)).round();

			fl += ol * g; fh += oh * g;
			Vc fl_b = Vc(fl * b_inv).round(), rl_b = fl - fl_b * b;
			const Vc fh_b = Vc(fh * b_inv).round(), rh_b = fh - fh_b * b;
			fh = fh_b;

			rl_b += rh_b * split;
			const Vc frl = Vc(rl_b * b_inv).round(); rl_b -= frl * b; fl_b += frl;
			fl = fl_b;

			const Vc h = Vc(rl_b * split_inv).round() * split;
			zli = rl_b - h; zhi = h;
		}

		fl_new = fl; fh_new = fh;
	}

	finline void mul_carry(const Vc & fl_prev, const Vc & fh_prev, Vc & fl_new, Vc & fh_new, const double g, const double b, const double b_inv, const double t2_n, Vc & err)
	{
		Vc fl = fl_prev, fh = fh_prev;

		for (size_t i = 0; i < 8; ++i)
		{
			Vc & zli = zl[i]; Vc & zhi = zh[i];
			const Vc ofl = zli * t2_n, ofh = zhi * (t2_n * split_inv), ol = ofl.round(), oh = ofh.round();
			err.max(Vc(ofl - ol).abs()); err.max(Vc(ofh - oh).abs());

			fl += ol * g; fh += oh * g;
			Vc fl_b = Vc(fl * b_inv).round(), rl_b = fl - fl_b * b;
			const Vc fh_b = Vc(fh * b_inv).round(), rh_b = fh - fh_b * b;
			fh = fh_b;

			rl_b += rh_b * split;
			const Vc frl = Vc(rl_b * b_inv).round(); rl_b -= frl * b; fl_b += frl;
			fl = fl_b;

			const Vc h = Vc(rl_b * split_inv).round() * split;
			zli = rl_b - h; zhi = h;
		}

		fl_new = fl; fh_new = fh;
	}

	finline void carry(const Vc & fl_i, const Vc & fh_i, const double b, const double b_inv)
	{
		Vc f = fl_i + fh_i * split;

		for (size_t i = 0; i < 8 - 1; ++i)
		{
			Vc & zli = zl[i]; Vc & zhi = zh[i];
			f += zli.round() + zhi.round();
			const Vc f_b = Vc(f * b_inv).round();
			const Vc r_b = f - f_b * b;
			f = f_b;
			const Vc h = Vc(r_b * split_inv).round() * split;
			zli = r_b - h; zhi = h;
			if (f.isZero()) return;
		}

		Vc & zli = zl[8 - 1]; Vc & zhi = zh[8 - 1];
		f += zli.round() + zhi.round();
		const Vc h = Vc(f * split_inv).round() * split;
		zli = f - h; zhi = h;
	}
};

template<size_t N, size_t VSIZE>
class transformCPUf64s : public transform
{
	using Vc = Vcx<VSIZE>;
	using Vr4 = Vradix4<VSIZE>;
	using Vr8 = Vradix8<VSIZE>;
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
	static const size_t zSize = index(N) * sizeof(Complex) + 1024;	// L1 line size is 4K
	static const size_t fcSize = 64 * n_io_inv * sizeof(Vc);		// num_threads <= 64

	static const size_t wOffset = 0;
	static const size_t wsOffset = wOffset + wSize;
	static const size_t zlOffset = wsOffset + wsSize;
	static const size_t zhOffset = zlOffset + zSize;
	static const size_t fclOffset = zhOffset + zSize;
	static const size_t fchOffset = fclOffset + fcSize;
	static const size_t zlpOffset = fchOffset + fcSize;
	static const size_t zhpOffset = zlpOffset + zSize;
	static const size_t zrOffset = zhpOffset + zSize;

	const size_t _num_threads;
	const double _b, _b_inv;
	const size_t _mem_size, _cache_size;
	bool _checkError;
	double _error;
	char * const _mem;
	char * const _mem_copy;

private:
	finline static void forward_out(Vc * const zl, Vc * const zh, const Complex * const w122i)
	{
		static const size_t stepi = index(n_io) / VSIZE;

		size_t s = (N / 4) / n_io / 2; for (; s >= 4 * 2; s /= 4);

		if (s == 4) { Vr8::forward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, zl); Vr8::forward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, zh); }
		else        { Vr4::forward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, zl); Vr4::forward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, zh); }

		for (size_t mi = index((s == 4) ? N / 32 : N / 16) / VSIZE; mi >= stepi; mi /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
				Vr4::forward4e(mi, stepi, 2 * 4 / VSIZE, &zl[k + 0 * 4 * mi], w0, w1); Vr4::forward4e(mi, stepi, 2 * 4 / VSIZE, &zh[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				Vr4::forward4o(mi, stepi, 2 * 4 / VSIZE, &zl[k + 1 * 4 * mi], w0, w2); Vr4::forward4o(mi, stepi, 2 * 4 / VSIZE, &zh[k + 1 * 4 * mi], w0, w2);
			}
		}
	}

	finline static void backward_out(Vc * const zl, Vc * const zh, const Complex * const w122i)
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
				Vr4::backward4e(mi, stepi, 2 * 4 / VSIZE, &zl[k + 0 * 4 * mi], w0, w1); Vr4::backward4e(mi, stepi, 2 * 4 / VSIZE, &zh[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				Vr4::backward4o(mi, stepi, 2 * 4 / VSIZE, &zl[k + 1 * 4 * mi], w0, w2); Vr4::backward4o(mi, stepi, 2 * 4 / VSIZE, &zh[k + 1 * 4 * mi], w0, w2);
			}
		}

		if (s == 1) { Vr8::backward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, zl); Vr8::backward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, zh); }
		else        { Vr4::backward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, zl); Vr4::backward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, zh); }
	}

	void pass1(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const zl = (Vc *)&_mem[zlOffset];
		Vc * const zh = (Vc *)&_mem[zhOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zl_l = &zl[index(n_io * l) / VSIZE];
			Vc * const zh_l = &zh[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::forward4e(n_io / 4 / VSIZE, zl_l, w0, w1); Vr4::forward4e(n_io / 4 / VSIZE, zh_l, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::forward4o(n_io / 4 / VSIZE, zl_l, w0, w2); Vr4::forward4o(n_io / 4 / VSIZE, zh_l, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zl_j = &zl_l[8 * m * j];
					Vc * const zh_j = &zh_l[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::forward4e(m, &zl_j[0 * 4 * m], w0, w1); Vr4::forward4e(m, &zh_j[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::forward4o(m, &zl_j[1 * 4 * m], w0, w2); Vr4::forward4o(m, &zh_j[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zl_j = &zl_l[32 / VSIZE * j];
					Vc * const zh_j = &zh_l[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::forward4e_4(&zl_j[0], w0, w1); Vr4::forward4e_4(&zh_j[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::forward4o_4(&zl_j[2], w0, w2); Vr4::forward4o_4(&zh_j[2], w0, w2);
				}
			}

			// square
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zl_j = &zl_l[8 * j];
				Vc * const zh_j = &zh_l[8 * j];
				Vc8s z8(zl_j, zh_j);
				z8.transpose_in();
				z8.square4e(wsl[j]);
				z8.store(zl_j, zh_j);
			}
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zl_j = &zl_l[8 * j];
				Vc * const zh_j = &zh_l[8 * j];
				Vc8s z8(zl_j, zh_j);
				z8.square4o(wsl[j]);
				z8.transpose_out();
				z8.store(zl_j, zh_j);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zl_j = &zl_l[32 / VSIZE * j];
					Vc * const zh_j = &zh_l[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::backward4e_4(&zl_j[0], w0, w1); Vr4::backward4e_4(&zh_j[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::backward4o_4(&zl_j[2], w0, w2); Vr4::backward4o_4(&zh_j[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zl_j = &zl_l[8 * m * j];
					Vc * const zh_j = &zh_l[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::backward4e(m, &zl_j[0 * 4 * m], w0, w1); Vr4::backward4e(m, &zh_j[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::backward4o(m, &zl_j[1 * 4 * m], w0, w2); Vr4::backward4o(m, &zh_j[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::backward4e(n_io / 4 / VSIZE, zl_l, w0, w1); Vr4::backward4e(n_io / 4 / VSIZE, zh_l, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::backward4o(n_io / 4 / VSIZE, zl_l, w0, w2); Vr4::backward4o(n_io / 4 / VSIZE, zh_l, w0, w2); }
			}
		}
	}

	void pass1multiplicand(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const zlp = (Vc *)&_mem[zlpOffset];
		Vc * const zhp = (Vc *)&_mem[zhpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zlp_l = &zlp[index(n_io * l) / VSIZE];
			Vc * const zhp_l = &zhp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::forward4e(n_io / 4 / VSIZE, zlp_l, w0, w1); Vr4::forward4e(n_io / 4 / VSIZE, zhp_l, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::forward4o(n_io / 4 / VSIZE, zlp_l, w0, w2); Vr4::forward4o(n_io / 4 / VSIZE, zhp_l, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zlp_j = &zlp_l[8 * m * j];
					Vc * const zhp_j = &zhp_l[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::forward4e(m, &zlp_j[0 * 4 * m], w0, w1); Vr4::forward4e(m, &zhp_j[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::forward4o(m, &zlp_j[1 * 4 * m], w0, w2); Vr4::forward4o(m, &zhp_j[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zlp_j = &zlp_l[32 / VSIZE * j];
					Vc * const zhp_j = &zhp_l[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::forward4e_4(&zlp_j[0], w0, w1); Vr4::forward4e_4(&zhp_j[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::forward4o_4(&zlp_j[2], w0, w2); Vr4::forward4o_4(&zhp_j[2], w0, w2);
				}
			}

			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zlp_j = &zlp_l[8 * j];
				Vc * const zhp_j = &zhp_l[8 * j];
				Vc8s zp8(zlp_j, zhp_j);
				zp8.transpose_in();
				zp8.mul4_forward(wsl[j]);
				zp8.store(zlp_j, zhp_j);
			}
		}
	}

	void pass1mul(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const zl = (Vc *)&_mem[zlOffset];
		Vc * const zh = (Vc *)&_mem[zhOffset];
		const Vc * const zlp = (Vc *)&_mem[zlpOffset];
		const Vc * const zhp = (Vc *)&_mem[zhpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zl_l = &zl[index(n_io * l) / VSIZE];
			Vc * const zh_l = &zh[index(n_io * l) / VSIZE];
			const Vc * const zlp_l = &zlp[index(n_io * l) / VSIZE];
			const Vc * const zhp_l = &zhp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::forward4e(n_io / 4 / VSIZE, zl_l, w0, w1); Vr4::forward4e(n_io / 4 / VSIZE, zh_l, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::forward4o(n_io / 4 / VSIZE, zl_l, w0, w2); Vr4::forward4o(n_io / 4 / VSIZE, zh_l, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zl_j = &zl_l[8 * m * j];
					Vc * const zh_j = &zh_l[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::forward4e(m, &zl_j[0 * 4 * m], w0, w1); Vr4::forward4e(m, &zh_j[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::forward4o(m, &zl_j[1 * 4 * m], w0, w2); Vr4::forward4o(m, &zh_j[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zl_j = &zl_l[32 / VSIZE * j];
					Vc * const zh_j = &zh_l[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::forward4e_4(&zl_j[0], w0, w1); Vr4::forward4e_4(&zh_j[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::forward4o_4(&zl_j[2], w0, w2); Vr4::forward4o_4(&zh_j[2], w0, w2);
				}
			}

			// mul
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zl_j = &zl_l[8 * j];
				Vc * const zh_j = &zh_l[8 * j];
				const Vc * const zlp_j = &zlp_l[8 * j];
				const Vc * const zhp_j = &zhp_l[8 * j];
				Vc8s z8(zl_j, zh_j); z8.transpose_in();
				Vc8s zp8(zlp_j, zhp_j); z8.mul4(zp8, wsl[j]);
				z8.transpose_out();
				z8.store(zl_j, zh_j);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zl_j = &zl_l[32 / VSIZE * j];
					Vc * const zh_j = &zh_l[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::backward4e_4(&zl_j[0], w0, w1); Vr4::backward4e_4(&zh_j[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::backward4o_4(&zl_j[2], w0, w2); Vr4::backward4o_4(&zh_j[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zl_j = &zl_l[8 * m * j];
					Vc * const zh_j = &zh_l[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::backward4e(m, &zl_j[0 * 4 * m], w0, w1); Vr4::backward4e(m, &zh_j[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::backward4o(m, &zl_j[1 * 4 * m], w0, w2); Vr4::backward4o(m, &zh_j[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::backward4e(n_io / 4 / VSIZE, zl_l, w0, w1); Vr4::backward4e(n_io / 4 / VSIZE, zh_l, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::backward4o(n_io / 4 / VSIZE, zl_l, w0, w2); Vr4::backward4o(n_io / 4 / VSIZE, zh_l, w0, w2); }
			}
		}
	}

	double pass2_0(const size_t thread_id, const double g)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		Vc * const zl = (Vc *)&_mem[zlOffset];
		Vc * const zh = (Vc *)&_mem[zhOffset];
		Vc * const fcl = (Vc *)&_mem[fclOffset]; Vc * const fl = &fcl[thread_id * n_io_inv];
		Vc * const fch = (Vc *)&_mem[fchOffset]; Vc * const fh = &fch[thread_id * n_io_inv];
		const double b = _b, b_inv = _b_inv;
		const bool checkError = _checkError;

		Vc err = Vc(0.0);

		const size_t num_threads = _num_threads;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			Vc * const zl_l = &zl[2 * 4 / VSIZE * lh];
			Vc * const zh_l = &zh[2 * 4 / VSIZE * lh];

			backward_out(zl_l, zh_l, w122i);

			for (size_t j = 0; j < n_io_inv; ++j)
			{
				Vc * const zl_j = &zl_l[index(n_io) * j];
				Vc * const zh_j = &zh_l[index(n_io) * j];
				Vc8s z8(zl_j, zh_j, index(n_io));
				z8.transpose_in();

				const Vc fl_prev = (lh != l_min) ? fl[j] : Vc(0.0);
				const Vc fh_prev = (lh != l_min) ? fh[j] : Vc(0.0);
				if (!checkError) z8.mul_carry(fl_prev, fh_prev, fl[j], fh[j], g, b, b_inv, 2.0 / N);
				else             z8.mul_carry(fl_prev, fh_prev, fl[j], fh[j], g, b, b_inv, 2.0 / N, err);

				if (lh != l_min) z8.transpose_out();
				z8.store(zl_j, zh_j, index(n_io));	// transposed if lh = l_min
			}

			if (lh != l_min) forward_out(zl_l, zh_l, w122i);
		}

		return err.max();
	}

	void pass2_1(const size_t thread_id)
	{
		const size_t num_threads = _num_threads;
		const size_t thread_id_prev = ((thread_id != 0) ? thread_id : num_threads) - 1;
		const size_t lh = thread_id * n_io_s / num_threads;	// l_min of pass2

		Vc * const zl = (Vc *)&_mem[zlOffset]; Vc * const zl_l = &zl[2 * 4 / VSIZE * lh];
		Vc * const zh = (Vc *)&_mem[zhOffset]; Vc * const zh_l = &zh[2 * 4 / VSIZE * lh];
		const Vc * const fcl = (Vc *)&_mem[fclOffset]; const Vc * const fl = &fcl[thread_id_prev * n_io_inv];
		const Vc * const fch = (Vc *)&_mem[fchOffset]; const Vc * const fh = &fch[thread_id_prev * n_io_inv];

		const double b = _b, b_inv = _b_inv;

		for (size_t j = 0; j < n_io_inv; ++j)
		{
			Vc * const zl_j = &zl_l[index(n_io) * j];
			Vc * const zh_j = &zh_l[index(n_io) * j];
			Vc8s z8(zl_j, zh_j, index(n_io));	// transposed

			Vc fl_prev = fl[j], fh_prev = fh[j];
			if (thread_id == 0)
			{
				fl_prev.shift(fl[((j == 0) ? n_io_inv : j) - 1], j == 0);
				fh_prev.shift(fh[((j == 0) ? n_io_inv : j) - 1], j == 0);
			}
			z8.carry(fl_prev, fh_prev, b, b_inv);

			z8.transpose_out();
			z8.store(zl_j, zh_j, index(n_io));
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		forward_out(zl_l, zh_l, w122i);
	}

public:
	transformCPUf64s(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError)
		: transform(N, n, b, ((VSIZE == 2) ? EKind::SBDTvec2 : ((VSIZE == 4) ? EKind::SBDTvec4 : EKind::SBDTvec8))),
		_num_threads(num_threads),
		_b(b), _b_inv(1.0 / b),
		_mem_size(wSize + wsSize + 2 * (zSize + fcSize + zSize + (num_regs - 1) * zSize) + 2 * 1024 * 1024),
		_cache_size(wSize + wsSize + 2 * (zSize + fcSize)), _checkError(checkError), _error(0),
		_mem((char *)alignNew(_mem_size, 2 * 1024 * 1024)), _mem_copy((char *)alignNew(2 * zSize, 1024))
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
		alignDelete((void *)_mem_copy);
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return _cache_size; }

protected:
	void getZi(int32_t * const zi) const override
	{
		const Vc * const zl = (Vc *)&_mem[zlOffset];
		const Vc * const zh = (Vc *)&_mem[zhOffset];

		Vc * const zl_copy = (Vc *)&_mem_copy[0];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zl_copy[k] = zl[k];
		Vc * const zh_copy = (Vc *)&_mem_copy[zSize];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zh_copy[k] = zh[k];

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			backward_out(&zl_copy[2 * 4 / VSIZE * lh], &zh_copy[2 * 4 / VSIZE * lh], w122i);
		}

		const double n_io_N = static_cast<double>(n_io) / N;

		for (size_t k = 0; k < N; k += VSIZE)
		{
			const Vc vc = zl_copy[index(k) / VSIZE] + zh_copy[index(k) / VSIZE];
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
		Vc * const zl = (Vc *)&_mem[zlOffset];
		Vc * const zh = (Vc *)&_mem[zhOffset];

		for (size_t k = 0; k < N; k += VSIZE)
		{
			Vc vc;
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const Complex zc(static_cast<double>(zi[k + i + 0 * N]), static_cast<double>(zi[k + i + 1 * N]));
				vc.set(i, zc);
			}
			const Vc h = Vc(vc * split_inv).round() * split;
			zl[index(k) / VSIZE] = vc - h;
			zh[index(k) / VSIZE] = h;
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&zl[2 * 4 / VSIZE * lh], &zh[2 * 4 / VSIZE * lh], w122i);
		}
	}

public:
	bool readContext(file & cFile, const size_t num_regs) override
	{
		int kind = 0;
		if (!cFile.read(reinterpret_cast<char *>(&kind), sizeof(kind))) return false;
		if (kind != static_cast<int>(getKind())) return false;

		if (!cFile.read(reinterpret_cast<char *>(&_error), sizeof(_error))) return false;

		Vc * const zl = (Vc *)&_mem[zlOffset];
		if (!cFile.read(reinterpret_cast<char *>(zl), zSize)) return false;
		Vc * const zh = (Vc *)&_mem[zhOffset];
		if (!cFile.read(reinterpret_cast<char *>(zh), zSize)) return false;
		if (num_regs > 1)
		{
			Vc * const zr = (Vc *)&_mem[zrOffset];
			if (!cFile.read(reinterpret_cast<char *>(zr), (num_regs - 1) * 2 * zSize)) return false;
		}

		return true;
	}

	void saveContext(file & cFile, const size_t num_regs) const override
	{
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		if (!cFile.write(reinterpret_cast<const char *>(&_error), sizeof(_error))) return;

		const Vc * const zl = (Vc *)&_mem[zlOffset];
		if (!cFile.write(reinterpret_cast<const char *>(zl), zSize)) return;
		const Vc * const zh = (Vc *)&_mem[zhOffset];
		if (!cFile.write(reinterpret_cast<const char *>(zh), zSize)) return;
		if (num_regs > 1)
		{
			const Vc * const zr = (Vc *)&_mem[zrOffset];
			if (!cFile.write(reinterpret_cast<const char *>(zr), (num_regs - 1) * 2 * zSize)) return;
		}
	}

	void set(const uint32_t a) override
	{
		Vc * const zl = (Vc *)&_mem[zlOffset];
		Vc * const zh = (Vc *)&_mem[zhOffset];
		zl[0] = Vc(a); zh[0] = Vc(0.0);
		for (size_t k = 1; k < index(N) / VSIZE; ++k) { zl[k] = zh[k] = Vc(0.0); }

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&zl[2 * 4 / VSIZE * lh], &zh[2 * 4 / VSIZE * lh], w122i);
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
		const Vc * const zl_src = (Vc *)&_mem[(src == 0) ? zlOffset : zrOffset + (src - 1) * 2 * zSize];
		const Vc * const zh_src = (Vc *)&_mem[(src == 0) ? zhOffset : zrOffset + (src - 1) * 2 * zSize + zSize];
		Vc * const zlp = (Vc *)&_mem[zlpOffset];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zlp[k] = zl_src[k];
		Vc * const zhp = (Vc *)&_mem[zhpOffset];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zhp[k] = zh_src[k];

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
		const Vc * const zl_src = (Vc *)&_mem[(src == 0) ? zlOffset : zrOffset + (src - 1) * 2 * zSize];
		const Vc * const zh_src = (Vc *)&_mem[(src == 0) ? zhOffset : zrOffset + (src - 1) * 2 * zSize + zSize];

		Vc * const zl_dst = (Vc *)&_mem[(dst == 0) ? zlOffset : zrOffset + (dst - 1) * 2 * zSize];
		Vc * const zh_dst = (Vc *)&_mem[(dst == 0) ? zhOffset : zrOffset + (dst - 1) * 2 * zSize + zSize];

		for (size_t k = 0; k < index(N) / VSIZE; ++k) zl_dst[k] = zl_src[k];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) zh_dst[k] = zh_src[k];
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
#endif
#if defined(SBDTRANSFORM)
	if      (n == 18) pTransform = new transformCPUf64s<(1 << 17), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 19) pTransform = new transformCPUf64s<(1 << 18), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 20) pTransform = new transformCPUf64s<(1 << 19), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 21) pTransform = new transformCPUf64s<(1 << 20), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 22) pTransform = new transformCPUf64s<(1 << 21), VSIZE>(b, n, num_threads, num_regs, checkError);
	else if (n == 23) pTransform = new transformCPUf64s<(1 << 22), VSIZE>(b, n, num_threads, num_regs, checkError);
#endif

	if (pTransform == nullptr) throw std::runtime_error("exponent is not supported");

	return pTransform;
}

}