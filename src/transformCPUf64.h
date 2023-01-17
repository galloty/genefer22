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
#include "f64vector.h"

namespace transformCPU_namespace
{

template<size_t N>
class Vcx8
{
	using Vc = Vcx<N>;

private:
	Vc z[8];

private:
	Vcx8() {}

public:
	finline explicit Vcx8(const Vc * const mem)
	{
		for (size_t i = 0; i < 8; ++i) z[i] = mem[i];
	}

	finline void store(Vc * const mem) const
	{
		for (size_t i = 0; i < 8; ++i) mem[i] = z[i];
	}

	finline explicit Vcx8(const Vc * const mem, const size_t step)
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			z[i] = mem[(step * i_h + i_l) / N];
		}
	}

	finline void store(Vc * const mem, const size_t step) const
	{
		for (size_t i = 0; i < 8; ++i)
		{
			const size_t i_h = (N * i) / 8, i_l = (N * i) % 8;
			mem[(step * i_h + i_l) / N] = z[i];
		}
	}

	finline void transpose_in() { Vc::transpose_in(z); }
	finline void transpose_out() { Vc::transpose_out(z); }

	finline void square4e(const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const Vc s0 = v0.sqr() + v1.sqr().mulW(w), s1 = (v0 + v0) * v1, s2 = v2.sqr() - v3.sqr().mulW(w), s3 = (v2 + v2) * v3;
		z[0] = s0 + s2; z[2] = Vc(s0 - s2).mulWconj(w); z[1] = s1 + s3; z[3] = Vc(s1 - s3).mulWconj(w);
	}

	finline void square4o(const Vc & w)
	{
		const Vc u4 = z[4], u6 = z[6].mulW(w), u5 = z[5], u7 = z[7].mulW(w);
		const Vc v4 = u4.addi(u6), v6 = u4.subi(u6), v5 = u5.addi(u7), v7 = u7.addi(u5);
		const Vc s4 = v5.sqr().mulW(w).subi(v4.sqr()), s5 = (v4 + v4) * v5, s6 = v6.sqr().addi(v7.sqr().mulW(w)), s7 = (v6 + v6) * v7;
		z[4] = s6.addi(s4); z[6] = s4.addi(s6).mulWconj(w); z[5] = s5.subi(s7); z[7] = s7.subi(s5).mulWconj(w);
	}

	finline void mul4_forward(const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		z[0] = u0 + u2; z[2] = u0 - u2; z[1] = u1 + u3; z[3] = u1 - u3;
		const Vc u4 = z[4], u6 = z[6].mulW(w), u5 = z[5], u7 = z[7].mulW(w);
		z[4] = u4.addi(u6); z[6] = u4.subi(u6); z[5] = u5.addi(u7); z[7] = u7.addi(u5);
	}

	finline void mul4(const Vcx8 & rhs, const Vc & w)
	{
		const Vc u0 = z[0], u2 = z[2].mulW(w), u1 = z[1], u3 = z[3].mulW(w);
		const Vc v0 = u0 + u2, v2 = u0 - u2, v1 = u1 + u3, v3 = u1 - u3;
		const Vc vp0 = rhs.z[0], vp2 = rhs.z[2], vp1 = rhs.z[1], vp3 = rhs.z[3];
		const Vc s0 = v0 * vp0 + Vc(v1 * vp1).mulW(w), s1 = v0 * vp1 + vp0 * v1;
		const Vc s2 = v2 * vp2 - Vc(v3 * vp3).mulW(w), s3 = v2 * vp3 + vp2 * v3;
		z[0] = s0 + s2; z[2] = Vc(s0 - s2).mulWconj(w); z[1] = s1 + s3; z[3] = Vc(s1 - s3).mulWconj(w);

		const Vc u4 = z[4], u6 = z[6].mulW(w), u5 = z[5], u7 = z[7].mulW(w);
		const Vc v4 = u4.addi(u6), v6 = u4.subi(u6), v5 = u5.addi(u7), v7 = u7.addi(u5);
		const Vc vp4 = rhs.z[4], vp6 = rhs.z[6], vp5 = rhs.z[5], vp7 = rhs.z[7];
		const Vc s4 = Vc(v5 * vp5).mulW(w).subi(v4 * vp4), s5 = v4 * vp5 + vp4 * v5;
		const Vc s6 = Vc(v6 * vp6).addi(Vc(v7 * vp7).mulW(w)), s7 = v6 * vp7 + vp6 * v7;
		z[4] = s6.addi(s4); z[6] = s4.addi(s6).mulWconj(w); z[5] = s5.subi(s7); z[7] = s7.subi(s5).mulWconj(w);
	}

	finline Vc mul_carry(const Vc & f_prev, const double g, const double b, const double b_inv, const double t2_n)
	{
		Vc f = f_prev;

		for (size_t i = 0; i < 8; ++i)
		{
			Vc & zi = z[i];
			const Vc of = zi * t2_n, o = of.round();
			const Vc o_b = Vc(o * b_inv).round();
			const Vc f_i = f + (o - o_b * b) * g;
			const Vc f_b = Vc(f_i * b_inv).round();
			f = f_b + o_b * g;
			zi = f_i - f_b * b;
		}

		return f;
	}

	finline Vc mul_carry(const Vc & f_prev, const double g, const double b, const double b_inv, const double t2_n, Vc & err)
	{
		Vc f = f_prev;

		for (size_t i = 0; i < 8; ++i)
		{
			Vc & zi = z[i];
			const Vc of = zi * t2_n, o = of.round();
			err.max(Vc(of - o).abs());
			const Vc o_b = Vc(o * b_inv).round();
			const Vc f_i = f + (o - o_b * b) * g;
			const Vc f_b = Vc(f_i * b_inv).round();
			f = f_b + o_b * g;
			zi = f_i - f_b * b;
		}

		return f;
	}

	finline Vc mul_carry_i(const Vc & f_prev, const double g, const double b, const double b_inv, const double t2_n,
						   const double sb, const double sb_inv, const double sbh, const double sbl)
	{
		Vc f = f_prev;

		for (size_t i = 0; i < 4; ++i)
		{
			Vc & z0 = z[2 * i + 0]; Vc & z1 = z[2 * i + 1];

			// const Vc o = Vc((z0 + z1 * sb) * t2_n).round();
			// const Vc f_i = f + o * g;
			// const Vc f_b = Vc(f_i * b_inv).round();
			// const Vc r = f_i - f_b * b;
			// f = f_b;

			const Vc of = (z0 + z1 * sb) * t2_n, o = of.round();
			const Vc o_b = Vc(o * b_inv).round();
			const Vc f_i = f + (o - o_b * b) * g;
			const Vc f_b = Vc(f_i * b_inv).round();
			const Vc r = f_i - f_b * b;
			f = f_b + o_b * g;

			const Vc irh = Vc(r * sb_inv).round();
			z0 = (r - irh * sbh) - irh * sbl; z1 = irh;
		}

		return f;
	}

	finline Vc mul_carry_i(const Vc & f_prev, const double g, const double b, const double b_inv, const double t2_n,
						   const double sb_inv, const double sbh, const double sbl, Vc & err)
	{
		Vc f = f_prev;

		for (size_t i = 0; i < 4; ++i)
		{
			Vc & z0 = z[2 * i + 0]; Vc & z1 = z[2 * i + 1];

			const Vc of = ((z0 + z1 * sbl) + z1 * sbh) * t2_n, o = of.round();
			err.max(Vc(of - o).abs());

			const Vc o_b = Vc(o * b_inv).round();
			const Vc f_i = f + (o - o_b * b) * g;
			const Vc f_b = Vc(f_i * b_inv).round();
			const Vc r = f_i - f_b * b;
			f = f_b + o_b * g;

			const Vc irh = Vc(r * sb_inv).round();
			z0 = (r - irh * sbh) - irh * sbl; z1 = irh;
		}

		return f;
	}

	finline void carry(const Vc & f_i, const double b, const double b_inv)
	{
		Vc f = f_i;

		for (size_t i = 0; i < 8 - 1; ++i)
		{
			Vc & zi = z[i];
			f += zi.round();
			const Vc f_o = Vc(f * b_inv).round();
			zi = f - f_o * b;
			f = f_o;
			if (f.isZero()) return;
		}

		Vc & zi = z[8 - 1];
		zi = f + zi.round();
	}

	finline void carry_i(const Vc & f_i, const double b, const double b_inv, const double sb, const double sb_inv, const double sbh, const double sbl)
	{
		Vc f = f_i;

		for (size_t i = 0; i < 4 - 1; ++i)
		{
			Vc & z0 = z[2 * i + 0]; Vc & z1 = z[2 * i + 1];
			f += Vc(z0 + z1 * sb).round();
			const Vc f_o = Vc(f * b_inv).round();
			const Vc r = f - f_o * b;
			f = f_o;
			const Vc irh = Vc(r * sb_inv).round();
			z0 = (r - irh * sbh) - irh * sbl; z1 = irh;
			if (f.isZero()) return;
		}

		Vc & z0 = z[2 * (4 - 1) + 0]; Vc & z1 = z[2 * (4 - 1) + 1];
		const Vc r = f + Vc(z0 + z1 * sb).round();
		const Vc irh = Vc(r * sb_inv).round();
		z0 = (r - irh * sbh) - irh * sbl; z1 = irh;
	}
};

template<size_t N, size_t VSIZE, bool IBASE>
class transformCPUf64 : public transform
{
	using Vc = Vcx<VSIZE>;
	using Vr4 = Vradix4<VSIZE>;
	using Vr8 = Vradix8<VSIZE>;
	using Vc8 = Vcx8<VSIZE>;

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
	static const size_t zSize = index(N) * sizeof(Complex);
	static const size_t fcSize = 64 * n_io_inv * sizeof(Vc);	// num_threads <= 64

	static const size_t wOffset = 0;
	static const size_t wsOffset = wOffset + wSize;
	static const size_t zOffset = wsOffset + wsSize;
	static const size_t fcOffset = zOffset + zSize;
	static const size_t zpOffset = fcOffset + fcSize;
	static const size_t zrOffset = zpOffset + zSize;

	const size_t _num_threads;
	const double _b, _b_inv, _sb, _sb_inv;
	const size_t _mem_size, _cache_size;
	double _sbh, _sbl;
	bool _checkError;
	double _error;
	char * const _mem;
	Vc * const _z_copy;

private:
	finline static void forward_out(Vc * const z, const Complex * const w122i)
	{
		static const size_t stepi = index(n_io) / VSIZE;

		size_t s = (N / 4) / n_io / 2; for (; s >= 4 * 2; s /= 4);

		if (s == 4) Vr8::forward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, z);
		else        Vr4::forward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, z);

		for (size_t mi = index((s == 4) ? N / 32 : N / 16) / VSIZE; mi >= stepi; mi /= 4, s *= 4)
		{
			for (size_t j = 0; j < s; ++j)
			{
				const size_t k = 8 * mi * j;
				const Complex * const w = &w122i[s + 3 * j];
				const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
				Vr4::forward4e(mi, stepi, 2 * 4 / VSIZE, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				Vr4::forward4o(mi, stepi, 2 * 4 / VSIZE, &z[k + 1 * 4 * mi], w0, w2);
			}
		}
	}

	finline static void backward_out(Vc * const z, const Complex * const w122i)
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
				Vr4::backward4e(mi, stepi, 2 * 4 / VSIZE, &z[k + 0 * 4 * mi], w0, w1);
				const Vc w2 = Vc::broadcast(w[2]);
				Vr4::backward4o(mi, stepi, 2 * 4 / VSIZE, &z[k + 1 * 4 * mi], w0, w2);
			}
		}

		if (s == 1) Vr8::backward8_0(index(N / 8) / VSIZE, stepi, 2 * 4 / VSIZE, z);
		else        Vr4::backward4_0(index(N / 4) / VSIZE, stepi, 2 * 4 / VSIZE, z);
	}

	void pass1(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const z = (Vc *)&_mem[zOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zl = &z[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::forward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::forward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::forward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::forward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::forward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::forward4o_4(&zj[2], w0, w2);
				}
			}

			// square
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zj = &zl[8 * j];
				Vc8 z8(zj);
				z8.transpose_in();
				z8.square4e(wsl[j]);
				z8.store(zj);
			}
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zj = &zl[8 * j];
				Vc8 z8(zj);
				z8.square4o(wsl[j]);
				z8.transpose_out();
				z8.store(zj);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::backward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::backward4o_4(&zj[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::backward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::backward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::backward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::backward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}
		}
	}

	void pass1multiplicand(const size_t thread_id)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		const Vc * const ws = (Vc *)&_mem[wsOffset];
		Vc * const zp = (Vc *)&_mem[zpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zpl = &zp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::forward4e(n_io / 4 / VSIZE, zpl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::forward4o(n_io / 4 / VSIZE, zpl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zpj = &zpl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::forward4e(m, &zpj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::forward4o(m, &zpj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zpj = &zpl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::forward4e_4(&zpj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::forward4o_4(&zpj[2], w0, w2);
				}
			}

			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zpj = &zpl[8 * j];
				Vc8 zp8(zpj);
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
		Vc * const z = (Vc *)&_mem[zOffset];
		const Vc * const zp = (Vc *)&_mem[zpOffset];

		const size_t num_threads = _num_threads, s_io = N / n_io;
		const size_t l_min = thread_id * s_io / num_threads, l_max = (thread_id + 1 == num_threads) ? s_io : (thread_id + 1) * s_io / num_threads;
		for (size_t l = l_min; l < l_max; ++l)
		{
			Vc * const zl = &z[index(n_io * l) / VSIZE];
			const Vc * const zpl = &zp[index(n_io * l) / VSIZE];
			const Vc * const wsl = &ws[l * n_io / 8 / VSIZE];

			// forward_in
			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::forward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::forward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}

			for (size_t m = n_io / 16 / VSIZE, s = 2; m >= ((VSIZE == 8) ? 16 : 4) / VSIZE; m /= 4, s *= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::forward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::forward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::forward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::forward4o_4(&zj[2], w0, w2);
				}
			}

			// mul
			for (size_t j = 0; j < n_io / 8 / VSIZE; ++j)
			{
				Vc * const zj = &zl[8 * j];
				const Vc * const zpj = &zpl[8 * j];
				Vc8 z8(zj); z8.transpose_in();
				Vc8 zp8(zpj); z8.mul4(zp8, wsl[j]);
				z8.transpose_out();
				z8.store(zj);
			}

			if (VSIZE == 8)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * (n_io / 32)];

				for (size_t j = 0; j < n_io / 32; j += 2)
				{
					Vc * const zj = &zl[32 / VSIZE * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0], w[3]), w1 = Vc::broadcast(w[1], w[4]);
					Vr4::backward4e_4(&zj[0], w0, w1);
					const Vc w2 = Vc::broadcast(w[2], w[5]);
					Vr4::backward4o_4(&zj[2], w0, w2);
				}
			}

			// backward_in
			for (size_t m = ((VSIZE == 8) ? 16 : 4) / VSIZE, s = 2 * n_io / 16 / VSIZE / m; m <= n_io / 16 / VSIZE; m *= 4, s /= 4)
			{
				const Complex * const w_s = &w122i[(s_io + 3 * l) * s];

				for (size_t j = 0; j < s; ++j)
				{
					Vc * const zj = &zl[8 * m * j];
					const Complex * const w = &w_s[3 * j];
					const Vc w0 = Vc::broadcast(w[0]), w1 = Vc::broadcast(w[1]);
					Vr4::backward4e(m, &zj[0 * 4 * m], w0, w1);
					const Vc w2 = Vc::broadcast(w[2]);
					Vr4::backward4o(m, &zj[1 * 4 * m], w0, w2);
				}
			}

			{
				const Complex * const w = &w122i[s_io / 2 + 3 * (l / 2)];
				const Vc w0 = Vc::broadcast(w[0]);

				if (l % 2 == 0) { const Vc w1 = Vc::broadcast(w[1]); Vr4::backward4e(n_io / 4 / VSIZE, zl, w0, w1); }
				else            { const Vc w2 = Vc::broadcast(w[2]); Vr4::backward4o(n_io / 4 / VSIZE, zl, w0, w2); }
			}
		}
	}

	double pass2_0(const size_t thread_id, const bool dup)
	{
		const Complex * const w122i = (Complex *)&_mem[wOffset];
		Vc * const z = (Vc *)&_mem[zOffset];
		Vc * const fc = (Vc *)&_mem[fcOffset]; Vc * const f = &fc[thread_id * n_io_inv];
		const double b = _b, b_inv = _b_inv, sb = _sb, sb_inv = _sb_inv, sbh = _sbh, sbl = _sbl, g = dup ? 2.0 : 1.0;
		const bool checkError = _checkError;

		Vc err = Vc(0.0);

		const size_t num_threads = _num_threads;
		const size_t l_min = thread_id * n_io_s / num_threads, l_max = (thread_id + 1 == num_threads) ? n_io_s : (thread_id + 1) * n_io_s / num_threads;
		for (size_t lh = l_min; lh < l_max; ++lh)
		{
			Vc * const zl = &z[2 * 4 / VSIZE * lh];

			backward_out(zl, w122i);

			for (size_t j = 0; j < n_io_inv; ++j)
			{
				Vc * const zj = &zl[index(n_io) * j];
				Vc8 z8(zj, index(n_io));
				z8.transpose_in();

				const Vc f_prev = (lh != l_min) ? f[j] : Vc(0.0);
				if (!checkError)
				{
					if (IBASE) f[j] = z8.mul_carry_i(f_prev, g, b, b_inv, 2.0 / N, sb, sb_inv, sbh, sbl);
					else f[j] = z8.mul_carry(f_prev, g, b, b_inv, 2.0 / N);
				}
				else
				{
					if (IBASE) f[j] = z8.mul_carry_i(f_prev, g, b, b_inv, 2.0 / N, sb_inv, sbh, sbl, err);
					else f[j] = z8.mul_carry(f_prev, g, b, b_inv, 2.0 / N, err);
				}

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

		Vc * const z = (Vc *)&_mem[zOffset]; Vc * const zl = &z[2 * 4 / VSIZE * lh];
		const Vc * const fc = (Vc *)&_mem[fcOffset]; const Vc * const f = &fc[thread_id_prev * n_io_inv];

		const double b = _b, b_inv = _b_inv, sb = _sb, sb_inv = _sb_inv, sbh = _sbh, sbl = _sbl;

		for (size_t j = 0; j < n_io_inv; ++j)
		{
			Vc * const zj = &zl[index(n_io) * j];
			Vc8 z8(zj, index(n_io));	// transposed

			Vc f_prev = f[j];
			if (thread_id == 0) f_prev.shift(f[((j == 0) ? n_io_inv : j) - 1], j == 0);
			if (IBASE) z8.carry_i(f_prev, b, b_inv, sb, sb_inv, sbh, sbl);
			else z8.carry(f_prev, b, b_inv);

			z8.transpose_out();
			z8.store(zj, index(n_io));
		}

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		forward_out(zl, w122i);
	}

public:
	transformCPUf64(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError)
		: transform(N, n, b, IBASE ? ((VSIZE == 2) ? EKind::IBDTvec2 : ((VSIZE == 4) ? EKind::IBDTvec4 : EKind::IBDTvec8))
								   : ((VSIZE == 2) ? EKind::DTvec2 : ((VSIZE == 4) ? EKind::DTvec4 : EKind::DTvec8))),
		_num_threads(num_threads),
		_b(b), _b_inv(1.0 / b), _sb(sqrt(static_cast<double>(b))), _sb_inv(1 / _sb),
		_mem_size(wSize + wsSize + zSize + fcSize + zSize + (num_regs - 1) * zSize + 2 * 1024 * 1024),
		_cache_size(wSize + wsSize + zSize + fcSize), _checkError(checkError), _error(0),
		_mem((char *)alignNew(_mem_size, 2 * 1024 * 1024)), _z_copy((Vc *)alignNew(zSize, 1024))
	{
		mpz_t sb2e64, t; mpz_init_set_ui(sb2e64, b); mpz_init(t);
		mpz_mul_2exp(sb2e64, sb2e64, 128); mpz_sqrt(sb2e64, sb2e64);

		const int shift = 16;
		mpz_div_2exp(t, sb2e64, 64 - shift);
		_sbh = std::ldexp(mpz_get_d(t), -shift);
		mpz_mod_2exp(t, sb2e64, 64 - shift);
		_sbl = std::ldexp(mpz_get_d(t), -64);

		mpz_clear(sb2e64); mpz_clear(t);

		const size_t a =
#if defined(CYCLO)
			3;
#else
			2;
#endif
		Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t s = N / 16; s >= 4; s /= 4)
		{
			Complex * const w_s = &w122i[2 * s / 4];
			for (size_t j = 0; j < s / 2; ++j)
			{
				const size_t r = bitRev(2 * j, 2 * s);
				w_s[3 * j + 0] = Complex::exp2iPi(a * r + 1, 4 * a * s);
				w_s[3 * j + 1] = Complex::exp2iPi(a * r + 1, 2 * 4 * a * s);
				w_s[3 * j + 2] = Complex::exp2iPi(a * (r + s) + 1, 2 * 4 * a * s);
			}
		}

		Vc * const ws = (Vc *)&_mem[wsOffset];
		for (size_t j = 0; j < N / 8 / VSIZE; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t r = bitRev(2 * (VSIZE * j + i), 2 * (N / 4));
				ws[j].set(i, Complex::exp2iPi(a * r + 1, 4 * a * (N / 4)));
			}
		}
	}

	virtual ~transformCPUf64()
	{
		alignDelete((void *)_mem);
		alignDelete((void *)_z_copy);
	}

	size_t getMemSize() const override { return _mem_size; }
	size_t getCacheSize() const override { return _cache_size; }

protected:
	void getZi(int32_t * const zi) const override
	{
		const Vc * const z = (Vc *)&_mem[zOffset];

		Vc * const z_copy = _z_copy;
		for (size_t k = 0; k < index(N) / VSIZE; ++k) z_copy[k] = z[k];

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			backward_out(&z_copy[2 * 4 / VSIZE * lh], w122i);
		}

		const double n_io_N = static_cast<double>(n_io) / N;

		if (IBASE)
		{
			const double sb = _sb;

			for (size_t k = 0; k < N / 2; k += VSIZE / 2)
			{
				const Vc vc = z_copy[index(2 * k) / VSIZE];
				for (size_t i = 0; i < VSIZE / 2; ++i)
				{
					const Complex z1 = vc[2 * i + 0], z2 = vc[2 * i + 1];
					zi[k + i + 0 * N / 2] = std::lround((z1.real + sb * z2.real) * n_io_N);
					zi[k + i + 1 * N / 2] = std::lround((z1.imag + sb * z2.imag) * n_io_N);
				}
			}
		}
		else
		{
			for (size_t k = 0; k < N; k += VSIZE)
			{
				const Vc vc = z_copy[index(k) / VSIZE];
				for (size_t i = 0; i < VSIZE; ++i)
				{
					const Complex zc = vc[i];
					zi[k + i + 0 * N] = std::lround(zc.real * n_io_N);
					zi[k + i + 1 * N] = std::lround(zc.imag * n_io_N);
				}
			}
		}
	}

	void setZi(const int32_t * const zi) override
	{
		Vc * const z = (Vc *)&_mem[zOffset];

		if (IBASE)
		{
			const Vd<VSIZE> sbh = Vd<VSIZE>::broadcast(_sbh), sbl = Vd<VSIZE>::broadcast(_sbl), sb_inv = Vd<VSIZE>::broadcast(_sb_inv);

			for (size_t k = 0; k < N / 2; k += VSIZE / 2)
			{
				Vd<VSIZE> r;
				for (size_t i = 0; i < VSIZE / 2; ++i)
				{
					r.set(2 * i + 0, static_cast<double>(zi[k + i + 0 * N / 2]));
					r.set(2 * i + 1, static_cast<double>(zi[k + i + 1 * N / 2]));
				}

				const Vd<VSIZE> irh = Vd<VSIZE>(r * sb_inv).round();
				const Vd<VSIZE> re = (r - irh * sbh) - irh * sbl, im = irh;

				Vc vc;
				for (size_t i = 0; i < VSIZE / 2; ++i)
				{
					vc.set(2 * i + 0, Complex(re[2 * i + 0], re[2 * i + 1]));
					vc.set(2 * i + 1, Complex(im[2 * i + 0], im[2 * i + 1]));
				}

				z[index(2 * k) / VSIZE] = vc;
			}
		}
		else
		{
			for (size_t k = 0; k < N; k += VSIZE)
			{
				Vc vc;
				for (size_t i = 0; i < VSIZE; ++i)
				{
					const Complex zc(static_cast<double>(zi[k + i + 0 * N]), static_cast<double>(zi[k + i + 1 * N]));
					vc.set(i, zc);
				}
				z[index(k) / VSIZE] = vc;
			}
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

		Vc * const z = (Vc *)&_mem[zOffset];
		if (!cFile.read(reinterpret_cast<char *>(z), zSize)) return false;
		if (num_regs > 1)
		{
			Vc * const zr = (Vc *)&_mem[zrOffset];
			if (!cFile.read(reinterpret_cast<char *>(zr), (num_regs - 1) * zSize)) return false;
		}

		return true;
	}

	void saveContext(file & cFile, const size_t num_regs) const override
	{
		const int kind = static_cast<int>(getKind());
		if (!cFile.write(reinterpret_cast<const char *>(&kind), sizeof(kind))) return;

		if (!cFile.write(reinterpret_cast<const char *>(&_error), sizeof(_error))) return;

		const Vc * const z = (Vc *)&_mem[zOffset];
		if (!cFile.write(reinterpret_cast<const char *>(z), zSize)) return;
		if (num_regs > 1)
		{
			const Vc * const zr = (Vc *)&_mem[zrOffset];
			if (!cFile.write(reinterpret_cast<const char *>(zr), (num_regs - 1) * zSize)) return;
		}
	}

	void set(const int32_t a) override
	{
		Vc * const z = (Vc *)&_mem[zOffset];
		z[0] = Vc(a);
		for (size_t k = 1; k < index(N) / VSIZE; ++k) z[k] = Vc(0.0);

		const Complex * const w122i = (Complex *)&_mem[wOffset];
		for (size_t lh = 0; lh < n_io / 4 / 2; ++lh)
		{
			forward_out(&z[2 * 4 / VSIZE * lh], w122i);
		}
	}

	void squareDup(const bool dup) override
	{
		const size_t num_threads = _num_threads;
		double e[num_threads];

		if (num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				pass1(thread_id);
#pragma omp barrier
				e[thread_id] = pass2_0(thread_id, dup);
#pragma omp barrier
				pass2_1(thread_id);
			}
		}
		else
		{
			pass1(0);
			e[0] = pass2_0(0, dup);
			pass2_1(0);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		_error = std::max(_error, err);
	}

	void initMultiplicand(const size_t src) override
	{
		const Vc * const z_src = (Vc *)&_mem[(src == 0) ? zOffset : zrOffset + (src - 1) * zSize];
		Vc * const zp = (Vc *)&_mem[zpOffset];
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
		double e[num_threads];

		if (num_threads > 1)
		{
#pragma omp parallel
			{
				const size_t thread_id = size_t(omp_get_thread_num());

				pass1mul(thread_id);
#pragma omp barrier
				e[thread_id] = pass2_0(thread_id, false);
#pragma omp barrier
				pass2_1(thread_id);
			}
		}
		else
		{
			pass1mul(0);
			e[0] = pass2_0(0, false);
			pass2_1(0);
		}

		double err = 0;
		for (size_t i = 0; i < num_threads; ++i) err = std::max(err, e[i]);
		_error = std::max(_error, err);
	}

	void copy(const size_t dst, const size_t src) const override
	{
		const Vc * const z_src = (Vc *)&_mem[(src == 0) ? zOffset : zrOffset + (src - 1) * zSize];
		Vc * const z_dst = (Vc *)&_mem[(dst == 0) ? zOffset : zrOffset + (dst - 1) * zSize];
		for (size_t k = 0; k < index(N) / VSIZE; ++k) z_dst[k] = z_src[k];
	}

	double getError() const override { return _error; }
};

template<size_t VSIZE>
inline transform * create_transformCPUf64(const uint32_t b, const uint32_t n, const size_t num_threads, const size_t num_regs, const bool checkError)
{
	transform * pTransform = nullptr;
#if defined(DTRANSFORM)
	if      (n == 12) pTransform = new transformCPUf64<(1 << 11), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 13) pTransform = new transformCPUf64<(1 << 12), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 14) pTransform = new transformCPUf64<(1 << 13), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 15) pTransform = new transformCPUf64<(1 << 14), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 16) pTransform = new transformCPUf64<(1 << 15), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 17) pTransform = new transformCPUf64<(1 << 16), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 18) pTransform = new transformCPUf64<(1 << 17), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 19) pTransform = new transformCPUf64<(1 << 18), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 20) pTransform = new transformCPUf64<(1 << 19), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 21) pTransform = new transformCPUf64<(1 << 20), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 22) pTransform = new transformCPUf64<(1 << 21), VSIZE, false>(b, n, num_threads, num_regs, checkError);
	else if (n == 23) pTransform = new transformCPUf64<(1 << 22), VSIZE, false>(b, n, num_threads, num_regs, checkError);
#elif defined(IBDTRANSFORM)
	if      (n == 12) pTransform = new transformCPUf64<(1 << 12), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 13) pTransform = new transformCPUf64<(1 << 13), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 14) pTransform = new transformCPUf64<(1 << 14), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 15) pTransform = new transformCPUf64<(1 << 15), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 16) pTransform = new transformCPUf64<(1 << 16), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 17) pTransform = new transformCPUf64<(1 << 17), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 18) pTransform = new transformCPUf64<(1 << 18), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 19) pTransform = new transformCPUf64<(1 << 19), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 20) pTransform = new transformCPUf64<(1 << 20), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 21) pTransform = new transformCPUf64<(1 << 21), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 22) pTransform = new transformCPUf64<(1 << 22), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 23) pTransform = new transformCPUf64<(1 << 23), VSIZE, true>(b, n, num_threads, num_regs, checkError);
#elif defined(SBDTRANSFORM)
	(void)b; (void)n; (void)num_threads; (void)num_regs; (void)checkError;
#else
	if      (n == 18) pTransform = new transformCPUf64<(1 << 18), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 19) pTransform = new transformCPUf64<(1 << 19), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 20) pTransform = new transformCPUf64<(1 << 20), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 21) pTransform = new transformCPUf64<(1 << 21), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	else if (n == 22)
	{
		if (b < 846398) pTransform = new transformCPUf64<(1 << 21), VSIZE, false>(b, n, num_threads, num_regs, checkError);
		else            pTransform = new transformCPUf64<(1 << 22), VSIZE, true>(b, n, num_threads, num_regs, checkError);
	}
	else if (n == 23) pTransform = new transformCPUf64<(1 << 22), VSIZE, false>(b, n, num_threads, num_regs, checkError);
#endif

	if (pTransform == nullptr) throw std::runtime_error("exponent is not supported");

	return pTransform;
}

}