/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

	// Gentleman-Sande and Cooley-Tukey FFT can be used to compute the product of two polynomials modulo x^n - 1 (cyclic convolution).
	// Bruun's method is based on a recursive factorization of polynomials.
	// If this polynomial is different of x^n - 1, the transform is a Z-transform and not a Fourier transform and the convolution is not cyclic.
	// It can be used to compute the product of two polynomials modulo P(x).
	// The discrete weighted transform approach (Crandall, Fagin) is not needed.

	// Radix-2 Decimation-In-Frequency, Gentleman-Sande, bit-reversed outputs
	for (size_t m = n / 2, s = 1; m >= 1; m /= 2, s *= 2)
	{
		for (size_t j = 0; j < m; ++j)
		{
			const FIELD w = FIELD::root_one(2 * m).pow(j);
			for (size_t i = 0; i < s; ++i)
			{
				const size_t k = 2 * m * i + j;
				const FIELD u0 = z[k], u1 = z[k + m];
				z[k] = u0 + u1;
				z[k + m] = (u0 - u1).mul(w);
			}
		}
	}

	// Radix-2 Decimation-In-Time, Cooley-Tukey, bit-reversed intputs, inverse transform
	for (size_t m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2)
	{
		for (size_t j = 0; j < m; ++j)
		{
			const FIELD w = FIELD::root_one(2 * m).pow(j);
			for (size_t i = 0; i < s; ++i)
			{
				const size_t k = 2 * m * i + j;
				const FIELD u0 = z[k], u1 = z[k + m].mulconj(w);
				z[k] = (u0 + u1).half();
				z[k + m] = (u0 - u1).half();
			}
		}
	}

	// Radix-2, Bruun's recursive polynomial factorization, bit-reversed outputs
	for (size_t m = n / 2, s = 1; m >= 1; m /= 2, s *= 2)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const FIELD w = FIELD::root_one(2 * s).pow(bitrev(j, s));
			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const FIELD u0 = z[k], u1 = z[k + m].mul(w);
				z[k] = u0 + u1;
				z[k + m] = u0 - u1;
			}
		}
	}

	// Radix-2, inverse transform of Bruun's method, bit-reversed intputs
	for (size_t m = 1, s = n / 2; m <= n / 2; m *= 2, s /= 2)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const FIELD w = FIELD::root_one(2 * s).pow(bitrev(j, s));
			for (size_t i = 0; i < m; ++i)
			{
				const size_t k = 2 * m * j + i;
				const FIELD u0 = z[k], u1 = z[k + m];
				z[k] = (u0 + u1).half();
				z[k + m] = (u0 - u1).half().mulconj(w);
			}
		}
	}
