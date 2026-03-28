/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <stdexcept>

#include "transform.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#if defined(_WIN32)

size_t transform::get_sve_size() { return 0; }

#else

#include <signal.h>
#include <setjmp.h>

static volatile sig_atomic_t g_sigill_caught = 0;
static sigjmp_buf g_jmpbuf;

static void sigill_handler(int)
{
	g_sigill_caught = 1;
	siglongjmp(g_jmpbuf, -1);
}

size_t transform::get_sve_size()
{
	uint64_t n = 0;

	struct sigaction act, old_act; memset(&act, 0, sizeof(act));
	act.sa_flags = SA_ONSTACK | SA_RESTART;
	act.sa_handler = sigill_handler;
	sigaction(SIGILL, &act, &old_act);
	sigaction(SIGSEGV, &act, &old_act);

	if (sigsetjmp(g_jmpbuf, 1) == 0)
	{
#ifdef __ARM_FEATURE_SVE
		n = svcntb() * 8;
#endif
	}

	sigaction(SIGILL, &old_act, nullptr);
	sigaction(SIGSEGV, &old_act, nullptr);

	return (g_sigill_caught != 0) ? 0 : n;
}
#endif
