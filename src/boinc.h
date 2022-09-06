/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#if defined(BOINC)
#include "boinc_api.h"
#include "boinc_opencl.h"
#else

// fake BOINC (for testing)

#include <cstring>
#include <iostream>

inline int boinc_init() { std::cout << "boinc_init()" << std::endl; return 0; }
inline int boinc_finish(const int status) { std::cout << "boinc_finish(" << status << ")" << std::endl; exit(status); /* never reached */ return 0; }

inline int boinc_resolve_filename(const char * const virtual_name, char * const physical_name, const int len)
{
	strncpy(physical_name, virtual_name, size_t(len - 1));
	return 0;
}

inline FILE * boinc_fopen(const char * const path, const char * const mode)
{
	return std::fopen(path, mode);
}

inline int boinc_time_to_checkpoint() { static int cnt = 0; if (++cnt == 20) { cnt = 0; return 1; } return 0; }
inline int boinc_checkpoint_completed() { return 0; }

inline int boinc_fraction_done(const double f) { std::cout << "boinc_fraction_done(" << f << ")" << std::endl; return 0; }

struct BOINC_STATUS { int no_heartbeat, suspended, quit_request, abort_request; };

inline int boinc_get_status(BOINC_STATUS * const status)
{
	status->no_heartbeat = status->suspended = status->quit_request = status->abort_request = 0;
	static int cnt = 0;
	if ((++cnt >= 10) && (cnt < 15)) status->suspended = 1;
	if (cnt >= 100) status->quit_request = 1;
	return 0;
}

#endif

