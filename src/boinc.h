/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
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

struct BOINC_OPTIONS
{
	int normal_thread_priority;
	int main_program;
	int check_heartbeat;
	int handle_process_control;
	int send_status_msgs;
	int direct_process_action;
	int multi_thread;
	int multi_process;
};

inline void boinc_options_defaults(BOINC_OPTIONS &) {}
inline int boinc_init_options(BOINC_OPTIONS *) { std::cout << "boinc_init()" << std::endl; return 0; }
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
	// std::cout << "boinc_get_status" << std::endl;
	status->no_heartbeat = status->suspended = status->quit_request = status->abort_request = 0;
	static int cnt = 0;
	if ((++cnt >= 10) && (cnt < 20)) status->suspended = 1;
	if (cnt >= 1000) status->suspended = status->abort_request = 1;
	return 0;
}

#endif

