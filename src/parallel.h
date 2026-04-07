/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <vector>
#include <thread>
#include <functional>
#include <utility>

#if defined(__aarch64__)
#define PAUSE()	__builtin_arm_isb(0xF)
#else
#define PAUSE()	__builtin_ia32_pause()
#endif

template<class T>
class parallel
{
private:
	struct task
	{
		std::thread thread;
		volatile bool running;
		void (T::*fn)(size_t);

		task(parallel * const p, const size_t thread_id) : thread(&parallel::work, p, thread_id), running(false) {}
	};
	T * const _transform;
	std::vector<task> _tasks;
	volatile bool _start, _stop;

public:
	parallel(T * const transform, const size_t num_threads) : _transform(transform), _start(false), _stop(false)
	{
		for (size_t i = 0; i < num_threads; ++i) _tasks.emplace_back(this, i);
		_start = true;
	}

	virtual ~parallel()
	{
		_stop = true;
		wait();

		for (auto & task : _tasks) if (task.thread.joinable()) task.thread.join();
	}

	void exec(void (T::*func)(size_t), const size_t i)
	{
		task & task = _tasks[i - 1];
		task.fn = func;
		task.running = true;
	}

	void wait() const
	{
		while (true)
		{
			bool running = false; for (const auto & task : _tasks) running |= task.running;
			if (!running) return;
			PAUSE();
		}
	}

private:
	void work(const size_t thread_id)
	{
		while (!_start) std::this_thread::sleep_for(std::chrono::milliseconds(100));

		task & task = _tasks[thread_id];

		while (true)
		{
			while (!task.running)
			{
				if (_stop) return;
				PAUSE();
			}

			(_transform->*(task.fn))(thread_id + 1);

			task.running = false;
		};
	}
};
