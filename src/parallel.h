/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <array>
#include <thread>
#include <atomic>

#if defined(__aarch64__)
#define PAUSE()	__builtin_arm_isb(0xF)
#else
#define PAUSE()	__builtin_ia32_pause()
#endif

template<class T>
class parallel
{
public:
	enum class EFunction : unsigned int { Pass1 = 1, Pass1mul = 2, Pass2_0 = 4, Pass2_1 = 8, Pass1multiplicand = 16 };

private:
	struct task
	{
		std::thread thread;
		std::atomic_uint fn = 0;

		task() {}
		task(const task &) = delete;
		task & operator=(const task &) = delete;
	};

	T * const _transform;
	const size_t _num_threads;
	std::atomic_bool _start = false, _stop = false;
	std::array<task, 64> _tasks;

public:
	parallel(T * const transform, const size_t num_threads) : _transform(transform), _num_threads(num_threads)
	{
		for (size_t i = 0; i < num_threads; ++i) { std::thread t = std::thread(&parallel::work, this, i); _tasks[i].thread.swap(t); }

		std::atomic_thread_fence(std::memory_order_release);
		_start.store(true, std::memory_order_relaxed);
	}

	parallel(const parallel &) = delete;
	parallel & operator=(const parallel &) = delete;

	virtual ~parallel()
	{
		const size_t num_threads = _num_threads;

		std::atomic_thread_fence(std::memory_order_release);
		_stop.store(true, std::memory_order_relaxed);

		wait();

		for (size_t i = 0; i < num_threads; ++i)
		{
			task & task = _tasks[i];
			if (task.thread.joinable()) task.thread.join();
		}
	}

	void exec(const size_t i, const EFunction fn)
	{
		task & task = _tasks[i - 1];
		std::atomic_thread_fence(std::memory_order_release);
		task.fn.store(static_cast<unsigned int>(fn), std::memory_order_relaxed);
	}

	void wait() const
	{
		const size_t num_threads = _num_threads;

		while (true)
		{
			unsigned int running = 0;
			for (size_t i = 0; i < num_threads; ++i) running |= _tasks[i].fn.load(std::memory_order_relaxed);
			if (running == 0)
			{
				std::atomic_thread_fence(std::memory_order_acquire);
				return;
			}
			PAUSE();
		}
	}

private:
	void work(const size_t thread_id)
	{
		while (!_start.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::milliseconds(100));
		std::atomic_thread_fence(std::memory_order_acquire);

		task & task = _tasks[thread_id];
 
		while (true)
		{
			const unsigned int fn = task.fn.load(std::memory_order_relaxed);
			if (fn == 0) { PAUSE(); }
			else
			{
				std::atomic_thread_fence(std::memory_order_acquire);

				if (fn == static_cast<unsigned int>(EFunction::Pass1)) _transform->pass1(thread_id + 1);
				else if (fn == static_cast<unsigned int>(EFunction::Pass1mul)) _transform->pass1mul(thread_id + 1);
				else if (fn == static_cast<unsigned int>(EFunction::Pass2_0)) _transform->pass2_0(thread_id + 1);
				else if (fn == static_cast<unsigned int>(EFunction::Pass2_1)) _transform->pass2_1(thread_id + 1);
				else if (fn == static_cast<unsigned int>(EFunction::Pass1multiplicand)) _transform->pass1multiplicand(thread_id + 1);

				std::atomic_thread_fence(std::memory_order_release);
				task.fn.store(0, std::memory_order_relaxed);
			}

			if (_stop.load(std::memory_order_relaxed))
			{
				std::atomic_thread_fence(std::memory_order_acquire);
				return;
			}
		};
	}
};
