/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <array>
#include <thread>
#include <atomic>

inline void PAUSE()
{
#if defined(__aarch64__)
#if defined(__clang__)
	__builtin_arm_isb(0xF);
#else
	__asm__ __volatile__ ("isb" : : : "memory");
#endif
#else
	__builtin_ia32_pause();	// rep nop
#endif
}

template<class T>
class parallel
{
public:
	enum class EFunction : unsigned int { None = 0, Pass1 = 1, Pass1mul = 2, Pass2_0 = 3, Pass2_1 = 4, Pass1multiplicand = 5 };

private:
	struct task
	{
		std::thread _thread;
		std::atomic_uint _fn = 0;

		task() {}
		task(const task &) = delete;
		task & operator=(const task &) = delete;

		EFunction get() const
		{
			const unsigned int fn = _fn.load(std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_acquire);	// aarch64: dmb ishld, x64: -
			return static_cast<EFunction>(fn);
		}

		void set(const EFunction fn)
		{
			std::atomic_thread_fence(std::memory_order_release);	// aarch64: dmb ish, x64: -
			_fn.store(static_cast<unsigned int>(fn), std::memory_order_relaxed);
		}
	};

	T * const _transform;
	const size_t _num_threads;
	std::atomic_bool _alive = false;
	std::array<task, 64> _tasks;

	bool is_alive() const
	{
		const bool b = _alive.load(std::memory_order_relaxed);
		if (!b) std::atomic_thread_fence(std::memory_order_acquire);
		return b;
	}

	void set_alive(const bool value)
	{
		std::atomic_thread_fence(std::memory_order_release);
		_alive.store(value, std::memory_order_relaxed);
	}

public:
	parallel(T * const transform, const size_t num_threads) : _transform(transform), _num_threads(num_threads)
	{
		for (size_t i = 0; i < num_threads; ++i) { std::thread t = std::thread(&parallel::work, this, i); _tasks[i]._thread.swap(t); }

		set_alive(true);
	}

	parallel(const parallel &) = delete;
	parallel & operator=(const parallel &) = delete;

	virtual ~parallel()
	{
		const size_t num_threads = _num_threads;

		wait();
		set_alive(false);

		for (size_t i = 0; i < num_threads; ++i)
		{
			task & task = _tasks[i];
			if (task._thread.joinable()) task._thread.join();
		}
	}

	void exec(const size_t i, const EFunction fn)
	{
		_tasks[i - 1].set(fn);
	}

	void wait() const
	{
		const size_t num_threads = _num_threads;

		unsigned int running;
		do
		{
			running = 0; for (size_t i = 0; i < num_threads; ++i) running |= _tasks[i]._fn.load(std::memory_order_relaxed);
		}
		while (static_cast<EFunction>(running) != EFunction::None);

		std::atomic_thread_fence(std::memory_order_acquire);
	}

private:
	void work(const size_t thread_id)
	{
		if (!is_alive()) std::this_thread::sleep_for(std::chrono::milliseconds(100));

		task & task = _tasks[thread_id];
 
		while (true)
		{
			const EFunction fn = task.get();
			if (fn != EFunction::None)
			{
				if (fn == EFunction::Pass1) _transform->pass1(thread_id + 1);
				else if (fn == EFunction::Pass1mul) _transform->pass1mul(thread_id + 1);
				else if (fn == EFunction::Pass2_0) _transform->pass2_0(thread_id + 1);
				else if (fn == EFunction::Pass2_1) _transform->pass2_1(thread_id + 1);
				else if (fn == EFunction::Pass1multiplicand) _transform->pass1multiplicand(thread_id + 1);

				task.set(EFunction::None);
			}
			else PAUSE();

			if (!is_alive()) return;
		};
	}
};
