/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

struct timer
{
	typedef std::chrono::high_resolution_clock::time_point time;

	static time currentTime()
	{
		return std::chrono::high_resolution_clock::now();
	}

	static double diffTime(const time & end, const time & start)
	{
		return std::chrono::duration<double>(end - start).count();
	}

	static std::string formatTime(const double time)
	{
		uint64_t seconds = uint64_t(time), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::stringstream ss;
		ss << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
		return ss.str();
	}
};

class watch
{
private:
	const double _elapsedTime;
	const timer::time _startTime;
	timer::time _currentTime;
	timer::time _displayStartTime;
	timer::time _recordStartTime;

public:
	watch(const double restoredTime = 0) : _elapsedTime(restoredTime), _startTime(timer::currentTime())
	{
		_currentTime = _displayStartTime = _recordStartTime = _startTime;
	}

	virtual ~watch() {}

	void get() { _currentTime = timer::currentTime(); }

	double getElapsedTime() const { return _elapsedTime + timer::diffTime(_currentTime, _startTime); }
	double getDisplayTime() const { return timer::diffTime(_currentTime, _displayStartTime); }
	double getRecordTime() const { return timer::diffTime(_currentTime, _recordStartTime); }

	void resetDisplayTime() { _displayStartTime = _currentTime; }
	void resetRecordTime() { _recordStartTime = _currentTime; }
};
