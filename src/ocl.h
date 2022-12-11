/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#define CL_TARGET_OPENCL_VERSION 110
#if defined(__APPLE__)
	#include <OpenCL/cl.h>
	#include <OpenCL/cl_ext.h>
#else
	#include <CL/cl.h>
#endif

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "pio.h"

class splitter
{
private:
	struct partition
	{
		size_t size;
		uint32_t p[8];
	};

	const bool b256, b1024;
	const size_t mMax;
	size_t size;
	partition part[32];

private:
	void split(const size_t m, const size_t i, partition & p)
	{
		if (b1024 && (m >= 10 + 5))
		{
			p.p[i] = 10;
			split(m - 10, i + 1, p);
		}
		if (b256 && (m >= 8 + 5))
		{
			p.p[i] = 8;
			split(m - 8, i + 1, p);
		}
		if (m >= 6 + 5)
		{
			p.p[i] = 6;
			split(m - 6, i + 1, p);
		}

		if ((5 <= m) && (m <= mMax))
		{
			for (size_t k = 0; k < i; ++k) part[size].p[k] = p.p[k];
			part[size].p[i] = static_cast<uint32_t>(m);
			part[size].size = i + 1;
			size++;
		}
	}

private:
	static size_t log_2(const size_t n) { size_t r = 0; for (size_t m = 1; m < n; m *= 2) ++r; return r; }

public:
	splitter(const size_t n, const size_t chunk256, const size_t chunk1024, const size_t sizeofRNS, const size_t mSquareMax,
		const cl_ulong localMemSize, const size_t maxWorkGroupSize) :
		b256(maxWorkGroupSize >= (256 / 4) * chunk256),
		b1024((maxWorkGroupSize >= (1024 / 4) * chunk1024) && (localMemSize / sizeofRNS >= 1024 * chunk1024)),
		mMax(std::min(mSquareMax, std::min(log_2(size_t(localMemSize / sizeofRNS)), log_2(maxWorkGroupSize * 4))))
	{
		size = 0;
		partition p;
		split(n, 0, p);
	}

	size_t getSize() const { return size; }
	size_t getPartSize(const size_t i) const { return part[i].size; }
	uint32_t getPart(const size_t i, const size_t j) const { return part[i].p[j]; }
};


// #define ocl_debug		1
#define ocl_fast_exec		1
#define ocl_device_type		CL_DEVICE_TYPE_GPU	// CL_DEVICE_TYPE_ALL

class oclObject
{
private:
	static const char * errorString(const cl_int & res)
	{
		switch (res)
		{
	#define oclCheck(err) case err: return #err
			oclCheck(CL_SUCCESS);
			oclCheck(CL_DEVICE_NOT_FOUND);
			oclCheck(CL_DEVICE_NOT_AVAILABLE);
			oclCheck(CL_COMPILER_NOT_AVAILABLE);
			oclCheck(CL_MEM_OBJECT_ALLOCATION_FAILURE);
			oclCheck(CL_OUT_OF_RESOURCES);
			oclCheck(CL_OUT_OF_HOST_MEMORY);
			oclCheck(CL_PROFILING_INFO_NOT_AVAILABLE);
			oclCheck(CL_MEM_COPY_OVERLAP);
			oclCheck(CL_IMAGE_FORMAT_MISMATCH);
			oclCheck(CL_IMAGE_FORMAT_NOT_SUPPORTED);
			oclCheck(CL_BUILD_PROGRAM_FAILURE);
			oclCheck(CL_MAP_FAILURE);
			oclCheck(CL_MISALIGNED_SUB_BUFFER_OFFSET);
			oclCheck(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
			oclCheck(CL_INVALID_VALUE);
			oclCheck(CL_INVALID_DEVICE_TYPE);
			oclCheck(CL_INVALID_PLATFORM);
			oclCheck(CL_INVALID_DEVICE);
			oclCheck(CL_INVALID_CONTEXT);
			oclCheck(CL_INVALID_QUEUE_PROPERTIES);
			oclCheck(CL_INVALID_COMMAND_QUEUE);
			oclCheck(CL_INVALID_HOST_PTR);
			oclCheck(CL_INVALID_MEM_OBJECT);
			oclCheck(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
			oclCheck(CL_INVALID_IMAGE_SIZE);
			oclCheck(CL_INVALID_SAMPLER);
			oclCheck(CL_INVALID_BINARY);
			oclCheck(CL_INVALID_BUILD_OPTIONS);
			oclCheck(CL_INVALID_PROGRAM);
			oclCheck(CL_INVALID_PROGRAM_EXECUTABLE);
			oclCheck(CL_INVALID_KERNEL_NAME);
			oclCheck(CL_INVALID_KERNEL_DEFINITION);
			oclCheck(CL_INVALID_KERNEL);
			oclCheck(CL_INVALID_ARG_INDEX);
			oclCheck(CL_INVALID_ARG_VALUE);
			oclCheck(CL_INVALID_ARG_SIZE);
			oclCheck(CL_INVALID_KERNEL_ARGS);
			oclCheck(CL_INVALID_WORK_DIMENSION);
			oclCheck(CL_INVALID_WORK_GROUP_SIZE);
			oclCheck(CL_INVALID_WORK_ITEM_SIZE);
			oclCheck(CL_INVALID_GLOBAL_OFFSET);
			oclCheck(CL_INVALID_EVENT_WAIT_LIST);
			oclCheck(CL_INVALID_EVENT);
			oclCheck(CL_INVALID_OPERATION);
			oclCheck(CL_INVALID_GL_OBJECT);
			oclCheck(CL_INVALID_BUFFER_SIZE);
			oclCheck(CL_INVALID_MIP_LEVEL);
			oclCheck(CL_INVALID_GLOBAL_WORK_SIZE);
			oclCheck(CL_INVALID_PROPERTY);
	#undef oclCheck
			default: return "CL_UNKNOWN_ERROR";
		}
	}

protected:
	static constexpr bool oclError(const cl_int res)
	{
		return (res == CL_SUCCESS);
	}

protected:
	static void oclFatal(const cl_int res)
	{
		if (!oclError(res))
		{
			std::ostringstream ss; ss << "opencl error: " << errorString(res);
			throw std::runtime_error(ss.str());
		}
	}
};

class platform : oclObject
{
private:
	struct deviceDesc
	{
		cl_platform_id platform_id;
		cl_device_id device_id;
		std::string name;
	};
	std::vector<deviceDesc> _devices;

protected:
	void findDevices(const bool gpu)
	{
		cl_uint num_platforms;
		cl_platform_id platforms[64];
		oclFatal(clGetPlatformIDs(64, platforms, &num_platforms));

		for (cl_uint p = 0; p < num_platforms; ++p)
		{
			char platformName[1024]; oclFatal(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, platformName, nullptr));

			cl_uint num_devices;
			cl_device_id devices[64];
			if (oclError(clGetDeviceIDs(platforms[p], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, 64, devices, &num_devices)))
			{
				for (cl_uint d = 0; d < num_devices; ++d)
				{
					char deviceName[1024]; oclFatal(clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, deviceName, nullptr));
					char deviceVendor[1024]; oclFatal(clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, 1024, deviceVendor, nullptr));

					std::ostringstream ss; ss << "device '" << deviceName << "', vendor '" << deviceVendor << "', platform '" << platformName << "'";
					deviceDesc device;
					device.platform_id = platforms[p];
					device.device_id = devices[d];
					device.name = ss.str();
					_devices.push_back(device);
				}
			}
		}
	}

public:
	platform()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl platform." << std::endl;
		pio::display(ss.str());
#endif
		findDevices(true);
		if (_devices.empty()) findDevices(false);
	}

	platform(const cl_platform_id platform_id, const cl_device_id device_id)
	{
		char platformName[1024]; oclFatal(clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1024, platformName, nullptr));
		char deviceName[1024]; oclFatal(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, deviceName, nullptr));
		char deviceVendor[1024]; oclFatal(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 1024, deviceVendor, nullptr));

		std::ostringstream ss; ss << "device '" << deviceName << "', vendor '" << deviceVendor << "', platform '" << platformName << "'";
		deviceDesc device;
		device.platform_id = platform_id;
		device.device_id = device_id;
		device.name = ss.str();
		_devices.push_back(device);
	}

public:
	virtual ~platform()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Delete ocl platform." << std::endl;
		pio::display(ss.str());
#endif
	}

public:
	size_t getDeviceCount() const { return _devices.size(); }

public:
	size_t displayDevices() const
	{
		const size_t n = _devices.size();
		std::ostringstream ss;
		for (size_t i = 0; i < n; ++i)
		{
			ss << i << " - " << _devices[i].name << "." << std::endl;
		}
		ss << std::endl;
		pio::print(ss.str());
		return n;
	}

public:
	cl_platform_id getPlatform(const size_t d) const { return _devices[d].platform_id; }
	cl_device_id getDevice(const size_t d) const { return _devices[d].device_id; }
};

class device : oclObject
{
private:
	const cl_platform_id _platform;
	const cl_device_id _device;
#if defined(ocl_debug)
	const size_t _d;
#endif
	bool _profile = false;
#if defined(__APPLE__)
	bool _isSync = true;
#else
	bool _isSync = false;
#endif
	size_t _syncCount = 0;
	cl_ulong _localMemSize = 0;
	size_t _maxWorkGroupSize = 0;
	cl_ulong _timerResolution = 0;
	cl_context _context = nullptr;
	cl_command_queue _queueF = nullptr;
	cl_command_queue _queueP = nullptr;
	cl_command_queue _queue = nullptr;
	cl_program _program = nullptr;

	enum class EVendor { Unknown, NVIDIA, AMD, INTEL };

	struct profile
	{
		std::string name;
		size_t count;
		cl_ulong time;

		profile() {}
		profile(const std::string & name) : name(name), count(0), time(0) {}
	};
	std::map<cl_kernel, profile> _profileMap;

public:
	device(const platform & parent, const size_t d, const bool verbose) : _platform(parent.getPlatform(d)), _device(parent.getDevice(d))
#if defined(ocl_debug)
		, _d(d)
#endif
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl device " << d << "." << std::endl;
		pio::display(ss.str());
#endif

		char deviceName[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_NAME, 1024, deviceName, nullptr));
		char deviceVendor[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_VENDOR, 1024, deviceVendor, nullptr));
		char deviceVersion[1024]; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_VERSION, 1024, deviceVersion, nullptr));
		char driverVersion[1024]; oclFatal(clGetDeviceInfo(_device, CL_DRIVER_VERSION, 1024, driverVersion, nullptr));

		cl_uint computeUnits; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr));
		cl_uint maxClockFrequency; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr));
		cl_ulong memSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr));
		cl_ulong memCacheSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(memCacheSize), &memCacheSize, nullptr));
		cl_uint memCacheLineSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(memCacheLineSize), &memCacheLineSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(_localMemSize), &_localMemSize, nullptr));
		cl_ulong memConstSize; oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(memConstSize), &memConstSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_maxWorkGroupSize), &_maxWorkGroupSize, nullptr));
		oclFatal(clGetDeviceInfo(_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(_timerResolution), &_timerResolution, nullptr));

		if (verbose)
		{
			std::ostringstream ssd;
			ssd << "Running on device '" << deviceName << "', vendor '" << deviceVendor
				<< "', version '" << deviceVersion << "', driver '" << driverVersion << "'";
			// ssd << computeUnits << " compUnits @ " << maxClockFrequency << "MHz, mem=" << (memSize >> 20) << "MB, cache="
			// 	<< (memCacheSize >> 10) << "kB, cacheLine=" << memCacheLineSize << "B, localMem=" << (_localMemSize >> 10)
			// 	<< "kB, constMem=" << (memConstSize >> 10) << "kB, maxWorkGroup=" << _maxWorkGroupSize << "." << std::endl;
			pio::print(ssd.str());
		}

		const cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)_platform, 0 };
		cl_int err_cc;
		_context = clCreateContext(contextProperties, 1, &_device, nullptr, nullptr, &err_cc);
		oclFatal(err_cc);
		cl_int err_ccq;
		_queueF = clCreateCommandQueue(_context, _device, 0, &err_ccq);
		_queueP = clCreateCommandQueue(_context, _device, CL_QUEUE_PROFILING_ENABLE, &err_ccq);
		_queue = _queueF;	// default queue is fast
		oclFatal(err_ccq);

		if (getVendor(deviceVendor) != EVendor::NVIDIA) _isSync = true;
	}

public:
	virtual ~device()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Delete ocl device " << _d << "." << std::endl;
		pio::display(ss.str());
#endif
		oclFatal(clReleaseCommandQueue(_queueP));
		oclFatal(clReleaseCommandQueue(_queueF));
		oclFatal(clReleaseContext(_context));
	}

public:
	size_t getMaxWorkGroupSize() const { return _maxWorkGroupSize; }
	size_t getLocalMemSize() const { return _localMemSize; }
	size_t getTimerResolution() const { return _timerResolution; }

private:
	static EVendor getVendor(const std::string & vendorString)
	{
		std::string lVendorString; lVendorString.resize(vendorString.size());
		std::transform(vendorString.begin(), vendorString.end(), lVendorString.begin(), [](char c){ return std::tolower(c); });

		if (strstr(lVendorString.c_str(), "nvidia") != nullptr) return EVendor::NVIDIA;
		if (strstr(lVendorString.c_str(), "amd") != nullptr) return EVendor::AMD;
		if (strstr(lVendorString.c_str(), "advanced micro devices") != nullptr) return EVendor::AMD;
		if (strstr(lVendorString.c_str(), "intel") != nullptr) return EVendor::INTEL;
		// must be tested after 'Intel' because 'ati' is in 'Intel(R) Corporation' string
		if (strstr(lVendorString.c_str(), "ati") != nullptr) return EVendor::AMD;
		return EVendor::Unknown;
	}

public:
	void resetProfiles()
	{
		for (auto it : _profileMap)
		{
			profile & prof = _profileMap[it.first];	// it.first is not a reference!
			prof.count = 0;
			prof.time = 0;
		}
	}

public:
	cl_ulong getProfileTime() const
	{
		cl_ulong time = 0;
		for (auto it : _profileMap) time += it.second.time;
		return time;
	}

public:
	void displayProfiles(const size_t count) const
	{
		cl_ulong ptime = 0;
		for (auto it : _profileMap) ptime += it.second.time;
		ptime /= count;

		std::ostringstream ss;
		for (auto it : _profileMap)
		{
			const profile & prof = it.second;
			if (prof.count != 0)
			{
				const size_t ncount = prof.count / count;
				const cl_ulong ntime = prof.time / count;
				ss << "- " << prof.name << ": " << ncount << ", " << std::setprecision(3)
					<< ntime * 100.0 / ptime << " %, " << ntime << " (" << (ntime / ncount) << ")" << std::endl;
			}
		}
		pio::display(ss.str());
	}

public:
	void setProfiling(const bool enable)
	{
		_profile = enable;
		_queue = enable ? _queueP : _queueF;
		resetProfiles();
	}

public:
	void loadProgram(const std::string & programSrc)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Load ocl program." << std::endl;
		pio::display(ss.str());
#endif
		const char * src[1]; src[0] = programSrc.c_str();
		cl_int err_cpws;
		_program = clCreateProgramWithSource(_context, 1, src, nullptr, &err_cpws);
		oclFatal(err_cpws);

		char pgmOptions[1024];
		strcpy(pgmOptions, "");
#if defined(ocl_debug)
		strcat(pgmOptions, " -cl-nv-verbose");
#endif
		const cl_int err = clBuildProgram(_program, 1, &_device, pgmOptions, nullptr, nullptr);

#if !defined(ocl_debug)
		if (err != CL_SUCCESS)
#endif		
		{
			size_t logSize; clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
			if (logSize > 2)
			{
				std::vector<char> buildLog(logSize + 1);
				clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
				buildLog[logSize] = '\0';
				std::ostringstream ss; ss << buildLog.data() << std::endl;
				pio::print(ss.str());
#if defined(ocl_debug)
				std::ofstream fileOut("pgm.log"); 
				fileOut << buildLog.data() << std::endl;
				fileOut.close();
#endif
			}
		}

		oclFatal(err);

#if defined(ocl_debug)
		size_t binSize; clGetProgramInfo(_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, nullptr);
		std::vector<char> binary(binSize);
		clGetProgramInfo(_program, CL_PROGRAM_BINARIES, sizeof(char *), &binary, nullptr);
		std::ofstream fileOut("pgm.bin", std::ios::binary);
		fileOut.write(binary.data(), std::streamsize(binSize));
		fileOut.close();
#endif	
	}

public:
	void clearProgram()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Clear ocl program." << std::endl;
		pio::display(ss.str());
#endif
		oclFatal(clReleaseProgram(_program));
		_program = nullptr;
		_profileMap.clear();
	}

private:
	void _sync()
	{
		_syncCount = 0;
		oclFatal(clFinish(_queue));
	}

public:
	cl_mem _createBuffer(const cl_mem_flags flags, const size_t size, const bool clear = true) const
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(_context, flags, size, nullptr, &err);
		oclFatal(err);
		if (clear)
		{
			std::vector<uint8_t> ptr(size);
			for (size_t i = 0; i < size; ++i) ptr[i] = 0x00;	// debug 0xff;
			oclFatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, 0, size, ptr.data(), 0, nullptr, nullptr));
		}
		return mem;
	}

public:
	static void _releaseBuffer(cl_mem & mem)
	{
		if (mem != nullptr)
		{
			oclFatal(clReleaseMemObject(mem));
			mem = nullptr;
		}
	}

protected:
	void _readBuffer(cl_mem & mem, void * const ptr, const size_t size)
	{
		_sync();
		oclFatal(clEnqueueReadBuffer(_queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr));
	}

protected:
	void _writeBuffer(cl_mem & mem, const void * const ptr, const size_t size)
	{
		_sync();
		oclFatal(clEnqueueWriteBuffer(_queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr));
	}

protected:
	cl_kernel _createKernel(const char * const kernelName)
	{
		cl_int err;
		cl_kernel kernel = clCreateKernel(_program, kernelName, &err);
		oclFatal(err);
		_profileMap[kernel] = profile(kernelName);
		return kernel;
	}

protected:
	static void _releaseKernel(cl_kernel & kernel)
	{
		if (kernel != nullptr)
		{
			oclFatal(clReleaseKernel(kernel));
			kernel = nullptr;
		}		
	}

protected:
	static void _setKernelArg(cl_kernel kernel, const cl_uint arg_index, const size_t arg_size, const void * const arg_value)
	{
#if !defined(ocl_fast_exec) || defined(ocl_debug)
		cl_int err =
#endif
		clSetKernelArg(kernel, arg_index, arg_size, arg_value);
#if !defined(ocl_fast_exec) || defined(ocl_debug)
		oclFatal(err);
#endif
	}

protected:
	void _executeKernel(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		if (!_profile)
		{
#if !defined(ocl_fast_exec) || defined(ocl_debug)
			cl_int err =
#endif
			clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, nullptr);
#if !defined(ocl_fast_exec) || defined(ocl_debug)
			oclFatal(err);
#endif
			if (_isSync)
			{
				++_syncCount;
				if (_syncCount == 1024) _sync();
			}
		}
		else
		{
			_sync();
			cl_event evt;
			oclFatal(clEnqueueNDRangeKernel(_queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, &evt));
			cl_ulong dt = 0;
			if (clWaitForEvents(1, &evt) == CL_SUCCESS)
			{
				cl_ulong start, end;
				cl_int err_s = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
				cl_int err_e = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
				if ((err_s == CL_SUCCESS) && (err_e == CL_SUCCESS)) dt = end - start;
			}
			clReleaseEvent(evt);

			profile & prof = _profileMap[kernel];
			prof.count++;
			prof.time += dt;
		}
	}
};
