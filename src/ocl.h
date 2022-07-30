/*
Copyright 2022, Yves Gallot

genefer22 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#define CL_TARGET_OPENCL_VERSION 110
#if defined (__APPLE__)
	#include <OpenCL/cl.h>
	#include <OpenCL/cl_ext.h>
#else
	#include <CL/cl.h>
#endif

#include <algorithm>

class Splitter
{
private:
	struct Part
	{
		size_t size;
		unsigned int p[8];
	};

	const bool b256, b1024;
	const size_t mMax;
	size_t size;
	Part part[32];

private:
	void Split(const size_t m, const size_t i, Part & p)
	{
		if (b1024 && (m >= 10 + 5))
		{
			p.p[i] = 10;
			Split(m - 10, i + 1, p);
		}
		if (b256 && (m >= 8 + 5))
		{
			p.p[i] = 8;
			Split(m - 8, i + 1, p);
		}
		if (m >= 6 + 5)
		{
			p.p[i] = 6;
			Split(m - 6, i + 1, p);
		}

		if ((5 <= m) && (m <= mMax))
		{
			for (size_t k = 0; k < i; ++k) part[size].p[k] = p.p[k];
			part[size].p[i] = (unsigned int)m;
			part[size].size = i + 1;
			size++;
		}
	}

private:
	static size_t Log2(const size_t n) { size_t r = 0; for (size_t m = 1; m < n; m *= 2) ++r; return r; }

public:
	Splitter(const size_t n, const size_t chunk256, const size_t chunk1024, const size_t sizeofRNS, const size_t mSquareMax,
		const cl_ulong localMemSize, const size_t maxWorkGroupSize) :
		b256(maxWorkGroupSize >= (256 / 4) * chunk256),
		b1024((maxWorkGroupSize >= (1024 / 4) * chunk1024) && (localMemSize / sizeofRNS >= 1024 * chunk1024)),
		mMax(std::min(mSquareMax, std::min(Log2((size_t)(localMemSize / sizeofRNS)), Log2(maxWorkGroupSize * 4))))
	{
		size = 0;
		Part p;
		Split(n, 0, p);
	}

	size_t GetSize() const { return size; }
	size_t GetPartSize(const size_t i) const { return part[i].size; }
	unsigned int GetPart(const size_t i, const size_t j) const { return part[i].p[j]; }
};

static void GetDevice(size_t & id, const cl_uint num_platforms, const cl_platform_id * const platforms, cl_device_id *device, cl_platform_id *platform)
{
	for (cl_uint p = 0; p != num_platforms; ++p)
	{
		cl_uint num_devices;
		cl_device_id devices[16];
		clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 16, devices, &num_devices);

		for (cl_uint d = 0; d != num_devices; ++d)
		{
			if (id == 0)
			{
				*platform = platforms[p];
				*device = devices[d];
			}
			++id;
		}
	}
}

static cl_int GetDevice(cl_device_id * device, cl_platform_id * platform)
{
	cl_uint num_platforms;
	cl_platform_id platforms[64];
	cl_int err = clGetPlatformIDs(64, platforms, &num_platforms);
	if ((err != CL_SUCCESS) || (num_platforms == 0))
	{
		return CL_DEVICE_NOT_FOUND;
	}

	size_t id = 0;
	GetDevice(id, num_platforms, platforms, device, platform);

	return err;
}

class HostProgram
{
private:
	const bool verbose;
	bool gpuSync;
	cl_int syncCount;
	cl_platform_id platform;
	cl_device_id device;
	cl_program program;
	cl_context context;
	cl_command_queue queue;
	bool selfTuning;
	cl_ulong timerResolution;
	cl_ulong localMemSize;
	size_t maxWorkGroupSize;

protected:
	const size_t n;
	const unsigned int ln;
	cl_int error;

public:
	HostProgram(const size_t size, const unsigned int lgSize, const bool verbose) :
		verbose(verbose), gpuSync(true), syncCount(0), platform(nullptr), selfTuning(size >= 1 * 1024), timerResolution(0), maxWorkGroupSize(0),
		n(size), ln(lgSize), error(CL_SUCCESS)
	{
		cl_int err = GetDevice(&device, &platform);
		if (err != CL_SUCCESS) { error = err; return; }

		clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(timerResolution), &timerResolution, nullptr);
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr);
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
	}

	~HostProgram()
	{
	}

public:
	void IsOK() const
	{
		if (error != CL_SUCCESS)
		{
			throw std::runtime_error("OpenCL");
		}
	}

public:
	void Sync()
	{
		syncCount = 0;
		error |= clFinish(queue);
	}

protected:
	bool SelfTuning() const { return selfTuning; }
	cl_ulong GetTimerResolution() const { return timerResolution; }
	cl_ulong GetLocalMemSize() const { return localMemSize; }
	size_t GetMaxWorkGroupSize() const { return maxWorkGroupSize; }

protected:
	cl_kernel CreateKernel(const char * kernelName)
	{
		cl_int err;
		cl_kernel kern = clCreateKernel(program, kernelName, &err);
		if (err != CL_SUCCESS)
		{
			if (verbose) fprintf(stderr, "Error: cannot create kernel '%s'.\n", kernelName);
			error = err;
			return nullptr;
		}
		return kern;
	}

protected:
	cl_mem CreateBuffer(cl_mem_flags flags, size_t size)
	{
		cl_int err;
		cl_mem mem = clCreateBuffer(context, flags, size, nullptr, &err);
		if (err != CL_SUCCESS)
		{
			if (verbose) fprintf(stderr, "Error: cannot create buffer.\n");
			error = err;
			return nullptr;
		}
		return mem;
	}

protected:
	void ReadBuffer(cl_mem mem, size_t size, void * ptr)
	{
		error |= clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr);
	}

protected:
	void WriteBuffer(cl_mem mem, size_t size, const void * ptr)
	{
		error |= clEnqueueWriteBuffer(queue, mem, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr);
	}

protected:
	cl_int BuildProgram(const char * const programSrc)
	{
		cl_int err;

		cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
		context = clCreateContext(contextProperties, 1, &device, nullptr, nullptr, &err);
		if (err != CL_SUCCESS) { if (verbose) fprintf(stderr, "Error: cannot create context.\n"); error = err; return err; }
		queue = clCreateCommandQueue(context, device, selfTuning ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
		if (err != CL_SUCCESS) { if (verbose) fprintf(stderr, "Error: cannot create command queue.\n"); error = err; return err; }

		const size_t programSrcSize = strlen(programSrc);
		char * const src = new char[programSrcSize + 1];
		strcpy(src, programSrc);
		program = clCreateProgramWithSource(context, 1, (const char **)&src, &programSrcSize, &err);
		delete[] src;
		if (err != CL_SUCCESS) { if (verbose) fprintf(stderr, "Error: cannot create program.\n"); error = err; return err; }

		char curPlatformName[1024];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1024, curPlatformName, nullptr);

		char pgmOptions[1024];
		strcpy(pgmOptions, "");
		// if (my_strcasestr(curPlatformName, "nvidia") != nullptr)
		// {
		// 	gpuSync = false;
		// 	strcat(pgmOptions, "-DNVIDIA_PTX=1");
		// 	if (verbose) strcat(pgmOptions, " -cl-nv-verbose");
		// }

		err = clBuildProgram(program, 1, &device, pgmOptions, nullptr, nullptr);

		if (err != CL_SUCCESS)
		{
			error = err;
			fprintf(stderr, "Error: build program failed.\n");
		}

		if ((err != CL_SUCCESS) || verbose)
		{
			size_t logSize; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
			if (logSize > 0)
			{
				char * buildLog = new char[logSize + 1];
				clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, nullptr);
				buildLog[logSize] = '\0';
				fprintf(stderr, "%s\n", buildLog);
				delete[] buildLog;
			}
		}

		if (verbose)
		{
			size_t binSize; clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, nullptr);
			char * binary = new char[binSize];
			clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char *), &binary, nullptr);
			FILE * const fp = fopen("genefer.acl", "wb");
			if (fp != nullptr)
			{
				fwrite(binary, 1, binSize, fp);
				fclose(fp);
			}
			delete[] binary;
		}

		return err;
	}

protected:
	void ClearProgram()
	{
		if (error == CL_SUCCESS)
		{
			clReleaseProgram(program);

			clReleaseCommandQueue(queue);
			clReleaseContext(context);
		}
	}

protected:
	void Execute(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		error |= clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, nullptr);
		if (gpuSync)
		{
			++syncCount;
			if (syncCount == 1024) Sync();
		}
	}

protected:
	cl_ulong ExecuteProfiling(cl_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize)
	{
		cl_ulong dt = 0;

		cl_event evt;
		cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, (localWorkSize == 0) ? nullptr : &localWorkSize, 0, nullptr, &evt);
		if (err != CL_SUCCESS) return (cl_ulong)0x0001000000000000ull; // excessive amount of time => this function will not be selected
		if (clWaitForEvents(1, &evt) == CL_SUCCESS)
		{
			cl_ulong start;
			if (clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr) == CL_SUCCESS)
			{
				cl_ulong end;
				if (clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr) == CL_SUCCESS)
				{
					dt = end - start;
				}
			}
		}
		clReleaseEvent(evt);
		Sync();
		return dt;
	}
};
