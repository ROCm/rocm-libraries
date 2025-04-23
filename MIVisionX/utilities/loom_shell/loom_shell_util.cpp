/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "loom_shell_util.h"
#if __APPLE__
#include <cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <algorithm>
#include <vector>
#include <map>

#if _WIN32
#include <windows.h>
#undef min
#undef max
#else
#include <chrono>
#include <strings.h>
#define _strnicmp strncasecmp
#define _stricmp  strcasecmp
#endif

// local cache for OpenCL buffer management
static std::map<cl_context, cl_command_queue> globalClCtx2CmdqMap;
static std::map<cl_mem, cl_command_queue> globalClMem2CmdqMap;
static std::map<cl_mem, vx_uint32> globalClMem2SizeMap;

//! \brief The macro for fread error checking and reporting.
#define ERROR_CHECK_FREAD_(call,value) {size_t retVal = (call); if(retVal != (size_t)value) { Error("ERROR: fread call expected to return [ %d elements ] but returned [ %d elements ] at " __FILE__ "#%d\n", (int)value, (int)retVal, __LINE__); }  }

int64_t GetClockCounter()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	return v.QuadPart;
#else
	return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

int64_t GetClockFrequency()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceFrequency(&v);
	return v.QuadPart;
#else
	return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
#endif
}

bool GetEnvVariable(const char * name, char * value, size_t valueSize)
{
#if _WIN32
	DWORD len = GetEnvironmentVariableA(name, value, (DWORD)valueSize);
	value[valueSize - 1] = 0;
	return (len > 0) ? true : false;
#else
	const char * v = getenv(name);
	if (v) {
		strncpy(value, v, valueSize);
		value[valueSize - 1] = 0;
	}
	return v ? true : false;
#endif
}

static void Message(const char * format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

static int Error(const char * format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	if (format[strlen(format) - 1] != '\n') printf("\n");
	fflush(stdout);
	return -1;
}

static cl_command_queue GetCmdqCached(cl_mem mem)
{
	cl_command_queue cmdq = nullptr;
	if (globalClMem2CmdqMap.find(mem) != globalClMem2CmdqMap.end()) {
		cmdq = globalClMem2CmdqMap[mem];
	}
	else {
		// get OpenCL context from mem object
		cl_context opencl_context;
		cl_int err = clGetMemObjectInfo(mem, CL_MEM_CONTEXT, sizeof(opencl_context), &opencl_context, nullptr);
		if (err) { Error("ERROR: clGetMemObjectInfo(*,CL_MEM_CONTEXT) failed (%d)", err); return nullptr; }
		if (globalClCtx2CmdqMap.find(opencl_context) != globalClCtx2CmdqMap.end()) {
			cmdq = globalClCtx2CmdqMap[opencl_context];
		}
		else {
			// First, get the size of device list data
			size_t device_list_size_;
			cl_int err; 
  			err = clGetContextInfo(opencl_context, CL_CONTEXT_DEVICES, 0, NULL, &device_list_size_);
			if (err) { Error("ERROR: clGetContextInfo(*,CL_CONTEXT_DEVICES) failed (%d)", err); return nullptr; }
			// create OpenCL cmd_queue using first device in the context
			cl_device_id *devices_; devices_ = (cl_device_id *)malloc(device_list_size_);
  			if(devices_==NULL){ Error("ERROR: cl_device_id is NULL"); return nullptr; }
			err = clGetContextInfo(opencl_context, CL_CONTEXT_DEVICES, device_list_size_, devices_, nullptr);
			if (err) { Error("ERROR: clGetContextInfo(*,CL_CONTEXT_DEVICES) failed (%d)", err); return nullptr; }
#if defined(CL_VERSION_2_0)
			cmdq = clCreateCommandQueueWithProperties(opencl_context, devices_[0], NULL, &err);
#else
			cmdq = clCreateCommandQueue(opencl_context, devices_[0], 0, &err);
#endif
			if (!cmdq) { Error("ERROR: clCreateCommandQueueWithProperties: failed (%d)", err); return nullptr; }
			// save the command-queue
			globalClCtx2CmdqMap[opencl_context] = cmdq;
			globalClMem2CmdqMap[mem] = cmdq;
			// free mem
			free(devices_);
		}
	}
	return cmdq;
}

vx_status initializeBuffer(cl_mem mem, vx_uint32 size, cl_int pattern)
{
	cl_command_queue cmdq = GetCmdqCached(mem); if (!cmdq) return -1;
	cl_int status = clEnqueueFillBuffer(cmdq, mem, &pattern, sizeof(cl_int), 0, size, 0, NULL, NULL);
	if (status) return status;

	return VX_SUCCESS;
}

static vx_status ReleaseCmdqCached(cl_context opencl_context)
{
	if (globalClCtx2CmdqMap.find(opencl_context) != globalClCtx2CmdqMap.end()) {
		// remove the entry from cache
		cl_command_queue cmdq = globalClCtx2CmdqMap[opencl_context];
		globalClCtx2CmdqMap.erase(opencl_context);
		// clear the cmdq entries from mem object
		for (auto it = globalClMem2CmdqMap.begin(); it != globalClMem2CmdqMap.end(); ) {
			if (it->second == cmdq) it = globalClMem2CmdqMap.erase(it);
			else it++;
		}
		// release command queue
		cl_int err = clReleaseCommandQueue(cmdq);
		if (err) return Error("ERROR: clReleaseCommandQueue() failed (%d)", err);
	}
	return VX_SUCCESS;
}

static vx_status ReleaseCmdqCached(cl_mem mem)
{
	cl_command_queue cmdq = nullptr;
	if (globalClMem2CmdqMap.find(mem) != globalClMem2CmdqMap.end()) {
		// remove the entry from cache
		cmdq = globalClMem2CmdqMap[mem];
		globalClMem2CmdqMap.erase(mem);
		// check if used by other mem objects
		for (auto it = globalClMem2CmdqMap.begin(); it != globalClMem2CmdqMap.end(); it++) {
			if (it->second == cmdq) {
				cmdq = nullptr;
				break;
			}
		}
	}
	if (cmdq) {
		// clear the cmdq entry from contexts
		cl_context opencl_context = nullptr;
		for (auto it = globalClCtx2CmdqMap.begin(); it != globalClCtx2CmdqMap.end(); it++) {
			if (it->second == cmdq) {
				opencl_context = it->first;
				break;
			}
		}
		if (opencl_context) {
			return ReleaseCmdqCached(opencl_context);
		}
	}
	return VX_SUCCESS;
}

vx_status ClearCmdqCache()
{
	for (auto it = globalClCtx2CmdqMap.begin(); it != globalClCtx2CmdqMap.end(); it++) {
		// release command queue
		cl_int err = clReleaseCommandQueue(it->second);
		if (err) return Error("ERROR: clReleaseCommandQueue() failed (%d)", err);
		it->second = nullptr;
	}
	globalClCtx2CmdqMap.clear();
	globalClMem2CmdqMap.clear();
	globalClMem2SizeMap.clear();
	return VX_SUCCESS;
}

vx_status run(ls_context context, vx_uint32 frameCount)
{
	vx_status status = VX_SUCCESS;
	double clk2msec = 1000.0 / GetClockFrequency();
	double msec_first = 0, msec_sum = 0, msec_min = 0, msec_max = 0, msec_count = 0;
	vx_uint32 count = 0;
	for (vx_uint32 i = 0; frameCount == 0 || i < frameCount; i++, count++) {
		int64_t clk = GetClockCounter();
		status = lsScheduleFrame(context);
		if (status == VX_ERROR_GRAPH_ABANDONED) break;
		if (status) return Error("ERROR: lsScheduleFrame() failed (%d) @iter:%d", status, i);
		status = lsWaitForCompletion(context);
		if (status == VX_ERROR_GRAPH_ABANDONED) break;
		if (status) return Error("ERROR: lsWaitForCompletion) failed (%d) @iter:%d", status, i);
		double msec = clk2msec * (GetClockCounter() - clk);
		if (i == 0) {
			msec_first = msec;
		}
		else if (i == 1) {
			msec_sum = msec_min = msec_max = msec;
			msec_count = 1;
		}
		else {
			msec_sum += msec;
			msec_count += 1;
			msec_min = (msec < msec_min) ? msec : msec_min;
			msec_max = (msec > msec_max) ? msec : msec_max;
		}
	}
	if (status) Message("WARNING: run: execution abandoned after %d frames\n", count);
	else        Message("OK: run: executed for %d frames\n", count);
	if (count == 1) {
		Message("OK: run: Time: %7.3lf ms\n", msec_first);
	}
	else if (msec_count > 0) {
		Message("OK: run: Time: %7.3lf ms (min); %7.3lf ms (avg); %7.3lf ms (max); %7.3lf ms (1st-frame) of %d frames\n", msec_min, msec_sum / msec_count, msec_max, msec_first, count);
	}
	return VX_SUCCESS;
}

vx_status runParallel(ls_context * context, vx_uint32 contextCount, vx_uint32 frameCount)
{
	vx_status status = VX_SUCCESS;
	double clk2msec = 1000.0 / GetClockFrequency();
	double msec_first = 0, msec_sum = 0, msec_min = 0, msec_max = 0, msec_count = 0;
	vx_uint32 count = 0;
	for (vx_uint32 i = 0; frameCount == 0 || i < frameCount; i++, count++) {
		int64_t clk = GetClockCounter();
		for (vx_uint32 j = 0; j < contextCount; j++) {
			if (context[j]) {
				status = lsScheduleFrame(context[j]);
				if (status == VX_ERROR_GRAPH_ABANDONED) break;
				if (status) return Error("ERROR: lsScheduleFrame(context[%d]) failed (%d) @iter:%d", j, status, i);
			}
		}
		if (status) break;
		for (vx_uint32 j = 0; j < contextCount; j++) {
			if (context[j]) {
				status = lsWaitForCompletion(context[j]);
				if (status == VX_ERROR_GRAPH_ABANDONED) break;
				if (status) return Error("ERROR: lsWaitForCompletion(context[%d]) failed (%d) @iter:%d", j, status, i);
			}
		}
		if (status) break;
		double msec = clk2msec * (GetClockCounter() - clk);
		if (i == 0) {
			msec_first = msec;
		}
		else if (i == 1) {
			msec_sum = msec_min = msec_max = msec;
			msec_count = 1;
		}
		else {
			msec_sum += msec;
			msec_count += 1;
			msec_min = (msec < msec_min) ? msec : msec_min;
			msec_max = (msec > msec_max) ? msec : msec_max;
		}
	}
	if (status) Message("WARNING: runParallel: execution abandoned after %d frames\n", count);
	else        Message("OK: runParallel: executed for %d frames\n", count);
	if (count == 1) {
		Message("OK: runParallel: Time: %7.3lf ms\n", msec_first);
	}
	else if (msec_count > 0) {
		Message("OK: runParallel: Time: %7.3lf ms (min); %7.3lf ms (avg); %7.3lf ms (max); %7.3lf ms (1st-frame) of %d frames\n", msec_min, msec_sum / msec_count, msec_max, msec_first, count);
	}
	return VX_SUCCESS;
}

vx_status showOutputConfig(ls_context context)
{
	vx_df_image buffer_format = 0;
	vx_uint32 buffer_width = 0, buffer_height = 0;
	vx_status status = lsGetOutputConfig(context, &buffer_format, &buffer_width, &buffer_height);
	if (status) return status;
	Message("..showOutputConfig: format:%4.4s buffer:%dx%d\n", &buffer_format, buffer_width, buffer_height);
	return VX_SUCCESS;
}

vx_status showCameraConfig(ls_context context)
{
	vx_uint32 num_rows = 0, num_cols = 0;
	vx_df_image buffer_format = 0;
	vx_uint32 buffer_width = 0, buffer_height = 0;
	vx_status status = lsGetCameraConfig(context, &num_rows, &num_cols, &buffer_format, &buffer_width, &buffer_height);
	if (status) return Error("ERROR: lsGetCameraConfig() failed (%d)", status);
	Message("..showCameraConfig: cameras:%dx%d format:%4.4s buffer:%dx%d\n", num_rows, num_cols, &buffer_format, buffer_width, buffer_height);
	for (vx_uint32 index = 0; index < num_rows*num_cols; index++) {
		camera_params camera_par = { 0 };
		status = lsGetCameraParams(context, index, &camera_par);
		if (status) return Error("ERROR: lsGetCameraParams(*,%d,*) failed (%d)", index, status);
		char lensType[64];
		if (camera_par.lens.lens_type == ptgui_lens_rectilinear) strcpy(lensType, "ptgui_lens_rectilinear");
		else if (camera_par.lens.lens_type == ptgui_lens_fisheye_ff) strcpy(lensType, "ptgui_lens_fisheye_ff");
		else if (camera_par.lens.lens_type == ptgui_lens_fisheye_circ) strcpy(lensType, "ptgui_lens_fisheye_circ");
		else if (camera_par.lens.lens_type == adobe_lens_rectilinear) strcpy(lensType, "adobe_lens_rectilinear");
		else if (camera_par.lens.lens_type == adobe_lens_fisheye) strcpy(lensType, "adobe_lens_fisheye");
		else {
			Message("ERROR: lsExportConfiguration: loom_shell: (ovr) lens_type=%d not supported\n", camera_par.lens.lens_type);
			return VX_ERROR_NOT_SUPPORTED;
		}
		Message("..showCameraConfig: index#%d {{%8.3f,%8.3f,%8.3f,%.0f,%.0f,%.0f},{%.3f,%.1f,%.1f,%.3f,%.3f,%s,%f,%f,%f}};\n", index,
			camera_par.focal.yaw, camera_par.focal.pitch, camera_par.focal.roll,
			camera_par.focal.tx, camera_par.focal.ty, camera_par.focal.tz,
			camera_par.lens.hfov, camera_par.lens.haw, camera_par.lens.r_crop,
			camera_par.lens.du0, camera_par.lens.dv0,
			lensType, camera_par.lens.k1, camera_par.lens.k2, camera_par.lens.k3);
	}
	return VX_SUCCESS;
}

vx_status showOverlayConfig(ls_context context)
{
	vx_uint32 num_rows = 0, num_cols = 0;
	vx_df_image buffer_format = 0;
	vx_uint32 buffer_width = 0, buffer_height = 0;
	vx_status status = lsGetOverlayConfig(context, &num_rows, &num_cols, &buffer_format, &buffer_width, &buffer_height);
	if (status) return Error("ERROR: lsGetOverlayConfig() failed (%d)", status);
	Message("..showOverlayConfig: overlays:%dx%d format:%4.4s buffer:%dx%d\n", num_rows, num_cols, &buffer_format, buffer_width, buffer_height);
	for (vx_uint32 index = 0; index < num_rows*num_cols; index++) {
		camera_params camera_par = { 0 };
		status = lsGetOverlayParams(context, index, &camera_par);
		if (status) return Error("ERROR: lsGetOverlayParams(*,%d,*) failed (%d)", index, status);
		char lensType[64];
		if (camera_par.lens.lens_type == ptgui_lens_rectilinear) strcpy(lensType, "ptgui_lens_rectilinear");
		else if (camera_par.lens.lens_type == ptgui_lens_fisheye_ff) strcpy(lensType, "ptgui_lens_fisheye_ff");
		else if (camera_par.lens.lens_type == ptgui_lens_fisheye_circ) strcpy(lensType, "ptgui_lens_fisheye_circ");
		else if (camera_par.lens.lens_type == adobe_lens_rectilinear) strcpy(lensType, "adobe_lens_rectilinear");
		else if (camera_par.lens.lens_type == adobe_lens_fisheye) strcpy(lensType, "adobe_lens_fisheye");
		else {
			Message("ERROR: lsExportConfiguration: loom_shell: (ovr) lens_type=%d not supported\n", camera_par.lens.lens_type);
			return VX_ERROR_NOT_SUPPORTED;
		}
		Message("..showOverlayConfig: index#%d {{%8.3f,%8.3f,%8.3f,%.0f,%.0f,%.0f},{%.3f,%.1f,%.1f,%.3f,%.3f,%s,%f,%f,%f}};\n", index,
			camera_par.focal.yaw, camera_par.focal.pitch, camera_par.focal.roll,
			camera_par.focal.tx, camera_par.focal.ty, camera_par.focal.tz,
			camera_par.lens.hfov, camera_par.lens.haw, camera_par.lens.r_crop,
			camera_par.lens.du0, camera_par.lens.dv0,
			lensType, camera_par.lens.k1, camera_par.lens.k2, camera_par.lens.k3);
	}
	return VX_SUCCESS;
}

vx_status showRigParams(ls_context context)
{
	rig_params rig_par = { 0 };
	vx_status status = lsGetRigParams(context, &rig_par);
	if (status) return Error("ERROR: showRigParams() failed (%d)", status);
	Message("..showRigParams: {%.3f,%.3f,%.3f,%.3f}\n", rig_par.yaw, rig_par.pitch, rig_par.roll, rig_par.d);
	return VX_SUCCESS;
}

vx_status showCameraModule(ls_context context)
{
	char module[256], kernelName[64], kernelArguments[1024];
	vx_status status = lsGetCameraModule(context, module, sizeof(module), kernelName, sizeof(kernelName), kernelArguments, sizeof(kernelArguments));
	if (status) return Error("ERROR: lsGetCameraModule() failed (%d)", status);
	Message("..showCameraModule: \"%s\" \"%s\" \"%s\"\n", module, kernelName, kernelArguments);
	return VX_SUCCESS;
}

vx_status showOutputModule(ls_context context)
{
	char module[256], kernelName[64], kernelArguments[1024];
	vx_status status = lsGetOutputModule(context, module, sizeof(module), kernelName, sizeof(kernelName), kernelArguments, sizeof(kernelArguments));
	if (status) return Error("ERROR: lsGetOutputModule() failed (%d)", status);
	Message("..showOutputModule: \"%s\" \"%s\" \"%s\"\n", module, kernelName, kernelArguments);
	return VX_SUCCESS;
}

vx_status showOverlayModule(ls_context context)
{
	char module[256], kernelName[64], kernelArguments[1024];
	vx_status status = lsGetOverlayModule(context, module, sizeof(module), kernelName, sizeof(kernelName), kernelArguments, sizeof(kernelArguments));
	if (status) return Error("ERROR: lsGetOverlayModule() failed (%d)", status);
	Message("..showOverlayModule: \"%s\" \"%s\" \"%s\"\n", module, kernelName, kernelArguments);
	return VX_SUCCESS;
}

vx_status showViewingModule(ls_context context)
{
	char module[256], kernelName[64], kernelArguments[1024];
	vx_status status = lsGetViewingModule(context, module, sizeof(module), kernelName, sizeof(kernelName), kernelArguments, sizeof(kernelArguments));
	if (status) return Error("ERROR: lsGetViewingModule() failed (%d)", status);
	Message("..showViewingModule: \"%s\" \"%s\" \"%s\"\n", module, kernelName, kernelArguments);
	return VX_SUCCESS;
}

vx_status showCameraBufferStride(ls_context context)
{
	vx_uint32 buffer_stride_in_bytes = 0;
	vx_status status = lsGetCameraBufferStride(context, &buffer_stride_in_bytes);
	if (status) return Error("ERROR: lsGetCameraBufferStride() failed (%d)", status);
	Message("..showCameraBufferStride: buffer_stride_in_bytes: %d\n", buffer_stride_in_bytes);
	return VX_SUCCESS;
}

vx_status showOutputBufferStride(ls_context context)
{
	vx_uint32 buffer_stride_in_bytes = 0;
	vx_status status = lsGetOutputBufferStride(context, &buffer_stride_in_bytes);
	if (status) return Error("ERROR: lsGetOutputBufferStride() failed (%d)", status);
	Message("..showOutputBufferStride: buffer_stride_in_bytes: %d\n", buffer_stride_in_bytes);
	return VX_SUCCESS;
}

vx_status showOverlayBufferStride(ls_context context)
{
	vx_uint32 buffer_stride_in_bytes = 0;
	vx_status status = lsGetOverlayBufferStride(context, &buffer_stride_in_bytes);
	if (status) return Error("ERROR: lsGetOverlayBufferStride() failed (%d)", status);
	Message("..showOverlayBufferStride: buffer_stride_in_bytes: %d\n", buffer_stride_in_bytes);
	return VX_SUCCESS;
}

vx_status showConfiguration(ls_context context, const char * exportType)
{
	return lsExportConfiguration(context, exportType, nullptr);
}

vx_status createBuffer(cl_context opencl_context, vx_uint32 size, cl_mem * mem)
{
	cl_int err;
	*mem = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE, size, NULL, &err);
	if (!*mem) return Error("ERROR: clCreateBuffer(...,%d,...): failed (%d)", size, err);
	globalClMem2SizeMap[*mem] = size;
	return VX_SUCCESS;
}

vx_status releaseBuffer(cl_mem * mem)
{
	if (ReleaseCmdqCached(*mem) != VX_SUCCESS) return -1;
	cl_int status = clReleaseMemObject(*mem);
	if (status) return Error("ERROR: clReleaseMemObject() failed (%d)", status);
	globalClMem2SizeMap.erase(*mem);
	*mem = nullptr;
	return VX_SUCCESS;
}

vx_status createOpenCLContext(const char * platform, const char * device, cl_context * opencl_context)
{
	// get OpenCL platform ID
	cl_platform_id platform_id[16]; cl_uint num_platform_id = 0;
	cl_int err = clGetPlatformIDs(16, platform_id, &num_platform_id);
	if (err) return Error("ERROR: clGetPlatformIDs failed (%d)", err);
	if (num_platform_id < 1) return -1;
	bool found = false; cl_uint platform_index = 0; char name[256] = "invalid";
	if (platform && platform[0] >= '0' && platform[0] <= '9') {
		platform_index = (cl_uint)atoi(platform);
		if (platform_index < num_platform_id) {
			err = clGetPlatformInfo(platform_id[platform_index], CL_PLATFORM_VENDOR, sizeof(name), name, NULL);
			if (err) return Error("ERROR: clGetPlatformInfo failed (%d)", err);
			found = true;
		}
	}
	else {
		for (platform_index = 0; platform_index < num_platform_id; platform_index++) {
			err = clGetPlatformInfo(platform_id[platform_index], CL_PLATFORM_VENDOR, sizeof(name), name, NULL);
			if (err) return Error("ERROR: clGetPlatformInfo failed (%d)", err);
			if (!platform || strstr(name, platform)) {
				found = true;
				break;
			}
		}
	}
	if (!found) return Error("ERROR: specified platform '%s' doesn't exist in this system", platform);
	// check if DirectGMA can be is supported
	bool direct_gma_available = false;
#if !__APPLE__
	clEnqueueWaitSignalAMD_fn clEnqueueWaitSignalAMD = (clEnqueueWaitSignalAMD_fn)clGetExtensionFunctionAddressForPlatform(platform_id[platform_index], "clEnqueueWaitSignalAMD");
	clEnqueueWriteSignalAMD_fn clEnqueueWriteSignalAMD = (clEnqueueWriteSignalAMD_fn)clGetExtensionFunctionAddressForPlatform(platform_id[platform_index], "clEnqueueWriteSignalAMD");
	clEnqueueMakeBuffersResidentAMD_fn clEnqueueMakeBuffersResidentAMD = (clEnqueueMakeBuffersResidentAMD_fn)clGetExtensionFunctionAddressForPlatform(platform_id[platform_index], "clEnqueueMakeBuffersResidentAMD");
	if (clEnqueueWaitSignalAMD && clEnqueueWriteSignalAMD && clEnqueueMakeBuffersResidentAMD) {
		direct_gma_available = true;
	}
#endif
	Message("..OpenCL platform#%d: %s %s\n", platform_index, name, direct_gma_available ? "[DirectGMA-OK]" : "[DirectGMA-No]");
	// get OpenCL device
	cl_device_id device_id[16]; cl_uint num_device_id = 0;
	err = clGetDeviceIDs(platform_id[platform_index], CL_DEVICE_TYPE_GPU, 16, device_id, &num_device_id);
	if (err) return Error("ERROR: clGetDeviceIDs failed (%d)", err);
	if (num_device_id < 1) return Error("ERROR: clGetDeviceIDs returned ZERO device IDs");
	found = false; cl_uint device_index = 0; strcpy(name, "invalid");
	if (device && device[0] >= '0' && device[0] <= '9') {
		device_index = (cl_uint)atoi(device);
		if (device_index < num_device_id) {
			clGetDeviceInfo(device_id[device_index], CL_DEVICE_NAME, sizeof(name), name, NULL);
			if (err) return Error("ERROR: clGetDeviceInfo failed (%d)", err);
			found = true;
		}
	}
	else {
		for (device_index = 0; device_index < num_device_id; device_index++) {
			clGetDeviceInfo(device_id[device_index], CL_DEVICE_NAME, sizeof(name), name, NULL);
			if (err) return Error("ERROR: clGetDeviceInfo failed (%d)", err);
			if (!device || !strcmp(name, device)) {
				found = true;
				break;
			}
		}
	}
	if (!found) return Error("ERROR: specified device doesn't exist in this system: (%s)", device);
	// get device name and check if DirectGMA is supported
	char ext[4096] = { 0 };
	err = clGetDeviceInfo(device_id[device_index], CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, NULL);
	if (err) return Error("ERROR: clGetDeviceInfo failed (%d)", err);
	if (!strstr(ext, "cl_amd_bus_addressable_memory"))
		direct_gma_available = false;
	printf("..OpenCL device#%d: %s %s\n", device_index, name, direct_gma_available ? "[DirectGMA-OK]" : "[DirectGMA-No]");
	// create OpenCL context
	cl_context_properties ctx_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id[platform_index], 0 };
	*opencl_context = clCreateContext(ctx_properties, 1, &device_id[device_index], NULL, NULL, &err);
	if (!*opencl_context) return Error("ERROR: clCreateContext() failed (%d)", err);
	return VX_SUCCESS;
}

vx_status releaseOpenCLContext(cl_context * opencl_context)
{
	if (ReleaseCmdqCached(*opencl_context) != VX_SUCCESS) return -1;
	cl_int status = clReleaseContext(*opencl_context);
	if (status) return Error("ERROR: clReleaseContext() failed (%d)", status);
	*opencl_context = nullptr;
	return VX_SUCCESS;
}

vx_status createOpenVXContext(vx_context * openvx_context)
{
	vx_context context = vxCreateContext();
	vx_status status = vxGetStatus((vx_reference)context);
	if (status) return Error("ERROR: vxCreateContext() failed (%d)", status);
	*openvx_context = context;
	return status;
}

vx_status releaseOpenVXContext(vx_context * openvx_context)
{
	vx_status status = vxReleaseContext(openvx_context);
	if (status) return Error("ERROR: vxReleaseContext() failed (%d)", status);
	return VX_SUCCESS;
}

vx_status setGlobalAttribute(vx_uint32 offset, float value)
{
	vx_status status = lsGlobalSetAttributes(offset, 1, &value);
	if (status) return Error("ERROR: setGlobalAttribute(%d,%.3f) failed (%d)", offset, value, status);
	return VX_SUCCESS;
}

vx_status showGlobalAttributes(vx_uint32 offset, vx_uint32 count)
{
	float * attr = new float[count]; if (!attr) return Error("ERROR: new[] failed");
	vx_status status = lsGlobalGetAttributes(offset, count, attr);
	if (status) { delete[] attr; return Error("ERROR: showGlobalAttributes(%d,%d,*) failed (%d)", offset, count, status); }
	for (vx_uint32 i = 0; i < count; i++) {
		Message("attr[%4d] = %13.6f (global)\n", i, attr[i]);
	}
	delete[] attr;
	return VX_SUCCESS;
}

vx_status loadGlobalAttributes(vx_uint32 offset, vx_uint32 count, const char * fileName)
{
	float * attr = new float[count]; if (!attr) return Error("ERROR: new[] failed");
	FILE * fp = fopen(fileName, "r"); if (!fp) { delete[] attr; return Error("ERROR: unable to open: %s", fileName); }
	for (vx_uint32 i = 0; i < count; i++) {
		if (fscanf(fp, "%f", &attr[i]) != 1) { delete[] attr; fclose(fp); return Error("ERROR: loadAttributes: too few values in %s", fileName); }
	}
	fclose(fp);
	vx_status status = lsGlobalSetAttributes(offset, count, attr);
	if (status) { delete[] attr; return Error("ERROR: loadGlobalAttributes(%d,%d,*) failed (%d)", offset, count, status); }
	delete[] attr;
	Message("OK: global attributes %d..%d (count %d) loaded from \"%s\"\n", offset, offset + count - 1, count, fileName);
	return VX_SUCCESS;
}

vx_status saveGlobalAttributes(vx_uint32 offset, vx_uint32 count, const char * fileName)
{
	float * attr = new float[count]; if (!attr) return Error("ERROR: new[] failed");
	vx_status status = lsGlobalGetAttributes(offset, count, attr);
	if (status) { delete[] attr; return Error("ERROR: saveGlobalAttributes(%d,%d,*) failed (%d)", offset, count, status); }
	FILE * fp = fopen(fileName, "w"); if (!fp) { delete[] attr; return Error("ERROR: unable to create: %s", fileName); }
	for (vx_uint32 i = 0; i < count; i++) {
		fprintf(fp, "%13.6f\n", attr[i]);
	}
	fclose(fp);
	delete[] attr;
	Message("OK: global attributes %d..%d (count %d) saved into \"%s\"\n", offset, offset + count - 1, count, fileName);
	return VX_SUCCESS;
}

vx_status setAttribute(ls_context context, vx_uint32 offset, float value)
{
	vx_status status = lsSetAttributes(context, offset, 1, &value);
	if (status) return Error("ERROR: setAttribute(*,%d,%.3f) failed (%d)", offset, value, status);
	return VX_SUCCESS;
}

vx_status showAttributes(ls_context context, vx_uint32 offset, vx_uint32 count)
{
	float * attr = new float[count]; if (!attr) return Error("ERROR: new[] failed");
	vx_status status = lsGetAttributes(context, offset, count, attr);
	if (status) { delete[] attr; return Error("ERROR: lsGetAttributes(*,%d,%d,*) failed (%d)", offset, count, status); }
	for (vx_uint32 i = 0; i < count; i++) {
		Message("attr[%4d] = %13.6f\n", i, attr[i]);
	}
	delete[] attr;
	return VX_SUCCESS;
}

vx_status loadAttributes(ls_context context, vx_uint32 offset, vx_uint32 count, const char * fileName)
{
	float * attr = new float[count]; if (!attr) return Error("ERROR: new[] failed");
	FILE * fp = fopen(fileName, "r"); if (!fp) { delete[] attr; return Error("ERROR: unable to open: %s", fileName); }
	for (vx_uint32 i = 0; i < count; i++) {
		if (fscanf(fp, "%f", &attr[i]) != 1) { delete[] attr; fclose(fp); return Error("ERROR: loadAttributes: too few values in %s", fileName); }
	}
	fclose(fp);
	vx_status status = lsSetAttributes(context, offset, count, attr);
	if (status) { delete[] attr; return Error("ERROR: lsSetAttributes(*,%d,%d,*) failed (%d)", offset, count, status); }
	delete[] attr;
	return VX_SUCCESS;
	Message("OK: attributes %d..%d (count %d) loaded from \"%s\"\n", offset, offset + count - 1, count, fileName);
}

vx_status saveAttributes(ls_context context, vx_uint32 offset, vx_uint32 count, const char * fileName)
{
	float * attr = new float[count]; if (!attr) return Error("ERROR: new[] failed");
	vx_status status = lsGetAttributes(context, offset, count, attr);
	if (status) { delete[] attr; return Error("ERROR: lsGetAttributes(*,%d,%d,*) failed (%d)", offset, count, status); }
	FILE * fp = fopen(fileName, "w"); if (!fp) { delete[] attr; return Error("ERROR: unable to create: %s", fileName); }
	for (vx_uint32 i = 0; i < count; i++) {
		fprintf(fp, "%13.6f\n", attr[i]);
	}
	fclose(fp);
	delete[] attr;
	Message("OK: attributes %d..%d (count %d) saved into \"%s\"\n", offset, offset + count - 1, count, fileName);
	return VX_SUCCESS;
}

vx_status loadBufferFromImage(cl_mem mem, const char * fileName, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes)
{
	return loadBufferFromMultipleImages(mem, fileName, 1, 1, buffer_format, buffer_width, buffer_height, stride_in_bytes);
}

vx_status saveBufferToImage(cl_mem mem, const char * fileName, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes)
{
	return saveBufferToMultipleImages(mem, fileName, 1, 1, buffer_format, buffer_width, buffer_height, stride_in_bytes);
	return VX_SUCCESS;
}

vx_status loadBufferFromMultipleImages(cl_mem mem, const char * fileName, vx_uint32 num_rows, vx_uint32 num_cols, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes)
{
	const char * fileNameExt = fileName + strlen(fileName) - 4;
	if ((buffer_format != VX_DF_IMAGE_RGB && buffer_format != VX_DF_IMAGE_RGBX) || !!_stricmp(fileNameExt, ".bmp")) {
		return Error("ERROR: loadBufferFromImage/loadBufferFromMultipleImages: supports only RGB images and BMP files");
	}
	cl_command_queue cmdq = GetCmdqCached(mem); if (!cmdq) return -1;
	vx_uint32 size = globalClMem2SizeMap[mem];
	cl_int err;
	unsigned char * img = (unsigned char *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err);
	if (err) return Error("ERROR: clEnqueueMapBuffer() failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	if (stride_in_bytes == 0) {
		if (buffer_format == VX_DF_IMAGE_RGBX) stride_in_bytes = buffer_width * 4;
		else stride_in_bytes = buffer_width * 3;
	}
	vx_uint32 width = buffer_width / num_cols;
	vx_uint32 height = buffer_height / num_rows;
	unsigned char * buf = nullptr;
	for (vx_uint32 row = 0, camIndex = 0; row < num_rows; row++) {
		for (vx_uint32 column = 0; column < num_cols; column++, camIndex++) {
			char bmpFileName[256] = { 0 };
			if (strstr(fileName, "%")) snprintf(bmpFileName, sizeof(bmpFileName), fileName, camIndex);
			else {
				vx_uint32 pos = 0;
				for (vx_uint32 skipItem = 0; skipItem < camIndex; skipItem++) {
					while (fileName[pos] && fileName[pos] != ',')
						pos++;
					if (fileName[pos] == ',')
						pos++;
				}
				if (!fileName[pos]) {
					return Error("ERROR: loadBufferFromMultipleImages: missing '%'/',' missing in the fileName: %s\n", fileName);
				}
				for (vx_uint32 i = 0; fileName[pos + i] && fileName[pos + i] != ','; i++)
					bmpFileName[i] = fileName[pos + i];
			}
			FILE * fp = fopen(bmpFileName, "rb"); if (!fp) return Error("ERROR: unable to open: %s", bmpFileName);
			if (buffer_format == VX_DF_IMAGE_RGB) {
				unsigned short bmpHeader[54 / 2];
				ERROR_CHECK_FREAD_(fread(bmpHeader, 1, sizeof(bmpHeader), fp), sizeof(bmpHeader));
				if (width != (vx_uint32)bmpHeader[9] || height != (vx_uint32)bmpHeader[11]) {
					return Error("ERROR: The BMP should be %dx%d: got %dx%d in %s\n", width, height, bmpHeader[9], bmpHeader[11], bmpFileName);
				}
				vx_uint32 size = (bmpHeader[2] << 16) + bmpHeader[1] - bmpHeader[5];
				if (bmpHeader[0] != 0x4d42 || bmpHeader[5] != 54 || bmpHeader[13] != 1 || bmpHeader[14] != 24 ||
					((size != width * height * 3) && (size != (width * height * 3 + 1)) && (size != (width * height * 3 + 2))))
				{
					return Error("ERROR: The BMP format is not supported or dimensions doesn't match buffer size in %s\n", bmpFileName);
				}
				if (!buf) { buf = new unsigned char[width * 3]; if (!buf) return Error("ERROR: new[%d] failed", width * 3); }
				for (vx_uint32 y = 0; y < height; y++) {
					ERROR_CHECK_FREAD_(fread(buf, 1, width * 3, fp), width * 3);
					unsigned char * q = img + (row * height + height - 1 - y)*stride_in_bytes + column * width * 3, *p = buf;
					for (vx_uint32 x = 0; x < width; x++, p += 3, q += 3) {
						q[0] = p[2];
						q[1] = p[1];
						q[2] = p[0];
					}
				}
			}
			else if (buffer_format == VX_DF_IMAGE_RGBX) {
				unsigned short bmpHeader[122 / 2];
				ERROR_CHECK_FREAD_(fread(bmpHeader, 1, 54, fp),54);
				if (bmpHeader[5] == 122){
					ERROR_CHECK_FREAD_(fread(&bmpHeader[54 / 2], 1, 122 - 54, fp), 122 - 54);
				}
				if (width != (vx_uint32)bmpHeader[9] || height != (vx_uint32)bmpHeader[11]) {
					return Error("ERROR: The BMP should be %dx%d: got %dx%d in %s\n", width, height, bmpHeader[9], bmpHeader[11], bmpFileName);
				}
				vx_uint32 size = (bmpHeader[2] << 16) + bmpHeader[1] - bmpHeader[5];
				if (bmpHeader[0] != 0x4d42 || bmpHeader[13] != 1 || bmpHeader[14] != 32 || (size != width * height * 4)) {
					return Error("ERROR: The BMP format is not supported or dimensions doesn't match buffer size in %s\n", bmpFileName);
				}
				if ((bmpHeader[5] != 54 && bmpHeader[5] != 122) || (bmpHeader[5] == 122 && bmpHeader[27] != 0x00ff)) {
					return Error("ERROR: The BMP format is not supported in %s\n", bmpFileName);
				}
				for (vx_uint32 y = 0; y < height; y++) {
					unsigned char * q = img + (row * height + height - 1 - y)*stride_in_bytes + column * width * 4;
					ERROR_CHECK_FREAD_(fread(q, 1, width * 4, fp),width*4);
					if (bmpHeader[27] != 0x00ff) {
						// convert BGRA to RGBA
						for (vx_uint32 x = 0; x < width; x++) {
							vx_uint32 color = *(vx_uint32 *)&q[x << 2];
							color = (color & 0xff00ff00) | ((color & 0x00ff0000) >> 16) | ((color & 0x000000ff) << 16);
							*(vx_uint32 *)&q[x << 2] = color;
						}
					}
				}
			}
			fclose(fp);
			Message("OK: loaded %s\n", bmpFileName);
		}
	}
	if(buf) delete[] buf;
	err = clEnqueueUnmapMemObject(cmdq, mem, img, 0, NULL, NULL);
	if (err) return Error("ERROR: clEnqueueUnmapMemObject failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	return VX_SUCCESS;
}

vx_status saveBufferToMultipleImages(cl_mem mem, const char * fileName, vx_uint32 num_rows, vx_uint32 num_cols, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes)
{
	const char * fileNameExt = fileName + strlen(fileName) - 4;
	if ((buffer_format != VX_DF_IMAGE_RGB && buffer_format != VX_DF_IMAGE_RGBX && buffer_format != VX_DF_IMAGE_UYVY && buffer_format != VX_DF_IMAGE_YUYV) || !!_stricmp(fileNameExt, ".bmp")) {
		return Error("ERROR: saveBufferToImage/saveBufferToMultipleImages: supports only RGB images and BMP files");
	}
	cl_command_queue cmdq = GetCmdqCached(mem); if (!cmdq) return -1;
	vx_uint32 size = globalClMem2SizeMap[mem];
	cl_int err;
	unsigned char * img = (unsigned char *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &err);
	if (err) return Error("ERROR: clEnqueueMapBuffer() failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	if (stride_in_bytes == 0) {
		if (buffer_format == VX_DF_IMAGE_RGBX) stride_in_bytes = buffer_width * 4;
		else if (buffer_format == VX_DF_IMAGE_UYVY || buffer_format == VX_DF_IMAGE_YUYV) stride_in_bytes = buffer_width * 2;
		else stride_in_bytes = buffer_width * 3;
	}
	vx_uint32 width = buffer_width / num_cols;
	vx_uint32 height = buffer_height / num_rows;
	if (buffer_format == VX_DF_IMAGE_RGB) {
		unsigned char * buf = new unsigned char[width * 3]; if (!buf) return Error("ERROR: new[%d] failed", width * 3);
		for (vx_uint32 row = 0, camIndex = 0; row < num_rows; row++) {
			for (vx_uint32 column = 0; column < num_cols; column++, camIndex++) {
				char bmpFileName[256] = { 0 };
				if (strstr(fileName, "%")) snprintf(bmpFileName, sizeof(bmpFileName), fileName, camIndex);
				else {
					vx_uint32 pos = 0;
					for (vx_uint32 skipItem = 0; skipItem < camIndex; skipItem++) {
						while (fileName[pos] && fileName[pos] != ',')
							pos++;
						if (fileName[pos] == ',')
							pos++;
					}
					if (!fileName[pos]) {
						return Error("ERROR: saveBufferToMultipleImages: missing '%'/',' missing in the fileName: %s\n", fileName);
					}
					for (vx_uint32 i = 0; fileName[pos + i] && fileName[pos + i] != ','; i++)
						bmpFileName[i] = fileName[pos + i];
				}
				FILE * fp = fopen(bmpFileName, "wb"); if (!fp) return Error("ERROR: unable to create: %s", bmpFileName);
				vx_uint32 size = 3 * width * height;
				short bmpHeader[54 / 2] = {
					0x4d42, (short)((size + 54) & 0xffff), (short)((size + 54) >> 16), 0, 0, 54, 0, 40, 0,
					(short)width, 0, (short)height, 0, 1, 24, 0, 0,
					(short)(size & 0xffff), (short)(size >> 16), 0, 0, 0, 0, 0, 0, 0, 0
				};
				fwrite(bmpHeader, 1, sizeof(bmpHeader), fp);
				for (vx_uint32 y = 0; y < height; y++) {
					unsigned char * p = img + (height * row + height - 1 - y)*stride_in_bytes + column * width * 3, *q = buf;
					for (vx_uint32 x = 0; x < width; x++, p += 3, q += 3) {
						q[0] = p[2];
						q[1] = p[1];
						q[2] = p[0];
					}
					fwrite(buf, 1, width * 3, fp);
				}
				fclose(fp);
				Message("OK: created %s\n", bmpFileName);
			}
		}
		delete[] buf;
	}
	else if (buffer_format == VX_DF_IMAGE_RGBX) {
		for (vx_uint32 row = 0, camIndex = 0; row < num_rows; row++) {
			for (vx_uint32 column = 0; column < num_cols; column++, camIndex++) {
				char bmpFileName[256]; snprintf(bmpFileName, sizeof(bmpFileName), fileName, camIndex);
				FILE * fp = fopen(bmpFileName, "wb"); if (!fp) return Error("ERROR: unable to create: %s", bmpFileName);
				vx_uint32 size = 4 * width * height;
				short bmpHeader[122 / 2] = {
					0x4d42, (short)((size + 122) & 0xffff), (short)((size + 122) >> 16), 0, 0, 122, 0, 108, 0,
					(short)width, 0, (short)height, 0, 1, 32, 3, 0,
					(short)(size & 0xffff), (short)(size >> 16), 0, 0, 0, 0, 0, 0, 0, 0,
					0x00ff, 0x0000, (short)0xff00, 0x0000, 0x0000, 0x00ff, 0x0000, (short)0xff00,
					0x6e20, 0x5769, 0
				};
				fwrite(bmpHeader, 1, sizeof(bmpHeader), fp);
				for (vx_uint32 y = 0; y < height; y++) {
					unsigned char * p = img + (height * row + height - 1 - y)*stride_in_bytes + column * width * 4;
					fwrite(p, 1, width * 4, fp);
				}
				fclose(fp);
				Message("OK: created %s\n", bmpFileName);
			}
		}
	}
	else if (buffer_format == VX_DF_IMAGE_UYVY || buffer_format == VX_DF_IMAGE_YUYV) {
		unsigned char * buf = new unsigned char[width * 3]; if (!buf) return Error("ERROR: new[%d] failed", width * 3);
		for (vx_uint32 row = 0, camIndex = 0; row < num_rows; row++) {
			for (vx_uint32 column = 0; column < num_cols; column++, camIndex++) {
				char bmpFileName[256]; snprintf(bmpFileName, sizeof(bmpFileName), fileName, camIndex);
				FILE * fp = fopen(bmpFileName, "wb"); if (!fp) return Error("ERROR: unable to create: %s", bmpFileName);
				vx_uint32 size = 3 * width * height;
				short bmpHeader[54 / 2] = {
					0x4d42, (short)((size + 54) & 0xffff), (short)((size + 54) >> 16), 0, 0, 54, 0, 40, 0,
					(short)width, 0, (short)height, 0, 1, 24, 0, 0,
					(short)(size & 0xffff), (short)(size >> 16), 0, 0, 0, 0, 0, 0, 0, 0
				};
				fwrite(bmpHeader, 1, sizeof(bmpHeader), fp);
				for (vx_uint32 y = 0; y < height; y++) {
					unsigned char * p = img + (height * row + height - 1 - y)*stride_in_bytes + column * width * 2, *q = buf;
					if (buffer_format == VX_DF_IMAGE_UYVY) {
						for (vx_uint32 x = 0; x < width; x += 2, p += 4, q += 6) {
							vx_int32 u = p[0] - 128, y0 = p[1], v = p[2] - 128, y1 = p[3];
							q[0] = (unsigned char)std::max(0.0f, std::min(255.0f, y0 + (1.770f * u)));
							q[1] = (unsigned char)std::max(0.0f, std::min(255.0f, y0 - (0.344f * u) - (0.714f * v)));
							q[2] = (unsigned char)std::max(0.0f, std::min(255.0f, y0 + (1.403f * v)));
							q[3] = (unsigned char)std::max(0.0f, std::min(255.0f, y1 + (1.770f * u)));
							q[4] = (unsigned char)std::max(0.0f, std::min(255.0f, y1 - (0.344f * u) - (0.714f * v)));
							q[5] = (unsigned char)std::max(0.0f, std::min(255.0f, y1 + (1.403f * v)));
						}
					}
					else {
						for (vx_uint32 x = 0; x < width; x += 2, p += 4, q += 6) {
							vx_int32 u = p[1] - 128, y0 = p[0], v = p[3] - 128, y1 = p[2];
							q[0] = (unsigned char)std::max(0.0f, std::min(255.0f, y0 + (1.770f * u)));
							q[1] = (unsigned char)std::max(0.0f, std::min(255.0f, y0 - (0.344f * u) - (0.714f * v)));
							q[2] = (unsigned char)std::max(0.0f, std::min(255.0f, y0 + (1.403f * v)));
							q[3] = (unsigned char)std::max(0.0f, std::min(255.0f, y1 + (1.770f * u)));
							q[4] = (unsigned char)std::max(0.0f, std::min(255.0f, y1 - (0.344f * u) - (0.714f * v)));
							q[5] = (unsigned char)std::max(0.0f, std::min(255.0f, y1 + (1.403f * v)));
						}
					}
					fwrite(buf, 1, width * 3, fp);
				}
				fclose(fp);
				Message("OK: created %s\n", bmpFileName);
			}
		}
		delete[] buf;
	}
	err = clEnqueueUnmapMemObject(cmdq, mem, img, 0, NULL, NULL);
	if (err) return Error("ERROR: clEnqueueUnmapMemObject failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	return VX_SUCCESS;
}

vx_status loadBuffer(cl_mem mem, const char * fileName, vx_uint32 offset)
{
	cl_command_queue cmdq = GetCmdqCached(mem); if (!cmdq) return -1;
	vx_uint32 size = globalClMem2SizeMap[mem];
	cl_int err;
	unsigned char * img = (unsigned char *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err);
	if (err) return Error("ERROR: clEnqueueMapBuffer() failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	FILE * fp = fopen(fileName, "rb"); if (!fp) return Error("ERROR: unable to open: %s", fileName);
	fseek(fp, offset, SEEK_SET);
	ERROR_CHECK_FREAD_(fread(img, 1, size, fp), size);
	fclose(fp);
	err = clEnqueueUnmapMemObject(cmdq, mem, img, 0, NULL, NULL);
	if (err) return Error("ERROR: clEnqueueUnmapMemObject failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	Message("OK: loaded %d bytes from %s\n", size, fileName);
	return VX_SUCCESS;
}

vx_status saveBuffer(cl_mem mem, const char * fileName, vx_uint32 flags)
{
	bool append = false;
	if (flags&1) {
		append = true;
	}
	else if (fileName[0] == '+') {
		fileName++;
		append = true;
	}
	cl_command_queue cmdq = GetCmdqCached(mem); if (!cmdq) return -1;
	vx_uint32 size = globalClMem2SizeMap[mem];
	cl_int err;
	unsigned char * img = (unsigned char *)clEnqueueMapBuffer(cmdq, mem, CL_TRUE, CL_MAP_READ, 0, size, 0, NULL, NULL, &err);
	if (err) return Error("ERROR: clEnqueueMapBuffer() failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	FILE * fp = fopen(fileName, append ? "ab" : "wb"); if (!fp) return Error("ERROR: unable to %s: %s", append ? "append" : "create", fileName);
	fwrite(img, 1, size, fp);
	fclose(fp);
	err = clEnqueueUnmapMemObject(cmdq, mem, img, 0, NULL, NULL);
	if (err) return Error("ERROR: clEnqueueUnmapMemObject failed (%d)", err);
	err = clFinish(cmdq); if (err) return Error("ERROR: clFinish() failed (%d)", err);
	Message("OK: saved %d bytes into %s\n", size, fileName);
	return VX_SUCCESS;
}

vx_status loadExpCompGains(ls_context stitch, size_t num_entries, const char * fileName)
{
	bool textFileFormat = (strlen(fileName) > 4 && !_stricmp(&fileName[strlen(fileName) - 4], ".txt")) ? true : false;
	FILE * fp = fopen(fileName, textFileFormat ? "r" : "rb");
	if (!fp) return Error("ERROR: unable to open: %s", fileName);
	std::vector<vx_float32> gains(num_entries);
	if (!textFileFormat) {
		size_t count = fread(gains.data(), sizeof(vx_float32), gains.size(), fp);
		if (count != gains.size()) {
			fclose(fp);
			return Error("ERROR: loadExpCompGains: missing entries in %s: got only %d\n", fileName, (vx_uint32)count);
		}
	}
	else {
		for (vx_size i = 0; i < gains.size(); i++) {
			vx_float32 f;
			if (fscanf(fp, "%g", &f) != 1) {
				fclose(fp);
				return Error("ERROR: loadExpCompGains: missing entries in %s: got only %d\n", fileName, (vx_uint32)i);
			}
			gains[i] = f;
		}
	}
	fclose(fp);
	vx_status status = lsSetExpCompGains(stitch, num_entries, gains.data());
	if (status) return Error("ERROR: lsSetExpCompGains: failed (%d)\n", status);
	Message("OK: loadExpCompGains: loaded %d values from %s\n", (vx_uint32)gains.size(), fileName);
	return VX_SUCCESS;
}

vx_status saveExpCompGains(ls_context stitch, size_t num_entries, const char * fileName)
{
	std::vector<vx_float32> gains(num_entries);
	vx_status status = lsGetExpCompGains(stitch, num_entries, gains.data());
	if (status) return Error("ERROR: lsGetExpCompGains: failed (%d)\n", status);
	FILE * fp = fopen(fileName, "w"); if (!fp) return Error("ERROR: unable to create: %s", fileName);
	for (vx_size i = 0; i < gains.size(); i++) {
		fprintf(fp, "%12g\n", gains[i]);
	}
	fclose(fp);
	Message("OK: saveExpCompGains: saved %d values into %s\n", (vx_uint32)gains.size(), fileName);
	return VX_SUCCESS;
}

vx_status showExpCompGains(ls_context stitch, size_t num_entries)
{
	std::vector<vx_float32> gains(num_entries);
	vx_status status = lsGetExpCompGains(stitch, num_entries, gains.data());
	if (status) return Error("ERROR: lsGetExpCompGains: failed (%d)\n", status);
	for (vx_size i = 0; i < gains.size(); i++) {
		Message(" %12g", gains[i]);
	}
	Message("\n");
	return VX_SUCCESS;
}

vx_status loadBlendWeights(ls_context stitch, const char * fileName)
{
	FILE * fp = fopen(fileName, "rb");
	if (!fp) return Error("ERROR: unable to open: %s", fileName);
	fseek(fp, 0, SEEK_END);  long size = ftell(fp); fseek(fp, 0, SEEK_SET);
	vx_uint8 * buf = new vx_uint8[size];
	if (!buf) {
		fclose(fp);
		return Error("ERROR: alloc(%d) failed", size);
	}
	ERROR_CHECK_FREAD_(fread(buf, 1, size, fp), size);
	fclose(fp);
	vx_status status = lsSetBlendWeights(stitch, buf, size);
	if (status) return Error("ERROR: lsSetBlendWeights(*,*,%d): failed (%d)\n", size, status);
	Message("OK: loaded %d bytes from %s as blend weights\n", size, fileName);
	delete[] buf;
	return VX_SUCCESS;
}
