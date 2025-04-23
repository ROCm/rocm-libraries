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

#include "kernels.h"
#if _WIN32
#include <Windows.h>
#else
#include <chrono>
#endif

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for publishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
	// register kernels
    ERROR_CHECK_STATUS(amd_media_decode_publish(context));
    ERROR_CHECK_STATUS(amd_media_encode_publish(context));
	return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for unpublishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxUnpublishKernels(vx_context context)
{
	// TBD: remove kernels
	return VX_SUCCESS;
}

vx_node createMediaNode(vx_graph graph, const char * kernelName, vx_reference params[], vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByName(context, kernelName);
    if (kernel) {
        node = vxCreateGenericNode(graph, kernel);
        if (node) {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++) {
                if (params[p]) {
                    status = vxSetParameterByIndex(node, p, params[p]);
                    if (status != VX_SUCCESS) {
                        vxAddLogEntry((vx_reference)graph, status, "stitchCreateNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
                        vxReleaseNode(&node);
                        node = 0;
                        break;
                    }
                }
            }
        }
        else {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "Failed to create node with kernel %s\n", kernelName);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "failed to retrieve kernel %s\n", kernelName);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

//////////////////////////////////////////////////////////////////////
//! \brief The common utility functions.

void av_log_callback(void* ptr, int level, const char* fmt, va_list vl)
{
	//vprintf(fmt, vl);
}

vx_status initialize_ffmpeg()
{
  static bool initialized = false;
  if (!initialized) { // make sure to initialize it only once
    initialized = true;
    av_log_set_callback(av_log_callback);
    av_log_set_level(AV_LOG_ERROR);
    //av_register_all();        // this is depracated from FFMpeg >4.0 
  }
  return VX_SUCCESS;
}

uint8_t * aligned_alloc(size_t size)
{
	uint8_t * buf = new uint8_t[size + 32 + sizeof(uint8_t *)];
	if (!buf) return nullptr;
	uint8_t * ptr = buf + (32 - (((intptr_t)buf) & 31)) + sizeof(uint8_t *);
	*(uint8_t **)(ptr - sizeof(uint8_t *)) = buf;
	return ptr;
}

void aligned_free(uint8_t * ptr)
{
	uint8_t * buf = *(uint8_t **)(ptr - sizeof(uint8_t *));
	delete[] buf;
}

int64_t ClockCounter()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	return v.QuadPart;
#else
	return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

int64_t ClockFrequency()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceFrequency(&v);
	return v.QuadPart;
#else
	return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
#endif
}

int64_t GetTimeInMicroseconds()
{
	static int64_t freq = 0; if (!freq) freq = ClockFrequency();
	return ClockCounter() * 1000000 / ClockFrequency();
}
