/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _VX_KERNELS_CUSTOM_H_
#define _VX_KERNELS_CUSTOM_H_

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>
#include <iostream>
#include "vx_amd_custom.h"

//////////////////////////////////////////////////////////////////////
// SHARED_PUBLIC - shared sybols for export
#if _WIN32
#define SHARED_PUBLIC extern "C" __declspec(dllexport)
#else
#define SHARED_PUBLIC extern "C" __attribute__ ((visibility ("default")))
#endif

#define VX_LIBRARY_CUSTOM   6

#define ERROR_CHECK_STATUS(call) {vx_status status = call; if(status != VX_SUCCESS) return status;}
#ifndef ERROR_CHECK_CUSTOM_STATUS
#define ERROR_CHECK_CUSTOM_STATUS(call) if(call) { \
    std::cerr << "ERROR: fatal error occured at " __FILE__ << "#" << __LINE__ << std::endl; \
    exit(1); \
    }
#endif
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for error message and return error code
#define ERRMSG(status, format, ...) printf("ERROR: " format, __VA_ARGS__), status



enum vx_kernel_ext_amd_custom_e
{
    VX_KERNEL_CUSTOM_LAYER = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_CUSTOM) + 0x0,
};

//! \brief The kernel registration functions.
vx_status publishCustomLayer(vx_context context);
vx_node createCustomNode(vx_graph graph, const char *kernelName, vx_reference params[], vx_uint32 num);


#endif
