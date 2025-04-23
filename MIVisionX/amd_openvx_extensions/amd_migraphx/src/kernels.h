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

#ifndef __VX_AMD_MIGRAPHX_KERNEL_H__
#define __VX_AMD_MIGRAPHX_KERNEL_H__

#include <VX/vx.h>
#include <vx_ext_amd.h>
#include <migraphx/migraphx.hpp>

//////////////////////////////////////////////////////////////////////
// SHARED_PUBLIC - shared sybols for export
#if _WIN32
#define SHARED_PUBLIC extern "C" __declspec(dllexport)
#else
#define SHARED_PUBLIC extern "C" __attribute__ ((visibility ("default")))
#endif

//////////////////////////////////////////////////////////////////////
//! \brief The AMD extension library for migraphx module
#define AMDOVX_LIBRARY_AMD_MIGRAPHX     4

//////////////////////////////////////////////////////////////////////
//! \brief The list of kernels in the migraphx module.
enum vx_kernel_amd_MIGRAPHX_e {
    //! \brief The MIGRAPHX kernel. Kernel name is "com.amd.amd_vx_migraphx".
    AMDOVX_KERNEL_AMD_MIGRAPHX = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_AMD_MIGRAPHX) + 0x001,
};

enum vx_amd_migraphx_type_e {
    VX_TYPE_MIGRAPHX_PROG     = 0x028,/*!< \brief A <tt>\ref VX_TYPE_MIGRAPHX_PROG</tt>. */
};

//////////////////////////////////////////////////////////////////////
//! \brief The macro for error checking from OpenVX status.
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS){ printf("ERROR: failed with status = (%d:0x%08x:%4.4s) at " __FILE__ "#%d\n", status, status, (const char *)&status, __LINE__); return status; }}
//! \brief The macro for error checking from OpenVX object.
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for error checking NULL object.
#define ERROR_CHECK_NULLPTR(obj)  { if(!(obj)){ printf("ERROR: found nullptr at " __FILE__ "#%d\n", __LINE__); return VX_FAILURE; }}
//! \brief The macro for error message and return error code
#define ERRMSG(status, format, ...) printf("ERROR: " format, __VA_ARGS__), status
//////////////////////////////////////////////////////////////////////

//! \brief The kernel registration functions.
vx_status amd_vx_migraphx_node_publish(vx_context context);
vx_node createMIGraphXNode(vx_graph graph, const char * kernelName, vx_reference params[], vx_uint32 num);

//////////////////////////////////////////////////////////////////////
#endif // __VX_AMD_MIGRAPHX_KERNEL_H__
