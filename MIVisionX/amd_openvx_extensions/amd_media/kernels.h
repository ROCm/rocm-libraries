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

#ifndef __VX_amd_MEDIA_H__
#define __VX_amd_MEDIA_H__

#include <VX/vx.h>
#include <vx_ext_amd.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/mathematics.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
}

//////////////////////////////////////////////////////////////////////
// SHARED_PUBLIC - shared sybols for export
// STITCH_API_ENTRY - export API symbols
#if _WIN32
#define SHARED_PUBLIC extern "C" __declspec(dllexport)
#else
#define SHARED_PUBLIC extern "C" __attribute__ ((visibility ("default")))
#endif

//////////////////////////////////////////////////////////////////////
//! \brief The AMD extension library for vx_amd_media module
#define AMDOVX_LIBRARY_AMD_MEDIA     3

//////////////////////////////////////////////////////////////////////
//! \brief The list of kernels in the LoomIO module.
enum vx_kernel_amd_media_e {
    //! \brief The LOOMIO Media Decoder kernel. Kernel name is "com.amd.amd_media.decode".
    AMDOVX_KERNEL_AMD_MEDIA_DECODE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_AMD_MEDIA) + 0x001,
    //! \brief The LOOMIO Media Encoder kernel. Kernel name is "com.amd.amd_media.encode".
    AMDOVX_KERNEL_AMD_MEDIA_ENCODE = VX_KERNEL_BASE(VX_ID_AMD, AMDOVX_LIBRARY_AMD_MEDIA) + 0x002,
};

//////////////////////////////////////////////////////////////////////
//! \brief The macro for error checking from OpenVX status.
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS){ printf("ERROR: failed with status = (%d:0x%08x:%4.4s) at " __FILE__ "#%d\n", status, status, (const char *)&status, __LINE__); return status; }}
//! \brief The macro for error checking from OpenVX object.
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for error checking NULL object.
#define ERROR_CHECK_NULLPTR(obj)  { if(!(obj)){ printf("ERROR: found nullptr at " __FILE__ "#%d\n", __LINE__); return VX_FAILURE; }}

//////////////////////////////////////////////////////////////////////
//! \brief The kernel registration functions.
vx_status amd_media_decode_publish(vx_context context);
vx_status amd_media_encode_publish(vx_context context);
vx_node createMediaNode(vx_graph graph, const char * kernelName, vx_reference params[], vx_uint32 num);

//////////////////////////////////////////////////////////////////////
//! \brief The common utility functions.
vx_status initialize_ffmpeg();
uint8_t * aligned_alloc(size_t size);
void aligned_free(uint8_t * ptr);
int64_t GetTimeInMicroseconds();

#endif // __VX_amd_MEDIA_H__
