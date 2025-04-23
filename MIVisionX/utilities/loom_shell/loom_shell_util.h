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
#ifndef __LOOM_SHELL_UTIL_H__
#define __LOOM_SHELL_UTIL_H__

#include "live_stitch_api.h"
#include <string.h>

vx_status run(ls_context context, vx_uint32 frameCount);
vx_status runParallel(ls_context * context, vx_uint32 contextCount, vx_uint32 frameCount);

vx_status showOutputConfig(ls_context context);
vx_status showCameraConfig(ls_context context);
vx_status showOverlayConfig(ls_context context);
vx_status showRigParams(ls_context context);
vx_status showCameraModule(ls_context context);
vx_status showOutputModule(ls_context context);
vx_status showOverlayModule(ls_context context);
vx_status showViewingModule(ls_context context);
vx_status showCameraBufferStride(ls_context context);
vx_status showOutputBufferStride(ls_context context);
vx_status showOverlayBufferStride(ls_context context);
vx_status showConfiguration(ls_context context, const char * exportType);
vx_status showAttributes(ls_context context, vx_uint32 offset, vx_uint32 count);
vx_status showGlobalAttributes(vx_uint32 offset, vx_uint32 count);

vx_status createBuffer(cl_context opencl_context, vx_uint32 size, cl_mem * mem);
vx_status releaseBuffer(cl_mem * mem);
vx_status createOpenCLContext(const char * platform, const char * device, cl_context * opencl_context);
vx_status releaseOpenCLContext(cl_context * opencl_context);
vx_status createOpenVXContext(vx_context * openvx_context);
vx_status releaseOpenVXContext(vx_context * openvx_context);

vx_status loadBufferFromImage(cl_mem mem, const char * fileName, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes = 0);
vx_status saveBufferToImage(cl_mem mem, const char * fileName, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes = 0);
vx_status loadBufferFromMultipleImages(cl_mem mem, const char * fileName, vx_uint32 num_rows, vx_uint32 num_cols, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes = 0);
vx_status saveBufferToMultipleImages(cl_mem mem, const char * fileName, vx_uint32 num_rows, vx_uint32 num_cols, vx_df_image buffer_format, vx_uint32 buffer_width, vx_uint32 buffer_height, vx_uint32 stride_in_bytes = 0);
vx_status loadBuffer(cl_mem mem, const char * fileName, vx_uint32 offset = 0);
vx_status saveBuffer(cl_mem mem, const char * fileName, vx_uint32 flags = 0);

vx_status setGlobalAttribute(vx_uint32 offset, float value);
vx_status saveGlobalAttributes(vx_uint32 offset, vx_uint32 count, const char * fileName);
vx_status loadGlobalAttributes(vx_uint32 offset, vx_uint32 count, const char * fileName);
vx_status setAttribute(ls_context context, vx_uint32 offset, float value);
vx_status saveAttributes(ls_context context, vx_uint32 offset, vx_uint32 count, const char * fileName);
vx_status loadAttributes(ls_context context, vx_uint32 offset, vx_uint32 count, const char * fileName);

vx_status showExpCompGains(ls_context stitch, size_t num_entries);
vx_status loadExpCompGains(ls_context stitch, size_t num_entries, const char * fileName);
vx_status saveExpCompGains(ls_context stitch, size_t num_entries, const char * fileName);

vx_status loadBlendWeights(ls_context stitch, const char * fileName);
vx_status initializeBuffer(cl_mem mem, vx_uint32 size, cl_int pattern);

vx_status ClearCmdqCache();
int64_t GetClockFrequency();
int64_t GetClockCounter();
bool GetEnvVariable(const char * name, char * value, size_t valueSize);

#endif // __LOOM_SHELL_UTIL_H__
