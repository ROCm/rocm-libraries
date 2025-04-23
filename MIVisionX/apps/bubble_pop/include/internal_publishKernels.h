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
#ifndef _PUBLISH_KERNELS_H_
#define _PUBLISH_KERNELS_H_

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__((visibility("default")))
#endif

#include "internal_opencvTunnel.h"

#define MAX_KERNELS 10

extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);

vx_status ADD_KERENEL(std::function<vx_status(vx_context)>);
vx_status get_kernels_to_publish();

vx_status VX_bubbles_pop_Register(vx_context);
vx_status VX_donut_pop_Register(vx_context);

#define VX_KERNEL_EXT_POP_BUBBLE_POP_NAME "org.pop.bubble_pop"
#define VX_KERNEL_EXT_POP_DONUT_POP_NAME "org.pop.donut_pop"

#endif //_PUBLISH_KERNELS_H_
