/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#pragma once
#if !defined(KARGS_H)
#define KARGS_H

#include "../../../shared/gpubuf.h"
#include <cstddef>
#include <vector>

#define KERN_ARGS_ARRAY_WIDTH 16

gpubuf_t<size_t> kargs_create(std::vector<size_t> length,
                              std::vector<size_t> inStride,
                              std::vector<size_t> outStride,
                              size_t              iDist,
                              size_t              oDist);

// data->node->devKernArg : points to the internal length device pointer
// data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH : points to the intenal in
// stride device pointer
// data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH : points to the internal out
// stride device pointer, only used in outof place kernels
static size_t* kargs_lengths(const gpubuf_t<size_t>& devKernArg)
{
    return devKernArg.data();
}

static size_t* kargs_stride_in(const gpubuf_t<size_t>& devKernArg)
{
    return devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH;
}

static size_t* kargs_stride_out(const gpubuf_t<size_t>& devKernArg)
{
    return devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH;
}

#endif // defined( KARGS_H )
