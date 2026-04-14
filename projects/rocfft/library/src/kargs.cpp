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

#include "kargs.h"
#include "../../shared/rocfft_hip.h"
#include <cassert>

// malloc device buffer; copy host buffer to device buffer
gpubuf_t<size_t> kargs_create(std::vector<size_t> length,
                              std::vector<size_t> inStride,
                              std::vector<size_t> outStride,
                              size_t              iDist,
                              size_t              oDist)
{
    gpubuf_t<size_t> devk;
    if(devk.alloc(3 * KERN_ARGS_ARRAY_WIDTH * sizeof(size_t)) != hipSuccess)
        return devk;

    size_t devkHost[3 * KERN_ARGS_ARRAY_WIDTH];

    size_t i = 0;
    while(i < 3 * KERN_ARGS_ARRAY_WIDTH)
        devkHost[i++] = 0;

    assert(length.size() == inStride.size());
    assert(length.size() == outStride.size());

    i = 0;
    while(i < length.size())
    {
        devkHost[i + 0 * KERN_ARGS_ARRAY_WIDTH] = length[i];
        devkHost[i + 1 * KERN_ARGS_ARRAY_WIDTH] = inStride[i];
        devkHost[i + 2 * KERN_ARGS_ARRAY_WIDTH] = outStride[i];
        i++;
    }

    // NB: iDist is right after the last inStride[dim-1], i.e. inStride[dim] = batch-in-stride
    //     oDist is right after the last outStride[dim-1], i.e. outStride[dim] = batch-out-stride
    devkHost[i + 1 * KERN_ARGS_ARRAY_WIDTH] = iDist;
    devkHost[i + 2 * KERN_ARGS_ARRAY_WIDTH] = oDist;

    if(hipMemcpy(
           devk.data(), devkHost, 3 * KERN_ARGS_ARRAY_WIDTH * sizeof(size_t), hipMemcpyHostToDevice)
       != hipSuccess)
        devk.free();
    return devk;
}
