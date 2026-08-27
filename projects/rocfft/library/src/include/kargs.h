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
#include "rtc_generator.h"
#include <cstddef>
#include <vector>

#define KERN_ARGS_ARRAY_WIDTH 16

// Device buffer holding the lengths and strides that a node's kernel
// needs, as three fixed-width arrays of KERN_ARGS_ARRAY_WIDTH
// elements: lengths, input strides, output strides.  The dist follows
// the last stride in each stride array, i.e. stride_in[dim] is the
// batch stride.
//
// Lengths are always 32-bit, since kernels only ever index and divide
// by them.  Strides and dists are as wide as the kint_type of the
// kernel that reads the buffer, so the layout - and therefore the
// offset of each array - depends on the KIntType this was created
// with.  The kernel's kint_type must be decided the same way, or it
// will read the arrays at the wrong width.
class KernelArgsBuffer
{
public:
    bool create(const std::vector<size_t>& length,
                const std::vector<size_t>& inStride,
                const std::vector<size_t>& outStride,
                size_t                     iDist,
                size_t                     oDist,
                KIntType                   itype);

    void* lengths() const
    {
        return buf.data_offset(0);
    }

    void* stride_in() const
    {
        return buf.data_offset(lengths_bytes());
    }

    void* stride_out() const
    {
        return buf.data_offset(lengths_bytes() + strides_bytes());
    }

private:
    static size_t lengths_bytes()
    {
        return KERN_ARGS_ARRAY_WIDTH * sizeof(unsigned int);
    }

    size_t strides_bytes() const
    {
        return KERN_ARGS_ARRAY_WIDTH * rtc_kint_type_size(itype);
    }

    gpubuf   buf;
    KIntType itype = KIntType::U32;
};

#endif // defined( KARGS_H )
