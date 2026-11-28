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
#include <array>
#include <cstddef>
#include <span>

#define KERN_ARGS_ARRAY_WIDTH 16

// The flat geometry shared by kernel generation, grid setup and argument packing.
// The last surviving batch axis is passed separately as batch and the two dists.
struct KernelView
{
    std::span<const size_t> lengths() const
    {
        return {input_lengths.data(), rank};
    }
    std::span<const size_t> output_lengths() const
    {
        return {output_spans.data(), rank};
    }
    std::span<const size_t> in_strides() const
    {
        return {input_strides.data(), rank};
    }
    std::span<const size_t> out_strides() const
    {
        return {output_strides.data(), rank};
    }

    size_t   batch;
    size_t   iDist;
    size_t   oDist;
    size_t   output_elements;
    KIntType itype;

private:
    friend class LeafNode;
    size_t                                        rank = 0;
    std::array<size_t, KERN_ARGS_ARRAY_WIDTH - 1> input_lengths, output_spans;
    std::array<size_t, KERN_ARGS_ARRAY_WIDTH - 1> input_strides, output_strides;
};

// Device buffer holding the lengths and strides that a node's kernel
// needs, as three fixed-width arrays of KERN_ARGS_ARRAY_WIDTH
// elements: lengths, input strides, output strides.  The dist follows
// the last stride in each stride array, i.e. stride_in[dim] is the
// batch stride.
//
// All three arrays are as wide as the kint_type of the kernel that
// reads the buffer, so the layout - and therefore the offset of each
// array - depends on the KIntType this was created with.  The kernel's
// kint_type must be decided the same way, or it will read the arrays
// at the wrong width.
//
// Lengths ride on the same width as the strides rather than getting a
// width of their own: the kernel view picks one width using the layout's
// maximum offset and stride. Kernel names need only the existing i32/i64 suffix.
class KernelArgsBuffer
{
public:
    bool  create(const KernelView& view);
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
    size_t lengths_bytes() const
    {
        return KERN_ARGS_ARRAY_WIDTH * rtc_kint_type_size(itype);
    }

    size_t strides_bytes() const
    {
        return KERN_ARGS_ARRAY_WIDTH * rtc_kint_type_size(itype);
    }

    // buf layout is as follows: | lengths | stride_in | stride_out |
    gpubuf   buf;
    KIntType itype = KIntType::U32;
};

#endif // defined( KARGS_H )
