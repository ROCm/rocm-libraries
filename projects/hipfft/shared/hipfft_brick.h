// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPFFT_BRICK_H
#define HIPFFT_BRICK_H

#include "ptrdiff.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "../library/include/hipfft/hipfft.h"
#include "../library/include/hipfft/hipfftXt.h"
#include "data_layout.h"
#include "fft_enums.h"

struct hipfft_brick
{
    // Device that the brick lives on
    int device = 0;

    // Row-major
    std::vector<size_t> field_lower;
    std::vector<size_t> field_upper;
    std::vector<size_t> brick_stride;

    // Compute the length of this brick
    std::vector<size_t> length() const
    {
        std::vector<size_t> ret;
        for(size_t i = 0; i < field_lower.size(); ++i)
            ret.push_back(field_upper[i] - field_lower[i]);
        return ret;
    }

    // Given brick index, return the offset in the field
    size_t field_offset(const std::vector<size_t>& brick_idx,
                        const std::vector<size_t>& field_stride) const
    {
        // Find the index in the field
        std::vector<size_t> field_idx;
        for(size_t i = 0; i < brick_idx.size(); ++i)
            field_idx.push_back(brick_idx[i] + field_lower[i]);

        // Based on the field's strides, return offset
        return std::inner_product(field_idx.begin(), field_idx.end(), field_stride.begin(), 0);
    }

    // Given abrick index, return the offset in this brick
    size_t brick_offset(const std::vector<size_t>& brick_idx) const
    {
        // Based on the brick's strides, return offset
        return std::inner_product(brick_idx.begin(), brick_idx.end(), brick_stride.begin(), 0);
    }
};

#endif
