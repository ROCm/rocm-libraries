// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <numeric>
#include <vector>

// Compute the farthest point from the original pointer.
static size_t compute_ptrdiff(const std::vector<size_t>& length,
                              const std::vector<size_t>& stride)
{
    // 1 + sum_i [ ( length_i - 1 ) * stride_i
    // = 1 + dot(length, stride) - sum(stride)
    // Since length is the one-past-the-end, we subtract the strides.
    // The length-zero vector is a scalar, so the buffer size is 1.
    return std::inner_product(length.begin(), length.end(), stride.begin(), 1)
        - std::accumulate(stride.begin(), stride.end(), 1, std::plus<size_t>());
}   

static size_t compute_ptrdiff(const std::vector<size_t>& length,
                              const std::vector<size_t>& stride,
                              const size_t               nbatch,
                              const size_t               dist)
{
    std::vector l = length;
    l.push_back(nbatch);
    std::vector s = stride;
    s.push_back(dist);
    return compute_ptrdiff(l, s);
}
