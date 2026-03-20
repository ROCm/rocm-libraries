// Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <stdexcept>
#include <type_traits>
#include <vector>

// Compute the farthest point from the original pointer for a C-style array.
// This provides the mininmal buffer allocation size for the array.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
static size_t compute_ptrdiff(const std::vector<intT1>& length, const std::vector<intT2>& stride)
{
    // Each length must have a matching stride:
    if(stride.size() < length.size())
        throw std::runtime_error("Inconsistent length/stride dimensions given to compute_ptrdiff");

    // If any lengths are zero, no memory is allocated:
    if(std::any_of(length.begin(), length.end(), [](const auto& l) { return l == 0; }))
        return 0;

    // Negative lengths or strides are not permitted:
    using lengthtype = std::remove_reference_t<decltype(length)>::value_type;
    if constexpr(std::is_signed<lengthtype>::value)
    {
        if(std::any_of(length.begin(), length.end(), [](const auto& l) { return l < 0; }))
            throw std::runtime_error("Negative lengths given to compute_ptrdiff");
    }
    using stridetype = std::remove_reference_t<decltype(stride)>::value_type;
    if constexpr(std::is_signed<stridetype>::value)
    {
        if(std::any_of(stride.begin(), stride.end(), [](const auto& s) { return s < 0; }))
            throw std::runtime_error("Negative strides given to compute_ptrdiff");
    }

    // We allow for weird data layouts with self-aliasing; this is not an array validator.

    // 1 + sum_i [ ( length_i - 1 ) * stride_i
    // = 1 + dot(length, stride) - sum(stride)
    // Since length is the one-past-the-end, we subtract the strides.
    // The length-zero vector is a scalar, so the buffer size is 1.
    // Computation is done as size_t to ensure large values are captured.
    return std::inner_product(length.begin(),
                              length.end(),
                              stride.begin(),
                              static_cast<size_t>(1),
                              std::plus<size_t>(),
                              std::multiplies<size_t>())
           - std::accumulate(
               stride.begin(), stride.end(), static_cast<size_t>(0), std::plus<size_t>());
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
