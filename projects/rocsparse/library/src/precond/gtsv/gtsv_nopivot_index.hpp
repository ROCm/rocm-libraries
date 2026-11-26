/*! \file */
/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rocsparse
{
    // Largest count with stride * (count - 1) + stride - 1 <= limit, for
    // 1 < stride <= limit.
    inline int64_t gtsv_max_count(int64_t stride, int64_t limit)
    {
        return (limit - stride + 1) / stride + 1;
    }

    // Largest number of right-hand sides one launch of the gtsv_no_pivot medium
    // kernels (m > 1024) can take. The kernels put one block per tile of num_rhs on
    // grid.y, at most max_grid_y. They index column c of the temporary buffer at
    // m * c + row in I (32-bit unsigned when I is 32-bit) and of the spike buffer at
    // num_spikes * c + spike in 32-bit unsigned. The result is a multiple of num_rhs
    // unless not even one tile fits.
    template <typename I>
    inline int64_t gtsv_no_pivot_medium_chunk(int64_t m,
                                              int64_t num_rhs,
                                              int64_t num_spikes,
                                              int64_t max_grid_y)
    {
        const int64_t uint32_limit = std::numeric_limits<uint32_t>::max();

        int64_t columns
            = std::min(max_grid_y * num_rhs, rocsparse::gtsv_max_count(num_spikes, uint32_limit));
        if(sizeof(I) == sizeof(int32_t))
        {
            columns = std::min(columns, rocsparse::gtsv_max_count(m, uint32_limit));
        }
        return (columns >= num_rhs) ? columns / num_rhs * num_rhs : columns;
    }

    // Largest number of systems one launch of the gtsv_no_pivot_strided_batch medium
    // kernels can take. The kernels put one block per system on grid.y, at most
    // max_grid_y, and offset the workspace by m * batch and num_spikes * batch in I,
    // the type of rocsparse_int.
    template <typename I>
    inline int64_t
        gtsv_no_pivot_strided_medium_chunk(int64_t m, int64_t num_spikes, int64_t max_grid_y)
    {
        const int64_t limit = std::numeric_limits<I>::max();
        return std::min({max_grid_y,
                         rocsparse::gtsv_max_count(m, limit),
                         rocsparse::gtsv_max_count(num_spikes, limit)});
    }
}
