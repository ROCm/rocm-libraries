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

#include <cstdint>
#include <limits>
#include <type_traits>

namespace rocsparse
{
    // stride * columns + inner fits in limit, without overflowing int64_t. columns == 0
    // checks inner alone.
    inline bool gtsv_index_fits(int64_t stride, int64_t columns, int64_t inner, int64_t limit)
    {
        if(stride < 0 || columns < 0 || inner < 0 || inner > limit)
        {
            return false;
        }
        if(columns == 0)
        {
            return true;
        }
        if(stride > std::numeric_limits<int64_t>::max() / columns)
        {
            return false;
        }
        return stride * columns <= limit - inner;
    }

    // Whether the gtsv_no_pivot medium path (m > 1024) can launch the straight-line
    // kernels for n > 0 right-hand sides in tiles of num_rhs. Each tile needs its own
    // block in grid.y, at most max_grid_y. Those kernels index the right-hand side
    // column with a 32-bit unsigned value: the temporary buffer at m * column + row
    // in I (32-bit unsigned when I is 32-bit), the spike buffer at
    // num_spikes * column + spike in 32-bit unsigned.
    template <typename I>
    inline bool gtsv_no_pivot_medium_straight(
        int64_t m, int64_t n, int64_t num_rhs, int64_t num_spikes, int64_t max_grid_y)
    {
        const int64_t tiles = (n - 1) / num_rhs + 1;
        if(tiles > max_grid_y)
        {
            return false;
        }
        const int64_t uint32_limit = std::numeric_limits<uint32_t>::max();
        const int64_t modified_limit
            = (sizeof(I) == sizeof(int32_t)) ? uint32_limit : std::numeric_limits<int64_t>::max();
        return gtsv_index_fits(m, n - 1, m - 1, modified_limit)
               && gtsv_index_fits(num_spikes, n - 1, num_spikes - 1, uint32_limit);
    }

    // Whether the gtsv_no_pivot_strided_batch medium path can launch the
    // straight-line kernels for batch_count > 0 systems. Each system needs its own
    // block in grid.y, at most max_grid_y. Those kernels offset the workspace by
    // m * batch and num_spikes * batch in I, the type of rocsparse_int.
    template <typename I>
    inline bool gtsv_no_pivot_strided_medium_straight(int64_t m,
                                                      int64_t batch_count,
                                                      int64_t num_spikes,
                                                      int64_t max_grid_y)
    {
        if(batch_count > max_grid_y)
        {
            return false;
        }
        const int64_t limit = std::numeric_limits<I>::max();
        return gtsv_index_fits(m, batch_count - 1, m - 1, limit)
               && gtsv_index_fits(num_spikes, batch_count - 1, num_spikes - 1, limit);
    }

    // launch(std::false_type{}) launches the straight-line kernel and
    // launch(std::true_type{}) the grid-stride one.
    template <typename F>
    inline decltype(auto) gtsv_no_pivot_dispatch_grid_stride(bool straight, F&& launch)
    {
        return straight ? launch(std::false_type{}) : launch(std::true_type{});
    }
}
