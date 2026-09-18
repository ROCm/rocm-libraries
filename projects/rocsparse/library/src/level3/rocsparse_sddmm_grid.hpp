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

namespace rocsparse
{
    //
    // grid.x sizing shared by every sddmm launch (AISPARSE-673).
    //
    // All of them cover a one dimensional work space at a fixed number of items
    // per block: nnz for COO, COO AoS and ELL, m for CSR, n for CSC. Sizing
    // grid.x as ceil(work_items / items_per_block) alone silently truncates the
    // launch once that count exceeds the device's grid.x limit, so the count is
    // clamped here and every kernel behind it grid-strides over grid.x. This is
    // the clamp the COO and COO AoS dense sample launches already carried,
    // lifted so that all five formats share one definition.
    //
    // Deliberately a new leaf header rather than an addition to
    // rocsparse_common.hpp: AISPARSE-696 (PR #11512) is adding a general
    // rocsparse::get_grid_size / rocsparse::ceil_div to that header, and
    // AISPARSE-677/678 are editing it concurrently. Once #11512 lands, this body
    // becomes
    //
    //     return rocsparse::get_grid_size(work_items, items_per_block, max_grid_x);
    //
    // and the header can be deleted outright.
    //
    // An empty work space keeps the historical single (fully masked) block
    // rather than dim3(0), which is not a launchable grid.
    //
    static inline int64_t
        sddmm_grid_size_x(int64_t work_items, int64_t items_per_block, int64_t max_grid_x)
    {
        const int64_t num_blocks = (work_items > 0) ? ((work_items - 1) / items_per_block + 1) : 1;
        const int64_t limit      = (max_grid_x > 1) ? max_grid_x : 1;

        return (num_blocks < limit) ? num_blocks : limit;
    }
}
