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

#include "rocsparse-types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocsparse
{
    // Host copy of the CSR row pointers and the CSR-adaptive row blocks built
    // from them. Default SpMV uses the row blocks to decide between adaptive
    // and nnz-split. nnz-split reads the row pointers for the longest row, so
    // this is not an adaptive-only object.
    template <typename I, typename J>
    struct csrmv_partition
    {
        std::vector<I> row_ptr;
        std::vector<I> row_blocks;
        std::vector<J> wg_ids;
        int64_t        first_row{};
        int64_t        last_row{};
        uint32_t       gen_wg{};

        // Copies the row pointers to the host and builds the adaptive row blocks.
        rocsparse_status compute(rocsparse_handle handle, J m, I nnz, const I* csr_row_ptr);

        // True when the row blocks are a poor fit, so default SpMV selects nnz-split.
        //
        // A zero row span is an intermediate chunk of one long row and is ignored.
        // Empty rows before the first and after the last nonempty row are not part
        // of any block's span, whether or not adaptive skips them.
        // The two poor partitions are:
        // - wide blocks, walked serially by one workgroup, that together span at
        //   least one eighth of the rows
        // - at least 8 blocks, a quarter or more of which hold fewer than
        //   csrmv_adaptive_constants::block_nnz / 8 nonzeros (empty blocks included). These are
        //   the runs of short rows that adaptive cuts off between long rows, so most
        //   of the grid does almost no numeric work.
        //
        // The fractions and the 8-block minimum are heuristics that separate the
        // scattered-row and alternating-row patterns from populated matrices and
        // from empty edges. The row count is row_ptr.size() - 1.
        bool is_poor_fit() const;

        // Longest row recorded by compute(). Zero until compute() runs.
        int64_t max_nnz_per_row() const;

    private:
        int64_t max_nnz_per_row_value{};
    };
}
