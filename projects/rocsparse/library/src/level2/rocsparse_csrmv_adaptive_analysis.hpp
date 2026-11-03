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

#include <cstddef>
#include <cstdint>

//
// Pure host-side analysis math for the adaptive CSR-Adaptive SpMV algorithm.
//
// These helpers were previously file-local (static) functions inside
//   library/src/level2/rocsparse_csrmv_template_adaptive.cpp
// They contain no device/kernel code and perform no allocations, so they are
// lifted here (as inline functions in namespace rocsparse) to make them
// independently unit-testable without linking the device translation unit.
//
namespace rocsparse
{
    // Round down to the largest power of two that is <= x (x > 0).
    inline uint32_t flp2(uint32_t x)
    {
        x |= (x >> 1);
        x |= (x >> 2);
        x |= (x >> 4);
        x |= (x >> 8);
        x |= (x >> 16);
        return x - (x >> 1);
    }

    // Short rows in CSR-Adaptive are batched together into a single row block.
    // If there are a relatively small number of these, then we choose to do
    // a horizontal reduction (groups of threads all reduce the same row).
    // If there are many threads (e.g. more threads than the maximum size
    // of our workgroup) then we choose to have each thread serially reduce
    // the row.
    // This function calculates the number of threads that could team up
    // to reduce these groups of rows. For instance, if you have a
    // workgroup size of 256 and 4 rows, you could have 64 threads
    // working on each row. If you have 5 rows, only 32 threads could
    // reliably work on each row because our reduction assumes power-of-2.
    //
    // Contract: num_rows must be >= 2 (num_rows - 1 must be non-zero for the
    // count-leading-zeros path); this matches the original static function.
    inline uint64_t numThreadsForReduction(uint64_t num_rows, uint64_t wg_size = 256)
    {
#if defined(__INTEL_COMPILER)
        return wg_size >> (_bit_scan_reverse(num_rows - 1) + 1);
#elif(defined(__clang__) && __has_builtin(__builtin_clz)) \
    || !defined(__clang) && defined(__GNUG__)             \
           && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
        return (wg_size >> (8 * sizeof(int) - __builtin_clz(num_rows - 1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
        uint64_t bit_returned;
        _BitScanReverse(&bit_returned, (num_rows - 1));
        return wg_size >> (bit_returned + 1);
#else
        return flp2(wg_size / num_rows);
#endif
    }

    // Nonzeros that close one adaptive row block. This is BLOCK_SIZE in
    // rocsparse_csrmv_template_adaptive.cpp.
    inline constexpr uint32_t csrmv_adaptive_block_nnz = 1024;

    // Rows of 32 workgroups of 256 threads. ComputeRowBlocks skips an empty
    // prefix or suffix longer than this, and csrmv_adaptive_partition_is_poor_fit
    // counts a block spanning more rows than this as wide. Sharing the limit
    // keeps an empty edge that adaptive does not skip from counting as wide.
    inline constexpr uint32_t csrmv_adaptive_wide_block_rows = 32 * 256;

    // True when the adaptive row blocks are a poor fit for this matrix, so that
    // csrmv_select_algorithm chooses nnz-split for csrmv_alg_default.
    //
    // A zero row span is an intermediate chunk of one long row and is ignored.
    // The two poor partitions are:
    // - wide blocks, walked serially by one workgroup, that together span at
    //   least one eighth of the rows
    // - at least 8 blocks, a quarter or more of which hold fewer than
    //   csrmv_adaptive_block_nnz / 8 nonzeros (empty blocks included). These are
    //   the runs of short rows that adaptive cuts off between long rows, so most
    //   of the grid does almost no numeric work.
    //
    // The fractions and the 8-block minimum are heuristics that separate the
    // scattered-row and alternating-row patterns from populated matrices and
    // from the empty edges adaptive skips. They were checked on gfx1201 (wave32)
    // and gfx90a (wave64).
    template <typename I>
    bool csrmv_adaptive_partition_is_poor_fit(const I*    row_blocks,
                                              std::size_t row_block_size,
                                              const I*    row_ptr,
                                              I           n_rows);
}
