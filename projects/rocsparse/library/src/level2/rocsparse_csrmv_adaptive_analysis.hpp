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

//
// Pure host-side analysis math for the adaptive CSR-Adaptive SpMV algorithm.
//
// These helpers were previously file-local (static) functions inside
//   library/src/level2/rocsparse_csrmv_template_adaptive.cpp
// They contain no device/kernel code and perform no allocations. The
// definitions are in rocsparse_csrmv_adaptive_analysis.cpp, which the host
// unit test compiles directly so it does not link the device translation unit.
//
namespace rocsparse
{
    // Round down to the largest power of two that is <= x (x > 0).
    uint32_t flp2(uint32_t x);

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
    uint64_t numThreadsForReduction(uint64_t num_rows, uint64_t wg_size = 256);

    // block_nnz closes one adaptive row block. This is BLOCK_SIZE in
    // rocsparse_csrmv_template_adaptive.cpp.
    // wide_block_rows is 32 workgroups of 256 threads. ComputeRowBlocks skips
    // an empty prefix or suffix longer than this, and csrmv_partition::is_poor_fit
    // counts a block spanning more rows than this as wide.
    struct csrmv_adaptive_constants
    {
        static constexpr uint32_t block_nnz       = 1024;
        static constexpr uint32_t wide_block_rows = 32 * 256;
    };
}
