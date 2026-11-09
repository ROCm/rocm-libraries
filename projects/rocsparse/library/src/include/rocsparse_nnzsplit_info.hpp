/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <cstdint>

typedef struct _rocsparse_nnzsplit_info
{

    bool use_starting_block_ids{};

    size_t size{};

    void* starting_ids{};
    // Interior pointer into starting_ids. One past that allocation when
    // use_starting_block_ids is false.
    void*  starting_block_ids{};
    size_t starting_ids_size_in_bytes{};
    // Bytes of the starting_block_ids region. Zero when that region is unused.
    size_t starting_block_ids_size_in_bytes{};

    // Launch tuning chosen at analysis time and replayed verbatim at compute
    // time so the two phases always agree (the block layout of starting_ids /
    // starting_block_ids depends on both).
    uint32_t block_size{}; // threads per block (wavefront-relative)
    uint32_t nnz_per_thread{}; // nnz-per-thread granularity (1 / 4 / 8)
    int64_t  max_row_nnz{}; // longest row length (row-skew signal)

public:
    ~_rocsparse_nnzsplit_info();
    void clear();

    // Allocates starting_ids and, when use_starting_block_ids is set, the
    // starting_block_ids region inside the same allocation. The flag is the
    // value already stored on this object.
    rocsparse_status
        allocate_starting_ids(hipStream_t stream, size_t required_blocks, size_t index_size);
} * rocsparse_nnzsplit_info;
