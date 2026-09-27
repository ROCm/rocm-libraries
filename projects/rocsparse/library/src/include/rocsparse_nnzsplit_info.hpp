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

#include <cstdint>

typedef struct _rocsparse_nnzsplit_info
{

    bool use_starting_block_ids{};

    size_t size{};

    void* starting_ids{};
    void* starting_block_ids{};

    // Launch tuning chosen at analysis time and replayed verbatim at compute
    // time so the two phases always agree (the block layout of starting_ids /
    // starting_block_ids depends on both).
    uint32_t block_size{}; // threads per block (wavefront-relative)
    uint32_t nnz_per_thread{}; // nnz-per-thread granularity (1 / 4 / 8)
    int64_t  max_row_nnz{}; // longest row length (row-skew signal)

public:
    ~_rocsparse_nnzsplit_info();
    void clear();
} * rocsparse_nnzsplit_info;
