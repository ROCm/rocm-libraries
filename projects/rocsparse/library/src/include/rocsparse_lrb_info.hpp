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

namespace rocsparse
{
    // LRB processes bins 0 to lrb_num_bins - 1, i.e. rows of up to 2^31 non-zeros.
    static constexpr int lrb_num_bins = 32;

    // ceil(log2(row length)) <= 63 for 64-bit offsets.
    static constexpr int lrb_num_bin_counters = 64;
}

typedef struct _rocsparse_lrb_info
{
    void* rows_offsets_scratch{}; // size of m
    void* rows_bins{}; // size of m
    void* n_rows_bins{}; // size of rocsparse::lrb_num_bin_counters

    size_t    size{};
    uint32_t* wg_flags{};

    int64_t nRowsBins[rocsparse::lrb_num_bins]{}; // host array

    ~_rocsparse_lrb_info();
    void clear();

} * rocsparse_lrb_info;
