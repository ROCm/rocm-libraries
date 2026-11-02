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

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    static constexpr uint32_t s_cscsv_blocksize          = 256;
    static constexpr uint32_t s_cscsv_analysis_groupsize = 16;

    inline size_t cscsv_aligned(size_t size_in_bytes)
    {
        return (size_in_bytes == 0) ? 0 : ((size_in_bytes - 1) / 256 + 1) * 256;
    }

    // Calls f(I{}, J{}) for the column pointer and row index types of the matrix.
    template <typename F>
    rocsparse_status cscsv_dispatch_indextype(rocsparse_indextype i, rocsparse_indextype j, F&& f)
    {
        if(i == rocsparse_indextype_i32 && j == rocsparse_indextype_i32)
        {
            return f(int32_t{}, int32_t{});
        }
        if(i == rocsparse_indextype_i32 && j == rocsparse_indextype_i64)
        {
            return f(int32_t{}, int64_t{});
        }
        if(i == rocsparse_indextype_i64 && j == rocsparse_indextype_i32)
        {
            return f(int64_t{}, int32_t{});
        }
        if(i == rocsparse_indextype_i64 && j == rocsparse_indextype_i64)
        {
            return f(int64_t{}, int64_t{});
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}
