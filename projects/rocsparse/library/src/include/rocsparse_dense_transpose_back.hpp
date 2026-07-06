/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"

namespace rocsparse
{
    rocsparse_status dense_transpose_back(rocsparse_handle   handle,
                                          int64_t            m,
                                          int64_t            n,
                                          rocsparse_datatype A_datatype,
                                          const void*        A,
                                          int64_t            lda,
                                          rocsparse_datatype B_datatype,
                                          void*              B,
                                          int64_t            ldb);

    rocsparse_status dense_transpose_back_strided_batched(rocsparse_handle   handle,
                                                          int64_t            batch_count,
                                                          int64_t            m,
                                                          int64_t            n,
                                                          rocsparse_datatype A_datatype,
                                                          const void*        A,
                                                          int64_t            lda,
                                                          int64_t            A_stride,
                                                          rocsparse_datatype B_datatype,
                                                          void*              B,
                                                          int64_t            ldb,
                                                          int64_t            B_stride);

    template <typename I, typename T>
    inline rocsparse_status dense_transpose_back(
        rocsparse_handle handle, I m, I n, const T* A, int64_t lda, T* B, int64_t ldb);

}

template <typename I, typename T>
inline rocsparse_status rocsparse::dense_transpose_back(
    rocsparse_handle handle, I m, I n, const T* A, int64_t lda, T* B, int64_t ldb)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose_back(
        handle, m, n, rocsparse::get_datatype<T>(), A, lda, rocsparse::get_datatype<T>(), B, ldb));
    return rocsparse_status_success;
}
