/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    template <typename I, typename T, typename U>
    rocsparse_status dense_transpose_template(rocsparse_handle handle,
                                              I                m,
                                              I                n,
                                              U                alpha_device_host,
                                              const T*         A,
                                              int64_t          lda,
                                              T*               B,
                                              int64_t          ldb);

    rocsparse_status dense_transpose(rocsparse_handle   handle,
                                     int64_t            m,
                                     int64_t            n,
                                     rocsparse_datatype source_datatype,
                                     const void*        source,
                                     int64_t            source_ld,
                                     rocsparse_datatype target__datatype,
                                     void*              target,
                                     int64_t            target_ld);

    template <typename I, typename T>
    rocsparse_status dense_transpose_back(
        rocsparse_handle handle, I m, I n, const T* A, int64_t lda, T* B, int64_t ldb);

    template <typename I, typename T>
    rocsparse_status conjugate(rocsparse_handle handle, I length, T* array);

    template <typename I, typename T>
    rocsparse_status valset(rocsparse_handle handle, I length, T value, T* array);

    template <typename I, typename T>
    rocsparse_status valset_2d(
        rocsparse_handle handle, I m, I n, int64_t ld, T value, T* array, rocsparse_order order);

    template <typename I, typename A, typename T>
    rocsparse_status
        scale_array(rocsparse_handle handle, I length, const T* scalar_device_host, A* array);

    template <typename I, typename A, typename T>
    rocsparse_status scale_2d_array(rocsparse_handle handle,
                                    I                m,
                                    I                n,
                                    int64_t          ld,
                                    int64_t          batch_count,
                                    int64_t          stride,
                                    const T*         scalar_device_host,
                                    A*               array,
                                    rocsparse_order  order);

    template <typename I, typename J>
    rocsparse_status copy(rocsparse_handle     handle,
                          int64_t              length,
                          const I*             in,
                          J*                   out,
                          rocsparse_index_base idx_base_in,
                          rocsparse_index_base idx_base_out);

    template <typename T>
    rocsparse_status copy_and_scale(
        rocsparse_handle handle, int64_t length, const T* in, T* out, const T* scalar_device_host);
}
