/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_conjugate.hpp"
#include "rocsparse_dense_transpose.hpp"
#include "rocsparse_dense_transpose_back.hpp"
#include "rocsparse_valset.hpp"

//
namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status valset(rocsparse_handle handle, I length, T value, T* array);

    template <typename I, typename T>
    rocsparse_status valset_2d(
        rocsparse_handle handle, I m, I n, int64_t ld, T value, T* array, rocsparse_order order);

    template <typename I, typename A, typename T>
    rocsparse_status
        scale_array(rocsparse_handle handle, I length, const T* scalar_device_host, A* array);

    template <typename I, typename X, typename Y, typename T>
    rocsparse_status axpby_array_batched(rocsparse_handle handle,
                                         I                length,
                                         rocsparse_int    num_extra,
                                         const T*         gamma_device_array,
                                         const X**        x_arrays,
                                         const T*         beta_device_host,
                                         Y*               y_array);

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
}
