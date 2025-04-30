/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "handle.hpp"

/*! \brief rocblas_internal_hemv_kernel_workspace_size
    workspace buffer for column reductions: number of blocks * cols * batch_count

    @param[in]
    outputType To*
        Type of output values
    @param[in]
    n rocblas_int
        Number of columns
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <typename To>
ROCBLAS_INTERNAL_EXPORT_NOINLINE size_t
    rocblas_internal_hemv_symv_kernel_workspace_size(rocblas_int n, rocblas_int batch_count = 1);

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_hemv_symv_arg_check(rocblas_handle handle,
                                                  rocblas_fill   uplo,
                                                  API_INT        n,
                                                  TScal          alpha,
                                                  rocblas_stride stride_alpha,
                                                  TConstPtr      A,
                                                  rocblas_stride offseta,
                                                  API_INT        lda,
                                                  rocblas_stride strideA,
                                                  TConstPtr      x,
                                                  rocblas_stride offsetx,
                                                  API_INT        incx,
                                                  rocblas_stride stridex,
                                                  TScal          beta,
                                                  rocblas_stride stride_beta,
                                                  TPtr           y,
                                                  rocblas_stride offsety,
                                                  API_INT        incy,
                                                  rocblas_stride stridey,
                                                  API_INT        batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(n < 0 || lda < n || lda < 1 || !incx || !incy || batch_count < 0)
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if(!handle->is_device_memory_size_query())
    {
        if(!beta || !alpha)
            return rocblas_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        // only supports stride_alpha and stride_beta for device memory alpha/beta
        if(stride_alpha || stride_beta)
            return rocblas_status_not_implemented;

        if(!handle->is_device_memory_size_query())
        {
            if(*alpha == 0 && *beta == 1)
                return rocblas_status_success;

            if(!y || (*alpha != 0 && (!A || !x)))
                return rocblas_status_invalid_pointer;
        }
    }

    return rocblas_status_continue;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   rocblas_stride stride_alpha,
                                   const T*       A,
                                   rocblas_stride offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T*       beta,
                                   rocblas_stride stride_beta,
                                   T*             y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T*             workspace);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symv_batched_template(rocblas_handle  handle,
                                           rocblas_fill    uplo,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           rocblas_stride  stride_alpha,
                                           const T* const* A,
                                           rocblas_stride  offseta,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           const T* const* x,
                                           rocblas_stride  offsetx,
                                           rocblas_int     incx,
                                           rocblas_stride  stridex,
                                           const T*        beta,
                                           rocblas_stride  stride_beta,
                                           T* const*       y,
                                           rocblas_stride  offsety,
                                           rocblas_int     incy,
                                           rocblas_stride  stridey,
                                           rocblas_int     batch_count,
                                           T*              workspace);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_template(rocblas_handle handle,
                                   rocblas_fill   uplo,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   rocblas_stride stride_alpha,
                                   const T*       A,
                                   rocblas_stride offseta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T*       beta,
                                   rocblas_stride stride_beta,
                                   T*             y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count,
                                   T*             workspace);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemv_batched_template(rocblas_handle  handle,
                                           rocblas_fill    uplo,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           rocblas_stride  stride_alpha,
                                           const T* const* A,
                                           rocblas_stride  offseta,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           const T* const* x,
                                           rocblas_stride  offsetx,
                                           rocblas_int     incx,
                                           rocblas_stride  stridex,
                                           const T*        beta,
                                           rocblas_stride  stride_beta,
                                           T* const*       y,
                                           rocblas_stride  offsety,
                                           rocblas_int     incy,
                                           rocblas_stride  stridey,
                                           rocblas_int     batch_count,
                                           T*              workspace);

//TODO :-Add rocblas_check_numerics_he_matrix_template for checking Matrix `A` which is a Hermitian Matrix
template <typename T, typename U>
rocblas_status rocblas_hemv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
template <typename T, typename U>
rocblas_status rocblas_symv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input);
