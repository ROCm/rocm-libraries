/*****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T, typename I>
void rocsolver_larf_getMemorySize(const rocblas_side side,
                                  const I m,
                                  const I n,
                                  const I batch_count,
                                  size_t* size_scalars,
                                  size_t* size_Abyx,
                                  size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || m == 0 || !batch_count)
    {
        *size_scalars = 0;
        *size_Abyx = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // if small size no workspace needed
    bool ssker_left
        = (side == rocblas_side_left && m <= LARF_SSKER_MAX_DIM && n <= LARF_SSKER_MIN_DIM);
    bool ssker_right
        = (side == rocblas_side_right && m <= LARF_SSKER_MIN_DIM && n <= LARF_SSKER_MAX_DIM);
    if(ssker_left || ssker_right)
    {
        *size_Abyx = 0;
        return;
    }

    // size of temporary result in Householder matrix generation
    if(side == rocblas_side_left)
        *size_Abyx = n;
    else if(side == rocblas_side_right)
        *size_Abyx = m;
    else
        *size_Abyx = std::max(m, n);
    *size_Abyx *= sizeof(T) * batch_count;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_larf_argCheck(rocblas_handle handle,
                                       const rocblas_side side,
                                       const I m,
                                       const I n,
                                       const I lda,
                                       const I incx,
                                       T x,
                                       T A,
                                       U alpha)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || !incx)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (left && m && (!alpha || !x)) || (!left && n && (!alpha || !x)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larf_template(rocblas_handle handle,
                                       const rocblas_side side,
                                       const I m,
                                       const I n,
                                       U x,
                                       const rocblas_stride shiftx,
                                       const I incx,
                                       const rocblas_stride stridex,
                                       const T* alpha,
                                       const rocblas_stride stridep,
                                       U A,
                                       const rocblas_stride shiftA,
                                       const I lda,
                                       const rocblas_stride stridea,
                                       const I batch_count,
                                       T* scalars,
                                       T* Abyx,
                                       T** workArr)
{
    ROCSOLVER_ENTER("larf", "side:", side, "m:", m, "n:", n, "shiftX:", shiftx, "incx:", incx,
                    "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if n is small, use small-size kernel
    bool ssker_left
        = (side == rocblas_side_left && m <= LARF_SSKER_MAX_DIM && n <= LARF_SSKER_MIN_DIM);
    bool ssker_right
        = (side == rocblas_side_right && m <= LARF_SSKER_MIN_DIM && n <= LARF_SSKER_MAX_DIM);
    if(ssker_left || ssker_right)
    {
        return larf_run_small(handle, side, m, n, x, shiftx, incx, stridex, alpha, stridep, A,
                              shiftA, lda, stridea, batch_count);
    }

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // determine side and order of H
    bool leftside = (side == rocblas_side_left);
    I order = m;
    rocblas_operation trans = rocblas_operation_none;
    if(leftside)
    {
        trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
        order = n;
    }

    // **** FOR NOW, IT DOES NOT DETERMINE "NON-ZERO" DIMENSIONS
    //      OF A AND X, AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
    //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
    //      ZERO ENTRIES ****

    // compute the matrix vector product  (W=-A'*X or W=-A*X)
    rocblasCall_gemv<T>(handle, trans, m, n, cast2constType<T>(scalars), 0, A, shiftA, lda, stridea,
                        x, shiftx, incx, stridex, cast2constType<T>(scalars + 1), 0, Abyx, 0, 1,
                        order, batch_count, workArr);

    // compute the rank-1 update  (A + tau*X*W'  or A + tau*W*X')
    if(leftside)
    {
        rocblasCall_ger<COMPLEX, T, I>(handle, m, n, alpha, stridep, x, shiftx, incx, stridex, Abyx,
                                       0, 1, order, A, shiftA, lda, stridea, batch_count, workArr);
    }
    else
    {
        rocblasCall_ger<COMPLEX, T, I>(handle, m, n, alpha, stridep, Abyx, 0, 1, order, x, shiftx,
                                       incx, stridex, A, shiftA, lda, stridea, batch_count, workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
