/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_ormtr_unmtr_hb2st_getMemorySize(const rocblas_side side,
                                               const rocblas_operation trans,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               const rocblas_int kd,
                                               const rocblas_int batch_count,
                                               size_t* size_scalars,
                                               size_t* size_work,
                                               size_t* size_workArr)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || kd == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // TODO: calculate memory requirements
    *size_scalars = 0;
    *size_work = 0;
    *size_workArr = 0;
}

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_ormtr_unmtr_hb2st_getMemorySize(const rocblas_side side,
                                               const rocblas_operation trans,
                                               const rocblas_int m,
                                               const rocblas_int n,
                                               const rocblas_int kd,
                                               const rocblas_int batch_count,
                                               size_t* size_scalars,
                                               size_t* size_work,
                                               size_t* size_work2,
                                               size_t* size_work3,
                                               size_t* size_work4,
                                               size_t* size_workArr,
                                               bool* optim_mem)
{
    *size_scalars = 0;
    *size_work = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_workArr = 0;
    *optim_mem = true;

    // if quick return no workspace needed
    if(m == 0 || n == 0 || kd == 0 || batch_count == 0)
    {
        return;
    }

    // TODO: calculate memory requirements
}

template <bool COMPLEX, typename T, typename U>
rocblas_status rocsolver_ormtr_hb2st_argCheck(rocblas_handle handle,
                                              const rocblas_side side,
                                              const rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int kd,
                                              T V,
                                              const rocblas_int ldv,
                                              U tau,  // why U? claude put last
                                              T C,
                                              const rocblas_int ldc)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    // rocblas_operation_conjugate_transpose ok for both real and complex.
    if(COMPLEX && trans == rocblas_operation_transpose)
        return rocblas_status_invalid_value;
    if(trans != rocblas_operation_none
       && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // 2. invalid size
    // todo: fix ldv validation
    if(m < 0 || n < 0 || kd < 0 || ldv < m || ldc < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    // todo: if (m > 0 && n > 0 && kd > 0 && (!V || !tau || !C))
    if((m > 0 && n > 0 && !V) || (kd > 0 && !tau) || (m && n && !C))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// todo: why are there 2 templates?
template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_hb2st_template(rocblas_handle handle,
                                                    const rocblas_side side,
                                                    const rocblas_operation trans,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    const rocblas_int kd,
                                                    U V,
                                                    const rocblas_int shiftV,
                                                    const rocblas_int ldv,
                                                    const rocblas_stride strideV,
                                                    T* tau,
                                                    const rocblas_stride strideT,
                                                    U C,
                                                    const rocblas_int shiftC,
                                                    const rocblas_int ldc,
                                                    const rocblas_stride strideC,
                                                    const rocblas_int batch_count,
                                                    T* scalars,
                                                    T* work,
                                                    T** workArr)
{
    ROCSOLVER_ENTER("ormtr_unmtr_hb2st", "side:", side, "trans:", trans,
                    "m:", m, "n:", n, "kd:", kd, "shiftV:", shiftV,
                    "ldv:", ldv, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // quick return
    // todo: x == 0 instead of !x ?
    if(!m || !n || !kd || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // TODO: implement the main logic
    // This will involve applying the Householder reflectors stored in V
    // to the matrix C

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_ormtr_unmtr_hb2st_template(rocblas_handle handle,
                                                    const rocblas_side side,
                                                    const rocblas_operation trans,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    const rocblas_int kd,
                                                    U V,
                                                    const rocblas_int shiftV,
                                                    const rocblas_int ldv,
                                                    const rocblas_stride strideV,
                                                    T* tau,
                                                    const rocblas_stride strideT,
                                                    U C,
                                                    const rocblas_int shiftC,
                                                    const rocblas_int ldc,
                                                    const rocblas_stride strideC,
                                                    const rocblas_int batch_count,
                                                    T* scalars,
                                                    T* work,
                                                    void* work2,
                                                    void* work3,
                                                    void* work4,
                                                    T** workArr,
                                                    bool optim_mem)
{
    ROCSOLVER_ENTER("ormtr_unmtr_hb2st", "side:", side, "trans:", trans,
                    "m:", m, "n:", n, "kd:", kd, "shiftV:", shiftV,
                    "ldv:", ldv, "shiftC:", shiftC, "ldc:", ldc,
                    "bc:", batch_count);

    // quick return
    if(!m || !n || !kd || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // TODO: implement the main logic
    // This will involve applying the Householder reflectors stored in V
    // to the matrix C

    return rocblas_status_success;
}

/** Adapts V and C to be of the same type **/
template <bool BATCHED, bool STRIDED, typename T>
rocblas_status rocsolver_ormtr_unmtr_hb2st_template(rocblas_handle handle,
                                                    const rocblas_side side,
                                                    const rocblas_operation trans,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    const rocblas_int kd,
                                                    T* const V[],
                                                    const rocblas_int shiftV,
                                                    const rocblas_int ldv,
                                                    const rocblas_stride strideV,
                                                    T* tau,
                                                    const rocblas_stride strideT,
                                                    T* C,
                                                    const rocblas_int shiftC,
                                                    const rocblas_int ldc,
                                                    const rocblas_stride strideC,
                                                    const rocblas_int batch_count,
                                                    T* scalars,
                                                    T* work,
                                                    T** workArr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, C, strideC,
                            batch_count);

    return rocsolver_ormtr_unmtr_hb2st_template<BATCHED, STRIDED>(
        handle, side, trans, m, n, kd, V, shiftV, ldv, strideV, tau, strideT,
        cast2constType(workArr), shiftC, ldc, strideC, batch_count, scalars, work,
        workArr + batch_count);
}

template <bool BATCHED, bool STRIDED, typename T>
rocblas_status rocsolver_ormtr_unmtr_hb2st_template(rocblas_handle handle,
                                                    const rocblas_side side,
                                                    const rocblas_operation trans,
                                                    const rocblas_int m,
                                                    const rocblas_int n,
                                                    const rocblas_int kd,
                                                    T* const V[],
                                                    const rocblas_int shiftV,
                                                    const rocblas_int ldv,
                                                    const rocblas_stride strideV,
                                                    T* tau,
                                                    const rocblas_stride strideT,
                                                    T* C,
                                                    const rocblas_int shiftC,
                                                    const rocblas_int ldc,
                                                    const rocblas_stride strideC,
                                                    const rocblas_int batch_count,
                                                    T* scalars,
                                                    T* work,
                                                    void* work2,
                                                    void* work3,
                                                    void* work4,
                                                    T** workArr,
                                                    bool optim_mem)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks = (batch_count - 1) / 256 + 1;
    ROCSOLVER_LAUNCH_KERNEL(get_array, dim3(blocks), dim3(256), 0, stream, workArr, C, strideC,
                            batch_count);

    return rocsolver_ormtr_unmtr_hb2st_template<BATCHED, STRIDED>(
        handle, side, trans, m, n, kd, V, shiftV, ldv, strideV, tau, strideT,
        cast2constType(workArr), shiftC, ldc, strideC, batch_count, scalars, work, work2, work3,
        work4, workArr + batch_count, optim_mem);
}

ROCSOLVER_END_NAMESPACE
