/****************************************************************************
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

#include "rocauxiliary_lacgv.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const rocblas_int n,
                                     const rocblas_int k,
                                     U V,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[j + i * ldf] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * Vp[i + j * ldv];
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + i * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * Vp[(n - k + i) + j * ldv];
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + (n - k + i) * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
    }
}

template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const rocblas_int n,
                                     const rocblas_int k,
                                     U V,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[j + i * ldf] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * conj(Vp[i + j * ldv]);
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + i * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                    Fp[j + i * ldf] = -tp[i] * conj(Vp[(n - k + i) + j * ldv]);
                else
                    Fp[j + i * ldf] = -tp[i] * Vp[j + (n - k + i) * ldv];
            }
            else
                Fp[j + i * ldf] = 0;
        }
    }
}

template <typename T>
ROCSOLVER_KERNEL void set_tau(const rocblas_int k, T* tau, const rocblas_stride strideT)
{
    const auto b = hipBlockIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k)
    {
        T* tp = tau + b * strideT;
        tp[i] = -tp[i];
    }
}

template <bool BATCHED, typename T>
void rocsolver_larft_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_workArr)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of re-usable workspace
    *size_work = sizeof(T) * k * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_larft_argCheck(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int ldv,
                                        const rocblas_int ldf,
                                        T V,
                                        U tau,
                                        U F)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool row = (storev == rocblas_row_wise);

    // 2. invalid size
    if(n < 0 || k < 1 || ldf < k)
        return rocblas_status_invalid_size;
    if((row && ldv < k) || (!row && ldv < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !V) || !tau || !F)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larft_template(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* F,
                                        const rocblas_int ldf,
                                        const rocblas_stride strideF,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T** workArr)
{
    ROCSOLVER_ENTER("larft", "direct:", direct, "storev:", storev, "n:", n, "k:", k,
                    "shiftV:", shiftV, "ldv:", ldv, "ldf:", ldf, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    rocblas_stride stridew = rocblas_stride(k);
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_fill uplo;
    rocblas_operation trans;

    // Fix diagonal of T, make zero the not used triangular part,
    // setup tau (changing signs) and account for the non-stored 1's on the
    // householder vectors
    rocblas_int blocks = (k - 1) / 32 + 1;
    ROCSOLVER_LAUNCH_KERNEL(set_triangular, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                            stream, n, k, V, shiftV, ldv, strideV, tau, strideT, F, ldf, strideF,
                            direct, storev);
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(32, 1), 0, stream, k, tau,
                            strideT);

    if(direct == rocblas_forward_direction)
    {
        uplo = rocblas_fill_upper;

        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        for(rocblas_int i = 1; i < k; ++i)
        {
            // compute the matrix vector product, using the householder vectors
            if(storev == rocblas_column_wise)
            {
                trans = rocblas_operation_conjugate_transpose;
                rocblasCall_gemv<T>(handle, trans, n - 1 - i, i, tau + i, strideT, V,
                                    shiftV + idx2D(i + 1, 0, ldv), ldv, strideV, V,
                                    shiftV + idx2D(i + 1, i, ldv), 1, strideV, scalars + 2, 0, F,
                                    idx2D(0, i, ldf), 1, strideF, batch_count, workArr);
            }
            else
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - i - 1, V, shiftV + idx2D(i, i + 1, ldv),
                                                ldv, strideV, batch_count);

                trans = rocblas_operation_none;
                rocblasCall_gemv<T>(handle, trans, i, n - 1 - i, tau + i, strideT, V,
                                    shiftV + idx2D(0, i + 1, ldv), ldv, strideV, V,
                                    shiftV + idx2D(i, i + 1, ldv), ldv, strideV, scalars + 2, 0, F,
                                    idx2D(0, i, ldf), 1, strideF, batch_count, workArr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - i - 1, V, shiftV + idx2D(i, i + 1, ldv),
                                                ldv, strideV, batch_count);
            }

            // multiply by the previous triangular factor
            trans = rocblas_operation_none;
            rocblasCall_trmv<T>(handle, uplo, trans, diag, i, F, 0, ldf, strideF, F,
                                idx2D(0, i, ldf), 1, strideF, work, stridew, batch_count);
        }
    }
    else
    {
        uplo = rocblas_fill_lower;

        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        for(rocblas_int i = k - 2; i >= 0; --i)
        {
            // compute the matrix vector product, using the householder vectors
            if(storev == rocblas_column_wise)
            {
                trans = rocblas_operation_conjugate_transpose;
                rocblasCall_gemv<T>(handle, trans, n - k + i, k - i - 1, tau + i, strideT, V,
                                    shiftV + idx2D(0, i + 1, ldv), ldv, strideV, V,
                                    shiftV + idx2D(0, i, ldv), 1, strideV, scalars + 2, 0, F,
                                    idx2D(i + 1, i, ldf), 1, strideF, batch_count, workArr);
            }
            else
            {
                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - k + i, V, shiftV + idx2D(i, 0, ldv),
                                                ldv, strideV, batch_count);

                trans = rocblas_operation_none;
                rocblasCall_gemv<T>(handle, trans, k - i - 1, n - k + i, tau + i, strideT, V,
                                    shiftV + idx2D(i + 1, 0, ldv), ldv, strideV, V,
                                    shiftV + idx2D(i, 0, ldv), ldv, strideV, scalars + 2, 0, F,
                                    idx2D(i + 1, i, ldf), 1, strideF, batch_count, workArr);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, n - k + i, V, shiftV + idx2D(i, 0, ldv),
                                                ldv, strideV, batch_count);
            }

            // multiply by the previous triangular factor
            trans = rocblas_operation_none;
            rocblasCall_trmv<T>(handle, uplo, trans, diag, k - i - 1, F, idx2D(i + 1, i + 1, ldf),
                                ldf, strideF, F, idx2D(i + 1, i, ldf), 1, strideF, work, stridew,
                                batch_count);
        }
    }

    // restore tau
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(32, 1), 0, stream, k, tau,
                            strideT);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
