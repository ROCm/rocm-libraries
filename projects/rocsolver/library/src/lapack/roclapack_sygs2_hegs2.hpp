/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
ROCSOLVER_KERNEL void sygs2_set_diag1(const rocblas_int k,
                                      U AA,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      U BB,
                                      const rocblas_int shiftB,
                                      const rocblas_int ldb,
                                      const rocblas_stride strideB,
                                      T* work,
                                      const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    constexpr rocblas_stride strideW = 3;

    if(b < batch_count)
    {
        T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
        T* B = load_ptr_batch<T>(BB, b, shiftB, strideB);
        T* W = work + b * strideW;

        T akk = A[k + k * lda];
        T bkk = B[k + k * ldb];
        akk /= bkk * bkk;
        A[k + k * lda] = akk;

        W[0] = T(1.0) / bkk;
        W[1] = T(-0.5) * akk;
    }
}
template <typename T, typename U>
ROCSOLVER_KERNEL void sygs2_set_diag2(const rocblas_int k,
                                      U AA,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      U BB,
                                      const rocblas_int shiftB,
                                      const rocblas_int ldb,
                                      const rocblas_stride strideB,
                                      T* work,
                                      const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    constexpr rocblas_stride strideW = 3;

    if(b < batch_count)
    {
        T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
        T* B = load_ptr_batch<T>(BB, b, shiftB, strideB);
        T* W = work + b * strideW;

        T akk = A[k + k * lda];
        T bkk = B[k + k * ldb];

        W[0] = bkk;
        W[1] = T(0.5) * akk;
        W[2] = akk * (bkk * bkk);
    }
}
template <typename T, typename U>
ROCSOLVER_KERNEL void sygs2_set_diag3(const rocblas_int k,
                                      U AA,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      T* work,
                                      const rocblas_int batch_count)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    constexpr rocblas_stride strideW = 3;

    if(b < batch_count)
    {
        T* A = load_ptr_batch<T>(AA, b, shiftA, strideA);
        T* W = work + b * strideW;

        A[k + k * lda] = W[2];
    }
}

template <bool BATCHED, typename T>
void rocsolver_sygs2_hegs2_getMemorySize(const rocblas_eform itype,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_store_wcs,
                                         size_t* size_workArr)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_store_wcs = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of stored value array
    *size_store_wcs = sizeof(T) * 3 * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    if(itype == rocblas_eform_ax)
    {
        // extra workspace (for calling TRSV)
        *size_store_wcs = std::max(*size_store_wcs, sizeof(rocblas_int) * batch_count);
        *size_work = 0;
    }
    else
    {
        // extra workspace (for calling TRMV)
        *size_work = sizeof(T) * n * batch_count;
    }
}

template <typename T>
rocblas_status rocsolver_sygs2_hegs2_argCheck(rocblas_handle handle,
                                              const rocblas_eform itype,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              const rocblas_int ldb,
                                              T A,
                                              T B,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !B))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_sygs2_hegs2_template(rocblas_handle handle,
                                              const rocblas_eform itype,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              U B,
                                              const rocblas_int shiftB,
                                              const rocblas_int ldb,
                                              const rocblas_stride strideB,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              void* work,
                                              void* store_wcs,
                                              T** workArr)
{
    ROCSOLVER_ENTER("sygs2_hegs2", "itype:", itype, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    rocblas_int blocks_batch = (batch_count - 1) / BS1 + 1;
    rocblas_int waves_batch = (batch_count - 1) / warpSize + 1;
    dim3 blocks(blocks_batch, 1, 1);
    dim3 threads(std::min(BS1, warpSize * waves_batch), 1, 1);

    if(itype == rocblas_eform_ax)
    {
        rocblas_stride strideS = 3;

        if(uplo == rocblas_fill_upper)
        {
            // Compute inv(U')*A*inv(U)
            for(rocblas_int k = 0; k < n; k++)
            {
                // Set A[k, k] and store coefficients in store_wcs
                ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag1, blocks, threads, 0, stream, k, A, shiftA,
                                        lda, strideA, B, shiftB, ldb, strideB, (T*)store_wcs,
                                        batch_count);

                if(k < n - 1)
                {
                    rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                                        shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

                    if(COMPLEX)
                    {
                        rocsolver_lacgv_template<T>(handle, n - k - 1, A,
                                                    shiftA + idx2D(k, k + 1, lda), lda, strideA,
                                                    batch_count);
                        rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                                    shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                                    batch_count);
                    }

                    rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                                        shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
                                        shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

                    rocblasCall_syr2_her2<T>(
                        handle, uplo, n - k - 1, scalars, A, shiftA + idx2D(k, k + 1, lda), lda,
                        strideA, B, shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
                        shiftA + idx2D(k + 1, k + 1, lda), lda, strideA, batch_count, workArr);

                    rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                                        shiftB + idx2D(k, k + 1, ldb), ldb, strideB, A,
                                        shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - k - 1, B,
                                                    shiftB + idx2D(k, k + 1, ldb), ldb, strideB,
                                                    batch_count);

                    rocblasCall_trsv(handle, uplo, rocblas_operation_conjugate_transpose,
                                     rocblas_diagonal_non_unit, n - k - 1, B,
                                     shiftB + idx2D(k + 1, k + 1, ldb), ldb, strideB, A,
                                     shiftA + idx2D(k, k + 1, lda), lda, strideA, batch_count,
                                     (rocblas_int*)store_wcs, workArr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - k - 1, A,
                                                    shiftA + idx2D(k, k + 1, lda), lda, strideA,
                                                    batch_count);
                }
            }
        }
        else
        {
            // Compute inv(L)*A*inv(L')
            for(rocblas_int k = 0; k < n; k++)
            {
                // Set A[k, k] and store coefficients in store_wcs
                ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag1, blocks, threads, 0, stream, k, A, shiftA,
                                        lda, strideA, B, shiftB, ldb, strideB, (T*)store_wcs,
                                        batch_count);

                if(k < n - 1)
                {
                    rocblasCall_scal<T>(handle, n - k - 1, (T*)store_wcs, strideS, A,
                                        shiftA + idx2D(k + 1, k, lda), 1, strideA, batch_count);

                    rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                                        shiftB + idx2D(k + 1, k, ldb), 1, strideB, A,
                                        shiftA + idx2D(k + 1, k, lda), 1, strideA, batch_count);

                    rocblasCall_syr2_her2<T>(
                        handle, uplo, n - k - 1, scalars, A, shiftA + idx2D(k + 1, k, lda), 1,
                        strideA, B, shiftB + idx2D(k + 1, k, ldb), 1, strideB, A,
                        shiftA + idx2D(k + 1, k + 1, lda), lda, strideA, batch_count, workArr);

                    rocblasCall_axpy<T>(handle, n - k - 1, ((T*)store_wcs) + 1, strideS, B,
                                        shiftB + idx2D(k + 1, k, ldb), 1, strideB, A,
                                        shiftA + idx2D(k + 1, k, lda), 1, strideA, batch_count);

                    rocblasCall_trsv(handle, uplo, rocblas_operation_none, rocblas_diagonal_non_unit,
                                     n - k - 1, B, shiftB + idx2D(k + 1, k + 1, ldb), ldb, strideB,
                                     A, shiftA + idx2D(k + 1, k, lda), 1, strideA, batch_count,
                                     (rocblas_int*)store_wcs, workArr);
                }
            }
        }
    }
    else
    {
        rocblas_stride strideS = 3;
        rocblas_stride strideW = rocblas_stride(n);

        if(uplo == rocblas_fill_upper)
        {
            // Compute U*A*U'
            for(rocblas_int k = 0; k < n; k++)
            {
                // Store coefficients in store_wcs
                ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag2, blocks, threads, 0, stream, k, A, shiftA,
                                        lda, strideA, B, shiftB, ldb, strideB, (T*)store_wcs,
                                        batch_count);

                rocblasCall_trmv<T>(handle, uplo, rocblas_operation_none, rocblas_diagonal_non_unit,
                                    k, B, shiftB, ldb, strideB, A, shiftA + idx2D(0, k, lda), 1,
                                    strideA, (T*)work, strideW, batch_count);

                rocblasCall_axpy<T>(handle, k, ((T*)store_wcs) + 1, strideS, B,
                                    shiftB + idx2D(0, k, ldb), 1, strideB, A,
                                    shiftA + idx2D(0, k, lda), 1, strideA, batch_count);

                rocblasCall_syr2_her2<T>(handle, uplo, k, scalars + 2, A, shiftA + idx2D(0, k, lda),
                                         1, strideA, B, shiftB + idx2D(0, k, ldb), 1, strideB, A,
                                         shiftA, lda, strideA, batch_count, workArr);

                rocblasCall_axpy<T>(handle, k, ((T*)store_wcs) + 1, strideS, B,
                                    shiftB + idx2D(0, k, ldb), 1, strideB, A,
                                    shiftA + idx2D(0, k, lda), 1, strideA, batch_count);

                rocblasCall_scal<T>(handle, k, (T*)store_wcs, strideS, A, shiftA + idx2D(0, k, lda),
                                    1, strideA, batch_count);

                // Set A[k, k]
                ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag3, blocks, threads, 0, stream, k, A, shiftA,
                                        lda, strideA, (T*)store_wcs, batch_count);
            }
        }
        else
        {
            // Compute L'*A*L
            for(rocblas_int k = 0; k < n; k++)
            {
                // Store coefficients in store_wcs
                ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag2, blocks, threads, 0, stream, k, A, shiftA,
                                        lda, strideA, B, shiftB, ldb, strideB, (T*)store_wcs,
                                        batch_count);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, k, A, shiftA + idx2D(k, 0, lda), lda,
                                                strideA, batch_count);

                rocblasCall_trmv<T>(handle, uplo, rocblas_operation_conjugate_transpose,
                                    rocblas_diagonal_non_unit, k, B, shiftB, ldb, strideB, A,
                                    shiftA + idx2D(k, 0, lda), lda, strideA, (T*)work, strideW,
                                    batch_count);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, k, B, shiftB + idx2D(k, 0, ldb), ldb,
                                                strideB, batch_count);

                rocblasCall_axpy<T>(handle, k, ((T*)store_wcs) + 1, strideS, B,
                                    shiftB + idx2D(k, 0, ldb), ldb, strideB, A,
                                    shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_syr2_her2<T>(handle, uplo, k, scalars + 2, A, shiftA + idx2D(k, 0, lda),
                                         lda, strideA, B, shiftB + idx2D(k, 0, ldb), ldb, strideB,
                                         A, shiftA, lda, strideA, batch_count, workArr);

                rocblasCall_axpy<T>(handle, k, ((T*)store_wcs) + 1, strideS, B,
                                    shiftB + idx2D(k, 0, ldb), ldb, strideB, A,
                                    shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, k, B, shiftB + idx2D(k, 0, ldb), ldb,
                                                strideB, batch_count);

                rocblasCall_scal<T>(handle, k, (T*)store_wcs, strideS, A, shiftA + idx2D(k, 0, lda),
                                    lda, strideA, batch_count);

                if(COMPLEX)
                    rocsolver_lacgv_template<T>(handle, k, A, shiftA + idx2D(k, 0, lda), lda,
                                                strideA, batch_count);

                // Set A[k, k]
                ROCSOLVER_LAUNCH_KERNEL(sygs2_set_diag3, blocks, threads, 0, stream, k, A, shiftA,
                                        lda, strideA, (T*)store_wcs, batch_count);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
