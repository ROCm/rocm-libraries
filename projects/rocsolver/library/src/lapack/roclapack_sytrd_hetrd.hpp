/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_latrd.hpp"
#include "rocblas.hpp"
#include "roclapack_sytd2_hetd2.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_sytrd_hetrd_getMemorySize(const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_norms,
                                         size_t* size_tmptau_W,
                                         size_t* size_workArr,
                                         bool recover_A = true)
{
    *size_scalars = 0;
    *size_work = 0;
    *size_norms = 0;
    *size_tmptau_W = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
        return;

    size_t s1 = 0, s2 = 0;
    size_t w1 = 0, w2 = 0;
    size_t n1 = 0, n2 = 0;
    size_t na;

    // extra requirements to call SYTD2/HETD2
    rocsolver_sytd2_hetd2_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w2, &n2, &s2,
                                                    size_workArr);

    if(n > xxTRD_xxTD2_SWITCHSIZE)
    {
        // size required to store temporary matrix W
        s1 = n * xxTRD_BLOCKSIZE;
        s1 *= sizeof(T) * batch_count;

        // extra requirements to call latrd_forsytrd
        rocsolver_latrd_forsytrd_getMemorySize<BATCHED, T>(n, xxTRD_BLOCKSIZE, batch_count,
                                                           size_scalars, &w1, &n1, size_workArr);
    }

    *size_tmptau_W = std::max(s1, s2);
    *size_work = std::max(w1, w2);
    *size_norms = std::max(n1, n2);

    // when recovering the non-referenced part of A is necessary,
    // add the required buffer size to hold a copy
    // TODO: Actually we only need to hold half of the elements (upper or lower part)
    // The methods to do the copies can be modified for this new data layout in the future
    if(recover_A)
        *size_work += sizeof(T) * n * n * batch_count;
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytrd_hetrd_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              S D,
                                              S E,
                                              U tau,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !D) || (n > 1 && !E) || (n > 1 && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, typename T, typename S, typename U>
rocblas_status rocsolver_sytrd_hetrd_template(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              T* tau,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work_Acpy,
                                              T* norms,
                                              T* tmptau_W,
                                              T** workArr,
                                              bool recover_A = true)
{
    ROCSOLVER_ENTER("sytrd_hetrd", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int k = xxTRD_BLOCKSIZE;
    rocblas_int kk = xxTRD_xxTD2_SWITCHSIZE;

    // if the matrix is too small, use the unblocked variant of the algorithm
    if(n <= kk)
        return rocsolver_sytd2_hetd2_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                              E, strideE, tau, strideP, batch_count, scalars,
                                              work_Acpy, norms, tmptau_W, workArr);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    const T minone = T(-1);
    const T one = T(1);

    rocblas_int ldw = n;
    rocblas_stride strideW = n * k;
    rocblas_int j;

    // make A a general matrix by copying its upper (lower) part
    // also copy the non referenced part of A to recover it if necessary
    T* work;
    T* Acpy;
    rocblas_int blocks = (n - 1) / BS2 + 1;
    rocblas_fill uplo2 = (uplo == rocblas_fill_upper) ? rocblas_fill_lower : rocblas_fill_upper;

    if(recover_A)
    {
        Acpy = work_Acpy;
        work = Acpy + n * n * batch_count;

        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2, 1),
                                0, stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, Acpy,
                                no_mask{}, uplo2, rocblas_diagonal_unit);
    }
    else
        work = work_Acpy;

    ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2, 1), 0, stream, rocblas_operation_conjugate_transpose, n,
                            n, A, shiftA, lda, strideA, A, shiftA, lda, strideA, no_mask{}, uplo,
                            rocblas_diagonal_unit);

    if(uplo == rocblas_fill_lower)
    {
        // reduce the lower part of A
        // main loop running forwards (for each block of columns)
        // when the unreduced part is not large enough, switch to unblocked algorithm
        j = 0;
        while(j < n - kk)
        {
            // reduce columns j:j+k-1
            rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A,
                                                 shiftA + idx2D(j, j, lda), lda, strideA, (E + j),
                                                 strideE, (tau + j), strideP, tmptau_W, 0, ldw,
                                                 strideW, batch_count, scalars, work, norms, workArr);

            // update trailing matrix
            // A = A - V*W' - W*V'
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                           n - j - k, n - j - k, k, &minone, A, shiftA + idx2D(j + k, j, lda), lda,
                           strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, &one, A,
                           shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                           n - j - k, n - j - k, k, &minone, tmptau_W, idx2D(k, 0, ldw), ldw,
                           strideW, A, shiftA + idx2D(j + k, j, lda), lda, strideA, &one, A,
                           shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);

            j += k;
        }

        // reduce last columns of A
        rocsolver_sytd2_hetd2_template<T>(handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                          strideA, (D + j), strideD, (E + j), strideE, (tau + j),
                                          strideP, batch_count, scalars, work, norms, tmptau_W,
                                          workArr);
    }

    else
    {
        // reduce the upper part of A
        // main loop running backwards (for each block of columns)
        // when the unreduced part is not large enough, switch to unblocked algorithm
        j = n - k;
        rocblas_int upkk = n - ((n - kk + k - 1) / k) * k;
        while(j >= upkk)
        {
            // reduce columns j:j+k-1
            rocsolver_latrd_forsytrd_template<T>(handle, uplo, j + k, k, A, shiftA, lda, strideA, E,
                                                 strideE, tau, strideP, tmptau_W, 0, ldw, strideW,
                                                 batch_count, scalars, work, norms, workArr);

            // update trailing matrix
            // A = A - V*W' - W*V'
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, j,
                           j, k, &minone, A, shiftA + idx2D(0, j, lda), lda, strideA, tmptau_W, 0,
                           ldw, strideW, &one, A, shiftA, lda, strideA, batch_count, workArr);
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, j,
                           j, k, &minone, tmptau_W, 0, ldw, strideW, A, shiftA + idx2D(0, j, lda),
                           lda, strideA, &one, A, shiftA, lda, strideA, batch_count, workArr);

            j -= k;
        }

        // reduce first columns of A
        rocsolver_sytd2_hetd2_template<T>(handle, uplo, upkk, A, shiftA, lda, strideA, D, strideD,
                                          E, strideE, tau, strideP, batch_count, scalars, work,
                                          norms, tmptau_W, workArr);
    }

    // Copy results (set tridiagonal form in A)
    blocks = (n - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, dim3(blocks, batch_count), dim3(BS1), 0, stream, uplo,
                            n, A, shiftA, lda, strideA, D, strideD, E, strideE);

    // recover non-referenced part of A if necessary
    if(recover_A)
    {
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2, 1),
                                0, stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, Acpy,
                                no_mask{}, uplo2, rocblas_diagonal_unit);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
