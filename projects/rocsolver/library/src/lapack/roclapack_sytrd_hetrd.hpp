/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsolver_device_workspace.hpp"

#include <rocprofiler-sdk-roctx/roctx.h>

ROCSOLVER_BEGIN_NAMESPACE

static const rocblas_int env_sytrd_blocksize = []() {
    const char* v = std::getenv("SYTRD_BLOCKSIZE");
    return v ? std::max(1, std::min(std::atoi(v), (int)xxTRD_BLOCKSIZE)) : (int)xxTRD_BLOCKSIZE;
}();

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

    // Panel size used for LATRD (0 = no LATRD call for this n, pure SYTD2)
    rocblas_int eff_k = (n <= 128) ? 0 : (n <= 256) ? 32 : xxTRD_BLOCKSIZE;

    if(eff_k > 0)
    {
        // size required to store temporary matrix W (n rows x eff_k cols)
        s1 = (size_t)n * eff_k * sizeof(T) * batch_count;

        // extra requirements to call latrd_forsytrd
        rocsolver_latrd_forsytrd_getMemorySize<BATCHED, T>(n, eff_k, batch_count, size_scalars, &w1,
                                                           &n1, size_workArr);
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
    roctxRangePush("rocsolver_sytrd_hetrd");

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // n <= 128: SYTD2 handles everything directly (small kernel, no LATRD overhead).
    // 128 < n <= 256: LATRD panel k=32, trailing remainder <= 32 goes to SYTD2 small kernel.
    // 256 < n <= 2048: LATRD panel k=64, trailing remainder <= 32 goes to SYTD2 small kernel.
    // n > 2048: unchanged (k=env_sytrd_blocksize, kk=xxTRD_xxTD2_SWITCHSIZE).
    if(n <= 128)
        return rocsolver_sytd2_hetd2_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                              E, strideE, tau, strideP, batch_count, scalars,
                                              work_Acpy, norms, tmptau_W, workArr);

    rocblas_int k, kk;
    if(n <= 256)
    {
        k = 32;
        kk = 32;
    }
    else if(n <= 2048)
    {
        k = env_sytrd_blocksize;
        kk = 32;
    }
    else
    {
        k = env_sytrd_blocksize;
        kk = xxTRD_xxTD2_SWITCHSIZE;
    }

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
    roctxRangePop();
    return rocblas_status_success;
}

template <bool BATCHED, typename T, typename S, typename U>
auto rocsolver_sytrd_hetrd_getWorkItems(rocblas_handle handle,
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
                                        bool recover_A = true)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // extra requirements for calling SYTD2/HETD2
    size_t size_norms, size_work, size_tmptau_W;
    // size of array of pointers to workspace (batched case)
    size_t size_workArr;
    rocsolver_sytrd_hetrd_getMemorySize<false, T>(n, batch_count, &size_scalars, &size_work,
                                                  &size_norms, &size_tmptau_W, &size_workArr);

    auto sytrd_work_items = create_work_item({"sytrd_scalars", size_scalars})
        + create_work_item({"sytrd_work", size_work}) + create_work_item({"sytrd_norms", size_norms})
        + create_work_item({"sytrd_tmptau_W", size_tmptau_W})
        + create_work_item({"sytrd_workArr", size_workArr});

    // eff_k mirrors the panel size used by getMemorySize (0 => pure SYTD2, no LATRD call)
    rocblas_int eff_k = (n <= 128) ? 0 : (n <= 256) ? 32 : xxTRD_BLOCKSIZE;
    rocblas_int ldw = n;
    rocblas_stride strideW = (rocblas_stride)n * (eff_k > 0 ? eff_k : 1);
    auto latrd_work_items = rocsolver_latrd_forsytrd_getWorkItems(
        handle, uplo, n, eff_k, A, shiftA, lda, strideA, E, strideE, tau, strideP, (T*)nullptr, 0,
        ldw, strideW, batch_count);

    return latrd_work_items + sytrd_work_items;
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
                                              rocsolver_device_workspace_ptr_t dwptr,
                                              /* T* scalars, */
                                              /* T* work_Acpy, */
                                              /* T* norms, */
                                              /* T* tmptau_W, */
                                              /* T** workArr, */
                                              bool recover_A = true)
{
    roctxRangePush("rocsolver_sytrd_hetrd_alt");
    ROCSOLVER_ENTER("sytrd_hetrd_alt", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);
    ROCSOLVER_INIT_DEVICE_WORKSPACE(dwptr,
                                    rocsolver_sytrd_hetrd_getWorkItems<BATCHED>(
                                        handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                        strideE, tau, strideP, batch_count, recover_A));

    T* scalars = (T*)dwptr->work("sytrd_scalars");
    T* work_Acpy = (T*)dwptr->work("sytrd_work");
    T* norms = (T*)dwptr->work("sytrd_norms");
    T* tmptau_W = (T*)dwptr->work("sytrd_tmptau_W");
    T** workArr = (T**)dwptr->work("sytrd_workArr");

    if(dwptr->size("sytrd_scalars") > 0)
        init_scalars(handle, (T*)scalars);

    /* // execution */
    /* return rocsolver_sytrd_hetrd_template<false, T>( */
    /*     handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count, */
    /*     (T*)scalars, (T*)work, (T*)norms, (T*)tmptau_W, (T**)workArr); */

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // n <= 128: SYTD2 handles everything directly (small kernel, no LATRD overhead).
    // 128 < n <= 256: LATRD panel k=32, trailing remainder <= 32 goes to SYTD2 small kernel.
    // 256 < n <= 2048: LATRD panel k=64, trailing remainder <= 32 goes to SYTD2 small kernel.
    // n > 2048: unchanged (k=env_sytrd_blocksize, kk=xxTRD_xxTD2_SWITCHSIZE).
    if(n <= 128)
        return rocsolver_sytd2_hetd2_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                              E, strideE, tau, strideP, batch_count, scalars,
                                              work_Acpy, norms, tmptau_W, workArr);

    rocblas_int k, kk;
    if(n <= 256)
    {
        k = 32;
        kk = 32;
    }
    else if(n <= 2048)
    {
        k = env_sytrd_blocksize;
        kk = 32;
    }
    else
    {
        k = env_sytrd_blocksize;
        kk = xxTRD_xxTD2_SWITCHSIZE;
    }

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
            rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A, shiftA + idx2D(j, j, lda),
                                                 lda, strideA, (E + j), strideE, (tau + j), strideP,
                                                 tmptau_W, 0, ldw, strideW, batch_count, dwptr);

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
                                                 batch_count, dwptr);

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
    roctxRangePop();
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
