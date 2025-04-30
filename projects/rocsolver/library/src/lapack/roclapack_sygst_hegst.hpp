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

#include "rocblas.hpp"
#include "roclapack_sygs2_hegs2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_sygst_hegst_getMemorySize(const rocblas_fill uplo,
                                         const rocblas_eform itype,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work_x_temp,
                                         size_t* size_workArr_temp_arr,
                                         size_t* size_store_wcs_invA,
                                         size_t* size_invA_arr,
                                         bool* optim_mem)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_x_temp = 0;
        *size_workArr_temp_arr = 0;
        *size_store_wcs_invA = 0;
        *size_invA_arr = 0;
        *optim_mem = true;
        return;
    }

    if(n < xxGST_BLOCKSIZE)
    {
        // requirements for calling a single SYGS2/HEGS2
        rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, n, batch_count, size_scalars,
                                                        size_work_x_temp, size_store_wcs_invA,
                                                        size_workArr_temp_arr);
        *size_invA_arr = 0;
        *optim_mem = true;
    }
    else
    {
        rocblas_int kb = xxGST_BLOCKSIZE;
        size_t temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

        // requirements for calling SYGS2/HEGS2 for the subblocks
        rocsolver_sygs2_hegs2_getMemorySize<BATCHED, T>(itype, kb, batch_count, size_scalars,
                                                        size_work_x_temp, size_store_wcs_invA,
                                                        size_workArr_temp_arr);
        *size_invA_arr = 0;

        if(itype == rocblas_eform_ax)
        {
            // extra requirements for calling TRSM
            if(uplo == rocblas_fill_upper)
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_left, rocblas_operation_conjugate_transpose, n - kb, kb,
                    batch_count, &temp1, &temp2, &temp3, &temp4, optim_mem);
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none,
                                                        n - kb, kb, batch_count, &temp5, &temp6,
                                                        &temp7, &temp8, optim_mem);
            }
            else
            {
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none,
                                                        n - kb, kb, batch_count, &temp1, &temp2,
                                                        &temp3, &temp4, optim_mem);
                rocsolver_trsm_mem<BATCHED, STRIDED, T>(
                    rocblas_side_right, rocblas_operation_conjugate_transpose, n - kb, kb,
                    batch_count, &temp5, &temp6, &temp7, &temp8, optim_mem);
            }

            *size_work_x_temp = std::max(*size_work_x_temp, std::max(temp1, temp5));
            *size_workArr_temp_arr = std::max(*size_workArr_temp_arr, std::max(temp2, temp6));
            *size_store_wcs_invA = std::max(*size_store_wcs_invA, std::max(temp3, temp7));
            *size_invA_arr = std::max(*size_invA_arr, std::max(temp4, temp8));
        }
        else
            *optim_mem = true;
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sygst_hegst_template(rocblas_handle handle,
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
                                              void* work_x_temp,
                                              void* workArr_temp_arr,
                                              void* store_wcs_invA,
                                              void* invA_arr,
                                              bool optim_mem)
{
    ROCSOLVER_ENTER("sygst_hegst", "itype:", itype, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftB:", shiftB, "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int nb = xxGST_BLOCKSIZE;

    // if the matrix is too small, use the unblocked variant of the algorithm
    if(n <= nb)
        return rocsolver_sygs2_hegs2_template<BATCHED, T>(
            handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
            scalars, work_x_temp, store_wcs_invA, (T**)workArr_temp_arr);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    S s_one = 1;
    T t_one = 1;
    T t_half = 0.5;
    T t_minone = -1;
    T t_minhalf = -0.5;

    if(itype == rocblas_eform_ax)
    {
        if(uplo == rocblas_fill_upper)
        {
            // Compute inv(U')*A*inv(U)
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = std::min(n - k, nb);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    store_wcs_invA, (T**)workArr_temp_arr);

                if(k + kb < n)
                {
                    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, kb, n - k - kb, B, shiftB + idx2D(k, k, ldb),
                        ldb, strideB, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count,
                        optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);

                    rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_syr2k_her2k<BATCHED, T>(
                        handle, uplo, rocblas_operation_conjugate_transpose, n - k - kb, kb,
                        &t_minone, A, shiftA + idx2D(k, k + kb, lda), lda, strideA, B,
                        shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &s_one, A,
                        shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, n - k - kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k, k + kb, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count);

                    rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                        handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
                        kb, n - k - kb, B, shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
                        shiftA + idx2D(k, k + kb, lda), lda, strideA, batch_count, optim_mem,
                        work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);
                }
            }
        }
        else
        {
            // Compute inv(L)*A*inv(L')
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = std::min(n - k, nb);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    store_wcs_invA, (T**)workArr_temp_arr);

                if(k + kb < n)
                {
                    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                        handle, rocblas_side_right, rocblas_operation_conjugate_transpose,
                        rocblas_diagonal_non_unit, n - k - kb, kb, B, shiftB + idx2D(k, k, ldb),
                        ldb, strideB, A, shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count,
                        optim_mem, work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);

                    rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, n - k - kb, kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count);

                    rocblasCall_syr2k_her2k<BATCHED, T>(
                        handle, uplo, rocblas_operation_none, n - k - kb, kb, &t_minone, A,
                        shiftA + idx2D(k + kb, k, lda), lda, strideA, B,
                        shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &s_one, A,
                        shiftA + idx2D(k + kb, k + kb, lda), lda, strideA, batch_count);

                    rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, n - k - kb, kb,
                                          &t_minhalf, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                                          shiftB + idx2D(k + kb, k, ldb), ldb, strideB, &t_one, A,
                                          shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count);

                    rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit,
                        n - k - kb, kb, B, shiftB + idx2D(k + kb, k + kb, ldb), ldb, strideB, A,
                        shiftA + idx2D(k + kb, k, lda), lda, strideA, batch_count, optim_mem,
                        work_x_temp, workArr_temp_arr, store_wcs_invA, invA_arr);
                }
            }
        }
    }
    else
    {
        if(uplo == rocblas_fill_upper)
        {
            // Compute U*A*U'
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = std::min(n - k, nb);

                rocblasCall_trmm(handle, rocblas_side_left, uplo, rocblas_operation_none,
                                 rocblas_diagonal_non_unit, k, kb, &t_one, 0, B, shiftB, ldb,
                                 strideB, A, shiftA + idx2D(0, k, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

                rocblasCall_syr2k_her2k<BATCHED, T>(
                    handle, uplo, rocblas_operation_none, k, kb, &t_one, A,
                    shiftA + idx2D(0, k, lda), lda, strideA, B, shiftB + idx2D(0, k, ldb), ldb,
                    strideB, &s_one, A, shiftA, lda, strideA, batch_count);

                rocblasCall_symm_hemm(handle, rocblas_side_right, uplo, k, kb, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(0, k, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(0, k, lda), lda, strideA, batch_count);

                rocblasCall_trmm(handle, rocblas_side_right, uplo,
                                 rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit,
                                 k, kb, &t_one, 0, B, shiftB + idx2D(k, k, ldb), ldb, strideB, A,
                                 shiftA + idx2D(0, k, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    store_wcs_invA, (T**)workArr_temp_arr);
            }
        }
        else
        {
            // Compute L'*A*L
            for(rocblas_int k = 0; k < n; k += nb)
            {
                rocblas_int kb = std::min(n - k, nb);

                rocblasCall_trmm(handle, rocblas_side_right, uplo, rocblas_operation_none,
                                 rocblas_diagonal_non_unit, kb, k, &t_one, 0, B, shiftB, ldb,
                                 strideB, A, shiftA + idx2D(k, 0, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, k, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(k, 0, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_syr2k_her2k<BATCHED, T>(
                    handle, uplo, rocblas_operation_conjugate_transpose, k, kb, &t_one, A,
                    shiftA + idx2D(k, 0, lda), lda, strideA, B, shiftB + idx2D(k, 0, ldb), ldb,
                    strideB, &s_one, A, shiftA, lda, strideA, batch_count);

                rocblasCall_symm_hemm(handle, rocblas_side_left, uplo, kb, k, &t_half, A,
                                      shiftA + idx2D(k, k, lda), lda, strideA, B,
                                      shiftB + idx2D(k, 0, ldb), ldb, strideB, &t_one, A,
                                      shiftA + idx2D(k, 0, lda), lda, strideA, batch_count);

                rocblasCall_trmm(handle, rocblas_side_left, uplo,
                                 rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit,
                                 kb, k, &t_one, 0, B, shiftB + idx2D(k, k, ldb), ldb, strideB, A,
                                 shiftA + idx2D(k, 0, lda), lda, strideA, batch_count,
                                 (T**)workArr_temp_arr);

                rocsolver_sygs2_hegs2_template<BATCHED, T>(
                    handle, itype, uplo, kb, A, shiftA + idx2D(k, k, lda), lda, strideA, B,
                    shiftB + idx2D(k, k, ldb), ldb, strideB, batch_count, scalars, work_x_temp,
                    store_wcs_invA, (T**)workArr_temp_arr);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
