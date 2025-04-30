/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_ormlq_unmlq.hpp"
#include "auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_gels.hpp"
#include "roclapack_geqrf.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_gels_outofplace_getMemorySize(const rocblas_operation trans,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             const rocblas_int nrhs,
                                             const rocblas_int batch_count,
                                             size_t* size_scalars,
                                             size_t* size_work_x_temp,
                                             size_t* size_workArr_temp_arr,
                                             size_t* size_diag_trfac_invA,
                                             size_t* size_trfact_workTrmm_invA_arr,
                                             size_t* size_ipiv,
                                             size_t* size_savedB,
                                             bool* optim_mem)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || nrhs == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_x_temp = 0;
        *size_workArr_temp_arr = 0;
        *size_diag_trfac_invA = 0;
        *size_trfact_workTrmm_invA_arr = 0;
        *size_ipiv = 0;
        *size_savedB = 0;
        *optim_mem = true;
        return;
    }

    size_t unused;

    rocsolver_gels_getMemorySize<BATCHED, STRIDED, T>(
        trans, m, n, nrhs, batch_count, size_scalars, size_work_x_temp, size_workArr_temp_arr,
        size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, &unused, optim_mem);

    *size_ipiv = sizeof(T) * std::min(m, n) * batch_count;

    if((trans == rocblas_operation_none && m >= n) || (trans != rocblas_operation_none && m < n))
        *size_savedB = sizeof(T) * std::max(m, n) * nrhs * batch_count;
    else
        *size_savedB = 0;
}

template <bool COMPLEX, typename T>
rocblas_status rocsolver_gels_outofplace_argCheck(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  T A,
                                                  const rocblas_int lda,
                                                  T B,
                                                  const rocblas_int ldb,
                                                  T X,
                                                  const rocblas_int ldx,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if((COMPLEX && trans == rocblas_operation_transpose)
       || (!COMPLEX && trans == rocblas_operation_conjugate_transpose))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(trans == rocblas_operation_none && (ldb < m || ldx < n))
        return rocblas_status_invalid_size;
    if(trans != rocblas_operation_none && (ldb < n || ldx < m))
        return rocblas_status_invalid_size;
    if(m < 0 || n < 0 || nrhs < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    rocblas_int rowsB = (trans == rocblas_operation_none ? m : n);
    rocblas_int rowsX = (trans == rocblas_operation_none ? n : m);
    if((m * n && !A) || (rowsB * nrhs && !B) || (rowsX * nrhs && !X) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gels_outofplace_template(rocblas_handle handle,
                                                  rocblas_operation trans,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  const rocblas_int nrhs,
                                                  U A,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  U B,
                                                  const rocblas_int shiftB,
                                                  const rocblas_int ldb,
                                                  const rocblas_stride strideB,
                                                  U X,
                                                  const rocblas_int shiftX,
                                                  const rocblas_int ldx,
                                                  const rocblas_stride strideX,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count,
                                                  T* scalars,
                                                  T* work_x_temp,
                                                  T* workArr_temp_arr,
                                                  T* diag_trfac_invA,
                                                  T** trfact_workTrmm_invA_arr,
                                                  T* ipiv,
                                                  T* savedB,
                                                  bool optim_mem)
{
    ROCSOLVER_ENTER("gels_outofplace", "trans:", trans, "m:", m, "n:", n, "nrhs:", nrhs,
                    "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB, "ldb:", ldb,
                    "shiftX:", shiftX, "ldx:", ldx, "bc:", batch_count);

    // quick return if zero instances in batch
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with a nonsingular matrix)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if B and X are empty
    if(nrhs == 0)
        return rocblas_status_success;

    // quick return if A is empty
    if(m == 0 || n == 0)
    {
        rocblas_int rowsX = (trans == rocblas_operation_none ? n : m);
        rocblas_int blocksx = (rowsX - 1) / 32 + 1;
        rocblas_int blocksy = (nrhs - 1) / 32 + 1;
        ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocksx, blocksy, batch_count), dim3(32, 32), 0,
                                stream, rowsX, nrhs, X, shiftX, ldx, strideX);

        return rocblas_status_success;
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants in host memory
    const rocblas_stride strideP = std::min(m, n);
    const rocblas_int check_threads = std::min(((std::min(m, n) - 1) / 64 + 1) * 64, BS1);
    const rocblas_int copyblocksmin = (std::min(m, n) - 1) / 32 + 1;
    const rocblas_int copyblocksmax = (std::max(m, n) - 1) / 32 + 1;
    const rocblas_int copyblocksy = (nrhs - 1) / 32 + 1;

    // TODO: apply scaling to improve accuracy over a larger range of values

    // If rows(B) < rows(X), then copy B to X and compute result in X.
    // If rows(B) > rows(X), then copy B to savedB, compute result in B, copy result from B to X, and restore B from savedB.

    if(m >= n)
    {
        // compute QR factorization of A
        rocsolver_geqrf_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count, scalars, work_x_temp,
            workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

        if(trans == rocblas_operation_none)
        {
            // save B in savedB
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmax, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, copymat_to_buffer, m, nrhs, B, shiftB,
                                    ldb, strideB, savedB);

            rocsolver_ormqr_unmqr_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose, m, nrhs, n, A,
                shiftA, lda, strideA, ipiv, strideP, B, shiftB, ldb, strideB, batch_count, scalars,
                work_x_temp, workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

            ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1),
                                    dim3(1, check_threads, 1), 0, stream, n, A, shiftA, lda,
                                    strideA, info);

            // solve RX = Q'B
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
                nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count, optim_mem,
                work_x_temp, workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

            // copy result to X
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmin, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, n, nrhs, B, shiftB, ldb, strideB, X,
                                    shiftX, ldx, strideX);

            // restore B from savedB
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmax, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, copymat_from_buffer, m, nrhs, B,
                                    shiftB, ldb, strideB, savedB);
        }
        else
        {
            ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1),
                                    dim3(1, check_threads, 1), 0, stream, n, A, shiftA, lda,
                                    strideA, info);

            // copy B to X
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmin, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, n, nrhs, B, shiftB, ldb, strideB, X,
                                    shiftX, ldx, strideX);

            // solve R'Y = B (here Y = Q'X)
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_diagonal_non_unit, n, nrhs, A, shiftA, lda, strideA, X, shiftX, ldx,
                strideX, batch_count, optim_mem, work_x_temp, workArr_temp_arr, diag_trfac_invA,
                trfact_workTrmm_invA_arr);

            // zero row n to m-1 of X in cases where info is zero
            const rocblas_int zeroblocksx = (m - n - 1) / 32 + 1;
            ROCSOLVER_LAUNCH_KERNEL((gels_set_zero<T, U>),
                                    dim3(zeroblocksx, copyblocksy, batch_count), dim3(32, 32), 0,
                                    stream, n, m, nrhs, X, shiftX, ldx, strideX, info);

            rocsolver_ormqr_unmqr_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_none, m, nrhs, n, A, shiftA, lda,
                strideA, ipiv, strideP, X, shiftX, ldx, strideX, batch_count, scalars, work_x_temp,
                workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);
        }
    }
    else
    {
        // compute LQ factorization of A
        rocsolver_gelqf_template<BATCHED, STRIDED>(
            handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count, scalars, work_x_temp,
            workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

        if(trans == rocblas_operation_none)
        {
            ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1),
                                    dim3(1, check_threads, 1), 0, stream, m, A, shiftA, lda,
                                    strideA, info);

            // copy B to X
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmin, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, m, nrhs, B, shiftB, ldb, strideB, X,
                                    shiftX, ldx, strideX);

            // solve LY = B (here Y = QX)
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, m,
                nrhs, A, shiftA, lda, strideA, X, shiftX, ldx, strideX, batch_count, optim_mem,
                work_x_temp, workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

            // zero row m to n-1 of X in cases where info is zero
            const rocblas_int zeroblocksx = (n - m - 1) / 32 + 1;
            ROCSOLVER_LAUNCH_KERNEL((gels_set_zero<T, U>),
                                    dim3(zeroblocksx, copyblocksy, batch_count), dim3(32, 32), 0,
                                    stream, m, n, nrhs, X, shiftX, ldx, strideX, info);

            rocsolver_ormlq_unmlq_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose, n, nrhs, m, A,
                shiftA, lda, strideA, ipiv, strideP, X, shiftX, ldx, strideX, batch_count, scalars,
                work_x_temp, workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);
        }
        else
        {
            // save B in savedB
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmax, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, copymat_to_buffer, n, nrhs, B, shiftB,
                                    ldb, strideB, savedB);

            rocsolver_ormlq_unmlq_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, rocblas_operation_none, n, nrhs, m, A, shiftA, lda,
                strideA, ipiv, strideP, B, shiftB, ldb, strideB, batch_count, scalars, work_x_temp,
                workArr_temp_arr, diag_trfac_invA, trfact_workTrmm_invA_arr);

            ROCSOLVER_LAUNCH_KERNEL(check_singularity<T>, dim3(batch_count, 1, 1),
                                    dim3(1, check_threads, 1), 0, stream, m, A, shiftA, lda,
                                    strideA, info);

            // solve L'X = QB
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                rocblas_diagonal_non_unit, m, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb,
                strideB, batch_count, optim_mem, work_x_temp, workArr_temp_arr, diag_trfac_invA,
                trfact_workTrmm_invA_arr);

            // copy result to X
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmin, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, m, nrhs, B, shiftB, ldb, strideB, X,
                                    shiftX, ldx, strideX);

            // restore B from savedB
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, U>), dim3(copyblocksmax, copyblocksy, batch_count),
                                    dim3(32, 32), 0, stream, copymat_from_buffer, n, nrhs, B,
                                    shiftB, ldb, strideB, savedB);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
