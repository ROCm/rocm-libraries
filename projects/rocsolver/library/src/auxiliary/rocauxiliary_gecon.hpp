/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ideal_sizes.hpp"
#include "lapack_device_functions.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"
#include "rocsolver_hybrid_storage.hpp"
#include "rocsolver_run_specialized_kernels.hpp"
#include <algorithm>
#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

#ifndef GECON_BLOCKSIZE
#define GECON_BLOCKSIZE 1024
#endif

/**
 * gecon_compute_rcond: Compute rcond from estimate on device
 * rcond = 1 / (||A|| * ||A^-1||) = 1 / (anorm * est)
 */
template <typename S>
ROCSOLVER_KERNEL void gecon_compute_rcond(S* rcond, const S* est, const S* anorm)
{
    if(hipThreadIdx_x == 0)
    {
        S est_val = *est;
        S anorm_val = *anorm;
        if(est_val != S(0) && anorm_val != S(0))
            *rcond = S(1) / (est_val * anorm_val);
        else
            *rcond = S(0);
    }
}

template <typename T, typename I, typename S>
void rocsolver_gecon_getMemorySize(const I n,
                                   const I lda,
                                   const I batch_count,
                                   size_t* size_work_v,
                                   size_t* size_work_x,
                                   size_t* size_work_isgn,
                                   size_t* size_scalar_est,
                                   size_t* size_scalar_max_idx,
                                   size_t* size_scalar_kase,
                                   size_t* size_scalar_jump,
                                   size_t* size_work_trsm_1,
                                   size_t* size_work_trsm_2,
                                   size_t* size_work_trsm_3,
                                   size_t* size_work_trsm_4,
                                   bool* optim_mem)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_work_v = 0;
        *size_work_x = 0;
        *size_work_isgn = 0;
        *size_scalar_est = 0;
        *size_scalar_max_idx = 0;
        *size_scalar_kase = 0;
        *size_scalar_jump = 0;
        *size_work_trsm_1 = 0;
        *size_work_trsm_2 = 0;
        *size_work_trsm_3 = 0;
        *size_work_trsm_4 = 0;
        *optim_mem = true;
        return;
    }

    // need n elements of type T per batch for v vector
    *size_work_v = sizeof(T) * n * batch_count;

    // need n elements of type T per batch for x vector
    *size_work_x = sizeof(T) * n * batch_count;

    // need n elements of type I per batch for isgn vector (only used by real types, not complex)
    *size_work_isgn = rocblas_is_complex<T> ? 0 : sizeof(I) * n * batch_count;

    // Scalars are reused across batches (not allocated per batch)
    // need one S (real) scalar for estimate
    *size_scalar_est = sizeof(S);

    // need one I scalar for max index
    *size_scalar_max_idx = sizeof(I);

    // need one rocblas_int scalar for kase state
    *size_scalar_kase = sizeof(rocblas_int);

    // need one rocblas_int scalar for jump state
    *size_scalar_jump = sizeof(rocblas_int);

    // workspace for trsm (solving triangular systems directly)
    rocsolver_trsm_mem<false, false, T, I>(rocblas_side_left, rocblas_operation_none, n, (I)1,
                                           batch_count, size_work_trsm_1, size_work_trsm_2,
                                           size_work_trsm_3, size_work_trsm_4, optim_mem);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_gecon_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I n,
                                        const I lda,
                                        T A,
                                        const S* anorm,
                                        S* rcond)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_infinity)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !anorm) || (n && !rcond))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// main gecon function
template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gecon_template(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I inca,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const S* anorm, // S = real_t<T>
                                        S* rcond,
                                        const I batch_count,
                                        T* work_v,
                                        T* work_x,
                                        I* work_isgn,
                                        S* scalar_est,
                                        I* scalar_max_idx,
                                        rocblas_int* scalar_kase,
                                        rocblas_int* scalar_jump,
                                        const bool optim_mem,
                                        void* work_trsm_1,
                                        void* work_trsm_2,
                                        void* work_trsm_3,
                                        void* work_trsm_4,
                                        const I max_iter)
{
    ROCSOLVER_ENTER("gecon", "norm_type:", norm_type, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    if(!batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return if no dimensions
    if(n == 0)
    {
        if(rcond)
        {
            rocblas_int blocks = (batch_count - 1) / BS1 + 1;
            dim3 grid(blocks, 1, 1);
            dim3 threads(BS1, 1, 1);
            ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, rcond, batch_count, S(1));
        }
        return rocblas_status_success;
    }

    // Use hybrid storage to get pointers for batched arrays
    rocsolver_hybrid_storage<T, I, U> hA;
    ROCBLAS_CHECK(hA.init_pointers_only(A, shiftA, strideA, batch_count, stream));

    // iterate over each batch
    // TODO: make a version that operates on batches efficiently
    // preferably, one that remains entirely on the GPU, and potentially simplifies into a single kernel
    std::vector<S> h_anorm(batch_count);
    HIP_CHECK(hipMemcpyAsync(h_anorm.data(), anorm, sizeof(S) * batch_count, hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for(I batch = 0; batch < batch_count; batch++)
    {
        // if anorm is zero for this batch, rcond is zero

        if(h_anorm[batch] == S(0))
        {
            S zero = S(0);
            HIP_CHECK(hipMemcpyAsync(rcond + batch, &zero, sizeof(S), hipMemcpyHostToDevice, stream));
            continue;
        }

        // get workspace pointers for this batch
        T* v = work_v + batch * n;
        T* x = work_x + batch * n;
        I* isgn = work_isgn + batch * n;
        S* d_est = scalar_est; // reuse single scalar for all batches
        const S* d_anorm = anorm + batch;
        I* d_max_idx = scalar_max_idx; // reuse single scalar
        rocblas_int* d_kase = scalar_kase; // reuse single scalar
        rocblas_int* d_jump = scalar_jump; // reuse single scalar

        // initialize lacn2 state
        rocblas_int h_kase = 0;
        rocblas_int h_jump = 0;
        I h_iters = 1;

        // main LACN2 and TRSM iteration loop
        do
        {
            // Call LACN2 to get next operation (simplified parameter list)
            rocsolver_lacn2_template<T, I, S>(handle, n, &x, &v, isgn, d_est, d_max_idx, d_kase,
                                              d_jump, &h_kase, &h_jump, &h_iters, max_iter, stream);

            if(h_kase == 0)
                break; // converged, d_est contains ||inv(A)||_1

            // determine operation based on norm_type and kase
            rocblas_operation opr;
            if(norm_type == rocsolver_norm_type_one)
            {
                opr = (h_kase == 1) ? rocblas_operation_none : rocblas_operation_conjugate_transpose;
            }
            else
            { // infinity norm
                opr = (h_kase == 1) ? rocblas_operation_conjugate_transpose : rocblas_operation_none;
            }

            // Solve triangular systems directly using TRSM (bypassing LASWP from GETRS)
            // For LU factorization PA = LU, we need to solve (LU)^{-1} * x or (LU)^{-H} * x
            // Row permutations P don't affect the norm, so we skip LASWP entirely.
            if(opr == rocblas_operation_none)
            {
                // Solve L*y = x (unit lower triangular), then U*x = y (non-unit upper triangular)
                rocsolver_trsm_lower<false, false, T, I>(
                    handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, n,
                    (I)1, (T*)hA[batch], 0, inca, lda, strideA, (T*)x, 0, (I)1, n, strideA, (I)1,
                    optim_mem, work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4);

                rocsolver_trsm_upper<false, false, T, I>(
                    handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
                    (I)1, (T*)hA[batch], 0, inca, lda, strideA, (T*)x, 0, (I)1, n, strideA, (I)1,
                    optim_mem, work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4);
            }
            else
            {
                // Solve U^H*y = x, then L^H*x = y (conjugate transpose for complex, transpose for real)
                rocsolver_trsm_upper<false, false, T, I>(
                    handle, rocblas_side_left, opr, rocblas_diagonal_non_unit, n, (I)1,
                    (T*)hA[batch], 0, inca, lda, strideA, (T*)x, 0, (I)1, n, strideA, (I)1,
                    optim_mem, work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4);

                rocsolver_trsm_lower<false, false, T, I>(
                    handle, rocblas_side_left, opr, rocblas_diagonal_unit, n, (I)1, (T*)hA[batch],
                    0, inca, lda, strideA, (T*)x, 0, (I)1, n, strideA, (I)1, optim_mem, work_trsm_1,
                    work_trsm_2, work_trsm_3, work_trsm_4);
            }
        } while(h_kase != 0);

        // Compute rcond from estimate on device
        // rcond = 1 / (||A|| * ||inv(A)||) = 1 / (anorm * est)
        ROCSOLVER_LAUNCH_KERNEL((gecon_compute_rcond<S>), dim3(1), dim3(1), 0, stream,
                                rcond + batch, d_est, d_anorm);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
