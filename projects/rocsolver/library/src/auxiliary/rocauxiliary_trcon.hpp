/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routines (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_device_functions.hpp"
#include "rocauxiliary_lantr.hpp"
#include "rocsolver_hybrid_storage.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Sets *flag to 1 if x has an element that is not finite. Launch with a single block. **/
template <typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS1)
    con_check_finite(const I n, const T* x, rocblas_int* flag)
{
    for(I i = threadIdx.x; i < n; i += BS1)
    {
        const T xi = x[i];
        bool finite;
        if constexpr(rocblas_is_complex<T>)
            finite = std::isfinite(xi.real()) && std::isfinite(xi.imag());
        else
            finite = std::isfinite(xi);
        if(!finite)
            *flag = 1;
    }
}

/** rcond = (1 / anorm) / ainvnm (TRCON) or (1 / ainvnm) / anorm (POCON), or zero if ainvnm is
    zero or a solve overflowed (flag set). **/
template <bool INVFIRST, typename S>
ROCSOLVER_KERNEL void
    con_compute_rcond(S* rcond, const S* ainvnm, const S* anorm, const rocblas_int* flag)
{
    if(threadIdx.x == 0)
    {
        const S est = *ainvnm;
        if(*flag || est == 0 || !std::isfinite(est))
            *rcond = 0;
        else
            *rcond = INVFIRST ? (S(1) / est) / *anorm : (S(1) / *anorm) / est;
    }
}

template <typename T, typename I, typename S>
void rocsolver_trcon_getMemorySize(const rocsolver_norm_type norm_type,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_work_norm,
                                   size_t* size_anorm,
                                   size_t* size_work_v,
                                   size_t* size_work_x,
                                   size_t* size_work_isgn,
                                   size_t* size_scalars,
                                   size_t* size_work_trsm_1,
                                   size_t* size_work_trsm_2,
                                   size_t* size_work_trsm_3,
                                   size_t* size_work_trsm_4,
                                   bool* optim_mem)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_work_norm = 0;
        *size_anorm = 0;
        *size_work_v = 0;
        *size_work_x = 0;
        *size_work_isgn = 0;
        *size_scalars = 0;
        *size_work_trsm_1 = 0;
        *size_work_trsm_2 = 0;
        *size_work_trsm_3 = 0;
        *size_work_trsm_4 = 0;
        *optim_mem = true;
        return;
    }

    // workspace for the norm of A, and the norms
    rocsolver_lantr_getMemorySize<T, I, S>(norm_type, n, n, batch_count, size_work_norm);
    *size_anorm = sizeof(S) * batch_count;

    // vectors v, x and isgn of LACN2 (only one matrix is processed at a time)
    *size_work_v = sizeof(T) * n;
    *size_work_x = sizeof(T) * n;
    *size_work_isgn = rocblas_is_complex<T> ? 0 : sizeof(I) * n;

    // scalars of LACN2 (estimate, index, kase, jump) and the overflow flag, in 8-byte slots
    *size_scalars = 5 * sizeof(int64_t);

    // workspace for the solves with A or A^H
    size_t w1, w2, w3, w4;
    rocsolver_trsm_mem<false, false, T, I>(rocblas_side_left, rocblas_operation_none, n, (I)1, (I)1,
                                           size_work_trsm_1, size_work_trsm_2, size_work_trsm_3,
                                           size_work_trsm_4, optim_mem);
    rocsolver_trsm_mem<false, false, T, I>(rocblas_side_left, rocblas_operation_conjugate_transpose,
                                           n, (I)1, (I)1, &w1, &w2, &w3, &w4, optim_mem);
    *size_work_trsm_1 = std::max(*size_work_trsm_1, w1);
    *size_work_trsm_2 = std::max(*size_work_trsm_2, w2);
    *size_work_trsm_3 = std::max(*size_work_trsm_3, w3);
    *size_work_trsm_4 = std::max(*size_work_trsm_4, w4);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_trcon_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const I n,
                                        const I lda,
                                        T A,
                                        S* rcond)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_infinity)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !rcond))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Estimates ||inv(A)|| with LACN2, where A is a triangular matrix (TRIANGULAR) or the product
    of a triangular matrix and its conjugate transpose given by the Cholesky factor (!TRIANGULAR),
    and computes rcond. Processes one matrix of the batch: Ab points to it. **/
template <bool TRIANGULAR, typename T, typename I, typename S>
rocblas_status rocsolver_con_estimate(rocblas_handle handle,
                                      const rocsolver_norm_type norm_type,
                                      const rocblas_fill uplo,
                                      const rocblas_diagonal diag,
                                      const I n,
                                      T* Ab,
                                      const I lda,
                                      const S* anorm,
                                      S* rcond,
                                      T* work_v,
                                      T* work_x,
                                      I* work_isgn,
                                      void* scalars,
                                      const bool optim_mem,
                                      void* work_trsm_1,
                                      void* work_trsm_2,
                                      void* work_trsm_3,
                                      void* work_trsm_4,
                                      const I max_iter,
                                      hipStream_t stream)
{
    // scalars in 8-byte slots (see getMemorySize)
    int64_t* slots = (int64_t*)scalars;
    S* d_est = (S*)slots;
    I* d_max_idx = (I*)(slots + 1);
    rocblas_int* d_kase = (rocblas_int*)(slots + 2);
    rocblas_int* d_jump = (rocblas_int*)(slots + 3);
    rocblas_int* d_flag = (rocblas_int*)(slots + 4);

    HIP_CHECK(hipMemsetAsync(d_flag, 0, sizeof(rocblas_int), stream));

    // the triangular solves pass their scalars on the host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    T* x = work_x;
    T* v = work_v;
    const bool upper = (uplo == rocblas_fill_upper);

    // initialize lacn2 state
    rocblas_int h_kase = 0;
    rocblas_int h_jump = 0;
    I h_iters = 1;

    // with kase1, solve A x = b; otherwise A^H x = b
    const rocblas_int kase1 = (norm_type == rocsolver_norm_type_infinity) ? 2 : 1;

    // A x = b, or the first solve of the Cholesky factorization
    auto solve = [&](const rocblas_operation trans, const rocblas_diagonal d) {
        if(upper)
            return rocsolver_trsm_upper<false, false, T, I>(
                handle, rocblas_side_left, trans, d, n, (I)1, Ab, 0, lda, rocblas_stride(0), x, 0,
                n, rocblas_stride(n), (I)1, optim_mem, work_trsm_1, work_trsm_2, work_trsm_3,
                work_trsm_4);
        else
            return rocsolver_trsm_lower<false, false, T, I>(
                handle, rocblas_side_left, trans, d, n, (I)1, Ab, 0, lda, rocblas_stride(0), x, 0,
                n, rocblas_stride(n), (I)1, optim_mem, work_trsm_1, work_trsm_2, work_trsm_3,
                work_trsm_4);
    };

    do
    {
        ROCBLAS_CHECK(rocsolver_lacn2_template<T, I, S>(handle, n, &x, &v, work_isgn, d_est,
                                                        d_max_idx, d_kase, d_jump, &h_kase, &h_jump,
                                                        &h_iters, max_iter, stream));
        if(h_kase == 0)
            break;

        if(TRIANGULAR)
        {
            if(h_kase == kase1)
            {
                ROCBLAS_CHECK(solve(rocblas_operation_none, diag));
            }
            else
            {
                ROCBLAS_CHECK(solve(rocblas_operation_conjugate_transpose, diag));
            }
        }
        else
        {
            // inv(A) is Hermitian, so both kase values solve A x = b:
            // U^H U x = b (upper) or L L^H x = b (lower)
            if(upper)
            {
                ROCBLAS_CHECK(solve(rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit));
                ROCBLAS_CHECK(solve(rocblas_operation_none, rocblas_diagonal_non_unit));
            }
            else
            {
                ROCBLAS_CHECK(solve(rocblas_operation_none, rocblas_diagonal_non_unit));
                ROCBLAS_CHECK(solve(rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit));
            }
        }

        // the solves are not scaled (unlike LAPACK's xLATRS): an overflow means that A is
        // numerically singular, and rcond is set to zero
        ROCSOLVER_LAUNCH_KERNEL((con_check_finite<T, I>), dim3(1), dim3(BS1), 0, stream, n, x,
                                d_flag);
    } while(h_kase != 0);

    ROCSOLVER_LAUNCH_KERNEL((con_compute_rcond<!TRIANGULAR, S>), dim3(1), dim3(1), 0, stream, rcond,
                            d_est, anorm, d_flag);

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_trcon_template(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        S* rcond,
                                        const I batch_count,
                                        S* work_norm,
                                        S* anorm,
                                        T* work_v,
                                        T* work_x,
                                        I* work_isgn,
                                        void* scalars,
                                        const bool optim_mem,
                                        void* work_trsm_1,
                                        void* work_trsm_2,
                                        void* work_trsm_3,
                                        void* work_trsm_4,
                                        const I max_iter)
{
    ROCSOLVER_ENTER("trcon", "norm_type:", norm_type, "uplo:", uplo, "diag:", diag, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return: rcond = 1 if n = 0
    if(n == 0)
    {
        if(rcond)
        {
            const I blocks = (batch_count - 1) / BS1 + 1;
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocks), dim3(BS1), 0, stream,
                                    rcond, batch_count, S(1));
        }
        return rocblas_status_success;
    }

    // norms of the matrices
    ROCBLAS_CHECK(rocsolver_lantr_template<T, I, S>(handle, norm_type, uplo, diag, n, n, A, shiftA,
                                                    lda, strideA, batch_count, anorm, work_norm));

    std::vector<S> h_anorm(batch_count);
    HIP_CHECK(hipMemcpyAsync(h_anorm.data(), anorm, sizeof(S) * batch_count, hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    rocsolver_hybrid_storage<T, I, U> hA;
    ROCBLAS_CHECK(hA.init_pointers_only(A, shiftA, strideA, batch_count, stream));

    // the estimation is driven from the host, one matrix at a time
    for(I b = 0; b < batch_count; b++)
    {
        // as in LAPACK, rcond = 0 unless the norm is positive
        if(!(h_anorm[b] > 0))
        {
            HIP_CHECK(hipMemsetAsync(rcond + b, 0, sizeof(S), stream));
            continue;
        }

        ROCBLAS_CHECK(rocsolver_con_estimate<true, T, I, S>(
            handle, norm_type, uplo, diag, n, (T*)hA[b], lda, anorm + b, rcond + b, work_v, work_x,
            work_isgn, scalars, optim_mem, work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4,
            max_iter, stream));
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
