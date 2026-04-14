/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
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

#include "auxiliary/rocauxiliary_orglq_unglq.hpp"
#include "auxiliary/rocauxiliary_orgqr_ungqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "roclapack_gesvdj.hpp"
#include "roclapack_syevj_heevj.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Argument checking **/
template <typename T, typename SS, typename W>
rocblas_status rocsolver_gesvdj_notransv_argCheck(rocblas_handle handle,
                                                  const rocblas_svect left_svect,
                                                  const rocblas_svect right_svect,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  W A,
                                                  const rocblas_int lda,
                                                  SS* residual,
                                                  const rocblas_int max_sweeps,
                                                  rocblas_int* n_sweeps,
                                                  SS* S,
                                                  T* U,
                                                  const rocblas_int ldu,
                                                  T* V,
                                                  const rocblas_int ldv,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(left_svect != rocblas_svect_all && left_svect != rocblas_svect_singular
       && left_svect != rocblas_svect_none)
        return rocblas_status_invalid_value;
    if(right_svect != rocblas_svect_all && right_svect != rocblas_svect_singular
       && right_svect != rocblas_svect_none)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || max_sweeps <= 0 || ldu < 1 || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;
    if((left_svect == rocblas_svect_all || left_svect == rocblas_svect_singular) && ldu < m)
        return rocblas_status_invalid_size;
    if((right_svect == rocblas_svect_all || right_svect == rocblas_svect_singular) && ldv < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n * m && !A) || (batch_count && !residual) || (batch_count && !n_sweeps)
       || (min(m, n) && !S) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if((left_svect == rocblas_svect_all && m && !U)
       || (left_svect == rocblas_svect_singular && min(m, n) && !U))
        return rocblas_status_invalid_pointer;
    if((right_svect == rocblas_svect_all && n && !V)
       || (right_svect == rocblas_svect_singular && min(m, n) && !V))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename SS>
void rocsolver_gesvdj_notransv_getMemorySize(const rocblas_svect left_svect,
                                             const rocblas_svect right_svect,
                                             const rocblas_int m,
                                             const rocblas_int n,
                                             const rocblas_int batch_count,
                                             size_t* size_scalars,
                                             size_t* size_VUtmp,
                                             size_t* size_work1_UVtmp,
                                             size_t* size_work2,
                                             size_t* size_work3,
                                             size_t* size_work4,
                                             size_t* size_work5_ipiv,
                                             size_t* size_work6_workArr)
{
    // if quick return, set workspace to zero
    if(n == 0 || m == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_VUtmp = 0;
        *size_work1_UVtmp = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_work5_ipiv = 0;
        *size_work6_workArr = 0;
        return;
    }

    bool leftv = left_svect != rocblas_svect_none;
    bool rightv = right_svect != rocblas_svect_none;
    bool left_full = left_svect == rocblas_svect_all;
    bool right_full = right_svect == rocblas_svect_all;
    size_t a1 = 0, a2 = 0;
    size_t b1 = 0, b2 = 0, b3 = 0;
    size_t c1 = 0, c2 = 0, c3 = 0;
    size_t d1 = 0, d2 = 0, d3 = 0;
    size_t e1 = 0, e2 = 0;
    size_t f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    size_t unused;

    *size_VUtmp = 0;

    if(m >= n)
    {
        // requirements for Jacobi eigensolver
        rocsolver_syevj_heevj_getMemorySize<BATCHED, T, SS>(rocblas_evect_original,
                                                            rocblas_fill_upper, n, batch_count, &a1,
                                                            &b1, &c1, &d1, &e1, &f1);

        // requirements for QR factorization
        rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, size_scalars, &b2, &c2, &d2,
                                                  &f2);
        if(left_svect != rocblas_svect_none)
            rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, (left_full ? m : n), n, batch_count,
                                                            &unused, &b3, &c3, &d3, &f3);

        // extra requirements for temporary V & U storage
        if(!rightv)
            *size_VUtmp = sizeof(T) * n * n * batch_count;
        if(!leftv)
            a2 = sizeof(T) * m * n * batch_count;
    }
    else
    {
        // requirements for Jacobi eigensolver
        rocsolver_syevj_heevj_getMemorySize<BATCHED, T, SS>(rocblas_evect_original,
                                                            rocblas_fill_upper, m, batch_count, &a1,
                                                            &b1, &c1, &d1, &e1, &f1);

        // requirements for QR factorization
        rocsolver_geqrf_getMemorySize<BATCHED, T>(n, m, batch_count, size_scalars, &b2, &c2, &d2,
                                                  &f2);
        if(right_svect != rocblas_svect_none)
            rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(n, (right_full ? n : m), m, batch_count,
                                                            &unused, &b3, &c3, &d3, &f3);

        // extra requirements for temporary U & V storage
        if(!leftv)
            *size_VUtmp = sizeof(T) * m * m * batch_count;
        if(!rightv)
            a2 = sizeof(T) * m * n * batch_count;
    }

    // extra requirements for temporary Householder scalars
    e2 = sizeof(T) * min(m, n) * batch_count;

    // size of array of pointers (batched cases)
    if(BATCHED)
        f4 = sizeof(T*) * 2 * batch_count;

    *size_work1_UVtmp = std::max({a1, a2});
    *size_work2 = std::max({b1, b2, b3});
    *size_work3 = std::max({c1, c2, c3});
    *size_work4 = std::max({d1, d2, d3});
    *size_work5_ipiv = std::max({e1, e2});
    *size_work6_workArr = std::max({f1, f2, f3, f4});
}

template <bool BATCHED, bool STRIDED, typename T, typename SS, typename W>
rocblas_status rocsolver_gesvdj_notransv_template(rocblas_handle handle,
                                                  const rocblas_svect left_svect,
                                                  const rocblas_svect right_svect,
                                                  const rocblas_int m,
                                                  const rocblas_int n,
                                                  W A,
                                                  const rocblas_int shiftA,
                                                  const rocblas_int lda,
                                                  const rocblas_stride strideA,
                                                  const SS abstol,
                                                  SS* residual,
                                                  const rocblas_int max_sweeps,
                                                  rocblas_int* n_sweeps,
                                                  SS* S,
                                                  const rocblas_stride strideS,
                                                  T* U,
                                                  const rocblas_int ldu,
                                                  const rocblas_stride strideU,
                                                  T* V,
                                                  const rocblas_int ldv,
                                                  const rocblas_stride strideV,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count,
                                                  T* scalars,
                                                  T* VUtmp,
                                                  void* work1_UVtmp,
                                                  void* work2,
                                                  void* work3,
                                                  void* work4,
                                                  void* work5_ipiv,
                                                  void* work6_workArr)
{
    ROCSOLVER_ENTER("gesvdj", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "abstol:", abstol, "max_sweeps:", max_sweeps,
                    "ldu:", ldu, "ldv:", ldv, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(m == 0 || n == 0)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, residual,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, n_sweeps,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        return rocblas_status_success;
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    bool leftv = left_svect != rocblas_svect_none;
    bool rightv = right_svect != rocblas_svect_none;
    bool left_full = left_svect == rocblas_svect_all;
    bool right_full = right_svect == rocblas_svect_all;
    T minone = T(-1);
    T one = T(1);
    T zero = T(0);

    if(m >= n)
    {
        // compute -A'A
        T* V_gemm = (rightv ? V : VUtmp);
        rocblas_int ldv_gemm = (rightv ? ldv : n);
        rocblas_int strideV_gemm = (rightv ? strideV : n * n);

        rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n,
                       m, &minone, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, V_gemm,
                       0, ldv_gemm, strideV_gemm, batch_count, (T**)work6_workArr);

        // apply eigenvalue decomposition to -A'A, obtaining V as eigenvectors
        rocsolver_syevj_heevj_template<false, STRIDED, T>(
            handle, rocblas_esort_ascending, rocblas_evect_original, rocblas_fill_upper, n, V_gemm,
            0, ldv_gemm, strideV_gemm, abstol, residual, max_sweeps, n_sweeps, S, strideS, info,
            batch_count, (T*)work1_UVtmp, (T*)work2, (SS*)work3, (rocblas_int*)work4,
            (rocblas_int*)work5_ipiv, (rocblas_int*)work6_workArr);

        // compute AV
        T* U_gemm = (leftv ? U : (T*)work1_UVtmp);
        rocblas_int ldu_gemm = (leftv ? ldu : m);
        rocblas_int strideU_gemm = (leftv ? strideU : m * n);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, n, &one, A,
                       shiftA, lda, strideA, V_gemm, 0, ldv_gemm, strideV_gemm, &zero, U_gemm, 0,
                       ldu_gemm, strideU_gemm, batch_count, (T**)work6_workArr);

        // apply QR factorization to AV, obtaining U = Q and S = R
        rocsolver_geqrf_template<false, STRIDED, T>(handle, m, n, U_gemm, 0, ldu_gemm, strideU_gemm,
                                                    (T*)work5_ipiv, n, batch_count, scalars, work2,
                                                    (T*)work3, (T*)work4, (T**)work6_workArr);

        rocblas_int blocks = (n - 1) / BS1 + 1;
        ROCSOLVER_LAUNCH_KERNEL(gesvdj_finalize<T>, dim3(blocks, batch_count, 1), dim3(BS1, 1, 1),
                                0, stream, n, S, strideS, U_gemm, ldu_gemm, strideU_gemm, V_gemm,
                                ldv_gemm, strideV_gemm);

        if(leftv)
            rocsolver_orgqr_ungqr_template<false, STRIDED, T>(
                handle, m, (left_full ? m : n), n, U_gemm, 0, ldu_gemm, strideU_gemm, (T*)work5_ipiv,
                n, batch_count, scalars, (T*)work2, (T*)work3, (T*)work4, (T**)work6_workArr);
    }
    else
    {
        // compute -AA'
        T* U_gemm = (leftv ? U : VUtmp);
        rocblas_int ldu_gemm = (leftv ? ldu : m);
        rocblas_int strideU_gemm = (leftv ? strideU : m * m);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, m, m,
                       n, &minone, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, U_gemm,
                       0, ldu_gemm, strideU_gemm, batch_count, (T**)work6_workArr);

        // apply eigenvalue decomposition to -AA', obtaining U as eigenvectors
        rocsolver_syevj_heevj_template<false, STRIDED, T>(
            handle, rocblas_esort_ascending, rocblas_evect_original, rocblas_fill_upper, m, U_gemm,
            0, ldu_gemm, strideU_gemm, abstol, residual, max_sweeps, n_sweeps, S, strideS, info,
            batch_count, (T*)work1_UVtmp, (T*)work2, (SS*)work3, (rocblas_int*)work4,
            (rocblas_int*)work5_ipiv, (rocblas_int*)work6_workArr);

        // compute A'U
        T* V_gemm = (rightv ? V : (T*)work1_UVtmp);
        rocblas_int ldv_gemm = (rightv ? ldv : n);
        rocblas_int strideV_gemm = (rightv ? strideV : n * m);

        rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, n, m,
                       m, &one, A, shiftA, lda, strideA, U_gemm, 0, ldu_gemm, strideU_gemm, &zero,
                       V_gemm, 0, ldv_gemm, strideV_gemm, batch_count, (T**)work6_workArr);

        // apply QR factorization to A'U, obtaining V = Q and S = R
        rocsolver_geqrf_template<false, STRIDED, T>(handle, n, m, V_gemm, 0, ldv_gemm, strideV_gemm,
                                                    (T*)work5_ipiv, m, batch_count, scalars, work2,
                                                    (T*)work3, (T*)work4, (T**)work6_workArr);

        rocblas_int blocks = (m - 1) / BS1 + 1;
        ROCSOLVER_LAUNCH_KERNEL(gesvdj_finalize<T>, dim3(blocks, batch_count, 1), dim3(BS1, 1, 1),
                                0, stream, m, S, strideS, V_gemm, ldv_gemm, strideV_gemm, U_gemm,
                                ldu_gemm, strideU_gemm);

        if(rightv)
            rocsolver_orgqr_ungqr_template<false, STRIDED, T>(
                handle, n, (right_full ? n : m), m, V_gemm, 0, ldv_gemm, strideV_gemm, (T*)work5_ipiv,
                m, batch_count, scalars, (T*)work2, (T*)work3, (T*)work4, (T**)work6_workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
