/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     April 2012
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

#include "auxiliary/rocauxiliary_orglq_unglq.hpp"
#include "auxiliary/rocauxiliary_orgqr_ungqr.hpp"
#include "rocblas.hpp"
#include "roclapack_gelqf.hpp"
#include "roclapack_geqrf.hpp"
#include "roclapack_syevd_heevd.hpp"
#include "roclapack_syevj_heevj.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S>
ROCSOLVER_KERNEL void gesdd_flip_signs(const rocblas_int n,
                                       S* SS,
                                       const rocblas_stride strideS,
                                       T* R,
                                       const rocblas_int ldr,
                                       const rocblas_stride strideR,
                                       T* Q,
                                       const rocblas_int ldq,
                                       const rocblas_stride strideQ,
                                       const rocblas_int batch_count)
{
    rocblas_int tid = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int bid = blockIdx.y;

    // local variables
    rocblas_int b, j, k;
    S sigma;

    for(b = bid; b < batch_count; b += gridDim.y)
    {
        // array pointers
        S* Sigma = SS + b * strideS;
        T* D = R + b * strideR;
        T* U = Q + b * strideQ;

        for(j = tid; j < n; j += gridDim.x * blockDim.x)
        {
            for(k = 0; k < n; k++)
            {
                sigma = std::real(D[k + k * ldr]);
                if(sigma < 0)
                {
                    U[j + k * ldq] = -U[j + k * ldq];
                }

                if(j == 0)
                {
                    Sigma[k] = std::abs(sigma);
                }
            }
        }
    }
}

/** Argument checking **/
template <typename T, typename SS, typename W>
rocblas_status rocsolver_gesdd_argCheck(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int lda,
                                        SS* S,
                                        T* U,
                                        const rocblas_int ldu,
                                        T* V,
                                        const rocblas_int ldv,
                                        rocblas_int* info,
                                        const rocblas_int batch_count = 1)
{
    // Order is important: must avoid pointer check if querying memory size;
    // also, unit tests may depend on order of checks

    auto is_svect_all = [](auto s) -> bool { return s == rocblas_svect_all; };

    auto is_svect_singular = [](auto s) -> bool { return s == rocblas_svect_singular; };

    auto is_svect_none = [](auto s) -> bool { return s == rocblas_svect_none; };

    auto is_svect = [&](auto s) -> bool { return is_svect_all(s) || is_svect_singular(s); };

    auto invalid_svect_option = [&](auto s) -> bool {
        return !(is_svect_all(s) || is_svect_singular(s) || is_svect_none(s));
    };

    //
    // 1. Invalid and non-supported values
    //
    if(invalid_svect_option(left_svect) || invalid_svect_option(right_svect))
    {
        return rocblas_status_invalid_value;
    }

    //
    // 2. Invalid sizes
    //
    if(n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || batch_count < 0)
    {
        return rocblas_status_invalid_size;
    }

    // Left singular vectors need ldu >= m
    if(is_svect(left_svect) && (ldu < m))
    {
        return rocblas_status_invalid_size;
    }

    // Right singular vectors need ldv >= n or ldv >= min(m, n),
    // depending on choice of svect_all or svect_singular
    if(is_svect_all(right_svect) && (ldv < n))
    {
        return rocblas_status_invalid_size;
    }

    if(is_svect_singular(right_svect) && (ldv < std::min(m, n)))
    {
        return rocblas_status_invalid_size;
    }

    // Skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_status_continue;
    }

    //
    // 3. Invalid pointers
    //
    auto invalid_pointer = [](auto ptr, auto size) -> bool { return (ptr == nullptr) && (size > 0); };

    if(batch_count && !info)
    {
        return rocblas_status_invalid_pointer;
    }

    if(invalid_pointer(A, m * n) || invalid_pointer(S, std::min(m, n)))
    {
        return rocblas_status_invalid_pointer;
    }

    if(is_svect_all(left_svect) && invalid_pointer(U, m))
    {
        return rocblas_status_invalid_pointer;
    }

    if(is_svect_singular(left_svect) && invalid_pointer(U, std::min(m, n)))
    {
        return rocblas_status_invalid_pointer;
    }

    if(is_svect_all(right_svect) && invalid_pointer(V, n))
    {
        return rocblas_status_invalid_pointer;
    }

    if(is_svect_singular(right_svect) && invalid_pointer(V, std::min(m, n)))
    {
        return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename SS>
void rocsolver_gesdd_getMemorySize(rocblas_handle handle,
                                   const rocblas_svect left_svect,
                                   const rocblas_svect right_svect,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int stride,
                                   const rocblas_int batch_count,
                                   size_t* size_VUtmp,
                                   size_t* size_UVtmpZ,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_work5_ipiv,
                                   size_t* size_splits,
                                   size_t* size_tmptau_W,
                                   size_t* size_tau,
                                   size_t* size_workArr,
                                   size_t* size_workArr2)
{
    // Make sure workspace is initialized
    *size_scalars = 0;
    *size_VUtmp = 0;
    *size_UVtmpZ = 0;
    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_work5_ipiv = 0;
    *size_splits = 0;
    *size_tmptau_W = 0;
    *size_tau = 0;
    *size_workArr = 0;
    *size_workArr2 = 0;

    // Quick return
    if(n == 0 || m == 0 || batch_count == 0)
    {
        return;
    }

    bool leftv = left_svect != rocblas_svect_none;
    bool rightv = right_svect != rocblas_svect_none;
    bool left_full = left_svect == rocblas_svect_all;
    bool right_full = right_svect == rocblas_svect_all;
    size_t size_UVtmp = 0;
    size_t a1 = 0, a2 = 0;
    size_t b1 = 0, b2 = 0, b3 = 0;
    size_t c1 = 0, c2 = 0, c3 = 0;
    size_t d1 = 0, d2 = 0, d3 = 0;
    size_t e1 = 0, e2 = 0;
    size_t f1 = 0, f2 = 0, f3 = 0, f4 = 0;
    size_t g1 = 0;
    size_t h1 = 0;
    size_t unused;

    *size_VUtmp = 0;

    if(m >= n)
    {
        // Requirements for Divide-and-Conquer eigensolver
        rocsolver_syevd_heevd_getMemorySize<BATCHED, T, SS>(
            handle, rocblas_evect_original, rocblas_fill_upper, n, batch_count, &a1, &b1, &c1, &d1,
            &e1, &f1, &g1, &h1, size_workArr2);

        // Requirements for QR factorization
        rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &a2, &b2, &c2, &d2, &f2);
        if(left_svect != rocblas_svect_none)
            rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, (left_full ? m : n), n, batch_count,
                                                            &unused, &b3, &c3, &d3, &f3);

        // Extra requirements for temporary V & U storage
        *size_VUtmp = sizeof(T) * n * n * batch_count;
        if(!leftv)
            size_UVtmp = sizeof(T) * m * n * batch_count;
    }
    else
    {
        // Requirements for Divide-and-Conquer eigensolver
        rocsolver_syevd_heevd_getMemorySize<BATCHED, T, SS>(
            handle, rocblas_evect_original, rocblas_fill_upper, m, batch_count, &a1, &b1, &c1, &d1,
            &e1, &f1, &g1, &h1, size_workArr2);

        // Requirements for LQ factorization
        rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, &a2, &b2, &c2, &d2, &f2);
        if(right_svect != rocblas_svect_none)
            rocsolver_orglq_unglq_getMemorySize<BATCHED, T>((right_full ? n : m), n, m, batch_count,
                                                            &unused, &b3, &c3, &d3, &f3);

        // Extra requirements for temporary U & V storage
        if(!leftv)
            *size_VUtmp = sizeof(T) * m * m * batch_count;
        if(!rightv)
            size_UVtmp = sizeof(T) * m * n * batch_count;
    }

    // Extra requirements for temporary Householder scalars
    e2 = sizeof(T) * min(m, n) * batch_count;

    // Size of array of pointers (batched cases)
    if(BATCHED)
        f4 = sizeof(T*) * 2 * batch_count;

    *size_UVtmpZ = std::max({e1, size_UVtmp});
    *size_scalars = std::max({a1, a2});
    *size_work1 = std::max({b1});
    *size_work2 = std::max({c1, b2, b3});
    *size_work3 = std::max({d1, c2, c3});
    *size_work4 = std::max({d2, d3});
    *size_work5_ipiv = std::max({e2});
    *size_splits = std::max({f1});
    *size_tmptau_W = std::max({g1});
    *size_tau = std::max({h1});
    *size_workArr = sizeof(T) * std::min({m, n}) * std::max({stride, 1}) * batch_count;
    *size_workArr = std::max({*size_workArr, f2, f3, f4});
}

template <bool BATCHED, bool STRIDED, typename T, typename SS, typename W>
rocblas_status rocsolver_gesdd_template(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
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
                                        T* VUtmp,
                                        void* UVtmpZ,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        void* work5_ipiv,
                                        void* splits,
                                        void* tmptau_W,
                                        void* tau,
                                        void* workArr,
                                        void* workArr2)
{
    ROCSOLVER_ENTER("gesdd", "leftsv:", left_svect, "rightsv:", right_svect, "m:", m, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "ldu:", ldu, "ldv:", ldv, "bc:", batch_count);

    // Quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Quick return
    if(m == 0 || n == 0)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        return rocblas_status_success;
    }

    // Everything is executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    bool leftv = left_svect != rocblas_svect_none;
    bool rightv = right_svect != rocblas_svect_none;
    bool left_full = left_svect == rocblas_svect_all;
    bool right_full = right_svect == rocblas_svect_all;
    T neg_one = T(-1);
    T one = T(1);
    T zero = T(0);

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    // The general idea is as follows: Given a m by n (m >= n) matrix A, we
    // compute the eigendecomposition of A^*A with Divide-and-Conquer to obtain
    //
    // A^*A |--> (V, D); such that V^*A^*AV = D.
    //
    // Discard the computed eigenvalues D, and use the eigenvectors V as the
    // right singular vectors of A, obtaining the left singular vectors (U) and
    // the singular values (S) from the QR decomposition of AV (see code for
    // necessary changes when m < n and AA^* is used instead of A^*A).

    if(m >= n)
    {
        // Compute -A'A; negative sign is necessary as `gesdd` outputs singular
        // values in non-ascending order, while `syevd` outputs eigenvalues in
        // non-decreasing order.
        T* V_gemm = VUtmp;
        rocblas_int ldv_gemm = n;
        rocblas_int strideV_gemm = n * n;

        rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, n, n,
                       m, &neg_one, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, V_gemm,
                       0, ldv_gemm, strideV_gemm, batch_count, (T**)workArr);

        rocsolver_syevd_heevd_template<false, STRIDED, T>(
            handle, rocblas_evect_original, rocblas_fill_upper, n, V_gemm, 0, ldv_gemm,
            strideV_gemm, S, strideS, (SS*)workArr, strideS, info, batch_count, scalars, work1,
            work2, work3, (SS*)UVtmpZ, (rocblas_int*)splits, (T*)tmptau_W, (T*)tau, (T**)workArr2);

        // Compute AV
        T* U_gemm = (leftv ? U : (T*)UVtmpZ);
        rocblas_int ldu_gemm = (leftv ? ldu : m);
        rocblas_int strideU_gemm = (leftv ? strideU : m * n);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, n, &one, A,
                       shiftA, lda, strideA, V_gemm, 0, ldv_gemm, strideV_gemm, &zero, U_gemm, 0,
                       ldu_gemm, strideU_gemm, batch_count, (T**)workArr);

        // Apply QR factorization to AV, obtaining U from Q and S from the
        // diagonal of R; notice that, since the QR decomposition is not
        // unique, we are required to make sure that all of the diagonal
        // elements of R are positive and flip the signs of the respective
        // columns of Q otherwise.
        rocsolver_geqrf_template<false, STRIDED, T>(handle, m, n, U_gemm, 0, ldu_gemm, strideU_gemm,
                                                    (T*)work5_ipiv, n, batch_count, scalars, work2,
                                                    (T*)work3, (T*)work4, (T**)workArr);

        rocblas_int blocks = (n - 1) / BS1 + 1;
        blocks = std::min(blocks, props->maxGridSize[0]);
        auto bc = std::min(batch_count, props->maxGridSize[1]);
        ROCSOLVER_LAUNCH_KERNEL(gesdd_flip_signs<T>, dim3(blocks, bc, 1), dim3(BS1, 1, 1), 0,
                                stream, n, S, strideS, U_gemm, ldu_gemm, strideU_gemm, V_gemm,
                                ldv_gemm, strideV_gemm, batch_count);

        if(leftv)
            rocsolver_orgqr_ungqr_template<false, STRIDED, T>(
                handle, m, (left_full ? m : n), n, U_gemm, 0, ldu_gemm, strideU_gemm, (T*)work5_ipiv,
                n, batch_count, scalars, (T*)work2, (T*)work3, (T*)work4, (T**)workArr);

        // Transpose V (for consistency with LAPACK's API)
        if(rightv)
        {
            rocblas_int blocks_n = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks_n, blocks_n, batch_count),
                                    dim3(BS2, BS2, 1), 0, stream,
                                    rocblas_operation_conjugate_transpose, n, n, V_gemm, 0,
                                    ldv_gemm, strideV_gemm, V, 0, ldv, strideV);
        }
    }
    else
    {
        // Compute -AA'
        T* U_gemm = (leftv ? U : VUtmp);
        rocblas_int ldu_gemm = (leftv ? ldu : m);
        rocblas_int strideU_gemm = (leftv ? strideU : m * m);

        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, m, m,
                       n, &neg_one, A, shiftA, lda, strideA, A, shiftA, lda, strideA, &zero, U_gemm,
                       0, ldu_gemm, strideU_gemm, batch_count, (T**)workArr);

        rocsolver_syevd_heevd_template<false, STRIDED, T>(
            handle, rocblas_evect_original, rocblas_fill_upper, m, U_gemm, 0, ldu_gemm,
            strideU_gemm, S, strideS, (SS*)workArr, strideS, info, batch_count, scalars, work1,
            work2, work3, (SS*)UVtmpZ, (rocblas_int*)splits, (T*)tmptau_W, (T*)tau, (T**)workArr2);

        // Compute U^*A
        T* V_gemm = (rightv ? V : (T*)UVtmpZ);
        rocblas_int ldv_gemm = (rightv ? ldv : m);
        rocblas_int strideV_gemm = (rightv ? strideV : m * n);

        rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, m, n,
                       m, &one, U_gemm, 0, ldu_gemm, strideU_gemm, A, shiftA, lda, strideA, &zero,
                       V_gemm, 0, ldv_gemm, strideV_gemm, batch_count, (T**)workArr);

        // Apply LQ factorization to U^*A, obtaining S from the diagonal of L and V^* from Q
        rocsolver_gelqf_template<false, STRIDED, T>(handle, m, n, V_gemm, 0, ldv_gemm, strideV_gemm,
                                                    (T*)work5_ipiv, m, batch_count, scalars, work2,
                                                    (T*)work3, (T*)work4, (T**)workArr);

        rocblas_int blocks = (m - 1) / BS1 + 1;
        blocks = std::min(blocks, props->maxGridSize[0]);
        auto bc = std::min(batch_count, props->maxGridSize[1]);
        ROCSOLVER_LAUNCH_KERNEL(gesdd_flip_signs<T>, dim3(blocks, bc, 1), dim3(BS1, 1, 1), 0,
                                stream, m, S, strideS, V_gemm, ldv_gemm, strideV_gemm, U_gemm,
                                ldu_gemm, strideU_gemm, batch_count);

        if(rightv)
            rocsolver_orglq_unglq_template<false, STRIDED, T>(
                handle, (right_full ? n : m), n, m, V_gemm, 0, ldv_gemm, strideV_gemm, (T*)work5_ipiv,
                m, batch_count, scalars, (T*)work2, (T*)work3, (T*)work4, (T**)workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
