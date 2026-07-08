/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_ormqr_unmqr.hpp"
#include "auxiliary/rocauxiliary_ormtr_unmtr.hpp"
#include "auxiliary/rocauxiliary_ormtr_unmtr_hb2st.hpp"
#include "auxiliary/rocauxiliary_sb2st_hb2st.hpp"
#include "auxiliary/rocauxiliary_stedc.hpp"
#include "auxiliary/rocauxiliary_sterf.hpp"
#include "auxiliary/rocauxiliary_sy2sb_he2hb.hpp"
#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "roclapack_sytrd_hetrd.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename S>
void rocsolver_syevd_heevd_getMemorySize(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work1,
                                         size_t* size_work2,
                                         size_t* size_work3,
                                         size_t* size_tmpz,
                                         size_t* size_splits,
                                         size_t* size_tmptau_W,
                                         size_t* size_tau,
                                         size_t* size_workArr,
                                         size_t* size_Aband,
                                         size_t* size_he2hb_work,
                                         size_t* size_V_hb2st,
                                         size_t* size_tau_hb2st)
{
    *size_Aband = 0;
    *size_he2hb_work = 0;
    *size_V_hb2st = 0;
    *size_tau_hb2st = 0;

    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_tmptau_W = 0;
        *size_tau = 0;
        *size_workArr = 0;
        *size_splits = 0;
        *size_tmpz = 0;
        return;
    }

    rocsolver_alg_mode alg_mode;
    rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &alg_mode);

    size_t unused;
    size_t w11 = 0, w12 = 0, w13 = 0;
    size_t w21 = 0, w22 = 0, w23 = 0;
    size_t w31 = 0, w32 = 0;
    size_t t1 = 0, t2 = 0;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w11, &w21, &t1,
                                                    &unused, false);

    if(alg_mode != rocsolver_alg_mode_hybrid || evect == rocblas_evect_original)
    {
        // extra requirements for computing eigenvalues and vectors (stedc)
        rocsolver_stedc_getMemorySize<BATCHED, T, S>(rocblas_evect_tridiagonal, n, batch_count,
                                                     &w31, &w22, &w12, size_tmpz, size_splits,
                                                     size_workArr);
    }
    else
    {
        *size_splits = 0;
        *size_tmpz = 0;
    }

    if(evect == rocblas_evect_original)
    {
        // extra requirements for ormtr/unmtr
        rocsolver_ormtr_unmtr_getMemorySize<BATCHED, T>(rocblas_side_left, uplo, n, n, batch_count,
                                                        &unused, &w23, &w13, &w32, &unused);
    }

    // size of array for temporary matrix products
    t2 = sizeof(T) * n * n * batch_count;

    // get max values
    *size_work1 = std::max({w11, w12, w13});
    *size_work2 = std::max({w21, w22, w23});
    *size_work3 = std::max(w31, w32);
    *size_tmptau_W = std::max(t1, t2);

    // size of array for temporary householder scalars
    *size_tau = sizeof(T) * n * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = std::max(*size_workArr, 2 * sizeof(T*) * batch_count);

    // requirements for 2-stage path (used when n > SYEVD_2STAGE_SWITCHSIZE)
    if(n > SYEVD_2STAGE_SWITCHSIZE)
    {
        const rocblas_int kd = SYEVD_2STAGE_KD;
        const rocblas_int nb = SYEVD_2STAGE_NB;

        // band matrix: (3*kd - 1) rows by n cols
        const rocblas_int ldab = 3 * kd - 1;
        *size_Aband = sizeof(T) * ldab * n * batch_count;

        // he2hb internal workspace
        size_t s_he2hb, s_D, s_V, s_W, s_X, s_Z, s_work, s_workArr;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, rocblas_int>(
            n, kd, nb, batch_count, &s_he2hb, &s_D, &s_V, &s_W, &s_X, &s_Z, &s_work, &s_workArr);
        *size_he2hb_work
            = s_he2hb + s_D + s_V + s_W + s_X + s_Z + s_work + s_workArr;

        // V and tau for hb2st and unmtr_hb2st:
        // nt = ceildiv(n - 1, kd); nv = kd * nt * (nt + 1) / 2
        const rocblas_int nt = (n - 2) / kd + 1;
        const rocblas_int nv = kd * nt * (nt + 1) / 2;
        const rocblas_int ldv = 2 * kd;
        *size_V_hb2st = sizeof(T) * ldv * nv * batch_count;
        *size_tau_hb2st = sizeof(T) * nv * batch_count;

        // workspace for unmtr_hb2st: Tr, W, Z, work, workArr
        // (scalars reuse the existing scalars buffer)
        rocblas_int max_parallel_2stage = 1;
        size_t s_Tr, s_W2, s_Z2, s_work2, s_workArr2, s_scalars2;
        rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, rocblas_int>(
            rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count, &max_parallel_2stage,
            &s_scalars2, &s_Tr, &s_W2, &s_Z2, &s_work2, &s_workArr2);
        *size_he2hb_work = std::max(
            *size_he2hb_work, s_Tr + s_W2 + s_Z2 + s_work2 + s_workArr2 + s_scalars2);

        // workspace for ormqr (he2hb back-transform), k = n - kd
        // (scalars reuse the existing scalars buffer)
        if(evect == rocblas_evect_original)
        {
            size_t s_AbyxORwork, s_diagORtmptr, s_trfact, s_workArr3, s_scalars3;
            rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                rocblas_side_left, n, n, std::max(n - kd, 0), batch_count, &s_scalars3,
                &s_AbyxORwork, &s_diagORtmptr, &s_trfact, &s_workArr3);
            *size_he2hb_work = std::max(
                *size_he2hb_work, s_AbyxORwork + s_diagORtmptr + s_trfact + s_workArr3 + s_scalars3);
        }
    }
}

/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevd_heevd_argCheck(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              S* D,
                                              S* E,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !E) || (n && !D) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_syevd_heevd_getMemorySize(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work1,
                                         size_t* size_work2,
                                         size_t* size_work3,
                                         size_t* size_work4,
                                         size_t* size_tmpz,
                                         size_t* size_splits,
                                         size_t* size_tmptau_W,
                                         size_t* size_tau,
                                         size_t* size_workArr,
                                         bool* optim_mem,
                                         size_t* size_Aband,
                                         size_t* size_he2hb_work,
                                         size_t* size_V_hb2st,
                                         size_t* size_tau_hb2st)
{
    *size_scalars = 0;
    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_tmptau_W = 0;
    *size_tau = 0;
    *size_workArr = 0;
    *size_splits = 0;
    *size_tmpz = 0;
    *optim_mem = true;
    *size_Aband = 0;
    *size_he2hb_work = 0;
    *size_V_hb2st = 0;
    *size_tau_hb2st = 0;

    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        return;
    }

    rocsolver_alg_mode alg_mode;
    rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &alg_mode);

    size_t unused;
    size_t w11 = 0, w12 = 0, w13 = 0;
    size_t w21 = 0, w22 = 0, w23 = 0;
    size_t w31 = 0, w32 = 0;
    size_t t1 = 0, t2 = 0;
    size_t s1 = 0, s2 = 0;
    size_t z1 = 0, z2 = 0;

    // requirements for tridiagonalization (sytrd/hetrd)
    rocsolver_sytrd_hetrd_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w11, &w21, &t1,
                                                    &unused, false);

    if(alg_mode != rocsolver_alg_mode_hybrid || evect == rocblas_evect_original)
    {
        // extra requirements for computing eigenvalues and vectors (stedc)
        rocsolver_stedc_getMemorySize<BATCHED, T, S>(rocblas_evect_tridiagonal, n, batch_count,
                                                     &w31, &w22, &w12, &z1, &s1, size_workArr);
    }

    if(evect == rocblas_evect_original)
    {
        // extra requirements for ormtr/unmtr
        rocsolver_ormtr_unmtr_getMemorySize<BATCHED, STRIDED, T>(
            rocblas_side_left, uplo, rocblas_operation_none, n, n, batch_count, &unused, &w23, &z2,
            &s2, size_work4, &w13, &w32, &unused, optim_mem);
    }

    // size of array for temporary matrix products
    t2 = sizeof(T) * n * n * batch_count;

    // get max values
    *size_work1 = std::max({w11, w12, w13});
    *size_work2 = std::max({w21, w22, w23});
    *size_work3 = std::max(w31, w32);
    *size_tmptau_W = std::max(t1, t2);
    *size_splits = std::max(s1, s2);
    *size_tmpz = std::max(z1, z2);

    // size of array for temporary householder scalars
    *size_tau = sizeof(T) * n * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = std::max(*size_workArr, 2 * sizeof(T*) * batch_count);

    // requirements for 2-stage path (used when n > SYEVD_2STAGE_SWITCHSIZE)
    if(n > SYEVD_2STAGE_SWITCHSIZE)
    {
        const rocblas_int kd = SYEVD_2STAGE_KD;
        const rocblas_int nb = SYEVD_2STAGE_NB;

        // band matrix: (3*kd - 1) rows by n cols
        const rocblas_int ldab = 3 * kd - 1;
        *size_Aband = sizeof(T) * ldab * n * batch_count;

        // he2hb internal workspace
        size_t s_he2hb, s_D, s_V, s_W, s_X, s_Z, s_work, s_workArr;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, rocblas_int>(
            n, kd, nb, batch_count, &s_he2hb, &s_D, &s_V, &s_W, &s_X, &s_Z, &s_work, &s_workArr);
        *size_he2hb_work
            = s_he2hb + s_D + s_V + s_W + s_X + s_Z + s_work + s_workArr;

        // V and tau for hb2st and unmtr_hb2st:
        // nt = ceildiv(n - 1, kd); nv = kd * nt * (nt + 1) / 2
        const rocblas_int nt = (n - 2) / kd + 1;
        const rocblas_int nv = kd * nt * (nt + 1) / 2;
        const rocblas_int ldv = 2 * kd;
        *size_V_hb2st = sizeof(T) * ldv * nv * batch_count;
        *size_tau_hb2st = sizeof(T) * nv * batch_count;

        // workspace for unmtr_hb2st: scalars, Tr, W, Z, work, workArr
        rocblas_int max_parallel_2stage = 1;
        size_t s_Tr, s_W2, s_Z2, s_work2, s_workArr2, s_scalars2;
        rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, rocblas_int>(
            rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count, &max_parallel_2stage,
            &s_scalars2, &s_Tr, &s_W2, &s_Z2, &s_work2, &s_workArr2);
        *size_he2hb_work = std::max(
            *size_he2hb_work, s_Tr + s_W2 + s_Z2 + s_work2 + s_workArr2 + s_scalars2);

        // workspace for ormqr (he2hb back-transform), k = n - kd
        // (scalars reuse the existing scalars buffer)
        if(evect == rocblas_evect_original)
        {
            size_t s_AbyxORwork, s_diagORtmptr, s_trfact, s_workArr3, s_scalars3;
            rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                rocblas_side_left, n, n, std::max(n - kd, 0), batch_count, &s_scalars3,
                &s_AbyxORwork, &s_diagORtmptr, &s_trfact, &s_workArr3);
            *size_he2hb_work = std::max(
                *size_he2hb_work, s_AbyxORwork + s_diagORtmptr + s_trfact + s_workArr3 + s_scalars3);
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              W A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              S* tmpz,
                                              rocblas_int* splits,
                                              T* tmptau_W,
                                              T* tau,
                                              T** workArr,
                                              T* Aband,
                                              T* he2hb_work,
                                              T* V_hb2st,
                                              T* tau_hb2st)
{
    ROCSOLVER_ENTER("syevd_heevd", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    {
        // memory workspace sizes:
        // size for constants in rocblas calls
        size_t size_scalars;
        // size of reusable workspaces
        size_t size_work1;
        size_t size_work2;
        size_t size_work3;
        size_t size_tmptau_W;
        // extra space for call stedc
        size_t size_splits, size_tmpz;
        // size of array of pointers (only for batched case)
        size_t size_workArr;
        // size for temporary householder scalars
        size_t size_tau;
        // 2-stage workspace sizes
        size_t size_Aband, size_he2hb_work, size_V_hb2st, size_tau_hb2st;

        rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(
            handle, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2,
            &size_work3, &size_tmpz, &size_splits, &size_tmptau_W, &size_tau, &size_workArr,
            &size_Aband, &size_he2hb_work, &size_V_hb2st, &size_tau_hb2st);

        // Memory in `scalars` has already been initialized at this point
        HIP_CHECK(hipMemsetAsync((void*)work1, 0, size_work1, stream));
        HIP_CHECK(hipMemsetAsync((void*)work2, 0, size_work2, stream));
        HIP_CHECK(hipMemsetAsync((void*)work3, 0, size_work3, stream));
        HIP_CHECK(hipMemsetAsync((void*)tmpz, 0, size_tmpz, stream));
        HIP_CHECK(hipMemsetAsync((void*)splits, 0, size_splits, stream));
        HIP_CHECK(hipMemsetAsync((void*)tmptau_W, 0, size_tmptau_W, stream));
        HIP_CHECK(hipMemsetAsync((void*)tau, 0, size_tau, stream));
        HIP_CHECK(hipMemsetAsync((void*)workArr, 0, size_workArr, stream));
    }

    rocsolver_alg_mode sterf_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &sterf_mode));

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(syev_scalar_case<T>, gridReset, threads, 0, stream, evect, A,
                                strideA, D, strideD, batch_count);
        return rocblas_status_success;
    }

    // TODO: Scale the matrix

    // 2-stage path: he2hb + hb2st + unmtr_hb2st + ormqr
    if(n > SYEVD_2STAGE_SWITCHSIZE)
    {
        const rocblas_int kd = SYEVD_2STAGE_KD;
        const rocblas_int nb = SYEVD_2STAGE_NB;
        const rocblas_int ldab = 3 * kd - 1;
        const rocblas_int ldv_hb2st = 2 * kd;
        const rocblas_int nt = (n - 2) / kd + 1;
        const rocblas_int nv = kd * nt * (nt + 1) / 2;

        // Strides for band and V arrays (non-batched strides = 0 for the first template overload)
        const rocblas_stride strideAband = 0;
        const rocblas_stride strideV_hb2st = 0;
        const rocblas_stride strideTau_hb2st = 0;

        // Partition he2hb_work into sub-workspaces
        size_t s_he2hb_scalars, s_D, s_V, s_W, s_X, s_Z, s_work, s_workArr_he2hb;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, rocblas_int>(
            n, kd, nb, batch_count, &s_he2hb_scalars, &s_D, &s_V, &s_W, &s_X, &s_Z, &s_work,
            &s_workArr_he2hb);
        T* he2hb_scalars = he2hb_work;
        T* he2hb_D = he2hb_scalars + s_he2hb_scalars / sizeof(T);
        T* he2hb_V = he2hb_D + s_D / sizeof(T);
        T* he2hb_W = he2hb_V + s_V / sizeof(T);
        T* he2hb_X = he2hb_W + s_W / sizeof(T);
        T* he2hb_Z = he2hb_X + s_X / sizeof(T);
        T* he2hb_work2 = he2hb_Z + s_Z / sizeof(T);
        T** he2hb_workArr = (T**)(he2hb_work2 + s_work / sizeof(T));

        // If uplo == upper, symmetrize A: copy upper triangle to lower via conjugate transpose
        if(uplo == rocblas_fill_upper)
        {
            const rocblas_int blocks = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                                    dim3(BS2, BS2, 1), 0, stream,
                                    rocblas_operation_conjugate_transpose, n, n, A, shiftA, lda,
                                    strideA, A, shiftA, lda, strideA, no_mask{},
                                    rocblas_fill_upper, rocblas_diagonal_unit);
        }

        // Initialize scalars for he2hb if needed
        if(s_he2hb_scalars > 0)
            init_scalars(handle, he2hb_scalars);

        // Stage 1a: reduce dense Hermitian to band form (he2hb)
        // tau (size n) stores the he2hb Householder scalars; A stores the Householder vectors
        ROCBLAS_CHECK(rocsolver_sy2sb_he2hb_template<BATCHED, STRIDED, T, rocblas_int>(
            handle, n, kd, nb, A, shiftA, lda, strideA, Aband, ldab, strideAband, tau, n,
            batch_count, he2hb_scalars, he2hb_D, he2hb_V, he2hb_W, he2hb_X, he2hb_Z, he2hb_work2,
            he2hb_workArr));

        // Stage 1b: reduce band to tridiagonal form (hb2st)
        // V_hb2st and tau_hb2st store the hb2st Householder data
        ROCBLAS_CHECK(rocsolver_sb2st_hb2st_template<BATCHED, STRIDED, T, rocblas_int>(
            handle, rocblas_fill_lower, n, kd, Aband, 0, ldab, strideAband, D, strideD, E, strideE,
            V_hb2st, ldv_hb2st, strideV_hb2st, tau_hb2st, strideTau_hb2st, batch_count));

        if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
        {
            // only in hybrid mode, compute eigenvalues using sterf
            rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                        (rocblas_int*)work1);
        }
        else
        {
            // compute eigenvalues and eigenvectors of the tridiagonal (stedc)
            constexpr bool ISBATCHED = BATCHED || STRIDED;
            const rocblas_int ldw = n;
            const rocblas_stride strideW = n * n;

            rocsolver_stedc_template<false, ISBATCHED, T>(
                handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0,
                ldw, strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits,
                (S**)workArr);

            // update the eigenvectors (if applicable)
            if(evect == rocblas_evect_original)
            {
                // Partition he2hb_work for unmtr_hb2st sub-workspaces
                rocblas_int max_parallel_2stage = 1;
                size_t s_Tr, s_W2, s_Z2, s_work2_unmtr, s_workArr2, s_scalars2;
                rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, rocblas_int>(
                    rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count,
                    &max_parallel_2stage, &s_scalars2, &s_Tr, &s_W2, &s_Z2, &s_work2_unmtr,
                    &s_workArr2);
                T* unmtr_scalars = he2hb_work;
                T* unmtr_Tr = unmtr_scalars + s_scalars2 / sizeof(T);
                T* unmtr_W = unmtr_Tr + s_Tr / sizeof(T);
                T* unmtr_Z = unmtr_W + s_W2 / sizeof(T);
                T* unmtr_work = unmtr_Z + s_Z2 / sizeof(T);
                T** unmtr_workArr = (T**)(unmtr_work + s_work2_unmtr / sizeof(T));
                if(s_scalars2 > 0)
                    init_scalars(handle, unmtr_scalars);

                // Back-transform stage 1b: apply Q_hb2st to eigenvector matrix (unmtr_hb2st)
                // C = Q_hb2st * tmptau_W (tmptau_W holds the eigenvectors from stedc)
                ROCBLAS_CHECK(rocsolver_ormtr_unmtr_hb2st_template<BATCHED, STRIDED, T, T*>(
                    handle, rocblas_side_left, rocblas_operation_none, n, n, kd, V_hb2st, 0,
                    ldv_hb2st, strideV_hb2st, tau_hb2st, strideTau_hb2st, tmptau_W, 0, ldw,
                    strideW, batch_count, max_parallel_2stage, unmtr_scalars, unmtr_Tr, unmtr_W,
                    unmtr_Z, unmtr_work, unmtr_workArr));

                // Back-transform stage 1a: apply Q_he2hb to eigenvector matrix (ormqr)
                // Q_he2hb is stored in lower part of A (below diagonal kd) and in tau
                // Partition he2hb_work for ormqr sub-workspaces
                const rocblas_int k_he2hb = std::max(n - kd, 0);
                size_t s_AbyxORwork, s_diagORtmptr, s_trfact, s_workArr3, s_scalars3;
                rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                    rocblas_side_left, n, n, k_he2hb, batch_count, &s_scalars3, &s_AbyxORwork,
                    &s_diagORtmptr, &s_trfact, &s_workArr3);
                T* ormqr_scalars = he2hb_work;
                T* ormqr_AbyxORwork = ormqr_scalars + s_scalars3 / sizeof(T);
                T* ormqr_diagORtmptr = ormqr_AbyxORwork + s_AbyxORwork / sizeof(T);
                T* ormqr_trfact = ormqr_diagORtmptr + s_diagORtmptr / sizeof(T);
                T** ormqr_workArr = (T**)(ormqr_trfact + s_trfact / sizeof(T));
                if(s_scalars3 > 0)
                    init_scalars(handle, ormqr_scalars);

                // Apply Q_he2hb from left: tmptau_W := Q_he2hb * tmptau_W
                // Householder vectors V are stored in A below diagonal kd
                // For BATCHED=true, A is T* const* and C is T*, so the adapter overload is
                // selected (void return). For BATCHED=false, A and C are both T* so the
                // status-returning overload is selected.
                if constexpr(BATCHED)
                {
                    // void adapter: T* const A[], T* C — workArr used as workArr2
                    rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n, n, k_he2hb, A,
                        shiftA, lda, strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count,
                        ormqr_scalars, ormqr_AbyxORwork, ormqr_diagORtmptr, ormqr_trfact,
                        ormqr_workArr, workArr);
                }
                else
                {
                    // status-returning overload: T* A, T* C
                    ROCBLAS_CHECK(rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n, n, k_he2hb, A,
                        shiftA, lda, strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count,
                        ormqr_scalars, ormqr_AbyxORwork, ormqr_diagORtmptr, ormqr_trfact,
                        ormqr_workArr));
                }

                // copy matrix product into A
                const rocblas_int copyblocks = (n - 1) / BS2 + 1;
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                        dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW,
                                        A, shiftA, lda, strideA);
            }
        }

        return rocblas_status_success;
    }

    // 1-stage path: hetrd + stedc + unmtr
    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                            strideE, tau, n, batch_count, scalars, (T*)work1,
                                            (T*)work2, tmptau_W, workArr, false);

    if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
    {
        // only in hybrid mode, compute eigenvalues using sterf
        rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                    (rocblas_int*)work1);
    }
    else
    {
        // for performance reasons, we use stedc to compute eigenvalues even if the eigenvectors are ignored
        constexpr bool ISBATCHED = BATCHED || STRIDED;
        const rocblas_int ldw = n;
        const rocblas_stride strideW = n * n;

        rocsolver_stedc_template<false, ISBATCHED, T>(
            handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0, ldw,
            strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);

        // update the eigenvectors (if applicable)
        if(evect == rocblas_evect_original)
        {
            rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda,
                strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count, scalars, (T*)work2,
                (T*)work1, (T*)work3, workArr);

            // copy matrix product into A
            const rocblas_int copyblocks = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW, A,
                                    shiftA, lda, strideA);
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              W A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              void* work4,
                                              S* tmpz,
                                              rocblas_int* splits,
                                              T* tmptau_W,
                                              T* tau,
                                              T** workArr,
                                              bool optim_mem,
                                              T* Aband,
                                              T* he2hb_work,
                                              T* V_hb2st,
                                              T* tau_hb2st)
{
    ROCSOLVER_ENTER("syevd_heevd", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    {
        // memory workspace sizes:
        // size for constants in rocblas calls
        size_t size_scalars;
        // size of reusable workspaces
        size_t size_work1;
        size_t size_work2;
        size_t size_work3;
        size_t size_work4;
        size_t size_tmptau_W;
        // extra space for stedc call
        size_t size_splits, size_tmpz;
        // size of array of pointers (only for batched case)
        size_t size_workArr;
        // size for temporary householder scalars
        size_t size_tau;
        // 2-stage workspace sizes
        size_t size_Aband, size_he2hb_work, size_V_hb2st, size_tau_hb2st;

        rocsolver_syevd_heevd_getMemorySize<BATCHED, STRIDED, T, S>(
            handle, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2,
            &size_work3, &size_work4, &size_tmpz, &size_splits, &size_tmptau_W, &size_tau,
            &size_workArr, &optim_mem, &size_Aband, &size_he2hb_work, &size_V_hb2st,
            &size_tau_hb2st);

        // Memory in `scalars` has already been initialized at this point
        HIP_CHECK(hipMemsetAsync((void*)work1, 0, size_work1, stream));
        HIP_CHECK(hipMemsetAsync((void*)work2, 0, size_work2, stream));
        HIP_CHECK(hipMemsetAsync((void*)work3, 0, size_work3, stream));
        HIP_CHECK(hipMemsetAsync((void*)work4, 0, size_work4, stream));
        HIP_CHECK(hipMemsetAsync((void*)tmpz, 0, size_tmpz, stream));
        HIP_CHECK(hipMemsetAsync((void*)splits, 0, size_splits, stream));
        HIP_CHECK(hipMemsetAsync((void*)tmptau_W, 0, size_tmptau_W, stream));
        HIP_CHECK(hipMemsetAsync((void*)tau, 0, size_tau, stream));
        HIP_CHECK(hipMemsetAsync((void*)workArr, 0, size_workArr, stream));
    }

    rocsolver_alg_mode sterf_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &sterf_mode));

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(syev_scalar_case<T>, gridReset, threads, 0, stream, evect, A,
                                strideA, D, strideD, batch_count);
        return rocblas_status_success;
    }

    // TODO: Scale the matrix

    // 2-stage path: he2hb + hb2st + unmtr_hb2st + ormqr
    if(n > SYEVD_2STAGE_SWITCHSIZE)
    {
        const rocblas_int kd = SYEVD_2STAGE_KD;
        const rocblas_int nb = SYEVD_2STAGE_NB;
        const rocblas_int ldab = 3 * kd - 1;
        const rocblas_int ldv_hb2st = 2 * kd;
        const rocblas_int nt = (n - 2) / kd + 1;
        const rocblas_int nv = kd * nt * (nt + 1) / 2;

        // Strides for band and V arrays
        const rocblas_stride strideAband = (rocblas_stride)ldab * n;
        const rocblas_stride strideV_hb2st = (rocblas_stride)ldv_hb2st * nv;
        const rocblas_stride strideTau_hb2st = nv;

        // Partition he2hb_work into sub-workspaces
        size_t s_he2hb_scalars, s_D, s_V, s_W, s_X, s_Z, s_work, s_workArr_he2hb;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, rocblas_int>(
            n, kd, nb, batch_count, &s_he2hb_scalars, &s_D, &s_V, &s_W, &s_X, &s_Z, &s_work,
            &s_workArr_he2hb);
        T* he2hb_scalars = he2hb_work;
        T* he2hb_D = he2hb_scalars + s_he2hb_scalars / sizeof(T);
        T* he2hb_V = he2hb_D + s_D / sizeof(T);
        T* he2hb_W = he2hb_V + s_V / sizeof(T);
        T* he2hb_X = he2hb_W + s_W / sizeof(T);
        T* he2hb_Z = he2hb_X + s_X / sizeof(T);
        T* he2hb_work2 = he2hb_Z + s_Z / sizeof(T);
        T** he2hb_workArr = (T**)(he2hb_work2 + s_work / sizeof(T));

        // If uplo == upper, symmetrize A: copy upper triangle to lower via conjugate transpose
        if(uplo == rocblas_fill_upper)
        {
            const rocblas_int blocks = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                                    dim3(BS2, BS2, 1), 0, stream,
                                    rocblas_operation_conjugate_transpose, n, n, A, shiftA, lda,
                                    strideA, A, shiftA, lda, strideA, no_mask{},
                                    rocblas_fill_upper, rocblas_diagonal_unit);
        }

        // Initialize scalars for he2hb if needed
        if(s_he2hb_scalars > 0)
            init_scalars(handle, he2hb_scalars);

        // Stage 1a: reduce dense Hermitian to band form (he2hb)
        // tau (size n) stores the he2hb Householder scalars; A stores the Householder vectors
        ROCBLAS_CHECK(rocsolver_sy2sb_he2hb_template<BATCHED, STRIDED, T, rocblas_int>(
            handle, n, kd, nb, A, shiftA, lda, strideA, Aband, ldab, strideAband, tau, n,
            batch_count, he2hb_scalars, he2hb_D, he2hb_V, he2hb_W, he2hb_X, he2hb_Z, he2hb_work2,
            he2hb_workArr));

        // Stage 1b: reduce band to tridiagonal form (hb2st)
        // V_hb2st and tau_hb2st store the hb2st Householder data
        ROCBLAS_CHECK(rocsolver_sb2st_hb2st_template<BATCHED, STRIDED, T, rocblas_int>(
            handle, rocblas_fill_lower, n, kd, Aband, 0, ldab, strideAband, D, strideD, E, strideE,
            V_hb2st, ldv_hb2st, strideV_hb2st, tau_hb2st, strideTau_hb2st, batch_count));

        if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
        {
            // only in hybrid mode, compute eigenvalues using sterf
            rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                        (rocblas_int*)work1);
        }
        else
        {
            // compute eigenvalues and eigenvectors of the tridiagonal (stedc)
            constexpr bool ISBATCHED = BATCHED || STRIDED;
            const rocblas_int ldw = n;
            const rocblas_stride strideW = n * n;

            rocsolver_stedc_template<false, ISBATCHED, T>(
                handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0,
                ldw, strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits,
                (S**)workArr);

            // update the eigenvectors (if applicable)
            if(evect == rocblas_evect_original)
            {
                // Partition he2hb_work for unmtr_hb2st sub-workspaces
                rocblas_int max_parallel_2stage = 1;
                size_t s_Tr, s_W2, s_Z2, s_work2_unmtr, s_workArr2, s_scalars2;
                rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, rocblas_int>(
                    rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count,
                    &max_parallel_2stage, &s_scalars2, &s_Tr, &s_W2, &s_Z2, &s_work2_unmtr,
                    &s_workArr2);
                T* unmtr_scalars = he2hb_work;
                T* unmtr_Tr = unmtr_scalars + s_scalars2 / sizeof(T);
                T* unmtr_W = unmtr_Tr + s_Tr / sizeof(T);
                T* unmtr_Z = unmtr_W + s_W2 / sizeof(T);
                T* unmtr_work = unmtr_Z + s_Z2 / sizeof(T);
                T** unmtr_workArr = (T**)(unmtr_work + s_work2_unmtr / sizeof(T));
                if(s_scalars2 > 0)
                    init_scalars(handle, unmtr_scalars);

                // Back-transform stage 1b: apply Q_hb2st to eigenvector matrix (unmtr_hb2st)
                // C = Q_hb2st * tmptau_W (tmptau_W holds the eigenvectors from stedc)
                ROCBLAS_CHECK(rocsolver_ormtr_unmtr_hb2st_template<BATCHED, STRIDED, T, T*>(
                    handle, rocblas_side_left, rocblas_operation_none, n, n, kd, V_hb2st, 0,
                    ldv_hb2st, strideV_hb2st, tau_hb2st, strideTau_hb2st, tmptau_W, 0, ldw,
                    strideW, batch_count, max_parallel_2stage, unmtr_scalars, unmtr_Tr, unmtr_W,
                    unmtr_Z, unmtr_work, unmtr_workArr));

                // Back-transform stage 1a: apply Q_he2hb to eigenvector matrix (ormqr)
                // Q_he2hb is stored in lower part of A (below diagonal kd) and in tau
                // Partition he2hb_work for ormqr sub-workspaces
                const rocblas_int k_he2hb = std::max(n - kd, 0);
                size_t s_AbyxORwork, s_diagORtmptr, s_trfact, s_workArr3, s_scalars3;
                rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                    rocblas_side_left, n, n, k_he2hb, batch_count, &s_scalars3, &s_AbyxORwork,
                    &s_diagORtmptr, &s_trfact, &s_workArr3);
                T* ormqr_scalars = he2hb_work;
                T* ormqr_AbyxORwork = ormqr_scalars + s_scalars3 / sizeof(T);
                T* ormqr_diagORtmptr = ormqr_AbyxORwork + s_AbyxORwork / sizeof(T);
                T* ormqr_trfact = ormqr_diagORtmptr + s_diagORtmptr / sizeof(T);
                T** ormqr_workArr = (T**)(ormqr_trfact + s_trfact / sizeof(T));
                if(s_scalars3 > 0)
                    init_scalars(handle, ormqr_scalars);

                // Apply Q_he2hb from left: tmptau_W := Q_he2hb * tmptau_W
                // Householder vectors V are stored in A below diagonal kd
                // For BATCHED=true, A is T* const* and C is T*, so the adapter overload is
                // selected (void return). For BATCHED=false, A and C are both T* so the
                // status-returning overload is selected.
                if constexpr(BATCHED)
                {
                    // void adapter: T* const A[], T* C — workArr used as workArr2
                    rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n, n, k_he2hb, A,
                        shiftA, lda, strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count,
                        ormqr_scalars, ormqr_AbyxORwork, ormqr_diagORtmptr, ormqr_trfact,
                        ormqr_workArr, workArr);
                }
                else
                {
                    // status-returning overload: T* A, T* C
                    ROCBLAS_CHECK(rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n, n, k_he2hb, A,
                        shiftA, lda, strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count,
                        ormqr_scalars, ormqr_AbyxORwork, ormqr_diagORtmptr, ormqr_trfact,
                        ormqr_workArr));
                }

                // copy matrix product into A
                const rocblas_int copyblocks = (n - 1) / BS2 + 1;
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                        dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW,
                                        A, shiftA, lda, strideA);
            }
        }

        return rocblas_status_success;
    }

    // 1-stage path: hetrd + stedc + unmtr
    // reduce A to tridiagonal form
    rocsolver_sytrd_hetrd_template<BATCHED>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD, E,
                                            strideE, tau, n, batch_count, scalars, (T*)work1,
                                            (T*)work2, tmptau_W, workArr, false);

    if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
    {
        // only in hybrid mode, compute eigenvalues using sterf
        rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                    (rocblas_int*)work1);
    }
    else
    {
        // for performance reasons, we use stedc to compute eigenvalues even if the eigenvectors are ignored
        constexpr bool ISBATCHED = BATCHED || STRIDED;
        const rocblas_int ldw = n;
        const rocblas_stride strideW = n * n;

        rocsolver_stedc_template<false, ISBATCHED, T>(
            handle, rocblas_evect_tridiagonal, n, D, 0, strideD, E, 0, strideE, tmptau_W, 0, ldw,
            strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);

        // update the eigenvectors (if applicable)
        if(evect == rocblas_evect_original)
        {
            rocsolver_ormtr_unmtr_template<BATCHED, STRIDED>(
                handle, rocblas_side_left, uplo, rocblas_operation_none, n, n, A, shiftA, lda,
                strideA, tau, n, tmptau_W, 0, ldw, strideW, batch_count, scalars, (T*)work2, tmpz,
                splits, work4, (T*)work1, (T*)work3, workArr, optim_mem);

            // copy matrix product into A
            const rocblas_int copyblocks = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW, A,
                                    shiftA, lda, strideA);
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
