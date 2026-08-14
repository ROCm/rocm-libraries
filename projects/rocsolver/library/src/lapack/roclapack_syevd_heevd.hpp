/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// Using original QR routines.
template <bool BATCHED, typename T, typename S, typename I>
void rocsolver_syevd_heevd_getMemorySize(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const I n,
                                         const I batch_count,
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

    rocsolver_alg_mode hetrd_mode;
    rocsolver_get_alg_mode(handle, rocsolver_function_hetrd, &hetrd_mode);

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

    // 2-stage path: he2hb + hb2st + unmtr_hb2st + unmqr
    // 2-stage BATCHED not currently working. Also update use_2stage in implementation.
    const bool use_2stage = !BATCHED
        && (hetrd_mode == rocsolver_alg_mode_2stage
            || (hetrd_mode == rocsolver_alg_mode_auto && n >= SYEVD_2STAGE_SWITCHSIZE));
    if(use_2stage)
    {
        const I kd = SYEVD_2STAGE_KD;
        const I nb = SYEVD_2STAGE_NB;

        // band matrix: (3*kd - 1) rows by n cols
        const I ldab = 3 * kd - 1;
        *size_Aband = sizeof(T) * ldab * n * batch_count;

        // he2hb internal workspace
        size_t size_scalars2, size_D, size_V, size_W, size_X, size_Z, size_work, size_workArr;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, I>(
            uplo, n, kd, nb, batch_count, &size_scalars2, &size_D, &size_V, &size_W, &size_X,
            &size_Z, &size_work, &size_workArr);
        *size_scalars = std::max(*size_scalars, size_scalars2);
        *size_he2hb_work = size_D + size_V + size_W + size_X + size_Z + size_work + size_workArr;

        // V and tau for hb2st and unmtr_hb2st
        const I nt = ceildiv(n - 1, kd);
        const I nv = kd * nt * (nt + 1) / 2;
        const I ldv = 2 * kd - 1;
        *size_V_hb2st = sizeof(T) * ldv * nv * batch_count;
        *size_tau_hb2st = sizeof(T) * nv * batch_count;

        if(evect == rocblas_evect_original)
        {
            // workspace for unmtr_hb2st: Tr, W, Z, work, workArr
            I max_parallel_2stage = 1;
            size_t size_Tr, size_W2, size_Z2, size_work2, size_workArr2;
            rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, I>(
                rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count,
                &max_parallel_2stage, &size_scalars2, &size_Tr, &size_W2, &size_Z2, &size_work2,
                &size_workArr2);
            *size_scalars = std::max(*size_scalars, size_scalars2);
            *size_he2hb_work = std::max(*size_he2hb_work,
                                        size_Tr + size_W2 + size_Z2 + size_work2 + size_workArr2);

            // workspace for unmqr (he2hb back-transform): applies Q (n-kd x n-kd) to W[kd:n, :]
            const I n_kd = std::max(n - kd, I(0));
            size_t size_AbyxORwork, size_diagORtmptr, size_trfact, size_workArr3;
            rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                rocblas_side_left, n_kd, n, n_kd, batch_count, &size_scalars2, &size_AbyxORwork,
                &size_diagORtmptr, &size_trfact, &size_workArr3);
            *size_scalars = std::max(*size_scalars, size_scalars2);
            // workArr2 is needed by the adapter overload when BATCHED=true to build a pointer
            // array for the strided C matrix (tmptau_W); sizeof(T*) * batch_count bytes.
            const size_t size_workArr2_unmqr = BATCHED ? sizeof(T*) * batch_count : 0;
            *size_he2hb_work = std::max(*size_he2hb_work,
                                        size_AbyxORwork + size_diagORtmptr + size_trfact
                                            + size_workArr3 + size_workArr2_unmqr);
        }
    }
}

/** Argument checking **/
template <typename T, typename S, typename I>
rocblas_status rocsolver_syevd_heevd_argCheck(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const I n,
                                              T A,
                                              const I lda,
                                              S* D,
                                              S* E,
                                              I* info,
                                              const I batch_count = 1)
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

// Using optimized QR routines.
template <bool BATCHED, bool STRIDED, typename T, typename S, typename I>
void rocsolver_syevd_heevd_getMemorySize(rocblas_handle handle,
                                         const rocblas_evect evect,
                                         const rocblas_fill uplo,
                                         const I n,
                                         const I batch_count,
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
                                         size_t* size_Aband,
                                         size_t* size_he2hb_work,
                                         size_t* size_V_hb2st,
                                         size_t* size_tau_hb2st,
                                         bool* optim_mem)
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
    *size_Aband = 0;
    *size_he2hb_work = 0;
    *size_V_hb2st = 0;
    *size_tau_hb2st = 0;
    *optim_mem = true;

    // if quick return, set workspace to zero
    if(n <= 1 || batch_count == 0)
    {
        return;
    }

    rocsolver_alg_mode alg_mode;
    rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &alg_mode);

    rocsolver_alg_mode hetrd_mode;
    rocsolver_get_alg_mode(handle, rocsolver_function_hetrd, &hetrd_mode);

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

    // 2-stage path: he2hb + hb2st + unmtr_hb2st + unmqr
    // 2-stage BATCHED not currently working. Also update use_2stage in implementation.
    const bool use_2stage = !BATCHED
        && (hetrd_mode == rocsolver_alg_mode_2stage
            || (hetrd_mode == rocsolver_alg_mode_auto && n >= SYEVD_2STAGE_SWITCHSIZE));
    if(use_2stage)
    {
        const I kd = SYEVD_2STAGE_KD;
        const I nb = SYEVD_2STAGE_NB;

        // band matrix: (3*kd - 1) rows by n cols
        const I ldab = 3 * kd - 1;
        *size_Aband = sizeof(T) * ldab * n * batch_count;

        // he2hb internal workspace
        size_t size_scalars2, size_D, size_V, size_W, size_X, size_Z, size_work, size_workArr;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, I>(
            uplo, n, kd, nb, batch_count, &size_scalars2, &size_D, &size_V, &size_W, &size_X,
            &size_Z, &size_work, &size_workArr);
        *size_scalars = std::max(*size_scalars, size_scalars2);
        *size_he2hb_work = size_D + size_V + size_W + size_X + size_Z + size_work + size_workArr;

        // V and tau for hb2st and unmtr_hb2st:
        // nt = ceildiv(n - 1, kd); nv = kd * nt * (nt + 1) / 2
        const I nt = ceildiv(n - 1, kd);
        const I nv = kd * nt * (nt + 1) / 2;
        const I ldv = 2 * kd - 1;
        *size_V_hb2st = sizeof(T) * ldv * nv * batch_count;
        *size_tau_hb2st = sizeof(T) * nv * batch_count;

        if(evect == rocblas_evect_original)
        {
            // workspace for unmtr_hb2st: Tr, W, Z, work, workArr
            I max_parallel_2stage = 1;
            size_t size_Tr, size_W2, size_Z2, size_work2, size_workArr2;
            rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, I>(
                rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count,
                &max_parallel_2stage, &size_scalars2, &size_Tr, &size_W2, &size_Z2, &size_work2,
                &size_workArr2);
            *size_scalars = std::max(*size_scalars, size_scalars2);
            *size_he2hb_work = std::max(*size_he2hb_work,
                                        size_Tr + size_W2 + size_Z2 + size_work2 + size_workArr2);

            // workspace for unmqr (he2hb back-transform): applies Q (n-kd x n-kd) to W[kd:n, :]
            const I n_kd = std::max(n - kd, I(0));
            size_t size_AbyxORwork, size_diagORtmptr, size_trfact, size_workArr3;
            rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                rocblas_side_left, n_kd, n, n_kd, batch_count, &size_scalars2, &size_AbyxORwork,
                &size_diagORtmptr, &size_trfact, &size_workArr3);
            *size_scalars = std::max(*size_scalars, size_scalars2);
            // workArr2 is needed by the adapter overload when BATCHED=true to build a pointer
            // array for the strided C matrix (tmptau_W); sizeof(T*) * batch_count bytes.
            const size_t size_workArr2_unmqr = BATCHED ? sizeof(T*) * batch_count : 0;
            *size_he2hb_work = std::max(*size_he2hb_work,
                                        size_AbyxORwork + size_diagORtmptr + size_trfact
                                            + size_workArr3 + size_workArr2_unmqr);
        }
    }
}

// Using original QR routines.
template <bool BATCHED, bool STRIDED, typename T, typename S, typename W, typename I>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const I n,
                                              W A,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              // info encodes O(n^2) submatrix row/col positions on
                                              // failure, so it is typed I (64-bit on the _64 path).
                                              I* info,
                                              const I batch_count,
                                              T* scalars,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              S* tmpz,
                                              I* splits,
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

    // size for constants in rocblas calls
    size_t size_scalars;
    {
        // memory workspace sizes:
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

    rocsolver_alg_mode hetrd_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_hetrd, &hetrd_mode));

    I blocksReset = (batch_count - 1) / BS1 + 1;
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

    // 2-stage path: he2hb + hb2st + unmtr_hb2st + unmqr
    // 2-stage BATCHED not currently working. Also update use_2stage in getMemorySize.
    const bool use_2stage = !BATCHED
        && (hetrd_mode == rocsolver_alg_mode_2stage
            || (hetrd_mode == rocsolver_alg_mode_auto && n >= SYEVD_2STAGE_SWITCHSIZE));
    if(use_2stage)
    {
        const I kd = SYEVD_2STAGE_KD;
        const I nb = SYEVD_2STAGE_NB;
        const I ldab = 3 * kd - 1;
        const I ldv_hb2st = 2 * kd - 1;
        const I nt = ceildiv(n - 1, kd);
        const I nv = kd * nt * (nt + 1) / 2;

        // Strides for band and V arrays, applies to both pointer batched and strided.
        const rocblas_stride strideAband = rocblas_stride(ldab * n);
        const rocblas_stride strideV_hb2st = rocblas_stride(ldv_hb2st * nv);
        const rocblas_stride strideTau_hb2st = rocblas_stride(nv);

        // Partition he2hb_work into sub-workspaces
        size_t size_D, size_V, size_W, size_X, size_Z, size_work, size_workArr_he2hb;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, I>(
            uplo, n, kd, nb, batch_count, &size_scalars, &size_D, &size_V, &size_W, &size_X,
            &size_Z, &size_work, &size_workArr_he2hb);
        T* he2hb_D = he2hb_work;
        T* he2hb_V = he2hb_D + size_D / sizeof(T);
        T* he2hb_W = he2hb_V + size_V / sizeof(T);
        T* he2hb_X = he2hb_W + size_W / sizeof(T);
        T* he2hb_Z = he2hb_X + size_X / sizeof(T);
        T* he2hb_work2 = he2hb_Z + size_Z / sizeof(T);
        T** he2hb_workArr = (T**)(he2hb_work2 + size_work / sizeof(T));

        // Stage 1: reduce dense Hermitian to band form (he2hb)
        // tau (size n) stores the he2hb Householder scalars; A stores the Householder vectors
        ROCBLAS_CHECK(rocsolver_sy2sb_he2hb_template<BATCHED, STRIDED, T, I>(
            handle, uplo, n, kd, nb, // opts
            A, shiftA, lda, strideA, // A
            Aband, ldab, strideAband, // Aband
            tau, n, // tau
            batch_count, scalars, he2hb_D, he2hb_V, he2hb_W, he2hb_X, he2hb_Z, he2hb_work2,
            he2hb_workArr));

        // Stage 2: reduce band to tridiagonal form (hb2st)
        // V_hb2st and tau_hb2st store the hb2st Householder data
        ROCBLAS_CHECK(rocsolver_sb2st_hb2st_template<BATCHED, STRIDED, T, I>(
            handle, rocblas_fill_lower, n, kd, // opts
            Aband, 0, ldab, strideAband, // A
            D, strideD, // D
            E, strideE, // E
            V_hb2st, ldv_hb2st, strideV_hb2st, // V
            tau_hb2st, strideTau_hb2st, // tau
            batch_count));

        if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
        {
            // only in hybrid mode, compute eigenvalues using sterf
            rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                        (I*)work1);
        }
        else
        {
            // compute eigenvalues and eigenvectors of the tridiagonal (stedc)
            constexpr bool ISBATCHED = BATCHED || STRIDED;
            const I ldw = n;
            const rocblas_stride strideW = n * n;

            rocsolver_stedc_template<false, ISBATCHED, T>(
                handle, rocblas_evect_tridiagonal, n, // opts
                D, 0, strideD, // D
                E, 0, strideE, // E
                tmptau_W, 0, ldw, // W
                strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);

            // update the eigenvectors (if applicable)
            if(evect == rocblas_evect_original)
            {
                // Partition he2hb_work for unmtr_hb2st sub-workspaces
                I max_parallel_2stage = 1;
                size_t size_Tr, size_W2, size_Z2, size_work2_unmtr, size_workArr2;
                rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, I>(
                    rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count,
                    &max_parallel_2stage, &size_scalars, &size_Tr, &size_W2, &size_Z2,
                    &size_work2_unmtr, &size_workArr2);
                T* unmtr_Tr = he2hb_work;
                T* unmtr_W = unmtr_Tr + size_Tr / sizeof(T);
                T* unmtr_Z = unmtr_W + size_W2 / sizeof(T);
                T* unmtr_work = unmtr_Z + size_Z2 / sizeof(T);
                T** unmtr_workArr = (T**)(unmtr_work + size_work2_unmtr / sizeof(T));

                // Back-transform stage 2: apply Q_hb2st to eigenvector matrix (unmtr_hb2st)
                // C = Q_hb2st * tmptau_W (tmptau_W holds the eigenvectors from stedc)
                ROCBLAS_CHECK(rocsolver_ormtr_unmtr_hb2st_template<BATCHED, STRIDED, T, T*>(
                    handle, rocblas_side_left, rocblas_operation_none, n, n, kd, // opts
                    V_hb2st, 0, ldv_hb2st, strideV_hb2st, // V
                    tau_hb2st, strideTau_hb2st, // tau
                    tmptau_W, 0, ldw, strideW, // W
                    batch_count, max_parallel_2stage, scalars, unmtr_Tr, unmtr_W, unmtr_Z,
                    unmtr_work, unmtr_workArr));

                // Back-transform stage 1: apply Q_he2hb to eigenvector matrix (unmqr)
                // Q_he2hb is stored in lower part of A (below diagonal kd) and in tau
                // Partition he2hb_work for unmqr sub-workspaces
                const I n_kd = std::max(n - kd, I(0));
                size_t size_AbyxORwork, size_diagORtmptr, size_trfact, size_workArr3;
                rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                    rocblas_side_left, n_kd, n, n_kd, batch_count, &size_scalars, &size_AbyxORwork,
                    &size_diagORtmptr, &size_trfact, &size_workArr3);
                T* unmqr_AbyxORwork = he2hb_work;
                T* unmqr_diagORtmptr = unmqr_AbyxORwork + size_AbyxORwork / sizeof(T);
                T* unmqr_trfact = unmqr_diagORtmptr + size_diagORtmptr / sizeof(T);
                T** unmqr_workArr = (T**)(unmqr_trfact + size_trfact / sizeof(T));
                T** unmqr_workArr2 = unmqr_workArr + size_workArr3 / sizeof(T*);

                // Apply Q_he2hb on the left to W[kd:n, 0:n]:
                //   V is in A[kd:n, 0:n-kd], Q is (n-kd) x (n-kd)
                // When BATCHED=true, A is T* const* and tmptau_W is T*; use the adapter overload
                // which builds a pointer array for C in unmqr_workArr2 before dispatching.
                // When BATCHED=false, both are T* and we call the strided overload directly.
                if constexpr(BATCHED)
                {
                    rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n_kd, n, n_kd, // opts
                        A, shiftA + idx2D(kd, 0, lda), lda, strideA, // A
                        tau, n, // tau
                        tmptau_W, idx2D(kd, 0, ldw), ldw, strideW, // W
                        batch_count, scalars, unmqr_AbyxORwork, unmqr_diagORtmptr, unmqr_trfact,
                        unmqr_workArr, unmqr_workArr2);
                }
                else
                {
                    ROCBLAS_CHECK(rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n_kd, n, n_kd, // opts
                        A, shiftA + idx2D(kd, 0, lda), lda, strideA, // A
                        tau, n, // tau
                        tmptau_W, idx2D(kd, 0, ldw), ldw, strideW, // W
                        batch_count, scalars, unmqr_AbyxORwork, unmqr_diagORtmptr, unmqr_trfact,
                        unmqr_workArr));
                }

                // copy matrix product into A
                const I copyblocks = ceildiv(n, BS2);
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                        dim3(BS2, BS2), 0, stream, n, n, // opts
                                        tmptau_W, 0, ldw, strideW, // W
                                        A, shiftA, lda, strideA); // A
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
        rocsolver_sterf_template<S>(handle, n, D, (I)0, strideD, E, (I)0, strideE, info,
                                    batch_count, (I*)work1);
    }
    else
    {
        // for performance reasons, we use stedc to compute eigenvalues even if the eigenvectors are ignored
        constexpr bool ISBATCHED = BATCHED || STRIDED;
        const I ldw = n;
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
            const I copyblocks = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW, A,
                                    shiftA, lda, strideA);
        }
    }

    return rocblas_status_success;
}

// Using optimized QR routines.
template <bool BATCHED, bool STRIDED, typename T, typename S, typename W, typename I>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const I n,
                                              W A,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              // info encodes O(n^2) submatrix row/col positions on
                                              // failure, so it is typed I (64-bit on the _64 path).
                                              I* info,
                                              const I batch_count,
                                              T* scalars,
                                              void* work1,
                                              void* work2,
                                              void* work3,
                                              void* work4,
                                              S* tmpz,
                                              I* splits,
                                              T* tmptau_W,
                                              T* tau,
                                              T** workArr,
                                              T* Aband,
                                              T* he2hb_work,
                                              T* V_hb2st,
                                              T* tau_hb2st,
                                              bool optim_mem)
{
    ROCSOLVER_ENTER("syevd_heevd", "evect:", evect, "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // size for constants in rocblas calls
    size_t size_scalars;
    {
        // memory workspace sizes:
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
            &size_workArr, &size_Aband, &size_he2hb_work, &size_V_hb2st, &size_tau_hb2st, &optim_mem);

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

    rocsolver_alg_mode hetrd_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_hetrd, &hetrd_mode));

    I blocksReset = (batch_count - 1) / BS1 + 1;
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

    // 2-stage path: he2hb + hb2st + unmtr_hb2st + unmqr
    // 2-stage BATCHED not currently working. Also update use_2stage in getMemorySize.
    const bool use_2stage = !BATCHED
        && (hetrd_mode == rocsolver_alg_mode_2stage
            || (hetrd_mode == rocsolver_alg_mode_auto && n >= SYEVD_2STAGE_SWITCHSIZE));
    if(use_2stage)
    {
        const I kd = SYEVD_2STAGE_KD;
        const I nb = SYEVD_2STAGE_NB;
        const I ldab = 3 * kd - 1;
        const I ldv_hb2st = 2 * kd - 1;
        const I nt = ceildiv(n - 1, kd);
        const I nv = kd * nt * (nt + 1) / 2;

        // Strides for band and V arrays, applies to both pointer batched and strided.
        const rocblas_stride strideAband = rocblas_stride(ldab * n);
        const rocblas_stride strideV_hb2st = rocblas_stride(ldv_hb2st * nv);
        const rocblas_stride strideTau_hb2st = rocblas_stride(nv);

        // Partition he2hb_work into sub-workspaces
        size_t size_D, size_V, size_W, size_X, size_Z, size_work, size_workArr_he2hb;
        rocsolver_sy2sb_he2hb_getMemorySize<BATCHED, T, I>(
            uplo, n, kd, nb, batch_count, &size_scalars, &size_D, &size_V, &size_W, &size_X,
            &size_Z, &size_work, &size_workArr_he2hb);
        T* he2hb_D = he2hb_work;
        T* he2hb_V = he2hb_D + size_D / sizeof(T);
        T* he2hb_W = he2hb_V + size_V / sizeof(T);
        T* he2hb_X = he2hb_W + size_W / sizeof(T);
        T* he2hb_Z = he2hb_X + size_X / sizeof(T);
        T* he2hb_work2 = he2hb_Z + size_Z / sizeof(T);
        T** he2hb_workArr = (T**)(he2hb_work2 + size_work / sizeof(T));

        // Stage 1: reduce dense Hermitian to band form (he2hb)
        // tau (size n) stores the he2hb Householder scalars; A stores the Householder vectors
        ROCBLAS_CHECK(rocsolver_sy2sb_he2hb_template<BATCHED, STRIDED, T, I>(
            handle, uplo, n, kd, nb, // opts
            A, shiftA, lda, strideA, // A
            Aband, ldab, strideAband, // Aband
            tau, n, // tau
            batch_count, scalars, he2hb_D, he2hb_V, he2hb_W, he2hb_X, he2hb_Z, he2hb_work2,
            he2hb_workArr));

        // Stage 2: reduce band to tridiagonal form (hb2st)
        // V_hb2st and tau_hb2st store the hb2st Householder data
        ROCBLAS_CHECK(rocsolver_sb2st_hb2st_template<BATCHED, STRIDED, T, I>(
            handle, rocblas_fill_lower, n, kd, // opts
            Aband, 0, ldab, strideAband, // Aband
            D, strideD, // D
            E, strideE, // E
            V_hb2st, ldv_hb2st, strideV_hb2st, // V
            tau_hb2st, strideTau_hb2st, batch_count));

        if(sterf_mode == rocsolver_alg_mode_hybrid && evect != rocblas_evect_original)
        {
            // only in hybrid mode, compute eigenvalues using sterf
            rocsolver_sterf_template<S>(handle, n, D, 0, strideD, E, 0, strideE, info, batch_count,
                                        (I*)work1);
        }
        else
        {
            // compute eigenvalues and eigenvectors of the tridiagonal (stedc)
            constexpr bool ISBATCHED = BATCHED || STRIDED;
            const I ldw = n;
            const rocblas_stride strideW = n * n;

            rocsolver_stedc_template<false, ISBATCHED, T>(
                handle, rocblas_evect_tridiagonal, n, // opts
                D, 0, strideD, // D
                E, 0, strideE, // E
                tmptau_W, 0, ldw, // W
                strideW, info, batch_count, work3, (S*)work2, (S*)work1, tmpz, splits, (S**)workArr);

            // update the eigenvectors (if applicable)
            if(evect == rocblas_evect_original)
            {
                // Partition he2hb_work for unmtr_hb2st sub-workspaces
                I max_parallel_2stage = 1;
                size_t size_Tr, size_W2, size_Z2, size_work2_unmtr, size_workArr2;
                rocsolver_ormtr_unmtr_hb2st_getMemorySize<BATCHED, T, I>(
                    rocblas_side_left, rocblas_operation_none, n, n, kd, batch_count,
                    &max_parallel_2stage, &size_scalars, &size_Tr, &size_W2, &size_Z2,
                    &size_work2_unmtr, &size_workArr2);
                T* unmtr_Tr = he2hb_work;
                T* unmtr_W = unmtr_Tr + size_Tr / sizeof(T);
                T* unmtr_Z = unmtr_W + size_W2 / sizeof(T);
                T* unmtr_work = unmtr_Z + size_Z2 / sizeof(T);
                T** unmtr_workArr = (T**)(unmtr_work + size_work2_unmtr / sizeof(T));

                // Back-transform stage 2: apply Q_hb2st to eigenvector matrix (unmtr_hb2st)
                // C = Q_hb2st * tmptau_W (tmptau_W holds the eigenvectors from stedc)
                ROCBLAS_CHECK(rocsolver_ormtr_unmtr_hb2st_template<BATCHED, STRIDED, T, T*>(
                    handle, rocblas_side_left, rocblas_operation_none, n, n, kd, // opts
                    V_hb2st, I(0), ldv_hb2st, strideV_hb2st, // V
                    tau_hb2st, strideTau_hb2st, // tau
                    tmptau_W, I(0), ldw, strideW, // W
                    batch_count, max_parallel_2stage, scalars, unmtr_Tr, unmtr_W, unmtr_Z,
                    unmtr_work, unmtr_workArr));

                // Back-transform stage 1: apply Q_he2hb to eigenvector matrix (unmqr)
                // Q_he2hb is stored in lower part of A (below diagonal kd) and in tau
                // Partition he2hb_work for unmqr sub-workspaces
                const I n_kd = std::max(n - kd, I(0));
                size_t size_AbyxORwork, size_diagORtmptr, size_trfact, size_workArr3;
                rocsolver_ormqr_unmqr_getMemorySize<BATCHED, T>(
                    rocblas_side_left, n_kd, n, n_kd, batch_count, &size_scalars, &size_AbyxORwork,
                    &size_diagORtmptr, &size_trfact, &size_workArr3);
                T* unmqr_AbyxORwork = he2hb_work;
                T* unmqr_diagORtmptr = unmqr_AbyxORwork + size_AbyxORwork / sizeof(T);
                T* unmqr_trfact = unmqr_diagORtmptr + size_diagORtmptr / sizeof(T);
                T** unmqr_workArr = (T**)(unmqr_trfact + size_trfact / sizeof(T));
                T** unmqr_workArr2 = unmqr_workArr + size_workArr3 / sizeof(T*);

                // Apply Q_he2hb on the left to W[kd:n, 0:n]:
                //   V is in A[kd:n, 0:n-kd], Q is (n-kd) x (n-kd)
                // When BATCHED=true, A is T* const* and tmptau_W is T*; use the adapter overload
                // which builds a pointer array for C in unmqr_workArr2 before dispatching.
                // When BATCHED=false, both are T* and we call the strided overload directly.
                if constexpr(BATCHED)
                {
                    rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n_kd, n, n_kd, A,
                        shiftA + idx2D(kd, 0, lda), lda, strideA, // A
                        tau, n, // tau
                        tmptau_W, idx2D(kd, 0, ldw), ldw, strideW, // W
                        batch_count, scalars, unmqr_AbyxORwork, unmqr_diagORtmptr, unmqr_trfact,
                        unmqr_workArr, unmqr_workArr2);
                }
                else
                {
                    ROCBLAS_CHECK(rocsolver_ormqr_unmqr_template<BATCHED, STRIDED, T>(
                        handle, rocblas_side_left, rocblas_operation_none, n_kd, n, n_kd, // opts
                        A, shiftA + idx2D(kd, 0, lda), lda, strideA, // A
                        tau, n, // tau
                        tmptau_W, idx2D(kd, 0, ldw), ldw, strideW, // W
                        batch_count, scalars, unmqr_AbyxORwork, unmqr_diagORtmptr, unmqr_trfact,
                        unmqr_workArr));
                }

                // copy matrix product into A
                const I copyblocks = ceildiv(n, BS2);
                ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                        dim3(BS2, BS2), 0, stream, n, n, // opts
                                        tmptau_W, 0, ldw, strideW, // W
                                        A, shiftA, lda, strideA); // A
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
        rocsolver_sterf_template<S>(handle, n, D, (I)0, strideD, E, (I)0, strideE, info,
                                    batch_count, (I*)work1);
    }
    else
    {
        // for performance reasons, we use stedc to compute eigenvalues even if the eigenvectors are ignored
        constexpr bool ISBATCHED = BATCHED || STRIDED;
        const I ldw = n;
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
            const I copyblocks = (n - 1) / BS2 + 1;
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(copyblocks, copyblocks, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, tmptau_W, 0, ldw, strideW, A,
                                    shiftA, lda, strideA);
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
