/************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "roclapack_potrf.hpp"
#include "roclapack_syevdx_heevdx.hpp"
#include "roclapack_sygst_hegst.hpp"
#include "roclapack_sygvx_hegvx.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S>
rocblas_status rocsolver_sygvdx_hegvdx_argCheck(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                T A,
                                                const rocblas_int lda,
                                                T B,
                                                const rocblas_int ldb,
                                                const S vl,
                                                const S vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                rocblas_int* nev,
                                                S* W,
                                                T Z,
                                                const rocblas_int ldz,
                                                rocblas_int* info,
                                                const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(itype != rocblas_eform_ax && itype != rocblas_eform_abx && itype != rocblas_eform_bax)
        return rocblas_status_invalid_value;
    if(evect != rocblas_evect_none && evect != rocblas_evect_original)
        return rocblas_status_invalid_value;
    if(erange != rocblas_erange_all && erange != rocblas_erange_value
       && erange != rocblas_erange_index)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || ldb < n || (evect != rocblas_evect_none && ldz < n) || batch_count < 0)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_value && vl >= vu)
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (il < 1 || iu < 0))
        return rocblas_status_invalid_size;
    if(erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu)))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !B) || (n && !W) || (batch_count && !nev) || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if(evect != rocblas_evect_none && n && !Z)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S>
void rocsolver_sygvdx_hegvdx_getMemorySize(const rocblas_eform itype,
                                           const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           size_t* size_scalars,
                                           size_t* size_work1,
                                           size_t* size_work2,
                                           size_t* size_work3,
                                           size_t* size_work4,
                                           size_t* size_work5,
                                           size_t* size_work6_ifail,
                                           size_t* size_D,
                                           size_t* size_E,
                                           size_t* size_iblock,
                                           size_t* size_isplit,
                                           size_t* size_tau,
                                           size_t* size_work7_workArr,
                                           size_t* size_iinfo,
                                           bool* optim_mem)
{
    // if quick return no need of workspace
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_work5 = 0;
        *size_work6_ifail = 0;
        *size_D = 0;
        *size_E = 0;
        *size_iblock = 0;
        *size_isplit = 0;
        *size_tau = 0;
        *size_work7_workArr = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    bool opt1, opt2, opt3 = true;
    size_t unused, temp1, temp2, temp3, temp4, temp5;

    // requirements for calling POTRF
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(n, uplo, batch_count, size_scalars,
                                                       size_work1, size_work2, size_work3, size_work4,
                                                       size_work7_workArr, size_iinfo, &opt1);
    *size_iinfo = std::max(*size_iinfo, sizeof(rocblas_int) * batch_count);

    // requirements for calling SYGST/HEGST
    rocsolver_sygst_hegst_getMemorySize<BATCHED, STRIDED, T>(uplo, itype, n, batch_count, &unused,
                                                             &temp1, &temp2, &temp3, &temp4, &opt2);
    *size_work1 = std::max(*size_work1, temp1);
    *size_work2 = std::max(*size_work2, temp2);
    *size_work3 = std::max(*size_work3, temp3);
    *size_work4 = std::max(*size_work4, temp4);

    // requirements for calling SYEVDX/HEEVDX
    rocsolver_syevdx_heevdx_getMemorySize<BATCHED, T, S>(
        evect, uplo, n, batch_count, &unused, &temp1, &temp2, &temp3, &temp4, size_work5,
        size_work6_ifail, size_D, size_E, size_iblock, size_isplit, size_tau, &temp5);
    *size_work1 = std::max(*size_work1, temp1);
    *size_work2 = std::max(*size_work2, temp2);
    *size_work3 = std::max(*size_work3, temp3);
    *size_work4 = std::max(*size_work4, temp4);
    *size_work7_workArr = std::max(*size_work7_workArr, temp5);

    if(evect == rocblas_evect_original)
    {
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            rocblas_operation trans
                = (uplo == rocblas_fill_upper ? rocblas_operation_none
                                              : rocblas_operation_conjugate_transpose);
            // requirements for calling TRSM
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, n, batch_count,
                                                    &temp1, &temp2, &temp3, &temp4, &opt3);
            *size_work1 = std::max(*size_work1, temp1);
            *size_work2 = std::max(*size_work2, temp2);
            *size_work3 = std::max(*size_work3, temp3);
            *size_work4 = std::max(*size_work4, temp4);
        }
    }

    *optim_mem = opt1 && opt2 && opt3;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sygvdx_hegvdx_template(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                U A,
                                                const rocblas_stride shiftA,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                U B,
                                                const rocblas_stride shiftB,
                                                const rocblas_int ldb,
                                                const rocblas_stride strideB,
                                                const S vl,
                                                const S vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                rocblas_int* nev,
                                                S* W,
                                                const rocblas_stride strideW,
                                                U Z,
                                                const rocblas_stride shiftZ,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* info,
                                                const rocblas_int batch_count,
                                                T* scalars,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                void* work5,
                                                rocblas_int* work6_ifail,
                                                S* D,
                                                S* E,
                                                rocblas_int* iblock,
                                                rocblas_int* isplit,
                                                T* tau,
                                                void* work7_workArr,
                                                rocblas_int* iinfo,
                                                bool optim_mem)
{
    ROCSOLVER_ENTER("sygvdx_hegvdx", "itype:", itype, "evect:", evect, "erange:", erange,
                    "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftB:", shiftB,
                    "ldb:", ldb, "vl:", vl, "vu:", vu, "il:", il, "iu:", iu, "shiftZ:", shiftZ,
                    "ldz:", ldz, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info=0 (starting with no errors)
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return with n = 0
    if(n == 0)
    {
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, nev, batch_count, 0);
        return rocblas_status_success;
    }

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // constants for rocblas functions calls
    T one = 1;

    // perform Cholesky factorization of B
    rocsolver_potrf_template<BATCHED, STRIDED, T, rocblas_int, rocblas_int, S>(
        handle, uplo, n, B, shiftB, ldb, strideB, info, batch_count, scalars, work1, work2, work3,
        work4, (T*)work7_workArr, iinfo, optim_mem);

    /** (TODO: Strictly speaking, computations should stop here if B is not positive definite.
        A should not be modified in this case as no eigenvalues or eigenvectors can be computed.
        Need to find a way to do this efficiently; for now A will be destroyed in the non
        positive-definite case) **/

    // reduce to standard eigenvalue problem and solve
    rocsolver_sygst_hegst_template<BATCHED, STRIDED, T, S>(
        handle, itype, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, batch_count,
        scalars, work1, work2, work3, work4, optim_mem);

    rocsolver_syevdx_heevdx_template<BATCHED, STRIDED, T>(
        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, nev, W, strideW, Z,
        shiftZ, ldz, strideZ, iinfo, batch_count, scalars, work1, work2, work3, work4, work5,
        work6_ifail, D, E, iblock, isplit, tau, (T**)work7_workArr);

    // combine info from POTRF with info from SYEVDX/HEEVDX
    ROCSOLVER_LAUNCH_KERNEL(sygvx_update_info, gridReset, threads, 0, stream, info, iinfo, nev, n,
                            batch_count);

    /** (TODO: Similarly, if only h_nev < n eigenvalues were returned, TRSM or TRMM below should not
            work with the entire matrix. Need to find a way to do this efficiently; for now we ignore
            nev and set h_nev = n) **/

    // backtransform eigenvectors
    if(evect == rocblas_evect_original)
    {
        rocblas_int h_nev = (erange == rocblas_erange_index ? iu - il + 1 : n);
        if(itype == rocblas_eform_ax || itype == rocblas_eform_abx)
        {
            if(uplo == rocblas_fill_upper)
                rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_non_unit, n,
                    h_nev, B, shiftB, ldb, strideB, Z, shiftZ, ldz, strideZ, batch_count, optim_mem,
                    work1, work2, work3, work4);
            else
                rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                    handle, rocblas_side_left, rocblas_operation_conjugate_transpose,
                    rocblas_diagonal_non_unit, n, h_nev, B, shiftB, ldb, strideB, Z, shiftZ, ldz,
                    strideZ, batch_count, optim_mem, work1, work2, work3, work4);
        }
        else
        {
            rocblas_operation trans
                = (uplo == rocblas_fill_upper ? rocblas_operation_conjugate_transpose
                                              : rocblas_operation_none);
            rocblasCall_trmm(handle, rocblas_side_left, uplo, trans, rocblas_diagonal_non_unit, n,
                             h_nev, &one, 0, B, shiftB, ldb, strideB, Z, shiftZ, ldz, strideZ,
                             batch_count, (T**)work7_workArr);
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
