/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_laswp.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"
#include "rocsolver_workspace_helper.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_getrs_argCheck(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const I n,
                                        const I nrhs,
                                        const I lda,
                                        const I ldb,
                                        T A,
                                        T B,
                                        const I* ipiv,
                                        const bool pivot,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nrhs < 0 || lda < n || ldb < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && batch_count && !A) || (nrhs && n && batch_count && !B))
    {
        return rocblas_status_invalid_pointer;
    }

    if(pivot)
    {
        if(n && batch_count && !ipiv)
        {
            return rocblas_status_invalid_pointer;
        }
    }

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
void rocsolver_getrs_getMemorySize(rocblas_handle handle,
                                   const rocblas_operation trans,
                                   const I n,
                                   const I nrhs,
                                   U A,
                                   const rocblas_stride shiftA,
                                   const I inca,
                                   const I lda,
                                   const rocblas_stride strideA,
                                   const I* ipiv,
                                   const rocblas_stride strideP,
                                   U B,
                                   const rocblas_stride shiftB,
                                   const I incb,
                                   const I ldb,
                                   const rocblas_stride strideB,
                                   const I batch_count,
                                   rocsolver_workspace_helper* work_helper,
                                   const bool pivot)
{
    // if quick return, no workspace is needed
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return;

    // workspace required for calling TRSM
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, trans, n, nrhs, batch_count,
                                            &size_work1, &size_work2, &size_work3, &size_work4,
                                            &optim_mem, lda, ldb, inca, incb);

    work_helper->set_optim_mem(optim_mem);
    work_helper->assign_sizes(
        {},
        {{"work1", size_work1}, {"work2", size_work2}, {"work3", size_work3}, {"work4", size_work4}});
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_getrs_template(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const I n,
                                        const I nrhs,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I inca,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const I* ipiv,
                                        const rocblas_stride strideP,
                                        U B,
                                        const rocblas_stride shiftB,
                                        const I incb,
                                        const I ldb,
                                        const rocblas_stride strideB,
                                        const I batch_count,
                                        rocsolver_workspace_helper* work_helper,
                                        const bool pivot)
{
    ROCSOLVER_ENTER((pivot) ? "getrs" : "getrs_npvt", "trans:", trans, "n:", n, "nrhs:", nrhs,
                    "shiftA:", shiftA, "inca:", inca, "lda:", lda, "shiftB:", shiftB, "incb:", incb,
                    "ldb:", ldb, "bc:", batch_count);

    // quick return
    if(n == 0 || nrhs == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // prepare workspace
    const bool optim_mem = work_helper->get_optim_mem();
    void* work1 = (void*)(*work_helper)["work1"];
    void* work2 = (void*)(*work_helper)["work2"];
    void* work3 = (void*)(*work_helper)["work3"];
    void* work4 = (void*)(*work_helper)["work4"];

    if(trans == rocblas_operation_none)
    {
        // first apply row interchanges to the right hand sides
        if(pivot)
        {
            rocblas_status const istat = rocsolver_laswp_template<T, I>(
                handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n, ipiv, 0, 1, strideP, batch_count);
            if(istat != rocblas_status_success)
            {
                rocblas_set_pointer_mode(handle, old_mode);
                return (istat);
            }
        }

        // solve L*X = B, overwriting B with X
        {
            rocblas_status const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, trans, rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                lda, strideA, B, shiftB, incb, ldb, strideB, batch_count, optim_mem, work1, work2,
                work3, work4);
            if(istat != rocblas_status_success)
            {
                rocblas_set_pointer_mode(handle, old_mode);
                return (istat);
            }
        }

        // solve U*X = B, overwriting B with X
        {
            rocblas_status const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, trans, rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                inca, lda, strideA, B, shiftB, incb, ldb, strideB, batch_count, optim_mem, work1,
                work2, work3, work4);
            if(istat != rocblas_status_success)
            {
                rocblas_set_pointer_mode(handle, old_mode);
                return (istat);
            }
        }
    }
    else
    {
        // solve U'*X = B or U**H *X = B, overwriting B with X
        {
            rocblas_status const istat = rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, trans, rocblas_diagonal_non_unit, n, nrhs, A, shiftA,
                inca, lda, strideA, B, shiftB, incb, ldb, strideB, batch_count, optim_mem, work1,
                work2, work3, work4);
            if(istat != rocblas_status_success)
            {
                rocblas_set_pointer_mode(handle, old_mode);
                return (istat);
            }
        }

        // solve L'*X = B, or L**H *X = B overwriting B with X
        {
            rocblas_status const istat = rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, trans, rocblas_diagonal_unit, n, nrhs, A, shiftA, inca,
                lda, strideA, B, shiftB, incb, ldb, strideB, batch_count, optim_mem, work1, work2,
                work3, work4);
            if(istat != rocblas_status_success)
            {
                rocblas_set_pointer_mode(handle, old_mode);
                return (istat);
            }
        }

        // then apply row interchanges to the solution vectors
        if(pivot)
        {
            rocblas_status const istat
                = rocsolver_laswp_template<T, I>(handle, nrhs, B, shiftB, incb, ldb, strideB, 1, n,
                                                 ipiv, 0, -1, strideP, batch_count);

            if(istat != rocblas_status_success)
            {
                rocblas_set_pointer_mode(handle, old_mode);
                return (istat);
            }
        }
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
