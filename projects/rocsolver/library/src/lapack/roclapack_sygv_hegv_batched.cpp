/* **************************************************************************
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

#include "roclapack_sygv_hegv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygv_hegv_batched_impl(rocblas_handle handle,
                                                const rocblas_eform itype,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                U A,
                                                const rocblas_int lda,
                                                U B,
                                                const rocblas_int ldb,
                                                S* D,
                                                const rocblas_stride strideD,
                                                S* E,
                                                const rocblas_stride strideE,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygv_batched" : "hegv_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb, "--strideD", strideD, "--strideE", strideE,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sygv_hegv_argCheck(handle, itype, evect, uplo, n, lda, ldb, A, B,
                                                     D, E, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (and for calling TRSM, SYGST/HEGST, and SYEV/HEEV)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling POTRF and SYEV/HEEV
    size_t size_pivots_workArr;
    // size of temporary info array
    size_t size_iinfo;
    rocsolver_sygv_hegv_getMemorySize<true, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_pivots_workArr, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivots_workArr,
                                                      size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivots_workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivots_workArr, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivots_workArr = mem[5];
    iinfo = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygv_hegv_template<true, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, D, strideD,
        E, strideE, info, batch_count, (T*)scalars, work1, work2, work3, work4, pivots_workArr,
        (rocblas_int*)iinfo, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygv_batched(rocblas_handle handle,
                                       const rocblas_eform itype,
                                       const rocblas_evect evect,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       float* const A[],
                                       const rocblas_int lda,
                                       float* const B[],
                                       const rocblas_int ldb,
                                       float* D,
                                       const rocblas_stride strideD,
                                       float* E,
                                       const rocblas_stride strideE,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygv_hegv_batched_impl<float>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_dsygv_batched(rocblas_handle handle,
                                       const rocblas_eform itype,
                                       const rocblas_evect evect,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       double* const A[],
                                       const rocblas_int lda,
                                       double* const B[],
                                       const rocblas_int ldb,
                                       double* D,
                                       const rocblas_stride strideD,
                                       double* E,
                                       const rocblas_stride strideE,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygv_hegv_batched_impl<double>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_chegv_batched(rocblas_handle handle,
                                       const rocblas_eform itype,
                                       const rocblas_evect evect,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       rocblas_float_complex* const A[],
                                       const rocblas_int lda,
                                       rocblas_float_complex* const B[],
                                       const rocblas_int ldb,
                                       float* D,
                                       const rocblas_stride strideD,
                                       float* E,
                                       const rocblas_stride strideE,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygv_hegv_batched_impl<rocblas_float_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_zhegv_batched(rocblas_handle handle,
                                       const rocblas_eform itype,
                                       const rocblas_evect evect,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       rocblas_double_complex* const A[],
                                       const rocblas_int lda,
                                       rocblas_double_complex* const B[],
                                       const rocblas_int ldb,
                                       double* D,
                                       const rocblas_stride strideD,
                                       double* E,
                                       const rocblas_stride strideE,
                                       rocblas_int* info,
                                       const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygv_hegv_batched_impl<rocblas_double_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, strideD, E, strideE, info, batch_count);
}

} // extern C
