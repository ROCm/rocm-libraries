/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_sygvdj_hegvdj.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvdj_hegvdj_impl(rocblas_handle handle,
                                            const rocblas_eform itype,
                                            const rocblas_evect evect,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int lda,
                                            U B,
                                            const rocblas_int ldb,
                                            S* D,
                                            rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvdj" : "hegvdj");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygvdj_hegvdj_argCheck(handle, itype, evect, uplo, n, lda, ldb, A, B, D, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideD = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling SYEVDJ/HEEVDJ
    size_t size_workE;
    size_t size_workTau;
    size_t size_workVec;
    size_t size_workSplits;
    size_t size_workArr;
    // size of temporary info array
    size_t size_iinfo;

    rocsolver_sygvdj_hegvdj_getMemorySize<false, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_workE, &size_workTau, &size_workVec, &size_workSplits, &size_iinfo,
        &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3, size_work4, size_workE,
            size_workTau, size_workVec, size_workSplits, size_iinfo, size_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *workE, *workTau, *workVec, *workSplits,
        *workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_workE, size_workTau, size_workVec, size_workSplits, size_iinfo,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    workE = mem[5];
    workTau = mem[6];
    workVec = mem[7];
    workSplits = mem[8];
    iinfo = mem[9];
    workArr = mem[10];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvdj_hegvdj_template<false, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, D, strideD,
        info, batch_count, (T*)scalars, work1, work2, work3, work4, (S*)workE, (T*)workTau,
        (T*)workVec, (rocblas_int*)workSplits, (rocblas_int*)iinfo, workArr, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvdj(rocblas_handle handle,
                                 const rocblas_eform itype,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 float* A,
                                 const rocblas_int lda,
                                 float* B,
                                 const rocblas_int ldb,
                                 float* D,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_sygvdj_hegvdj_impl<float>(handle, itype, evect, uplo, n, A, lda, B,
                                                          ldb, D, info);
}

rocblas_status rocsolver_dsygvdj(rocblas_handle handle,
                                 const rocblas_eform itype,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 double* A,
                                 const rocblas_int lda,
                                 double* B,
                                 const rocblas_int ldb,
                                 double* D,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_sygvdj_hegvdj_impl<double>(handle, itype, evect, uplo, n, A, lda, B,
                                                           ldb, D, info);
}

rocblas_status rocsolver_chegvdj(rocblas_handle handle,
                                 const rocblas_eform itype,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 rocblas_float_complex* A,
                                 const rocblas_int lda,
                                 rocblas_float_complex* B,
                                 const rocblas_int ldb,
                                 float* D,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_sygvdj_hegvdj_impl<rocblas_float_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);
}

rocblas_status rocsolver_zhegvdj(rocblas_handle handle,
                                 const rocblas_eform itype,
                                 const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int n,
                                 rocblas_double_complex* A,
                                 const rocblas_int lda,
                                 rocblas_double_complex* B,
                                 const rocblas_int ldb,
                                 double* D,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_sygvdj_hegvdj_impl<rocblas_double_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, D, info);
}

} // extern C
