/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_potri.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_potri_batched_impl(rocblas_handle handle,
                                            const rocblas_fill uplo,
                                            const rocblas_int n,
                                            U A,
                                            const rocblas_int lda,
                                            rocblas_int* info,
                                            const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("potri_batched", "--uplo", uplo, "-n", n, "--lda", lda, "--batch_count",
                        batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_potri_argCheck(handle, uplo, n, lda, A, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size of workspace (for calling TRTRI)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4, size_tmpcopy;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_potri_getMemorySize<true, false, T>(n, batch_count, &size_work1, &size_work2,
                                                  &size_work3, &size_work4, &size_tmpcopy,
                                                  &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4, size_tmpcopy, size_workArr);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4, *tmpcopy, *workArr;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4, size_tmpcopy,
                              size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];
    tmpcopy = mem[4];
    workArr = mem[5];

    // execution
    return rocsolver_potri_template<true, false, T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                                                    batch_count, work1, work2, work3, work4,
                                                    (T*)tmpcopy, (T**)workArr, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_spotri_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_potri_batched_impl<float>(handle, uplo, n, A, lda, info, batch_count);
}

rocblas_status rocsolver_dpotri_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_potri_batched_impl<double>(handle, uplo, n, A, lda, info,
                                                           batch_count);
}

rocblas_status rocsolver_cpotri_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_potri_batched_impl<rocblas_float_complex>(handle, uplo, n, A, lda,
                                                                          info, batch_count);
}

rocblas_status rocsolver_zpotri_batched(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_potri_batched_impl<rocblas_double_complex>(handle, uplo, n, A, lda,
                                                                           info, batch_count);
}
}
