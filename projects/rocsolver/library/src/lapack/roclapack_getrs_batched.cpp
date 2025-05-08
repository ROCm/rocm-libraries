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

#include "roclapack_getrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_getrs_batched_impl(rocblas_handle handle,
                                            const rocblas_operation trans,
                                            const I n,
                                            const I nrhs,
                                            U A,
                                            const I lda,
                                            const I* ipiv,
                                            const rocblas_stride strideP,
                                            U B,
                                            const I ldb,
                                            const I batch_count)
{
    ROCSOLVER_ENTER_TOP("getrs_batched", "--trans", trans, "-n", n, "--nrhs", nrhs, "--lda", lda,
                        "--strideP", strideP, "--ldb", ldb, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_getrs_argCheck(handle, trans, n, nrhs, lda, ldb, A, B, ipiv, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftB = 0;

    // batched execution
    I inca = 1;
    I incb = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;

    // memory workspace sizes:
    // size of workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_getrs_getMemorySize<true, false, T>(trans, n, nrhs, batch_count, &size_work1,
                                                  &size_work2, &size_work3, &size_work4, &optim_mem,
                                                  lda, ldb);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];

    // execution
    return rocsolver_getrs_template<true, false, T>(
        handle, trans, n, nrhs, A, shiftA, inca, lda, strideA, ipiv, strideP, B, shiftB, incb, ldb,
        strideB, batch_count, work1, work2, work3, work4, optim_mem, true);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetrs_batched(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        float* const A[],
                                        const rocblas_int lda,
                                        const rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        float* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrs_batched_impl<float>(handle, trans, n, nrhs, A, lda, ipiv,
                                                          strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_dgetrs_batched(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        double* const A[],
                                        const rocblas_int lda,
                                        const rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        double* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrs_batched_impl<double>(handle, trans, n, nrhs, A, lda, ipiv,
                                                           strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_cgetrs_batched(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        const rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_float_complex* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrs_batched_impl<rocblas_float_complex>(
        handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_zgetrs_batched(rocblas_handle handle,
                                        const rocblas_operation trans,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        const rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_double_complex* const B[],
                                        const rocblas_int ldb,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getrs_batched_impl<rocblas_double_complex>(
        handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_sgetrs_batched_64(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const int64_t n,
                                           const int64_t nrhs,
                                           float* const A[],
                                           const int64_t lda,
                                           const int64_t* ipiv,
                                           const rocblas_stride strideP,
                                           float* const B[],
                                           const int64_t ldb,
                                           const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_batched_impl<float>(handle, trans, n, nrhs, A, lda, ipiv,
                                                          strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgetrs_batched_64(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const int64_t n,
                                           const int64_t nrhs,
                                           double* const A[],
                                           const int64_t lda,
                                           const int64_t* ipiv,
                                           const rocblas_stride strideP,
                                           double* const B[],
                                           const int64_t ldb,
                                           const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_batched_impl<double>(handle, trans, n, nrhs, A, lda, ipiv,
                                                           strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgetrs_batched_64(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const int64_t n,
                                           const int64_t nrhs,
                                           rocblas_float_complex* const A[],
                                           const int64_t lda,
                                           const int64_t* ipiv,
                                           const rocblas_stride strideP,
                                           rocblas_float_complex* const B[],
                                           const int64_t ldb,
                                           const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_batched_impl<rocblas_float_complex>(
        handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgetrs_batched_64(rocblas_handle handle,
                                           const rocblas_operation trans,
                                           const int64_t n,
                                           const int64_t nrhs,
                                           rocblas_double_complex* const A[],
                                           const int64_t lda,
                                           const int64_t* ipiv,
                                           const rocblas_stride strideP,
                                           rocblas_double_complex* const B[],
                                           const int64_t ldb,
                                           const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_batched_impl<rocblas_double_complex>(
        handle, trans, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
