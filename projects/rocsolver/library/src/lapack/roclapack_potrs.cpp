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

#include "roclapack_potrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_potrs_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const I n,
                                    const I nrhs,
                                    T* A,
                                    const I lda,
                                    T* B,
                                    const I ldb)
{
    ROCSOLVER_ENTER_TOP("potrs", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_potrs_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size of workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_potrs_getMemorySize<false, false, T>(n, nrhs, batch_count, &size_work1, &size_work2,
                                                   &size_work3, &size_work4, &optim_mem);

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
    return rocsolver_potrs_template<false, false, T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA,
                                                     B, shiftB, ldb, strideB, batch_count, work1,
                                                     work2, work3, work4, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {
rocblas_status rocsolver_spotrs(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                float* A,
                                const rocblas_int lda,
                                float* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_potrs_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb);
}

rocblas_status rocsolver_dpotrs(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                double* A,
                                const rocblas_int lda,
                                double* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_potrs_impl<double>(handle, uplo, n, nrhs, A, lda, B, ldb);
}

rocblas_status rocsolver_cpotrs(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_potrs_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                                  ldb);
}

rocblas_status rocsolver_zpotrs(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_potrs_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                                   ldb);
}

rocblas_status rocsolver_spotrs_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   float* A,
                                   const int64_t lda,
                                   float* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrs_impl<float>(handle, uplo, n, nrhs, A, lda, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dpotrs_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   double* A,
                                   const int64_t lda,
                                   double* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrs_impl<double>(handle, uplo, n, nrhs, A, lda, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cpotrs_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   rocblas_float_complex* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrs_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                                  ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zpotrs_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   rocblas_double_complex* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_potrs_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda, B,
                                                                   ldb);
#else
    return rocblas_status_not_implemented;
#endif
}
}
