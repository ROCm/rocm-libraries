/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_sytrs2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*

===============================================================
sytrs2 is not intended for inclusion in the public API.
It exists to provide an alternative, sometimes faster, implementation of sytrs.
==============================================================
*/

template <typename T, typename I>
rocblas_status rocsolver_sytrs2_strided_batched_impl(rocblas_handle handle,
                                                     const rocblas_fill uplo,
                                                     const I n,
                                                     const I nrhs,
                                                     T* A,
                                                     const I lda,
                                                     const rocblas_stride strideA,
                                                     I* ipiv,
                                                     const rocblas_stride strideP,
                                                     T* B,
                                                     const I ldb,
                                                     const rocblas_stride strideB,
                                                     const I batch_count)
{
    ROCSOLVER_ENTER_TOP("sytrs2_strided_batched", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda",
                        lda, "--strideA", strideA, "--strideP", strideP, "--ldb", ldb, "--strideB",
                        strideB, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sytrs2_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B, ipiv, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftB = 0;

    // ----------------------
    // memory workspace sizes:
    // ----------------------
    size_t size_work = 0;
    ROCBLAS_CHECK(
        rocsolver_sytrs2_getMemorySize<T>(handle, n, nrhs, batch_count, lda, ldb, &size_work));

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // ---------------------------
    // memory workspace allocation
    // ---------------------------
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    void* const work = static_cast<void*>(mem[0]);

    // execution
    return rocsolver_sytrs2_template<T>(handle, uplo, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP,
                                        B, shiftB, ldb, strideB, batch_count, work, size_work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrs2_strided_batched(rocblas_handle handle,
                                                                  const rocblas_fill uplo,
                                                                  const rocblas_int n,
                                                                  const rocblas_int nrhs,
                                                                  float* A,
                                                                  const rocblas_int lda,
                                                                  const rocblas_stride strideA,
                                                                  rocblas_int* ipiv,
                                                                  const rocblas_stride strideP,
                                                                  float* B,
                                                                  const rocblas_int ldb,
                                                                  const rocblas_stride strideB,
                                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<float>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrs2_strided_batched(rocblas_handle handle,
                                                                  const rocblas_fill uplo,
                                                                  const rocblas_int n,
                                                                  const rocblas_int nrhs,
                                                                  double* A,
                                                                  const rocblas_int lda,
                                                                  const rocblas_stride strideA,
                                                                  rocblas_int* ipiv,
                                                                  const rocblas_stride strideP,
                                                                  double* B,
                                                                  const rocblas_int ldb,
                                                                  const rocblas_stride strideB,
                                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<double>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_csytrs2_strided_batched(rocblas_handle handle,
                                                                  const rocblas_fill uplo,
                                                                  const rocblas_int n,
                                                                  const rocblas_int nrhs,
                                                                  rocblas_float_complex* A,
                                                                  const rocblas_int lda,
                                                                  const rocblas_stride strideA,
                                                                  rocblas_int* ipiv,
                                                                  const rocblas_stride strideP,
                                                                  rocblas_float_complex* B,
                                                                  const rocblas_int ldb,
                                                                  const rocblas_stride strideB,
                                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<rocblas_float_complex>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytrs2_strided_batched(rocblas_handle handle,
                                                                  const rocblas_fill uplo,
                                                                  const rocblas_int n,
                                                                  const rocblas_int nrhs,
                                                                  rocblas_double_complex* A,
                                                                  const rocblas_int lda,
                                                                  const rocblas_stride strideA,
                                                                  rocblas_int* ipiv,
                                                                  const rocblas_stride strideP,
                                                                  rocblas_double_complex* B,
                                                                  const rocblas_int ldb,
                                                                  const rocblas_stride strideB,
                                                                  const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<rocblas_double_complex>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ssytrs2_strided_batched_64(rocblas_handle handle,
                                                                     const rocblas_fill uplo,
                                                                     const int64_t n,
                                                                     const int64_t nrhs,
                                                                     float* A,
                                                                     const int64_t lda,
                                                                     const rocblas_stride strideA,
                                                                     int64_t* ipiv,
                                                                     const rocblas_stride strideP,
                                                                     float* B,
                                                                     const int64_t ldb,
                                                                     const rocblas_stride strideB,
                                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<float, int64_t>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsytrs2_strided_batched_64(rocblas_handle handle,
                                                                     const rocblas_fill uplo,
                                                                     const int64_t n,
                                                                     const int64_t nrhs,
                                                                     double* A,
                                                                     const int64_t lda,
                                                                     const rocblas_stride strideA,
                                                                     int64_t* ipiv,
                                                                     const rocblas_stride strideP,
                                                                     double* B,
                                                                     const int64_t ldb,
                                                                     const rocblas_stride strideB,
                                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<double, int64_t>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_csytrs2_strided_batched_64(rocblas_handle handle,
                                                                     const rocblas_fill uplo,
                                                                     const int64_t n,
                                                                     const int64_t nrhs,
                                                                     rocblas_float_complex* A,
                                                                     const int64_t lda,
                                                                     const rocblas_stride strideA,
                                                                     int64_t* ipiv,
                                                                     const rocblas_stride strideP,
                                                                     rocblas_float_complex* B,
                                                                     const int64_t ldb,
                                                                     const rocblas_stride strideB,
                                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<rocblas_float_complex, int64_t>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zsytrs2_strided_batched_64(rocblas_handle handle,
                                                                     const rocblas_fill uplo,
                                                                     const int64_t n,
                                                                     const int64_t nrhs,
                                                                     rocblas_double_complex* A,
                                                                     const int64_t lda,
                                                                     const rocblas_stride strideA,
                                                                     int64_t* ipiv,
                                                                     const rocblas_stride strideP,
                                                                     rocblas_double_complex* B,
                                                                     const int64_t ldb,
                                                                     const rocblas_stride strideB,
                                                                     const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs2_strided_batched_impl<rocblas_double_complex, int64_t>(
        handle, uplo, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
