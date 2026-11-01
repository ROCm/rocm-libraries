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

#include "rocauxiliary_lacpy.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_lacpy_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const I m,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    T* B,
                                    const I ldb)
try
{
    ROCSOLVER_ENTER_TOP("lacpy", "--uplo", uplo, "-m", m, "-n", n, "--lda", lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lacpy_argCheck(handle, uplo, m, n, lda, ldb, A, B);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    I batch_count = 1;

    // no workspace is needed
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_lacpy_template<T>(handle, uplo, m, n, A, shiftA, lda, strideA, B, shiftB, ldb,
                                       strideB, batch_count);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slacpy(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_lacpy_impl<float, rocblas_int>(handle, uplo, m, n, A, lda, B, ldb);
}

rocblas_status rocsolver_dlacpy(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_lacpy_impl<double, rocblas_int>(handle, uplo, m, n, A, lda, B, ldb);
}

rocblas_status rocsolver_clacpy(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_lacpy_impl<rocblas_float_complex, rocblas_int>(handle, uplo, m, n,
                                                                               A, lda, B, ldb);
}

rocblas_status rocsolver_zlacpy(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_lacpy_impl<rocblas_double_complex, rocblas_int>(handle, uplo, m, n,
                                                                                A, lda, B, ldb);
}

rocblas_status rocsolver_slacpy_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lacpy_impl<float, int64_t>(handle, uplo, m, n, A, lda, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dlacpy_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   double* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lacpy_impl<double, int64_t>(handle, uplo, m, n, A, lda, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_clacpy_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   rocblas_float_complex* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lacpy_impl<rocblas_float_complex, int64_t>(handle, uplo, m, n, A,
                                                                           lda, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zlacpy_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   rocblas_double_complex* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lacpy_impl<rocblas_double_complex, int64_t>(handle, uplo, m, n, A,
                                                                            lda, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
