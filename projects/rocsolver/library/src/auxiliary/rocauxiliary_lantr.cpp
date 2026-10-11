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

#include "rocauxiliary_lantr.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
rocblas_status rocsolver_lantr_impl(rocblas_handle handle,
                                    const rocsolver_norm_type norm_type,
                                    const rocblas_fill uplo,
                                    const rocblas_diagonal diag,
                                    const I m,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    S* norm)
try
{
    ROCSOLVER_ENTER_TOP("lantr", "--norm_type", norm_type, "--uplo", uplo, "--diag", diag, "-m", m,
                        "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lantr_argCheck(handle, norm_type, uplo, diag, m, n, lda, A, norm);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size of workspace for the partial sums, maxima or sums of squares
    size_t size_work;
    rocsolver_lantr_getMemorySize<T, I, S>(norm_type, m, n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_lantr_template<T, I, S>(handle, norm_type, uplo, diag, m, n, A, shiftA, lda,
                                             strideA, batch_count, norm, (S*)work);
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

rocblas_status rocsolver_slantr(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* norm)
{
    return rocsolver::rocsolver_lantr_impl<float, rocblas_int, float>(handle, norm_type, uplo, diag,
                                                                      m, n, A, lda, norm);
}

rocblas_status rocsolver_dlantr(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* norm)
{
    return rocsolver::rocsolver_lantr_impl<double, rocblas_int, double>(handle, norm_type, uplo,
                                                                        diag, m, n, A, lda, norm);
}

rocblas_status rocsolver_clantr(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* norm)
{
    return rocsolver::rocsolver_lantr_impl<rocblas_float_complex, rocblas_int, float>(
        handle, norm_type, uplo, diag, m, n, A, lda, norm);
}

rocblas_status rocsolver_zlantr(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* norm)
{
    return rocsolver::rocsolver_lantr_impl<rocblas_double_complex, rocblas_int, double>(
        handle, norm_type, uplo, diag, m, n, A, lda, norm);
}

rocblas_status rocsolver_slantr_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* norm)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lantr_impl<float, int64_t, float>(handle, norm_type, uplo, diag, m,
                                                                  n, A, lda, norm);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dlantr_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   double* norm)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lantr_impl<double, int64_t, double>(handle, norm_type, uplo, diag,
                                                                    m, n, A, lda, norm);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_clantr_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   float* norm)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lantr_impl<rocblas_float_complex, int64_t, float>(
        handle, norm_type, uplo, diag, m, n, A, lda, norm);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zlantr_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   double* norm)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lantr_impl<rocblas_double_complex, int64_t, double>(
        handle, norm_type, uplo, diag, m, n, A, lda, norm);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
