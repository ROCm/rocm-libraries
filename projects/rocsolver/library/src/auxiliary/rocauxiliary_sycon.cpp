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

#include "rocauxiliary_sycon.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
rocblas_status rocsolver_sycon_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    I* ipiv,
                                    const S* anorm,
                                    S* rcond,
                                    const I max_iter = 5)
try
{
    ROCSOLVER_ENTER_TOP("sycon", "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sycon_argCheck(handle, uplo, n, lda, A, ipiv, anorm, rcond);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    I batch_count = 1;

    // memory workspace sizes:
    size_t size_work_v, size_work_x, size_work_isgn, size_scalars, size_work_sytrs;
    ROCBLAS_CHECK(rocsolver_sycon_getMemorySize<T, I, S>(handle, n, lda, batch_count, &size_work_v,
                                                         &size_work_x, &size_work_isgn,
                                                         &size_scalars, &size_work_sytrs));

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_v, size_work_x,
                                                      size_work_isgn, size_scalars, size_work_sytrs);

    // memory workspace allocation
    void *work_v, *work_x, *work_isgn, *scalars, *work_sytrs;
    rocblas_device_malloc mem(handle, size_work_v, size_work_x, size_work_isgn, size_scalars,
                              size_work_sytrs);

    if(!mem)
        return rocblas_status_memory_error;

    work_v = mem[0];
    work_x = mem[1];
    work_isgn = mem[2];
    scalars = mem[3];
    work_sytrs = mem[4];

    // execution
    return rocsolver_sycon_template<false, false, T>(
        handle, uplo, n, A, shiftA, lda, strideA, ipiv, strideP, anorm, rcond, batch_count,
        (T*)work_v, (T*)work_x, (I*)work_isgn, scalars, work_sytrs, size_work_sytrs, max_iter);
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

rocblas_status rocsolver_ssycon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                const float* anorm,
                                float* rcond)
{
    return rocsolver::rocsolver_sycon_impl<float, rocblas_int, float>(handle, uplo, n, A, lda, ipiv,
                                                                      anorm, rcond);
}

rocblas_status rocsolver_dsycon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                const double* anorm,
                                double* rcond)
{
    return rocsolver::rocsolver_sycon_impl<double, rocblas_int, double>(handle, uplo, n, A, lda,
                                                                        ipiv, anorm, rcond);
}

rocblas_status rocsolver_csycon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                const float* anorm,
                                float* rcond)
{
    return rocsolver::rocsolver_sycon_impl<rocblas_float_complex, rocblas_int, float>(
        handle, uplo, n, A, lda, ipiv, anorm, rcond);
}

rocblas_status rocsolver_zsycon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                const double* anorm,
                                double* rcond)
{
    return rocsolver::rocsolver_sycon_impl<rocblas_double_complex, rocblas_int, double>(
        handle, uplo, n, A, lda, ipiv, anorm, rcond);
}

rocblas_status rocsolver_ssycon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   const float* anorm,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sycon_impl<float, int64_t, float>(handle, uplo, n, A, lda, ipiv,
                                                                  anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dsycon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   const double* anorm,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sycon_impl<double, int64_t, double>(handle, uplo, n, A, lda, ipiv,
                                                                    anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_csycon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   const float* anorm,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sycon_impl<rocblas_float_complex, int64_t, float>(
        handle, uplo, n, A, lda, ipiv, anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zsycon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   int64_t* ipiv,
                                   const double* anorm,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sycon_impl<rocblas_double_complex, int64_t, double>(
        handle, uplo, n, A, lda, ipiv, anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
