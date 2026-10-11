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

#include "rocauxiliary_laqge.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
rocblas_status rocsolver_laqge_impl(rocblas_handle handle,
                                    const I m,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    const S* R,
                                    const S* C,
                                    const S* rowcnd,
                                    const S* colcnd,
                                    const S* amax,
                                    rocsolver_equilibration* equed)
try
{
    ROCSOLVER_ENTER_TOP("laqge", "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_laqge_argCheck(handle, m, n, lda, A, R, C, rowcnd, colcnd, amax, equed);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideR = 0;
    rocblas_stride strideC = 0;
    I batch_count = 1;

    // no workspace is needed
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_laqge_template<T, I, S>(handle, m, n, A, shiftA, lda, strideA, R, strideR, C,
                                             strideC, rowcnd, colcnd, amax, equed, batch_count);
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

rocblas_status rocsolver_slaqge(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float* R,
                                const float* C,
                                const float* rowcnd,
                                const float* colcnd,
                                const float* amax,
                                rocsolver_equilibration* equed)
{
    return rocsolver::rocsolver_laqge_impl<float, rocblas_int, float>(handle, m, n, A, lda, R, C,
                                                                      rowcnd, colcnd, amax, equed);
}

rocblas_status rocsolver_dlaqge(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double* R,
                                const double* C,
                                const double* rowcnd,
                                const double* colcnd,
                                const double* amax,
                                rocsolver_equilibration* equed)
{
    return rocsolver::rocsolver_laqge_impl<double, rocblas_int, double>(handle, m, n, A, lda, R, C,
                                                                        rowcnd, colcnd, amax, equed);
}

rocblas_status rocsolver_claqge(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float* R,
                                const float* C,
                                const float* rowcnd,
                                const float* colcnd,
                                const float* amax,
                                rocsolver_equilibration* equed)
{
    return rocsolver::rocsolver_laqge_impl<rocblas_float_complex, rocblas_int, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, equed);
}

rocblas_status rocsolver_zlaqge(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double* R,
                                const double* C,
                                const double* rowcnd,
                                const double* colcnd,
                                const double* amax,
                                rocsolver_equilibration* equed)
{
    return rocsolver::rocsolver_laqge_impl<rocblas_double_complex, rocblas_int, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, equed);
}

rocblas_status rocsolver_slaqge_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   const float* R,
                                   const float* C,
                                   const float* rowcnd,
                                   const float* colcnd,
                                   const float* amax,
                                   rocsolver_equilibration* equed)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_laqge_impl<float, int64_t, float>(handle, m, n, A, lda, R, C,
                                                                  rowcnd, colcnd, amax, equed);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dlaqge_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   const double* R,
                                   const double* C,
                                   const double* rowcnd,
                                   const double* colcnd,
                                   const double* amax,
                                   rocsolver_equilibration* equed)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_laqge_impl<double, int64_t, double>(handle, m, n, A, lda, R, C,
                                                                    rowcnd, colcnd, amax, equed);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_claqge_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   const float* R,
                                   const float* C,
                                   const float* rowcnd,
                                   const float* colcnd,
                                   const float* amax,
                                   rocsolver_equilibration* equed)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_laqge_impl<rocblas_float_complex, int64_t, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, equed);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zlaqge_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   const double* R,
                                   const double* C,
                                   const double* rowcnd,
                                   const double* colcnd,
                                   const double* amax,
                                   rocsolver_equilibration* equed)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_laqge_impl<rocblas_double_complex, int64_t, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, equed);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
