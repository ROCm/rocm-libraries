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

#include "rocauxiliary_geequ.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <bool POW2, typename T, typename I, typename S>
rocblas_status rocsolver_geequ_impl(rocblas_handle handle,
                                    const I m,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    S* R,
                                    S* C,
                                    S* rowcnd,
                                    S* colcnd,
                                    S* amax,
                                    I* info)
try
{
    ROCSOLVER_ENTER_TOP((POW2 ? "geequb" : "geequ"), "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_geequ_argCheck(handle, m, n, lda, A, R, C, rowcnd, colcnd, amax, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideR = 0;
    rocblas_stride strideC = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size of workspace for the partial maxima of the rows and columns
    size_t size_work;
    rocsolver_geequ_getMemorySize<T, I, S>(m, n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_geequ_template<POW2, T, I, S>(handle, m, n, A, shiftA, lda, strideA, R,
                                                   strideR, C, strideC, rowcnd, colcnd, amax, info,
                                                   batch_count, (S*)work);
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

rocblas_status rocsolver_sgeequ(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* R,
                                float* C,
                                float* rowcnd,
                                float* colcnd,
                                float* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<false, float, rocblas_int, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_dgeequ(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* R,
                                double* C,
                                double* rowcnd,
                                double* colcnd,
                                double* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<false, double, rocblas_int, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_cgeequ(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* R,
                                float* C,
                                float* rowcnd,
                                float* colcnd,
                                float* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<false, rocblas_float_complex, rocblas_int, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_zgeequ(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* R,
                                double* C,
                                double* rowcnd,
                                double* colcnd,
                                double* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<false, rocblas_double_complex, rocblas_int, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_sgeequ_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* R,
                                   float* C,
                                   float* rowcnd,
                                   float* colcnd,
                                   float* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<false, float, int64_t, float>(handle, m, n, A, lda, R, C,
                                                                         rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgeequ_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   double* R,
                                   double* C,
                                   double* rowcnd,
                                   double* colcnd,
                                   double* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<false, double, int64_t, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgeequ_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   float* R,
                                   float* C,
                                   float* rowcnd,
                                   float* colcnd,
                                   float* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<false, rocblas_float_complex, int64_t, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgeequ_64(rocblas_handle handle,
                                   const int64_t m,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   double* R,
                                   double* C,
                                   double* rowcnd,
                                   double* colcnd,
                                   double* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<false, rocblas_double_complex, int64_t, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_sgeequb(rocblas_handle handle,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 float* A,
                                 const rocblas_int lda,
                                 float* R,
                                 float* C,
                                 float* rowcnd,
                                 float* colcnd,
                                 float* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<true, float, rocblas_int, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_dgeequb(rocblas_handle handle,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 double* A,
                                 const rocblas_int lda,
                                 double* R,
                                 double* C,
                                 double* rowcnd,
                                 double* colcnd,
                                 double* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<true, double, rocblas_int, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_cgeequb(rocblas_handle handle,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 rocblas_float_complex* A,
                                 const rocblas_int lda,
                                 float* R,
                                 float* C,
                                 float* rowcnd,
                                 float* colcnd,
                                 float* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<true, rocblas_float_complex, rocblas_int, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_zgeequb(rocblas_handle handle,
                                 const rocblas_int m,
                                 const rocblas_int n,
                                 rocblas_double_complex* A,
                                 const rocblas_int lda,
                                 double* R,
                                 double* C,
                                 double* rowcnd,
                                 double* colcnd,
                                 double* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_geequ_impl<true, rocblas_double_complex, rocblas_int, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
}

rocblas_status rocsolver_sgeequb_64(rocblas_handle handle,
                                    const int64_t m,
                                    const int64_t n,
                                    float* A,
                                    const int64_t lda,
                                    float* R,
                                    float* C,
                                    float* rowcnd,
                                    float* colcnd,
                                    float* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<true, float, int64_t, float>(handle, m, n, A, lda, R, C,
                                                                        rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgeequb_64(rocblas_handle handle,
                                    const int64_t m,
                                    const int64_t n,
                                    double* A,
                                    const int64_t lda,
                                    double* R,
                                    double* C,
                                    double* rowcnd,
                                    double* colcnd,
                                    double* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<true, double, int64_t, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgeequb_64(rocblas_handle handle,
                                    const int64_t m,
                                    const int64_t n,
                                    rocblas_float_complex* A,
                                    const int64_t lda,
                                    float* R,
                                    float* C,
                                    float* rowcnd,
                                    float* colcnd,
                                    float* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<true, rocblas_float_complex, int64_t, float>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgeequb_64(rocblas_handle handle,
                                    const int64_t m,
                                    const int64_t n,
                                    rocblas_double_complex* A,
                                    const int64_t lda,
                                    double* R,
                                    double* C,
                                    double* rowcnd,
                                    double* colcnd,
                                    double* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_geequ_impl<true, rocblas_double_complex, int64_t, double>(
        handle, m, n, A, lda, R, C, rowcnd, colcnd, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
