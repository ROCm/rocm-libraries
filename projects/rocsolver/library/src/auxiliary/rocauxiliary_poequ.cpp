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

#include "rocauxiliary_poequ.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <bool POW2, typename T, typename I, typename S>
rocblas_status rocsolver_poequ_impl(rocblas_handle handle,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    S* Sc,
                                    S* scond,
                                    S* amax,
                                    I* info)
try
{
    ROCSOLVER_ENTER_TOP((POW2 ? "poequb" : "poequ"), "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_poequ_argCheck(handle, n, lda, A, Sc, scond, amax, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideS = 0;
    I batch_count = 1;

    // no workspace is needed
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_poequ_template<POW2, T, I, S>(handle, n, A, shiftA, lda, strideA, Sc, strideS,
                                                   scond, amax, info, batch_count);
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

rocblas_status rocsolver_spoequ(rocblas_handle handle,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* S,
                                float* scond,
                                float* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<false, float, rocblas_int, float>(handle, n, A, lda, S,
                                                                             scond, amax, info);
}

rocblas_status rocsolver_dpoequ(rocblas_handle handle,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* S,
                                double* scond,
                                double* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<false, double, rocblas_int, double>(handle, n, A, lda, S,
                                                                               scond, amax, info);
}

rocblas_status rocsolver_cpoequ(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* S,
                                float* scond,
                                float* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<false, rocblas_float_complex, rocblas_int, float>(
        handle, n, A, lda, S, scond, amax, info);
}

rocblas_status rocsolver_zpoequ(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* S,
                                double* scond,
                                double* amax,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<false, rocblas_double_complex, rocblas_int, double>(
        handle, n, A, lda, S, scond, amax, info);
}

rocblas_status rocsolver_spoequ_64(rocblas_handle handle,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* S,
                                   float* scond,
                                   float* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<false, float, int64_t, float>(handle, n, A, lda, S,
                                                                         scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dpoequ_64(rocblas_handle handle,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   double* S,
                                   double* scond,
                                   double* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<false, double, int64_t, double>(handle, n, A, lda, S,
                                                                           scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cpoequ_64(rocblas_handle handle,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   float* S,
                                   float* scond,
                                   float* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<false, rocblas_float_complex, int64_t, float>(
        handle, n, A, lda, S, scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zpoequ_64(rocblas_handle handle,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   double* S,
                                   double* scond,
                                   double* amax,
                                   int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<false, rocblas_double_complex, int64_t, double>(
        handle, n, A, lda, S, scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_spoequb(rocblas_handle handle,
                                 const rocblas_int n,
                                 float* A,
                                 const rocblas_int lda,
                                 float* S,
                                 float* scond,
                                 float* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<true, float, rocblas_int, float>(handle, n, A, lda, S,
                                                                            scond, amax, info);
}

rocblas_status rocsolver_dpoequb(rocblas_handle handle,
                                 const rocblas_int n,
                                 double* A,
                                 const rocblas_int lda,
                                 double* S,
                                 double* scond,
                                 double* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<true, double, rocblas_int, double>(handle, n, A, lda, S,
                                                                              scond, amax, info);
}

rocblas_status rocsolver_cpoequb(rocblas_handle handle,
                                 const rocblas_int n,
                                 rocblas_float_complex* A,
                                 const rocblas_int lda,
                                 float* S,
                                 float* scond,
                                 float* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<true, rocblas_float_complex, rocblas_int, float>(
        handle, n, A, lda, S, scond, amax, info);
}

rocblas_status rocsolver_zpoequb(rocblas_handle handle,
                                 const rocblas_int n,
                                 rocblas_double_complex* A,
                                 const rocblas_int lda,
                                 double* S,
                                 double* scond,
                                 double* amax,
                                 rocblas_int* info)
{
    return rocsolver::rocsolver_poequ_impl<true, rocblas_double_complex, rocblas_int, double>(
        handle, n, A, lda, S, scond, amax, info);
}

rocblas_status rocsolver_spoequb_64(rocblas_handle handle,
                                    const int64_t n,
                                    float* A,
                                    const int64_t lda,
                                    float* S,
                                    float* scond,
                                    float* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<true, float, int64_t, float>(handle, n, A, lda, S, scond,
                                                                        amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dpoequb_64(rocblas_handle handle,
                                    const int64_t n,
                                    double* A,
                                    const int64_t lda,
                                    double* S,
                                    double* scond,
                                    double* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<true, double, int64_t, double>(handle, n, A, lda, S,
                                                                          scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cpoequb_64(rocblas_handle handle,
                                    const int64_t n,
                                    rocblas_float_complex* A,
                                    const int64_t lda,
                                    float* S,
                                    float* scond,
                                    float* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<true, rocblas_float_complex, int64_t, float>(
        handle, n, A, lda, S, scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zpoequb_64(rocblas_handle handle,
                                    const int64_t n,
                                    rocblas_double_complex* A,
                                    const int64_t lda,
                                    double* S,
                                    double* scond,
                                    double* amax,
                                    int64_t* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_poequ_impl<true, rocblas_double_complex, int64_t, double>(
        handle, n, A, lda, S, scond, amax, info);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
