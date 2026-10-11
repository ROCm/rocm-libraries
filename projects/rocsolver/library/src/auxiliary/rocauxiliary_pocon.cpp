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

#include "rocauxiliary_pocon.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
rocblas_status rocsolver_pocon_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    const S* anorm,
                                    S* rcond,
                                    const I max_iter = 5)
try
{
    ROCSOLVER_ENTER_TOP("pocon", "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_pocon_argCheck(handle, uplo, n, lda, A, anorm, rcond);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    I batch_count = 1;

    // memory workspace sizes:
    size_t size_work_v, size_work_x, size_work_isgn, size_scalars;
    size_t size_work_trsm_1, size_work_trsm_2, size_work_trsm_3, size_work_trsm_4;
    bool optim_mem;
    rocsolver_pocon_getMemorySize<T, I, S>(
        n, batch_count, &size_work_v, &size_work_x, &size_work_isgn, &size_scalars,
        &size_work_trsm_1, &size_work_trsm_2, &size_work_trsm_3, &size_work_trsm_4, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_work_v, size_work_x, size_work_isgn, size_scalars, size_work_trsm_1,
            size_work_trsm_2, size_work_trsm_3, size_work_trsm_4);

    // memory workspace allocation
    void *work_v, *work_x, *work_isgn, *scalars;
    void *work_trsm_1, *work_trsm_2, *work_trsm_3, *work_trsm_4;
    rocblas_device_malloc mem(handle, size_work_v, size_work_x, size_work_isgn, size_scalars,
                              size_work_trsm_1, size_work_trsm_2, size_work_trsm_3, size_work_trsm_4);

    if(!mem)
        return rocblas_status_memory_error;

    work_v = mem[0];
    work_x = mem[1];
    work_isgn = mem[2];
    scalars = mem[3];
    work_trsm_1 = mem[4];
    work_trsm_2 = mem[5];
    work_trsm_3 = mem[6];
    work_trsm_4 = mem[7];

    // execution
    return rocsolver_pocon_template<false, false, T>(
        handle, uplo, n, A, shiftA, lda, strideA, anorm, rcond, batch_count, (T*)work_v, (T*)work_x,
        (I*)work_isgn, scalars, optim_mem, work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4,
        max_iter);
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

rocblas_status rocsolver_spocon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float* anorm,
                                float* rcond)
{
    return rocsolver::rocsolver_pocon_impl<float, rocblas_int, float>(handle, uplo, n, A, lda,
                                                                      anorm, rcond);
}

rocblas_status rocsolver_dpocon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double* anorm,
                                double* rcond)
{
    return rocsolver::rocsolver_pocon_impl<double, rocblas_int, double>(handle, uplo, n, A, lda,
                                                                        anorm, rcond);
}

rocblas_status rocsolver_cpocon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float* anorm,
                                float* rcond)
{
    return rocsolver::rocsolver_pocon_impl<rocblas_float_complex, rocblas_int, float>(
        handle, uplo, n, A, lda, anorm, rcond);
}

rocblas_status rocsolver_zpocon(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double* anorm,
                                double* rcond)
{
    return rocsolver::rocsolver_pocon_impl<rocblas_double_complex, rocblas_int, double>(
        handle, uplo, n, A, lda, anorm, rcond);
}

rocblas_status rocsolver_spocon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   const float* anorm,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_pocon_impl<float, int64_t, float>(handle, uplo, n, A, lda, anorm,
                                                                  rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dpocon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   const double* anorm,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_pocon_impl<double, int64_t, double>(handle, uplo, n, A, lda, anorm,
                                                                    rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cpocon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   const float* anorm,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_pocon_impl<rocblas_float_complex, int64_t, float>(
        handle, uplo, n, A, lda, anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zpocon_64(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   const double* anorm,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_pocon_impl<rocblas_double_complex, int64_t, double>(
        handle, uplo, n, A, lda, anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
