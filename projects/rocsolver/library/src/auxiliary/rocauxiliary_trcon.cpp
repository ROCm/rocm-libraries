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

#include "rocauxiliary_trcon.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
rocblas_status rocsolver_trcon_impl(rocblas_handle handle,
                                    const rocsolver_norm_type norm_type,
                                    const rocblas_fill uplo,
                                    const rocblas_diagonal diag,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    S* rcond,
                                    const I max_iter = 5)
try
{
    ROCSOLVER_ENTER_TOP("trcon", "--norm_type", norm_type, "--uplo", uplo, "--diag", diag, "-n", n,
                        "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_trcon_argCheck(handle, norm_type, uplo, diag, n, lda, A, rcond);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    I batch_count = 1;

    // memory workspace sizes:
    size_t size_work_norm, size_anorm, size_work_v, size_work_x, size_work_isgn, size_scalars;
    size_t size_work_trsm_1, size_work_trsm_2, size_work_trsm_3, size_work_trsm_4;
    bool optim_mem;
    rocsolver_trcon_getMemorySize<T, I, S>(norm_type, n, batch_count, &size_work_norm, &size_anorm,
                                           &size_work_v, &size_work_x, &size_work_isgn,
                                           &size_scalars, &size_work_trsm_1, &size_work_trsm_2,
                                           &size_work_trsm_3, &size_work_trsm_4, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_work_norm, size_anorm, size_work_v, size_work_x, size_work_isgn,
            size_scalars, size_work_trsm_1, size_work_trsm_2, size_work_trsm_3, size_work_trsm_4);

    // memory workspace allocation
    void *work_norm, *anorm, *work_v, *work_x, *work_isgn, *scalars;
    void *work_trsm_1, *work_trsm_2, *work_trsm_3, *work_trsm_4;
    rocblas_device_malloc mem(handle, size_work_norm, size_anorm, size_work_v, size_work_x,
                              size_work_isgn, size_scalars, size_work_trsm_1, size_work_trsm_2,
                              size_work_trsm_3, size_work_trsm_4);

    if(!mem)
        return rocblas_status_memory_error;

    work_norm = mem[0];
    anorm = mem[1];
    work_v = mem[2];
    work_x = mem[3];
    work_isgn = mem[4];
    scalars = mem[5];
    work_trsm_1 = mem[6];
    work_trsm_2 = mem[7];
    work_trsm_3 = mem[8];
    work_trsm_4 = mem[9];

    // execution
    return rocsolver_trcon_template<false, false, T>(
        handle, norm_type, uplo, diag, n, A, shiftA, lda, strideA, rcond, batch_count,
        (S*)work_norm, (S*)anorm, (T*)work_v, (T*)work_x, (I*)work_isgn, scalars, optim_mem,
        work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4, max_iter);
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

rocblas_status rocsolver_strcon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* rcond)
{
    return rocsolver::rocsolver_trcon_impl<float, rocblas_int, float>(handle, norm_type, uplo, diag,
                                                                      n, A, lda, rcond);
}

rocblas_status rocsolver_dtrcon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* rcond)
{
    return rocsolver::rocsolver_trcon_impl<double, rocblas_int, double>(handle, norm_type, uplo,
                                                                        diag, n, A, lda, rcond);
}

rocblas_status rocsolver_ctrcon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* rcond)
{
    return rocsolver::rocsolver_trcon_impl<rocblas_float_complex, rocblas_int, float>(
        handle, norm_type, uplo, diag, n, A, lda, rcond);
}

rocblas_status rocsolver_ztrcon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_fill uplo,
                                const rocblas_diagonal diag,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* rcond)
{
    return rocsolver::rocsolver_trcon_impl<rocblas_double_complex, rocblas_int, double>(
        handle, norm_type, uplo, diag, n, A, lda, rcond);
}

rocblas_status rocsolver_strcon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_trcon_impl<float, int64_t, float>(handle, norm_type, uplo, diag, n,
                                                                  A, lda, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dtrcon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_trcon_impl<double, int64_t, double>(handle, norm_type, uplo, diag,
                                                                    n, A, lda, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ctrcon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_trcon_impl<rocblas_float_complex, int64_t, float>(
        handle, norm_type, uplo, diag, n, A, lda, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ztrcon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const rocblas_fill uplo,
                                   const rocblas_diagonal diag,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_trcon_impl<rocblas_double_complex, int64_t, double>(
        handle, norm_type, uplo, diag, n, A, lda, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
