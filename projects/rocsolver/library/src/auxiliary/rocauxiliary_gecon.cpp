/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_gecon.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S>
rocblas_status rocsolver_gecon_impl(rocblas_handle handle,
                                    const rocsolver_norm_type norm_type,
                                    const I n,
                                    T* A,
                                    const I lda,
                                    const S* anorm,
                                    S* rcond,
                                    const I max_iter = 5)
{
    ROCSOLVER_ENTER_TOP("gecon", "--norm_type", norm_type, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gecon_argCheck(handle, norm_type, n, lda, A, anorm, rcond);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    I batch_count = 1;

    // memory workspace sizes:
    size_t size_work_v, size_work_x, size_work_isgn;
    size_t size_scalar_est, size_scalar_max_idx, size_scalar_kase, size_scalar_jump;
    size_t size_work_trsm_1, size_work_trsm_2, size_work_trsm_3, size_work_trsm_4;
    bool optim_mem;
    rocsolver_gecon_getMemorySize<T, I, S>(
        n, lda, batch_count, &size_work_v, &size_work_x, &size_work_isgn, &size_scalar_est,
        &size_scalar_max_idx, &size_scalar_kase, &size_scalar_jump, &size_work_trsm_1,
        &size_work_trsm_2, &size_work_trsm_3, &size_work_trsm_4, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_work_v, size_work_x, size_work_isgn, size_scalar_est, size_scalar_max_idx,
            size_scalar_kase, size_scalar_jump, size_work_trsm_1, size_work_trsm_2,
            size_work_trsm_3, size_work_trsm_4);

    // memory workspace allocation
    void *work_v, *work_x, *work_isgn, *scalar_est, *scalar_max_idx;
    void *scalar_kase, *scalar_jump;
    void *work_trsm_1, *work_trsm_2, *work_trsm_3, *work_trsm_4;
    rocblas_device_malloc mem(handle, size_work_v, size_work_x, size_work_isgn, size_scalar_est,
                              size_scalar_max_idx, size_scalar_kase, size_scalar_jump,
                              size_work_trsm_1, size_work_trsm_2, size_work_trsm_3, size_work_trsm_4);

    if(!mem)
        return rocblas_status_memory_error;

    work_v = mem[0];
    work_x = mem[1];
    work_isgn = mem[2];
    scalar_est = mem[3];
    scalar_max_idx = mem[4];
    scalar_kase = mem[5];
    scalar_jump = mem[6];
    work_trsm_1 = mem[7];
    work_trsm_2 = mem[8];
    work_trsm_3 = mem[9];
    work_trsm_4 = mem[10];

    // execution
    return rocsolver_gecon_template<false, false, T>(
        handle, norm_type, n, A, shiftA, (I)1, lda, strideA, anorm, rcond, batch_count, (T*)work_v,
        (T*)work_x, (I*)work_isgn, (S*)scalar_est, (I*)scalar_max_idx, (rocblas_int*)scalar_kase,
        (rocblas_int*)scalar_jump, optim_mem, work_trsm_1, work_trsm_2, work_trsm_3, work_trsm_4,
        max_iter);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgecon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const float* anorm,
                                float* rcond)
{
    return rocsolver::rocsolver_gecon_impl<float, rocblas_int, float>(handle, norm_type, n, A, lda,
                                                                      anorm, rcond);
}

rocblas_status rocsolver_dgecon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const double* anorm,
                                double* rcond)
{
    return rocsolver::rocsolver_gecon_impl<double, rocblas_int, double>(handle, norm_type, n, A,
                                                                        lda, anorm, rcond);
}

rocblas_status rocsolver_cgecon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const float* anorm,
                                float* rcond)
{
    return rocsolver::rocsolver_gecon_impl<rocblas_float_complex, rocblas_int, float>(
        handle, norm_type, n, A, lda, anorm, rcond);
}

rocblas_status rocsolver_zgecon(rocblas_handle handle,
                                const rocsolver_norm_type norm_type,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const double* anorm,
                                double* rcond)
{
    return rocsolver::rocsolver_gecon_impl<rocblas_double_complex, rocblas_int, double>(
        handle, norm_type, n, A, lda, anorm, rcond);
}

rocblas_status rocsolver_sgecon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const int64_t n,
                                   float* A,
                                   const int64_t lda,
                                   const float* anorm,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_gecon_impl<float, int64_t, float>(handle, norm_type, n, A, lda,
                                                                  anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgecon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const int64_t n,
                                   double* A,
                                   const int64_t lda,
                                   const double* anorm,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_gecon_impl<double, int64_t, double>(handle, norm_type, n, A, lda,
                                                                    anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgecon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const int64_t n,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   const float* anorm,
                                   float* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_gecon_impl<rocblas_float_complex, int64_t, float>(
        handle, norm_type, n, A, lda, anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgecon_64(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
                                   const int64_t n,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   const double* anorm,
                                   double* rcond)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_gecon_impl<rocblas_double_complex, int64_t, double>(
        handle, norm_type, n, A, lda, anorm, rcond);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
