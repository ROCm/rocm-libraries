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

#include "roclapack_gehrd.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_gehrd_strided_batched_impl(rocblas_handle handle,
                                                    const I n,
                                                    const I ilo,
                                                    const I ihi,
                                                    U A,
                                                    const I lda,
                                                    const rocblas_stride strideA,
                                                    T* tau,
                                                    const rocblas_stride strideP,
                                                    const I batch_count)
{
    ROCSOLVER_ENTER_TOP("gehrd_strided_batched", "-n", n, "--ilo", ilo, "--ihi", ihi, "--lda", lda,
                        "--strideA", strideA, "--strideP", strideP, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gehrd_argCheck(handle, n, ilo, ihi, lda, A, tau, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // memory workspace sizes
    size_t size_scalars;
    size_t size_work_workArr;
    size_t size_norms_tmptr;
    size_t size_diag_beta;
    size_t size_F;
    size_t size_work_vec;
    size_t size_Y;
    rocsolver_gehrd_getMemorySize<false, T>(n, ilo, ihi, batch_count, &size_scalars,
                                            &size_work_workArr, &size_norms_tmptr, &size_diag_beta,
                                            &size_F, &size_work_vec, &size_Y);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_norms_tmptr, size_diag_beta, size_F,
                                                      size_work_vec, size_Y);

    // memory workspace allocation
    void *scalars, *work_workArr, *norms_tmptr, *diag_beta, *F, *work_vec, *Y;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_norms_tmptr,
                              size_diag_beta, size_F, size_work_vec, size_Y);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    norms_tmptr = mem[2];
    diag_beta = mem[3];
    F = mem[4];
    work_vec = mem[5];
    Y = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gehrd_template<false, true, T>(
        handle, n, ilo, ihi, A, shiftA, lda, strideA, tau, strideP, batch_count, (T*)scalars,
        work_workArr, (T*)norms_tmptr, diag_beta, (T*)F, (T*)work_vec, (T*)Y);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgehrd_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int ilo,
                                                const rocblas_int ihi,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehrd_strided_batched_impl<float>(
        handle, n, ilo, ihi, A, lda, strideA, tau, strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgehrd_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int ilo,
                                                const rocblas_int ihi,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehrd_strided_batched_impl<double>(
        handle, n, ilo, ihi, A, lda, strideA, tau, strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgehrd_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int ilo,
                                                const rocblas_int ihi,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_float_complex* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehrd_strided_batched_impl<rocblas_float_complex>(
        handle, n, ilo, ihi, A, lda, strideA, tau, strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgehrd_strided_batched(rocblas_handle handle,
                                                const rocblas_int n,
                                                const rocblas_int ilo,
                                                const rocblas_int ihi,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_double_complex* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehrd_strided_batched_impl<rocblas_double_complex>(
        handle, n, ilo, ihi, A, lda, strideA, tau, strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
