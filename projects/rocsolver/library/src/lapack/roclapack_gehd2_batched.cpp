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

#include "exceptions.hpp"
#include "roclapack_gehd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_gehd2_batched_impl(rocblas_handle handle,
                                            const I n,
                                            const I ilo,
                                            const I ihi,
                                            U A,
                                            const I lda,
                                            T* tau,
                                            const rocblas_stride strideP,
                                            const I batch_count)
try
{
    ROCSOLVER_ENTER_TOP("gehd2_batched", "-n", n, "--ilo", ilo, "--ihi", ihi, "--lda", lda,
                        "--strideP", strideP, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gehd2_argCheck(handle, n, ilo, ihi, lda, A, tau, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARF and LARFG
    size_t size_Abyx_norms;
    // size of temporary array to store diagonal elements
    size_t size_diag;
    rocsolver_gehd2_getMemorySize<true, T>(n, ilo, ihi, batch_count, &size_scalars,
                                           &size_work_workArr, &size_Abyx_norms, &size_diag);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_Abyx_norms, size_diag);

    // memory workspace allocation
    void *scalars, *work_workArr, *Abyx_norms, *diag;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_Abyx_norms, size_diag);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    Abyx_norms = mem[2];
    diag = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gehd2_template<T>(handle, n, ilo, ihi, A, shiftA, lda, strideA, tau, strideP,
                                       batch_count, (T*)scalars, work_workArr, (T*)Abyx_norms,
                                       (T*)diag);
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

rocblas_status rocsolver_sgehd2_batched(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int ilo,
                                        const rocblas_int ihi,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehd2_batched_impl<float>(handle, n, ilo, ihi, A, lda, tau, strideP,
                                                          batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgehd2_batched(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int ilo,
                                        const rocblas_int ihi,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehd2_batched_impl<double>(handle, n, ilo, ihi, A, lda, tau,
                                                           strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgehd2_batched(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int ilo,
                                        const rocblas_int ihi,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_float_complex* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehd2_batched_impl<rocblas_float_complex>(
        handle, n, ilo, ihi, A, lda, tau, strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgehd2_batched(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int ilo,
                                        const rocblas_int ihi,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_double_complex* tau,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_gehd2_batched_impl<rocblas_double_complex>(
        handle, n, ilo, ihi, A, lda, tau, strideP, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
