/* **************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_getri_outofplace.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_getri_outofplace_batched_impl(rocblas_handle handle,
                                                       const rocblas_int n,
                                                       U A,
                                                       const rocblas_int lda,
                                                       rocblas_int* ipiv,
                                                       const rocblas_stride strideP,
                                                       U C,
                                                       const rocblas_int ldc,
                                                       rocblas_int* info,
                                                       const bool pivot,
                                                       const rocblas_int batch_count)
{
    const char* name = (pivot ? "getri_outofplace_batched" : "getri_npvt_outofplace_batched");
    ROCSOLVER_ENTER_TOP(name, "-n", n, "--lda", lda, "--strideP", strideP, "--ldc", ldc,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getri_outofplace_argCheck(handle, n, lda, ldc, A, C, ipiv, info,
                                                            pivot, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;
    rocblas_int shiftC = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideC = 0;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_getri_outofplace_getMemorySize<true, false, T>(
        handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, C, shiftC, ldc, strideC, info,
        batch_count, &work_helper, pivot);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());

    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // Execution
    return rocsolver_getri_outofplace_template<true, false, T>(
        handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, C, shiftC, ldc, strideC, info,
        batch_count, &work_helper, pivot);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   float* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   float* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getri_outofplace_batched_impl<float>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, true, batch_count);
}

rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   double* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   double* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getri_outofplace_batched_impl<double>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, true, batch_count);
}

rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   rocblas_float_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_float_complex* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, true, batch_count);
}

rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle handle,
                                                   const rocblas_int n,
                                                   rocblas_double_complex* const A[],
                                                   const rocblas_int lda,
                                                   rocblas_int* ipiv,
                                                   const rocblas_stride strideP,
                                                   rocblas_double_complex* const C[],
                                                   const rocblas_int ldc,
                                                   rocblas_int* info,
                                                   const rocblas_int batch_count)
{
    return rocsolver::rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, strideP, C, ldc, info, true, batch_count);
}

rocblas_status rocsolver_sgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        float* const A[],
                                                        const rocblas_int lda,
                                                        float* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_batched_impl<float>(handle, n, A, lda, ipiv, 0, C,
                                                                     ldc, info, false, batch_count);
}

rocblas_status rocsolver_dgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        double* const A[],
                                                        const rocblas_int lda,
                                                        double* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_batched_impl<double>(handle, n, A, lda, ipiv, 0, C,
                                                                      ldc, info, false, batch_count);
}

rocblas_status rocsolver_cgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        rocblas_float_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_float_complex* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_batched_impl<rocblas_float_complex>(
        handle, n, A, lda, ipiv, 0, C, ldc, info, false, batch_count);
}

rocblas_status rocsolver_zgetri_npvt_outofplace_batched(rocblas_handle handle,
                                                        const rocblas_int n,
                                                        rocblas_double_complex* const A[],
                                                        const rocblas_int lda,
                                                        rocblas_double_complex* const C[],
                                                        const rocblas_int ldc,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_batched_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, 0, C, ldc, info, false, batch_count);
}

} // extern C
