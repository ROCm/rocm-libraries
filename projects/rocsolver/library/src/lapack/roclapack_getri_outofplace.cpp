/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
rocblas_status rocsolver_getri_outofplace_impl(rocblas_handle handle,
                                               const rocblas_int n,
                                               U A,
                                               const rocblas_int lda,
                                               rocblas_int* ipiv,
                                               U C,
                                               const rocblas_int ldc,
                                               rocblas_int* info,
                                               const bool pivot)
{
    const char* name = (pivot ? "getri_outofplace" : "getri_npvt_outofplace");
    ROCSOLVER_ENTER_TOP(name, "-n", n, "--lda", lda, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_getri_outofplace_argCheck(handle, n, lda, ldc, A, C, ipiv, info, pivot);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftP = 0;
    rocblas_int shiftC = 0;

    // normal execution
    rocblas_stride strideA = 0;
    rocblas_stride strideC = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace (for calling GETRS)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;

    rocsolver_getri_outofplace_getMemorySize<false, false, T>(
        n, batch_count, &size_work1, &size_work2, &size_work3, &size_work4, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4);

    if(!mem)
        return rocblas_status_memory_error;
    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];

    // Execution
    return rocsolver_getri_outofplace_template<false, false, T>(
        handle, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, C, shiftC, ldc, strideC, info,
        batch_count, work1, work2, work3, work4, optim_mem, pivot);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           float* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           float* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver::rocsolver_getri_outofplace_impl<float>(handle, n, A, lda, ipiv, C, ldc, info,
                                                             true);
}

rocblas_status rocsolver_dgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           double* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           double* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver::rocsolver_getri_outofplace_impl<double>(handle, n, A, lda, ipiv, C, ldc, info,
                                                              true);
}

rocblas_status rocsolver_cgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           rocblas_float_complex* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           rocblas_float_complex* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver::rocsolver_getri_outofplace_impl<rocblas_float_complex>(handle, n, A, lda, ipiv,
                                                                             C, ldc, info, true);
}

rocblas_status rocsolver_zgetri_outofplace(rocblas_handle handle,
                                           const rocblas_int n,
                                           rocblas_double_complex* A,
                                           const rocblas_int lda,
                                           rocblas_int* ipiv,
                                           rocblas_double_complex* C,
                                           const rocblas_int ldc,
                                           rocblas_int* info)
{
    return rocsolver::rocsolver_getri_outofplace_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, C, ldc, info, true);
}

rocblas_status rocsolver_sgetri_npvt_outofplace(rocblas_handle handle,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                float* C,
                                                const rocblas_int ldc,
                                                rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_impl<float>(handle, n, A, lda, ipiv, C, ldc, info,
                                                             false);
}

rocblas_status rocsolver_dgetri_npvt_outofplace(rocblas_handle handle,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                double* C,
                                                const rocblas_int ldc,
                                                rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_impl<double>(handle, n, A, lda, ipiv, C, ldc, info,
                                                              false);
}

rocblas_status rocsolver_cgetri_npvt_outofplace(rocblas_handle handle,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                rocblas_float_complex* C,
                                                const rocblas_int ldc,
                                                rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_impl<rocblas_float_complex>(handle, n, A, lda, ipiv,
                                                                             C, ldc, info, false);
}

rocblas_status rocsolver_zgetri_npvt_outofplace(rocblas_handle handle,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                rocblas_double_complex* C,
                                                const rocblas_int ldc,
                                                rocblas_int* info)
{
    rocblas_int* ipiv = nullptr;
    return rocsolver::rocsolver_getri_outofplace_impl<rocblas_double_complex>(
        handle, n, A, lda, ipiv, C, ldc, info, false);
}

} // extern C
