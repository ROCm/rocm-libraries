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

#include "roclapack_gesv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_gesv_impl(rocblas_handle handle,
                                   const rocblas_int n,
                                   const rocblas_int nrhs,
                                   T* A,
                                   const rocblas_int lda,
                                   rocblas_int* ipiv,
                                   T* B,
                                   const rocblas_int ldb,
                                   rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gesv", "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesv_argCheck(handle, n, nrhs, lda, ldb, A, B, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_gesv_getMemorySize<false, false, T>(handle, n, nrhs, A, shiftA, lda, strideA, ipiv,
                                                  strideP, B, shiftB, ldb, strideB, info,
                                                  batch_count, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());

    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // execution
    return rocsolver_gesv_template<false, false, T>(handle, n, nrhs, A, shiftA, lda, strideA, ipiv,
                                                    strideP, B, shiftB, ldb, strideB, info,
                                                    batch_count, &work_helper);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_sgesv(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          float* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_impl<float>(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

extern "C" rocblas_status rocsolver_dgesv(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          double* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_impl<double>(handle, n, nrhs, A, lda, ipiv, B, ldb, info);
}

extern "C" rocblas_status rocsolver_cgesv(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          rocblas_float_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_impl<rocblas_float_complex>(handle, n, nrhs, A, lda, ipiv, B,
                                                                 ldb, info);
}

extern "C" rocblas_status rocsolver_zgesv(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int lda,
                                          rocblas_int* ipiv,
                                          rocblas_double_complex* B,
                                          const rocblas_int ldb,
                                          rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_impl<rocblas_double_complex>(handle, n, nrhs, A, lda, ipiv, B,
                                                                  ldb, info);
}
