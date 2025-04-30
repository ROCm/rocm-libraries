/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_gesv_outofplace.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    gesv_outofplace is not intended for inclusion in the public API. It
 *    exists to provide a gesv method with a signature identical to
 *    the cuSOLVER implementation, for use exclusively in hipSOLVER.
 * ===========================================================================
 */

template <typename T>
rocblas_status rocsolver_gesv_outofplace_impl(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nrhs,
                                              T* A,
                                              const rocblas_int lda,
                                              rocblas_int* ipiv,
                                              T* B,
                                              const rocblas_int ldb,
                                              T* X,
                                              const rocblas_int ldx,
                                              rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gesv_outofplace", "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb,
                        "--ldx", ldx);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gesv_outofplace_argCheck(handle, n, nrhs, lda, ldb, ldx, A, B, X, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftX = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace for calling GETRF and GETRS
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETRF
    size_t size_pivotval, size_pivotidx, size_iipiv, size_iinfo;
    rocsolver_gesv_outofplace_getMemorySize<false, false, T>(
        n, nrhs, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivotval,
                                                      size_pivotidx, size_iipiv, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo, *iipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivotval, size_pivotidx, size_iipiv, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivotval = mem[5];
    pivotidx = mem[6];
    iipiv = mem[7];
    iinfo = mem[8];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesv_outofplace_template<false, false, T>(
        handle, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP, B, shiftB, ldb, strideB, X, shiftX,
        ldx, strideX, info, batch_count, (T*)scalars, work1, work2, work3, work4, (T*)pivotval,
        (rocblas_int*)pivotidx, (rocblas_int*)iipiv, (rocblas_int*)iinfo, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           float* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           float* B,
                                                           const rocblas_int ldb,
                                                           float* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_outofplace_impl<float>(handle, n, nrhs, A, lda, ipiv, B, ldb,
                                                            X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           double* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           double* B,
                                                           const rocblas_int ldb,
                                                           double* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_outofplace_impl<double>(handle, n, nrhs, A, lda, ipiv, B, ldb,
                                                             X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           rocblas_float_complex* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           rocblas_float_complex* B,
                                                           const rocblas_int ldb,
                                                           rocblas_float_complex* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_outofplace_impl<rocblas_float_complex>(
        handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgesv_outofplace(rocblas_handle handle,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           rocblas_double_complex* A,
                                                           const rocblas_int lda,
                                                           rocblas_int* ipiv,
                                                           rocblas_double_complex* B,
                                                           const rocblas_int ldb,
                                                           rocblas_double_complex* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_outofplace_impl<rocblas_double_complex>(
        handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, info);
}

} // extern C
