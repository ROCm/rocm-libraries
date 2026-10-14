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

#include "roclapack_sysv.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_sysv_impl(rocblas_handle handle,
                                   const rocblas_fill uplo,
                                   const rocblas_int n,
                                   const rocblas_int nrhs,
                                   T* A,
                                   const rocblas_int lda,
                                   rocblas_int* ipiv,
                                   T* B,
                                   const rocblas_int ldb,
                                   rocblas_int* info)
try
{
    ROCSOLVER_ENTER_TOP("sysv", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sysv_argCheck(handle, uplo, n, nrhs, lda, ldb, A, ipiv, B, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_stride strideB = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of workspace for SYTRF and SYTRS, and for a copy of B
    size_t size_work_sytrf, size_work_sytrs, size_savedB;
    ROCBLAS_CHECK(rocsolver_sysv_getMemorySize<false, false, T>(
        handle, n, nrhs, lda, ldb, batch_count, &size_work_sytrf, &size_work_sytrs, &size_savedB));

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_sytrf, size_work_sytrs,
                                                      size_savedB);

    // memory workspace allocation
    void *work_sytrf, *work_sytrs, *savedB;
    rocblas_device_malloc mem(handle, size_work_sytrf, size_work_sytrs, size_savedB);

    if(!mem)
        return rocblas_status_memory_error;

    work_sytrf = mem[0];
    work_sytrs = mem[1];
    savedB = mem[2];

    // execution
    return rocsolver_sysv_template<false, false, T>(
        handle, uplo, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP, B, shiftB, ldb, strideB,
        info, batch_count, (T*)work_sytrf, work_sytrs, size_work_sytrs, (T*)savedB);
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

rocblas_status rocsolver_ssysv(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               float* A,
                               const rocblas_int lda,
                               rocblas_int* ipiv,
                               float* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_sysv_impl<float>(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
}

rocblas_status rocsolver_dsysv(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               double* A,
                               const rocblas_int lda,
                               rocblas_int* ipiv,
                               double* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_sysv_impl<double>(handle, uplo, n, nrhs, A, lda, ipiv, B, ldb, info);
}

rocblas_status rocsolver_csysv(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               rocblas_float_complex* A,
                               const rocblas_int lda,
                               rocblas_int* ipiv,
                               rocblas_float_complex* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_sysv_impl<rocblas_float_complex>(handle, uplo, n, nrhs, A, lda,
                                                                 ipiv, B, ldb, info);
}

rocblas_status rocsolver_zsysv(rocblas_handle handle,
                               const rocblas_fill uplo,
                               const rocblas_int n,
                               const rocblas_int nrhs,
                               rocblas_double_complex* A,
                               const rocblas_int lda,
                               rocblas_int* ipiv,
                               rocblas_double_complex* B,
                               const rocblas_int ldb,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_sysv_impl<rocblas_double_complex>(handle, uplo, n, nrhs, A, lda,
                                                                  ipiv, B, ldb, info);
}

} // extern C
