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

#include "roclapack_sytrf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_sytrf_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int lda,
                                    rocblas_int* ipiv,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("sytrf", "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sytf2_sytrf_argCheck(handle, uplo, n, lda, A, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // using unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;
    rocsolver_sytrf_getMemorySize<T>(n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_sytrf_template<T>(handle, uplo, n, A, shiftA, lda, strideA, ipiv, strideP,
                                       info, batch_count, (T*)work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sytrf_impl<float>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_dsytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sytrf_impl<double>(handle, uplo, n, A, lda, ipiv, info);
}

rocblas_status rocsolver_csytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sytrf_impl<rocblas_float_complex>(handle, uplo, n, A, lda, ipiv,
                                                                  info);
}

rocblas_status rocsolver_zsytrf(rocblas_handle handle,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_int* ipiv,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sytrf_impl<rocblas_double_complex>(handle, uplo, n, A, lda, ipiv,
                                                                   info);
}

} // extern C
