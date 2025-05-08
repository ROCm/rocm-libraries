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

#include "roclapack_sytd2_hetd2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          S* D,
                                                          const rocblas_stride strideD,
                                                          S* E,
                                                          const rocblas_stride strideE,
                                                          T* tau,
                                                          const rocblas_stride strideP,
                                                          const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sytd2_strided_batched" : "hetd2_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--uplo", uplo, "-n", n, "--lda", lda, "--strideA", strideA,
                        "--strideD", strideD, "--strideE", strideE, "--strideP", strideP,
                        "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sytd2_hetd2_argCheck(handle, uplo, n, lda, A, D, E, tau, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // extra requirements for calling LARFG
    size_t size_norms, size_work;
    // size of temporary householder scalars
    size_t size_tmptau;
    // size of array of pointers to workspace (batched case)
    size_t size_workArr;
    rocsolver_sytd2_hetd2_getMemorySize<false, T>(n, batch_count, &size_scalars, &size_work,
                                                  &size_norms, &size_tmptau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work, size_norms,
                                                      size_tmptau, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *norms, *tmptau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_norms, size_tmptau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    norms = mem[2];
    tmptau = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sytd2_hetd2_template<T>(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                             E, strideE, tau, strideP, batch_count, (T*)scalars,
                                             (T*)work, (T*)norms, (T*)tmptau, (T**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                float* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytd2_hetd2_strided_batched_impl<float>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_dsytd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                double* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytd2_hetd2_strided_batched_impl<double>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_chetd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                float* D,
                                                const rocblas_stride strideD,
                                                float* E,
                                                const rocblas_stride strideE,
                                                rocblas_float_complex* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytd2_hetd2_strided_batched_impl<rocblas_float_complex>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

rocblas_status rocsolver_zhetd2_strided_batched(rocblas_handle handle,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                double* D,
                                                const rocblas_stride strideD,
                                                double* E,
                                                const rocblas_stride strideE,
                                                rocblas_double_complex* tau,
                                                const rocblas_stride strideP,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sytd2_hetd2_strided_batched_impl<rocblas_double_complex>(
        handle, uplo, n, A, lda, strideA, D, strideD, E, strideE, tau, strideP, batch_count);
}

} // extern C
