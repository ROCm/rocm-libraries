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

#include "roclapack_syev_heev.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename W>
rocblas_status rocsolver_syev_heev_strided_batched_impl(rocblas_handle handle,
                                                        const rocblas_evect evect,
                                                        const rocblas_fill uplo,
                                                        const rocblas_int n,
                                                        W A,
                                                        const rocblas_int lda,
                                                        const rocblas_stride strideA,
                                                        S* D,
                                                        const rocblas_stride strideD,
                                                        S* E,
                                                        const rocblas_stride strideE,
                                                        rocblas_int* info,
                                                        const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "syev_strided_batched" : "heev_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--uplo", uplo, "-n", n, "--lda", lda, "--strideA",
                        strideA, "--strideD", strideD, "--strideE", strideE, "--batch_count",
                        batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_syev_heev_argCheck(handle, evect, uplo, n, A, lda, D, E, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace and stack
    size_t size_work_stack;
    // extra requirements to call SYTRD/HETRD and ORGTR/UNGTR
    size_t size_Abyx_norms_tmptr, size_tmptau_trfact;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_tau;

    rocsolver_syev_heev_getMemorySize<false, T, S>(evect, uplo, n, batch_count, &size_scalars,
                                                   &size_work_stack, &size_Abyx_norms_tmptr,
                                                   &size_tmptau_trfact, &size_tau, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_stack,
                                                      size_Abyx_norms_tmptr, size_tmptau_trfact,
                                                      size_tau, size_workArr);

    // memory workspace allocation
    void *scalars, *work_stack, *Abyx_norms_tmptr, *tmptau_trfact, *tau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work_stack, size_Abyx_norms_tmptr,
                              size_tmptau_trfact, size_tau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_stack = mem[1];
    Abyx_norms_tmptr = mem[2];
    tmptau_trfact = mem[3];
    tau = mem[4];
    workArr = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syev_heev_template<false, true, T>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, info, batch_count,
        (T*)scalars, work_stack, (T*)Abyx_norms_tmptr, (T*)tmptau_trfact, (T*)tau, (T**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyev_strided_batched(rocblas_handle handle,
                                               const rocblas_evect evect,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               float* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               float* D,
                                               const rocblas_stride strideD,
                                               float* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syev_heev_strided_batched_impl<float>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_dsyev_strided_batched(rocblas_handle handle,
                                               const rocblas_evect evect,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               double* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               double* D,
                                               const rocblas_stride strideD,
                                               double* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syev_heev_strided_batched_impl<double>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_cheev_strided_batched(rocblas_handle handle,
                                               const rocblas_evect evect,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               rocblas_float_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               float* D,
                                               const rocblas_stride strideD,
                                               float* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syev_heev_strided_batched_impl<rocblas_float_complex>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

rocblas_status rocsolver_zheev_strided_batched(rocblas_handle handle,
                                               const rocblas_evect evect,
                                               const rocblas_fill uplo,
                                               const rocblas_int n,
                                               rocblas_double_complex* A,
                                               const rocblas_int lda,
                                               const rocblas_stride strideA,
                                               double* D,
                                               const rocblas_stride strideD,
                                               double* E,
                                               const rocblas_stride strideE,
                                               rocblas_int* info,
                                               const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syev_heev_strided_batched_impl<rocblas_double_complex>(
        handle, evect, uplo, n, A, lda, strideA, D, strideD, E, strideE, info, batch_count);
}

} // extern C
