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

#include "roclapack_syevj_heevj.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_syevj_heevj_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_esort esort,
                                                          const rocblas_evect evect,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const S abstol,
                                                          S* residual,
                                                          const rocblas_int max_sweeps,
                                                          rocblas_int* n_sweeps,
                                                          S* W,
                                                          const rocblas_stride strideW,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "syevj_strided_batched" : "heevj_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--esort", esort, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--strideA", strideA, "--abstol", abstol, "--max_sweeps", max_sweeps,
                        "--strideW", strideW, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevj_heevj_argCheck(
        handle, esort, evect, uplo, n, A, lda, residual, max_sweeps, n_sweeps, W, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // memory workspace sizes:
    // size of temporary workspace
    size_t size_Acpy, size_J, size_norms, size_top, size_bottom, size_completed;

    rocsolver_syevj_heevj_getMemorySize<false, T, S>(evect, uplo, n, batch_count, &size_Acpy,
                                                     &size_J, &size_norms, &size_top, &size_bottom,
                                                     &size_completed);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_Acpy, size_J, size_norms,
                                                      size_top, size_bottom, size_completed);

    // memory workspace allocation
    void *Acpy, *J, *norms, *top, *bottom, *completed;
    rocblas_device_malloc mem(handle, size_Acpy, size_J, size_norms, size_top, size_bottom,
                              size_completed);

    if(!mem)
        return rocblas_status_memory_error;

    Acpy = mem[0];
    J = mem[1];
    norms = mem[2];
    top = mem[3];
    bottom = mem[4];
    completed = mem[5];

    // execution
    return rocsolver_syevj_heevj_template<false, true, T>(
        handle, esort, evect, uplo, n, A, shiftA, lda, strideA, abstol, residual, max_sweeps,
        n_sweeps, W, strideW, info, batch_count, (T*)Acpy, (T*)J, (S*)norms, (rocblas_int*)top,
        (rocblas_int*)bottom, (rocblas_int*)completed);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevj_strided_batched(rocblas_handle handle,
                                                const rocblas_esort esort,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const float abstol,
                                                float* residual,
                                                const rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                float* W,
                                                const rocblas_stride strideW,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevj_heevj_strided_batched_impl<float>(
        handle, esort, evect, uplo, n, A, lda, strideA, abstol, residual, max_sweeps, n_sweeps, W,
        strideW, info, batch_count);
}

rocblas_status rocsolver_dsyevj_strided_batched(rocblas_handle handle,
                                                const rocblas_esort esort,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const double abstol,
                                                double* residual,
                                                const rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                double* W,
                                                const rocblas_stride strideW,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevj_heevj_strided_batched_impl<double>(
        handle, esort, evect, uplo, n, A, lda, strideA, abstol, residual, max_sweeps, n_sweeps, W,
        strideW, info, batch_count);
}

rocblas_status rocsolver_cheevj_strided_batched(rocblas_handle handle,
                                                const rocblas_esort esort,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const float abstol,
                                                float* residual,
                                                const rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                float* W,
                                                const rocblas_stride strideW,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevj_heevj_strided_batched_impl<rocblas_float_complex>(
        handle, esort, evect, uplo, n, A, lda, strideA, abstol, residual, max_sweeps, n_sweeps, W,
        strideW, info, batch_count);
}

rocblas_status rocsolver_zheevj_strided_batched(rocblas_handle handle,
                                                const rocblas_esort esort,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const double abstol,
                                                double* residual,
                                                const rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                double* W,
                                                const rocblas_stride strideW,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevj_heevj_strided_batched_impl<rocblas_double_complex>(
        handle, esort, evect, uplo, n, A, lda, strideA, abstol, residual, max_sweeps, n_sweeps, W,
        strideW, info, batch_count);
}

} // extern C
