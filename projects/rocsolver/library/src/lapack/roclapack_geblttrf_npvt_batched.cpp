/* **************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_geblttrf_npvt.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_geblttrf_npvt_batched_impl(rocblas_handle handle,
                                                    const rocblas_int nb,
                                                    const rocblas_int nblocks,
                                                    U A,
                                                    const rocblas_int lda,
                                                    U B,
                                                    const rocblas_int ldb,
                                                    U C,
                                                    const rocblas_int ldc,
                                                    rocblas_int* info,
                                                    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("geblttrf_npvt_batched", "--nb", nb, "--nblocks", nblocks, "--lda", lda,
                        "--ldb", ldb, "--ldc", ldc, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_geblttrf_npvt_argCheck(handle, nb, nblocks, lda, ldb, ldc, A, B,
                                                         C, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftC = 0;

    // batched execution
    rocblas_int inca = 1;
    rocblas_int incb = 1;
    rocblas_int incc = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideC = 0;

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_geblttrf_npvt_getMemorySize<true, false, T>(
        handle, nb, nblocks, A, shiftA, inca, lda, strideA, B, shiftB, incb, ldb, strideB, C,
        shiftC, incc, ldc, strideC, info, batch_count, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());

    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    // Execution
    return rocsolver_geblttrf_npvt_template<true, false, T>(
        handle, nb, nblocks, A, shiftA, inca, lda, strideA, B, shiftB, incb, ldb, strideB, C,
        shiftC, incc, ldc, strideC, info, batch_count, &work_helper);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgeblttrf_npvt_batched(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                float* const A[],
                                                const rocblas_int lda,
                                                float* const B[],
                                                const rocblas_int ldb,
                                                float* const C[],
                                                const rocblas_int ldc,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_geblttrf_npvt_batched_impl<float>(handle, nb, nblocks, A, lda, B,
                                                                  ldb, C, ldc, info, batch_count);
}

rocblas_status rocsolver_dgeblttrf_npvt_batched(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                double* const A[],
                                                const rocblas_int lda,
                                                double* const B[],
                                                const rocblas_int ldb,
                                                double* const C[],
                                                const rocblas_int ldc,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_geblttrf_npvt_batched_impl<double>(handle, nb, nblocks, A, lda, B,
                                                                   ldb, C, ldc, info, batch_count);
}

rocblas_status rocsolver_cgeblttrf_npvt_batched(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                rocblas_float_complex* const A[],
                                                const rocblas_int lda,
                                                rocblas_float_complex* const B[],
                                                const rocblas_int ldb,
                                                rocblas_float_complex* const C[],
                                                const rocblas_int ldc,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_geblttrf_npvt_batched_impl<rocblas_float_complex>(
        handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, batch_count);
}

rocblas_status rocsolver_zgeblttrf_npvt_batched(rocblas_handle handle,
                                                const rocblas_int nb,
                                                const rocblas_int nblocks,
                                                rocblas_double_complex* const A[],
                                                const rocblas_int lda,
                                                rocblas_double_complex* const B[],
                                                const rocblas_int ldb,
                                                rocblas_double_complex* const C[],
                                                const rocblas_int ldc,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_geblttrf_npvt_batched_impl<rocblas_double_complex>(
        handle, nb, nblocks, A, lda, B, ldb, C, ldc, info, batch_count);
}

} // extern C
