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

#include "roclapack_geblttrs_npvt.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_geblttrs_npvt_impl(rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            const rocblas_int nrhs,
                                            U A,
                                            const rocblas_int lda,
                                            U B,
                                            const rocblas_int ldb,
                                            U C,
                                            const rocblas_int ldc,
                                            U X,
                                            const rocblas_int ldx)
{
    ROCSOLVER_ENTER_TOP("geblttrs_npvt", "--nb", nb, "--nblocks", nblocks, "--nrhs", nrhs, "--lda",
                        lda, "--ldb", ldb, "--ldc", ldc, "--ldx", ldx);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_geblttrs_npvt_argCheck(handle, nb, nblocks, nrhs, lda, ldb, ldc,
                                                         ldx, A, B, C, X);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftC = 0;
    rocblas_int shiftX = 0;

    // normal (non-batched non-strided) execution
    rocblas_int inca = 1;
    rocblas_int incb = 1;
    rocblas_int incc = 1;
    rocblas_int incx = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideC = 0;
    rocblas_stride strideX = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // requirements for calling GETRS
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;

    rocsolver_geblttrs_npvt_getMemorySize<false, false, T>(nb, nblocks, nrhs, batch_count,
                                                           &size_work1, &size_work2, &size_work3,
                                                           &size_work4, &optim_mem);

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
    return rocsolver_geblttrs_npvt_template<false, false, T>(
        handle, nb, nblocks, nrhs, A, shiftA, inca, lda, strideA, B, shiftB, incb, ldb, strideB, C,
        shiftC, incc, ldc, strideC, X, shiftX, incx, ldx, strideX, batch_count, work1, work2, work3,
        work4, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        float* A,
                                        const rocblas_int lda,
                                        float* B,
                                        const rocblas_int ldb,
                                        float* C,
                                        const rocblas_int ldc,
                                        float* X,
                                        const rocblas_int ldx)
{
    return rocsolver::rocsolver_geblttrs_npvt_impl<float>(handle, nb, nblocks, nrhs, A, lda, B, ldb,
                                                          C, ldc, X, ldx);
}

rocblas_status rocsolver_dgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        double* A,
                                        const rocblas_int lda,
                                        double* B,
                                        const rocblas_int ldb,
                                        double* C,
                                        const rocblas_int ldc,
                                        double* X,
                                        const rocblas_int ldx)
{
    return rocsolver::rocsolver_geblttrs_npvt_impl<double>(handle, nb, nblocks, nrhs, A, lda, B,
                                                           ldb, C, ldc, X, ldx);
}

rocblas_status rocsolver_cgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        rocblas_float_complex* A,
                                        const rocblas_int lda,
                                        rocblas_float_complex* B,
                                        const rocblas_int ldb,
                                        rocblas_float_complex* C,
                                        const rocblas_int ldc,
                                        rocblas_float_complex* X,
                                        const rocblas_int ldx)
{
    return rocsolver::rocsolver_geblttrs_npvt_impl<rocblas_float_complex>(
        handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx);
}

rocblas_status rocsolver_zgeblttrs_npvt(rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        rocblas_double_complex* A,
                                        const rocblas_int lda,
                                        rocblas_double_complex* B,
                                        const rocblas_int ldb,
                                        rocblas_double_complex* C,
                                        const rocblas_int ldc,
                                        rocblas_double_complex* X,
                                        const rocblas_int ldx)
{
    return rocsolver::rocsolver_geblttrs_npvt_impl<rocblas_double_complex>(
        handle, nb, nblocks, nrhs, A, lda, B, ldb, C, ldc, X, ldx);
}

} // extern C
