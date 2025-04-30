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

#include "roclapack_sygvj_hegvj.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvj_hegvj_impl(rocblas_handle handle,
                                          const rocblas_eform itype,
                                          const rocblas_evect evect,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          U A,
                                          const rocblas_int lda,
                                          U B,
                                          const rocblas_int ldb,
                                          const S abstol,
                                          S* residual,
                                          const rocblas_int max_sweeps,
                                          rocblas_int* n_sweeps,
                                          S* W,
                                          rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvj" : "hegvj");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--ldb", ldb, "--abstol", abstol, "--max_sweeps", max_sweeps);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sygvj_hegvj_argCheck(handle, itype, evect, uplo, n, A, lda, B,
                                                       ldb, residual, max_sweeps, n_sweeps, W, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (and for calling POTRF, TRSM, TRMM, SYGST/HEGST, SYEVJ/HEEVJ)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4, size_work5, size_work6;
    // size of temporary info array
    size_t size_iinfo;
    rocsolver_sygvj_hegvj_getMemorySize<false, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_work5, &size_work6, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_work5,
                                                      size_work6, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *work5, *work6, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_work5, size_work6, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    work5 = mem[5];
    work6 = mem[6];
    iinfo = mem[7];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvj_hegvj_template<false, false, T>(
        handle, itype, evect, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, abstol,
        residual, max_sweeps, n_sweeps, W, strideW, info, batch_count, (T*)scalars, work1, work2,
        work3, work4, work5, work6, (rocblas_int*)iinfo, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* B,
                                const rocblas_int ldb,
                                const float abstol,
                                float* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                float* W,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sygvj_hegvj_impl<float>(handle, itype, evect, uplo, n, A, lda, B,
                                                        ldb, abstol, residual, max_sweeps, n_sweeps,
                                                        W, info);
}

rocblas_status rocsolver_dsygvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* B,
                                const rocblas_int ldb,
                                const double abstol,
                                double* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                double* W,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sygvj_hegvj_impl<double>(handle, itype, evect, uplo, n, A, lda, B,
                                                         ldb, abstol, residual, max_sweeps,
                                                         n_sweeps, W, info);
}

rocblas_status rocsolver_chegvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* B,
                                const rocblas_int ldb,
                                const float abstol,
                                float* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                float* W,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sygvj_hegvj_impl<rocblas_float_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, max_sweeps, n_sweeps, W,
        info);
}

rocblas_status rocsolver_zhegvj(rocblas_handle handle,
                                const rocblas_eform itype,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* B,
                                const rocblas_int ldb,
                                const double abstol,
                                double* residual,
                                const rocblas_int max_sweeps,
                                rocblas_int* n_sweeps,
                                double* W,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sygvj_hegvj_impl<rocblas_double_complex>(
        handle, itype, evect, uplo, n, A, lda, B, ldb, abstol, residual, max_sweeps, n_sweeps, W,
        info);
}

} // extern C
