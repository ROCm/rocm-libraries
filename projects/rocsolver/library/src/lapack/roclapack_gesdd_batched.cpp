/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_gesdd.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename SS, typename W>
rocblas_status rocsolver_gesdd_batched_impl(rocblas_handle handle,
                                            const rocblas_svect left_svect,
                                            const rocblas_svect right_svect,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            W A,
                                            const rocblas_int lda,
                                            SS* S,
                                            const rocblas_stride strideS,
                                            T* U,
                                            const rocblas_int ldu,
                                            const rocblas_stride strideU,
                                            T* V,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            rocblas_int* info,
                                            const rocblas_int batch_count)
{
    ROCSOLVER_ENTER_TOP("gesdd_batched", "--left_svect", left_svect, "--right_svect", right_svect,
                        "-m", m, "-n", n, "--lda", lda, "--strideS", strideS, "--ldu", ldu,
                        "--strideU", strideU, "--ldv", ldv, "--strideV", strideV, "--batch_count",
                        batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesdd_argCheck(handle, left_svect, right_svect, m, n, A, lda, S,
                                                 U, ldu, V, ldv, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size for temporary matrix storage
    size_t size_VUtmp;
    // extra requirements for calling SYEVJ/HEEVJ, GEQRF, ORGQR/UNGQR, GELQF, ORGLQ/UNGLQ
    size_t size_UVtmpZ, size_work1, size_work2, size_work3, size_work4, size_work5_ipiv,
        size_splits, size_tmptau_W, size_tau, size_workArr, size_workArr2;

    rocsolver_gesdd_getMemorySize<true, T, SS>(
        handle, left_svect, right_svect, m, n, strideS, batch_count, &size_VUtmp, &size_UVtmpZ,
        &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4, &size_work5_ipiv,
        &size_splits, &size_tmptau_W, &size_tau, &size_workArr, &size_workArr2);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_VUtmp, size_UVtmpZ, size_scalars,
                                                      size_work1, size_work2, size_work3, size_work4,
                                                      size_work5_ipiv, size_splits, size_tmptau_W,
                                                      size_tau, size_workArr, size_workArr2);

    // memory workspace allocation
    void *scalars, *VUtmp, *UVtmpZ, *work1, *work2, *work3, *work4, *work5_ipiv, *splits, *tmptau_W,
        *tau, *workArr, *workArr2;
    rocblas_device_malloc mem(handle, size_VUtmp, size_UVtmpZ, size_scalars, size_work1, size_work2,
                              size_work3, size_work4, size_work5_ipiv, size_splits, size_tmptau_W,
                              size_tau, size_workArr, size_workArr, size_workArr2);

    if(!mem)
        return rocblas_status_memory_error;

    VUtmp = mem[0];
    UVtmpZ = mem[1];
    scalars = mem[2];
    work1 = mem[3];
    work2 = mem[4];
    work3 = mem[5];
    work4 = mem[6];
    work5_ipiv = mem[7];
    splits = mem[8];
    tmptau_W = mem[9];
    tau = mem[10];
    workArr = mem[11];
    workArr2 = mem[12];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesdd_template<true, false, T>(
        handle, left_svect, right_svect, m, n, A, shiftA, lda, strideA, S, strideS, U, ldu, strideU,
        V, ldv, strideV, info, batch_count, (T*)VUtmp, UVtmpZ, (T*)scalars, work1, work2, work3,
        work4, work5_ipiv, splits, tmptau_W, tau, workArr, workArr2);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgesdd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* S,
                                        const rocblas_stride strideS,
                                        float* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        float* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesdd_batched_impl<float>(handle, left_svect, right_svect, m, n, A,
                                                          lda, S, strideS, U, ldu, strideU, V, ldv,
                                                          strideV, info, batch_count);
}

rocblas_status rocsolver_dgesdd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* S,
                                        const rocblas_stride strideS,
                                        double* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        double* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesdd_batched_impl<double>(handle, left_svect, right_svect, m, n, A,
                                                           lda, S, strideS, U, ldu, strideU, V, ldv,
                                                           strideV, info, batch_count);
}

rocblas_status rocsolver_cgesdd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        float* S,
                                        const rocblas_stride strideS,
                                        rocblas_float_complex* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        rocblas_float_complex* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesdd_batched_impl<rocblas_float_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, strideS, U, ldu, strideU, V, ldv, strideV,
        info, batch_count);
}

rocblas_status rocsolver_zgesdd_batched(rocblas_handle handle,
                                        const rocblas_svect left_svect,
                                        const rocblas_svect right_svect,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        double* S,
                                        const rocblas_stride strideS,
                                        rocblas_double_complex* U,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        rocblas_double_complex* V,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesdd_batched_impl<rocblas_double_complex>(
        handle, left_svect, right_svect, m, n, A, lda, S, strideS, U, ldu, strideU, V, ldv, strideV,
        info, batch_count);
}

} // extern C
