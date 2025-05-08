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

#include "rocauxiliary_labrd.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_labrd_impl(rocblas_handle handle,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int lda,
                                    S* D,
                                    S* E,
                                    T* tauq,
                                    T* taup,
                                    U X,
                                    const rocblas_int ldx,
                                    U Y,
                                    const rocblas_int ldy)
{
    ROCSOLVER_ENTER_TOP("labrd", "-m", m, "-n", n, "-k", k, "--lda", lda, "--ldx", ldx, "--ldy", ldy);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_labrd_argCheck(handle, m, n, k, lda, ldx, ldy, A, D, E, tauq, taup, X, Y);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftX = 0;
    rocblas_int shiftY = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideY = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideQ = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARFG
    size_t size_norms;
    rocsolver_labrd_getMemorySize<false, T>(m, n, k, batch_count, &size_scalars, &size_work_workArr,
                                            &size_norms);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_norms);

    // memory workspace allocation
    void *scalars, *work_workArr, *norms;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_norms);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    norms = mem[2];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_labrd_template<T>(handle, m, n, k, A, shiftA, lda, strideA, D, strideD, E,
                                       strideE, tauq, strideQ, taup, strideP, X, shiftX, ldx,
                                       strideX, Y, shiftY, ldy, strideY, batch_count, (T*)scalars,
                                       work_workArr, (T*)norms);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                float* tauq,
                                float* taup,
                                float* X,
                                const rocblas_int ldx,
                                float* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_labrd_impl<float>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx,
                                                  Y, ldy);
}

rocblas_status rocsolver_dlabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                double* tauq,
                                double* taup,
                                double* X,
                                const rocblas_int ldx,
                                double* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_labrd_impl<double>(handle, m, n, k, A, lda, D, E, tauq, taup, X,
                                                   ldx, Y, ldy);
}

rocblas_status rocsolver_clabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_float_complex* tauq,
                                rocblas_float_complex* taup,
                                rocblas_float_complex* X,
                                const rocblas_int ldx,
                                rocblas_float_complex* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_labrd_impl<rocblas_float_complex>(handle, m, n, k, A, lda, D, E,
                                                                  tauq, taup, X, ldx, Y, ldy);
}

rocblas_status rocsolver_zlabrd(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_double_complex* tauq,
                                rocblas_double_complex* taup,
                                rocblas_double_complex* X,
                                const rocblas_int ldx,
                                rocblas_double_complex* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_labrd_impl<rocblas_double_complex>(handle, m, n, k, A, lda, D, E,
                                                                   tauq, taup, X, ldx, Y, ldy);
}

} // extern C
