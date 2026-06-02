/* **************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_lahr2.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_lahr2_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    const rocblas_int nb,
                                    T* A,
                                    const rocblas_int lda,
                                    T* tau,
                                    T* Tmat,
                                    const rocblas_int ldt,
                                    T* Y,
                                    const rocblas_int ldy)
{
    ROCSOLVER_ENTER_TOP("lahr2", "-n", n, "-k", k, "--nb", nb, "--lda", lda, "--ldt", ldt, "--ldy",
                        ldy);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lahr2_argCheck(handle, n, k, nb, lda, ldt, ldy, A, tau, Tmat, Y);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftY = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideT = 0;
    rocblas_stride strideN = 0;
    rocblas_stride strideY = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARFG and EI value storage
    size_t size_norms;
    // dedicated w vector buffer for update step (separate from Tmat to avoid aliasing)
    size_t size_work_vec;
    rocsolver_lahr2_getMemorySize<false, T>(n, k, nb, batch_count, &size_scalars,
                                            &size_work_workArr, &size_norms, &size_work_vec);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work_workArr,
                                                      size_norms, size_work_vec);

    // memory workspace allocation
    void *scalars, *work_workArr, *norms, *work_vec;
    rocblas_device_malloc mem(handle, size_scalars, size_work_workArr, size_norms, size_work_vec);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_workArr = mem[1];
    norms = mem[2];
    work_vec = mem[3];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_lahr2_template<T>(handle, n, k, nb, A, shiftA, lda, strideA, tau, strideT,
                                       Tmat, ldt, strideN, Y, shiftY, ldy, strideY, batch_count,
                                       (T*)scalars, work_workArr, (T*)norms, (T*)work_vec);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slahr2(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int k,
                                const rocblas_int nb,
                                float* A,
                                const rocblas_int lda,
                                float* tau,
                                float* T,
                                const rocblas_int ldt,
                                float* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_lahr2_impl<float>(handle, n, k, nb, A, lda, tau, T, ldt, Y, ldy);
}

rocblas_status rocsolver_dlahr2(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int k,
                                const rocblas_int nb,
                                double* A,
                                const rocblas_int lda,
                                double* tau,
                                double* T,
                                const rocblas_int ldt,
                                double* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_lahr2_impl<double>(handle, n, k, nb, A, lda, tau, T, ldt, Y, ldy);
}

rocblas_status rocsolver_clahr2(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int k,
                                const rocblas_int nb,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* tau,
                                rocblas_float_complex* T,
                                const rocblas_int ldt,
                                rocblas_float_complex* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_lahr2_impl<rocblas_float_complex>(handle, n, k, nb, A, lda, tau, T,
                                                                  ldt, Y, ldy);
}

rocblas_status rocsolver_zlahr2(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int k,
                                const rocblas_int nb,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* tau,
                                rocblas_double_complex* T,
                                const rocblas_int ldt,
                                rocblas_double_complex* Y,
                                const rocblas_int ldy)
{
    return rocsolver::rocsolver_lahr2_impl<rocblas_double_complex>(handle, n, k, nb, A, lda, tau, T,
                                                                   ldt, Y, ldy);
}

} // extern C
