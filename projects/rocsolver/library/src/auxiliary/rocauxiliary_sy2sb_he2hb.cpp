/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_sy2sb_he2hb.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_sy2sb_he2hb_impl(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nb,
                                          const rocblas_int k,
                                          U A,
                                          const rocblas_int lda,
                                          T* V,
                                          T* W)
{
    ROCSOLVER_ENTER_TOP("sy2sb_he2hb", "-n", n, "-nb", nb, "-k", k, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sy2sb_he2hb_argCheck(handle, n, nb, k, A, lda, V, W);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements
    size_t size_workT, size_workS1, size_workS2, size_workW;
    rocsolver_sy2sb_he2hb_getMemorySize<false, T>(n, nb, k, batch_count, &size_scalars, &size_workT,
                                        &size_workS1, &size_workS2, &size_workW, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_workT,
           size_workS1, size_workS2, size_workW, size_workArr);

    // memory workspace allocation
    void *scalars, *workT, *workS1, *workS2, *workW, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_workT,
       size_workS1, size_workS2, size_workW, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    workT = mem[1];
    workS1 = mem[2];
    workS2 = mem[3];
    workW = mem[4];
    workArr = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    rocblas_int ldv = n - nb;
    rocblas_int ldw = n - nb;
    return rocsolver_sy2sb_he2hb_template<false, false, T>(handle, n, nb, k, A, shiftA, lda, strideA, V, ldv,
                                             strideV, W, ldw, strideW, batch_count,
                                             (T*)scalars, (T*)workT, (T*)workS1, (T*)workS2, (T*)workW, (T**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_ssy2sb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nb,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* V,
                                float* W)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<float>(handle, n, nb, k, A, lda, V, W);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsy2sb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nb,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* V,
                                double* W)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<double>(handle, n, nb, k, A, lda, V, W);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_che2hb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nb,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* V,
                                rocblas_float_complex* W)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_float_complex>(handle, n, nb, k, A, lda, V, W);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhe2hb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int nb,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* V,
                                rocblas_double_complex* W)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_double_complex>(handle, n, nb, k, A, lda, V, W);
}

} // extern C

