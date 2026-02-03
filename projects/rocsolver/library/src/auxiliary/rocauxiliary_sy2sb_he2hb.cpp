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
                                          const rocblas_int kd,
                                          const rocblas_int nb,
                                          U A,
                                          const rocblas_int lda,
                                          T* Aband,
                                          const rocblas_int ldab,
                                          T* tau)
{
    ROCSOLVER_ENTER_TOP("sy2sb_he2hb", "-n", n, "-nb", nb, "-k", k, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sy2sb_he2hb_argCheck(
            handle, n, kd, nb, A, lda, Aband, ldab, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements
    size_t size_D, size_V, size_W, size_X, size_Z, size_work;
    rocsolver_sy2sb_he2hb_getMemorySize<false, T>(
            n, kd, nb, batch_count,
            &size_scalars, &size_D, &size_V, &size_W, &size_X, &size_Z,
            &size_work, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
                handle, size_scalars, size_D, size_V, size_W, size_X, size_Z,
                size_work, size_workArr);

    // memory workspace allocation
    rocblas_device_malloc mem(
            handle, size_scalars, size_D, size_V, size_W, size_X, size_Z,
            size_work, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    T* scalars = (T*) mem[0];
    T* D       = (T*) mem[1];
    T* V       = (T*) mem[2];
    T* W       = (T*) mem[3];
    T* X       = (T*) mem[4];
    T* Z       = (T*) mem[5];
    T* work    = (T*) mem[6];
    T** workArr = (T**) mem[7];
    if(size_scalars > 0)
        init_scalars(handle, scalars);

    // TODO: do D, V, W, X, Z need strides?
    // execution
    rocblas_int ldd = nb;
    rocblas_int ldv = n;
    rocblas_int ldw = n;
    rocblas_int ldx = n;
    rocblas_int ldz = n;
    return rocsolver_sy2sb_he2hb_template<false, false, T>(
            handle, n, kd, nb,
            A, shiftA, lda, strideA,
            tau,
            Aband, ldab, strideAb,
            D, ldd,
            V, ldv,
            W, ldw,
            X, ldx,
            Z, ldz, batch_count,
            scalars, V, ldv, W, ldw, X, ldx, Z, ldz, work, workArr);
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
                                const rocblas_int kd,
                                const rocblas_int nb,
                                float* A,
                                const rocblas_int lda,
                                float* tau,
                                float* Aband,
                                const rocblas_int lda)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<float>(
            handle, n, kd, nb, A, lda, tau, Aband, ldab);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsy2sb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int kd,
                                const rocblas_int nb,
                                double* A,
                                const rocblas_int lda,
                                double* tau,
                                double* Aband,
                                const rocblas_int lda)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<double>(
            handle, n, kd, nb, A, lda, tau, Aband, ldab);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_che2hb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int kd,
                                const rocblas_int nb,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* tau,
                                rocblas_float_complex* Aband,
                                const rocblas_int lda)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_float_complex>(
            handle, n, kd, nb, A, lda, tau, Aband, ldab);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhe2hb(rocblas_handle handle,
                                const rocblas_int n,
                                const rocblas_int kd,
                                const rocblas_int nb,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* tau,
                                rocblas_double_complex* Aband,
                                const rocblas_int lda)
{
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_double_complex>(
            handle, n, kd, nb, A, lda, tau, Aband, ldab);
}

} // extern C
