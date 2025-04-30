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

#include "rocauxiliary_ormbr_unmbr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_ormbr_unmbr_impl(rocblas_handle handle,
                                          const rocblas_storev storev,
                                          const rocblas_side side,
                                          const rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int k,
                                          T* A,
                                          const rocblas_int lda,
                                          T* ipiv,
                                          T* C,
                                          const rocblas_int ldc)
{
    const char* name = (!rocblas_is_complex<T> ? "ormbr" : "unmbr");
    ROCSOLVER_ENTER_TOP(name, "--storev", storev, "--side", side, "--trans", trans, "-m", m, "-n",
                        n, "-k", k, "--lda", lda, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_ormbr_argCheck<COMPLEX>(handle, storev, side, trans, m, n, k, lda,
                                                          ldc, A, C, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_int strideA = 0;
    rocblas_int strideP = 0;
    rocblas_int strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // requirements for calling ORMQR/UNMQR or ORMLQ/UNMLQ
    size_t size_scalars;
    size_t size_AbyxORwork, size_diagORtmptr;
    size_t size_trfact;
    size_t size_workArr;
    rocsolver_ormbr_unmbr_getMemorySize<false, T>(storev, side, m, n, k, batch_count, &size_scalars,
                                                  &size_AbyxORwork, &size_diagORtmptr, &size_trfact,
                                                  &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_AbyxORwork,
                                                      size_diagORtmptr, size_trfact, size_workArr);

    // memory workspace allocation
    void *scalars, *AbyxORwork, *diagORtmptr, *trfact, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_AbyxORwork, size_diagORtmptr, size_trfact,
                              size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    AbyxORwork = mem[1];
    diagORtmptr = mem[2];
    trfact = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_ormbr_unmbr_template<false, false, T>(
        handle, storev, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc,
        strideC, batch_count, (T*)scalars, (T*)AbyxORwork, (T*)diagORtmptr, (T*)trfact, (T**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sormbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv,
                                float* C,
                                const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormbr_unmbr_impl<float>(handle, storev, side, trans, m, n, k, A,
                                                        lda, ipiv, C, ldc);
}

rocblas_status rocsolver_dormbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv,
                                double* C,
                                const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormbr_unmbr_impl<double>(handle, storev, side, trans, m, n, k, A,
                                                         lda, ipiv, C, ldc);
}

rocblas_status rocsolver_cunmbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv,
                                rocblas_float_complex* C,
                                const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormbr_unmbr_impl<rocblas_float_complex>(
        handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

rocblas_status rocsolver_zunmbr(rocblas_handle handle,
                                const rocblas_storev storev,
                                const rocblas_side side,
                                const rocblas_operation trans,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv,
                                rocblas_double_complex* C,
                                const rocblas_int ldc)
{
    return rocsolver::rocsolver_ormbr_unmbr_impl<rocblas_double_complex>(
        handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

} // extern C
