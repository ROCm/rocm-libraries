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

#include "rocauxiliary_org2l_ung2l.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_org2l_ung2l_impl(rocblas_handle handle,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int k,
                                          T* A,
                                          const rocblas_int lda,
                                          T* ipiv)
{
    const char* name = (!rocblas_is_complex<T> ? "org2l" : "ung2l");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "-k", k, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_org2l_orgql_argCheck(handle, m, n, k, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // extra requirements for calling LARF
    size_t size_Abyx;
    rocsolver_org2l_ung2l_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_Abyx,
                                                  &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_Abyx, size_workArr);

    // memory workspace allocation
    void *scalars, *Abyx, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_Abyx, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    Abyx = mem[1];
    workArr = mem[2];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_org2l_ung2l_template<T>(handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP,
                                             batch_count, (T*)scalars, (T*)Abyx, (T**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sorg2l(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver::rocsolver_org2l_ung2l_impl<float>(handle, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_dorg2l(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver::rocsolver_org2l_ung2l_impl<double>(handle, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_cung2l(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver::rocsolver_org2l_ung2l_impl<rocblas_float_complex>(handle, m, n, k, A, lda,
                                                                        ipiv);
}

rocblas_status rocsolver_zung2l(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver::rocsolver_org2l_ung2l_impl<rocblas_double_complex>(handle, m, n, k, A, lda,
                                                                         ipiv);
}

} // extern C
