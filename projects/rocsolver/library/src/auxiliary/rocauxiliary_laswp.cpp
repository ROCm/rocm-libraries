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

#include "rocauxiliary_laswp.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_laswp_impl(rocblas_handle handle,
                                    const I n,
                                    U A,
                                    const I lda,
                                    const I k1,
                                    const I k2,
                                    const I* ipiv,
                                    const I incp)
{
    ROCSOLVER_ENTER_TOP("laswp", "-n", n, "--lda", lda, "--k1", k1, "--k2", k2);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_laswp_argCheck(handle, n, lda, k1, k2, A, ipiv, incp);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftP = 0;

    // normal (non-batched non-strided) execution
    I inca = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    I batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_laswp_template<T>(handle, n, A, shiftA, inca, lda, strideA, k1, k2, ipiv,
                                       shiftP, incp, strideP, batch_count);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slaswp(rocblas_handle handle,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver::rocsolver_laswp_impl<float>(handle, n, A, lda, k1, k2, ipiv, incp);
}

rocblas_status rocsolver_dlaswp(rocblas_handle handle,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver::rocsolver_laswp_impl<double>(handle, n, A, lda, k1, k2, ipiv, incp);
}

rocblas_status rocsolver_claswp(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver::rocsolver_laswp_impl<rocblas_float_complex>(handle, n, A, lda, k1, k2, ipiv,
                                                                  incp);
}

rocblas_status rocsolver_zlaswp(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const rocblas_int k1,
                                const rocblas_int k2,
                                const rocblas_int* ipiv,
                                const rocblas_int incp)
{
    return rocsolver::rocsolver_laswp_impl<rocblas_double_complex>(handle, n, A, lda, k1, k2, ipiv,
                                                                   incp);
}

} // extern C
