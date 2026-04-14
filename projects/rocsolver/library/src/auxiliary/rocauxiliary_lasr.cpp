/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_lasr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename SS>
rocblas_status rocsolver_lasr_impl(rocblas_handle handle,
                                   const rocblas_side side,
                                   const rocblas_pivot pivot,
                                   const rocblas_direct direct,
                                   const rocblas_int m,
                                   const rocblas_int n,
                                   SS* C,
                                   SS* S,
                                   T* A,
                                   const rocblas_int lda)
{
    ROCSOLVER_ENTER_TOP("lasr", "--side", side, "--pivot", pivot, "--direct", direct, "-m", m, "-n",
                        n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lasr_argCheck(handle, side, pivot, direct, m, n, C, S, A, lda);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideC = 0;
    rocblas_stride strideS = 0;
    rocblas_stride strideA = 0;
    rocblas_int batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    //  execution
    return rocsolver_lasr_template<T>(handle, side, pivot, direct, m, n, C, strideC, S, strideS, A,
                                      shiftA, lda, strideA, batch_count);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slasr(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_pivot pivot,
                               const rocblas_direct direct,
                               const rocblas_int m,
                               const rocblas_int n,
                               float* C,
                               float* S,
                               float* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_lasr_impl<float>(handle, side, pivot, direct, m, n, C, S, A, lda);
}

rocblas_status rocsolver_dlasr(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_pivot pivot,
                               const rocblas_direct direct,
                               const rocblas_int m,
                               const rocblas_int n,
                               double* C,
                               double* S,
                               double* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_lasr_impl<double>(handle, side, pivot, direct, m, n, C, S, A, lda);
}

rocblas_status rocsolver_clasr(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_pivot pivot,
                               const rocblas_direct direct,
                               const rocblas_int m,
                               const rocblas_int n,
                               float* C,
                               float* S,
                               rocblas_float_complex* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_lasr_impl<rocblas_float_complex>(handle, side, pivot, direct, m, n,
                                                                 C, S, A, lda);
}

rocblas_status rocsolver_zlasr(rocblas_handle handle,
                               const rocblas_side side,
                               const rocblas_pivot pivot,
                               const rocblas_direct direct,
                               const rocblas_int m,
                               const rocblas_int n,
                               double* C,
                               double* S,
                               rocblas_double_complex* A,
                               const rocblas_int lda)
{
    return rocsolver::rocsolver_lasr_impl<rocblas_double_complex>(handle, side, pivot, direct, m, n,
                                                                  C, S, A, lda);
}

} // extern C
