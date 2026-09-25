/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_laset.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    laset is not currently included in the public API.
 *    It is used in he2hb, hb2st, and the unmtr_hb2st tester.
 *    It has broad utility, so may move to the public API later.
 * ===========================================================================
 */

template <typename T>
rocblas_status rocsolver_laset_impl(rocblas_handle handle,
                                    const rocblas_fill uplo,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const T offdiag,
                                    const T diag,
                                    T* A,
                                    const rocblas_int lda)
try
{
    ROCSOLVER_ENTER_TOP("laset", "--uplo", uplo, "-m", m, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_laset_argCheck(handle, uplo, m, n, lda, A);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_int batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_laset_template<T>(handle, uplo, m, n, offdiag, diag, // opts
                                       A, shiftA, lda, strideA, // A
                                       batch_count);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slaset(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const float offdiag,
                                                 const float diag,
                                                 float* A,
                                                 const rocblas_int lda)
{
    return rocsolver::rocsolver_laset_impl<float>(handle, uplo, m, n, offdiag, diag, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlaset(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const double offdiag,
                                                 const double diag,
                                                 double* A,
                                                 const rocblas_int lda)
{
    return rocsolver::rocsolver_laset_impl<double>(handle, uplo, m, n, offdiag, diag, A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_claset(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_float_complex offdiag,
                                                 const rocblas_float_complex diag,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda)
{
    return rocsolver::rocsolver_laset_impl<rocblas_float_complex>(handle, uplo, m, n, offdiag, diag,
                                                                  A, lda);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlaset(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_double_complex offdiag,
                                                 const rocblas_double_complex diag,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda)
{
    return rocsolver::rocsolver_laset_impl<rocblas_double_complex>(handle, uplo, m, n, offdiag,
                                                                   diag, A, lda);
}

} // extern C
