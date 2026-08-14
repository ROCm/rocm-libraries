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

#include "rocauxiliary_lahr2.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_lahr2_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    const rocblas_int nb,
                                    T* A,
                                    const rocblas_int lda,
                                    T* tau,
                                    T* F,
                                    const rocblas_int ldf,
                                    T* Y,
                                    const rocblas_int ldy)
try
{
    ROCSOLVER_ENTER_TOP("lahr2", "-n", n, "-k", k, "--nb", nb, "--lda", lda, "--ldt", ldf, "--ldy",
                        ldy);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lahr2_argCheck(handle, n, k, nb, lda, ldf, ldy, A, tau, F, Y);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftY = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideT = 0;
    rocblas_stride strideF = 0;
    rocblas_stride strideY = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling LARFG
    size_t size_norms;
    // dedicated w vector buffer for update step (separate from F to avoid aliasing)
    size_t size_work_vec;
    // one scalar beta per batch instance
    size_t size_beta;
    rocsolver_lahr2_getMemorySize<false, T>(n, k, nb, batch_count, &size_work_workArr, &size_norms,
                                            &size_work_vec, &size_beta);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_workArr, size_norms,
                                                      size_work_vec, size_beta);

    // memory workspace allocation
    void *work_workArr, *norms, *work_vec, *beta;
    rocblas_device_malloc mem(handle, size_work_workArr, size_norms, size_work_vec, size_beta);

    if(!mem)
        return rocblas_status_memory_error;

    work_workArr = mem[0];
    norms = mem[1];
    work_vec = mem[2];
    beta = mem[3];

    // execution
    return rocsolver_lahr2_template<T>(handle, n, k, nb, A, shiftA, lda, strideA, tau, strideT, F,
                                       ldf, strideF, Y, shiftY, ldy, strideY, batch_count,
                                       work_workArr, (T*)norms, (T*)work_vec, (T*)beta);
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

/*! @{
    \brief LAHR2 reduces the first ``nb`` columns of a general n-by-(n-k+1) matrix A
    so that elements below the k-th subdiagonal are zero. This is an auxiliary routine
    used in the blocked reduction to upper Hessenberg form (GEHRD).

    \details
    The reduced form is given by:

    \f[
        B = Q^H  A  Q
    \f]

    where the elements of B below the k-th subdiagonal are zero and Q is an orthogonal
    matrix represented as the product of Householder matrices

    \f[
        \begin{array}{cl}
        Q = H(nb-1)H(nb-2)\cdots H(1)
        \end{array}
    \f]

    The reduction is performed by ``nb`` Householder reflectors. Each Householder
    matrix \f$H(i)\f$ is given by

    \f[
        H(i) = I - \text{tau}[i] \cdot v_i^{} v_i^H
    \f]

    where \f$\text{tau}[i]\f$ is the corresponding Householder scalar and the first \f$k+i\f$
    elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[k+i] = 1\f$.

    LAHR2 returns the triangular factor ``T`` that is upper triangular where

    \f[
        H = I - VTV^H
    \f]

    and matrix ``Y`` that is given by

    \f[
        Y = A V T
    \f]

    where the \f$i\f$th column of matrix ``V`` contains the Householder vector associated with \f$H(i)\f$.
    Together, \f$(V, T, Y)\f$ allow the unreduced trailing part of A to be updated via rank-nb operations
    instead of nb individual rank-1 updates.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    n           rocblas_int. n >= 0.
                The order of the matrix A.
    @param[in]
    k           rocblas_int. 1 <= k < n.
                The offset (1-based column index) at which the reduction begins.
    @param[in]
    nb          rocblas_int. 1 <= nb < n - k + 1.
                The number of columns to reduce.
    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.
                On entry, the n-by-(n-k+1) matrix to be reduced.
                On exit, the first nb Householder vectors are stored in the
                lower triangle of the submatrix A(k:n-1, k:k+nb-1),
                and the rest of A is updated accordingly.
    @param[in]
    lda         rocblas_int. lda >= max(1, n).
                The leading dimension of A.
    @param[out]
    tau         pointer to type. Array of nb scalars on the GPU.
                The vector of all the Householder scalars.
    @param[out]
    T           pointer to type. Array on the GPU of dimension ldt*nb.
                The upper triangular factor.
    @param[in]
    ldt         rocblas_int. ldt >= nb.
                The leading dimension of T.
    @param[out]
    Y           pointer to type. Array on the GPU of dimension ldy*nb.
                The n-by-nb matrix \f$Y = A V T\f$.
    @param[in]
    ldy         rocblas_int. ldy >= n.
                The leading dimension of Y.
    ********************************************************************/
ROCSOLVER_EXPORT rocblas_status rocsolver_slahr2(rocblas_handle handle,
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
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_lahr2_impl<float>(handle, n, k, nb, A, lda, tau, T, ldt, Y, ldy);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlahr2(rocblas_handle handle,
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
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_lahr2_impl<double>(handle, n, k, nb, A, lda, tau, T, ldt, Y, ldy);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_clahr2(rocblas_handle handle,
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
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_lahr2_impl<rocblas_float_complex>(handle, n, k, nb, A, lda, tau, T,
                                                                  ldt, Y, ldy);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlahr2(rocblas_handle handle,
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
#if defined(ROCSOLVER_ENABLE_HESSENBERG)
    return rocsolver::rocsolver_lahr2_impl<rocblas_double_complex>(handle, n, k, nb, A, lda, tau, T,
                                                                   ldt, Y, ldy);
#else
    return rocblas_status_not_implemented;
#endif
}
//! @}

} // extern C
