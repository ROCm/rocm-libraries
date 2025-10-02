/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef HIPSPARSE_GEMMI_H
#define HIPSPARSE_GEMMI_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Dense matrix sparse matrix multiplication using CSC storage format
*
*  \details
*  \p hipsparseXgemmi multiplies the scalar \f$\alpha\f$ with a dense column-oriented \f$m \times k\f$
*  matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSC
*  storage format and adds the result to the dense column-oriented \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot A \cdot B + \beta \cdot C
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \f$A\f$.
*  @param[in]
*  n           number of columns of the sparse CSC matrix \f$op(B)\f$ and \f$C\f$.
*  @param[in]
*  k           number of columns of the dense matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSC matrix \f$B\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  A           array of dimension \f$lda \times k\f$ (\f$op(A) == A\f$) or
*              \f$lda \times m\f$ (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*  @param[in]
*  lda         leading dimension of \f$A\f$, must be at least \f$m\f$
*              (\f$op(A) == A\f$) or \f$k\f$ (\f$op(A) == A^T\f$ or
*              \f$op(A) == A^H\f$).
*  @param[in]
*  cscValB     array of \p nnz elements of the sparse CSC matrix \f$B\f$.
*  @param[in]
*  cscColPtrB  array of \p n+1 elements that point to the start of every column of the
*              sparse CSC matrix \f$B\f$.
*  @param[in]
*  cscRowIndB  array of \p nnz elements containing the column indices of the sparse CSC
*              matrix \f$B\f$.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  C           array of dimension \f$ldc \times n\f$ that holds the values of \f$C\f$.
*  @param[in]
*  ldc         leading dimension of \f$C\f$, must be at least \f$m\f$.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnz,
*              \p lda, \p ldc, \p alpha, \p A, \p cscValB, \p cscColPtrB, \p cscRowIndB,
*              \p beta or \p C is invalid.
*
*  \par Example
*  \snippet example_hipsparse_gemmi.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const float*      alpha,
                                  const float*      A,
                                  int               lda,
                                  const float*      cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const float*      beta,
                                  float*            C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const double*     alpha,
                                  const double*     A,
                                  int               lda,
                                  const double*     cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const double*     beta,
                                  double*           C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const hipComplex* alpha,
                                  const hipComplex* A,
                                  int               lda,
                                  const hipComplex* cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const hipComplex* beta,
                                  hipComplex*       C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemmi(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  const hipDoubleComplex* cscValB,
                                  const int*              cscColPtrB,
                                  const int*              cscRowIndB,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       C,
                                  int                     ldc);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GEMMI_H */
