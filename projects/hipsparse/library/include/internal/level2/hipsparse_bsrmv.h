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
#ifndef HIPSPARSE_BSRMV_H
#define HIPSPARSE_BSRMV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p hipsparseXbsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$m \times n\f$ matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and where \f$m = mb \times blockDim\f$ and \f$n= nb \times blockDim\f$.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        matrix storage of BSR blocks.
*  @param[in]
*  transA      matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim     block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb*blockDim elements (\f$op(A) = A\f$) or \p mb*blockDim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*blockDim elements (\f$op(A) = A\f$) or \p nb*blockDim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb,
*              \p blockDim, \p descr, \p alpha, \p bsrSortedValA, \p bsrSortedRowPtrA,
*              \p bsrSortedColIndA, \p x, \p beta or \p y is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p trans != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_bsrmv.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSRMV_H */
