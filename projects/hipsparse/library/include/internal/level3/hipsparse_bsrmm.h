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
#ifndef HIPSPARSE_BSRMM_H
#define HIPSPARSE_BSRMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using BSR storage format
 *
 *  \details
 *  \p hipsparseXbsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in BSR storage format, and the column-oriented dense \f$k \times n\f$
 *  matrix \f$B\f$ and adds the result to the column-oriented dense \f$m \times n\f$ matrix \f$C\f$ that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if transA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and where \f$k = blockDim \times kb\f$ and \f$m = blockDim \times mb\f$.
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
 *  dirA        the storage format of the blocks. Can be \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  transA      matrix \f$A\f$ operation type. Currently, only \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
 *  @param[in]
 *  transB      matrix \f$B\f$ operation type. Currently, only \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and \ref HIPSPARSE_OPERATION_TRANSPOSE
 *              are supported.
 *  @param[in]
 *  mb          number of block rows of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
 *  @param[in]
 *  kb          number of block columns of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  nnzb        number of non-zero blocks of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descrA      descriptor of the sparse BSR matrix \f$A\f$. Currently, only
 *              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  bsrValA     array of \p nnzb*blockDim*blockDim elements of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse
 *              BSR matrix \f$A\f$.
 *  @param[in]
 *  blockDim    size of the blocks in the sparse BSR matrix.
 *  @param[in]
 *  B           array of dimension \p ldb*n (\f$op(B) == B\f$),
 *              \p ldb*k otherwise.
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$ (\f$ op(B) == B\f$) where \p k=blockDim*kb,
 *              \f$\max{(1, n)}\f$ otherwise.
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           array of dimension \p ldc*n.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$ (\f$ op(A) == A\f$) where \p m=blockDim*mb,
 *              \f$\max{(1, k)}\f$ where \p k=blockDim*kb otherwise.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p n, \p kb, \p nnzb, \p ldb,
 *              \p ldc, \p descr, \p alpha, \p bsrValA, \p bsrRowPtrA, \p bsrColIndA,
 *              \p B, \p beta or \p C is invalid.
 *  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
 *              \p transB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 *
 *  \par Example
 *  \snippet example_hipsparse_bsrmm.cpp doc example
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSRMM_H */
