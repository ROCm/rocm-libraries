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
#ifndef HIPSPARSE_BSRSM_H
#define HIPSPARSE_BSRSM_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level3_module
*  \details
*  \p hipsparseXbsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during \ref hipsparseSbsrsm2_analysis "hipsparseXbsrsm2_analysis()"
*  or \ref hipsparseSbsrsm2_solve "hipsparseXbsrsm2_solve()" computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXbsrsm2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level3_module
*  \details
*  \p hipsparseXbsrsm2_buffer_size returns the size of the temporary storage buffer in bytes
*  that is required by \ref hipsparseSbsrsm2_analysis "hipsparseXbsrsm2_analysis()" and
*  \ref hipsparseSbsrsm2_solve "hipsparseXbsrsm2_solve()". The temporary storage buffer must
*  be allocated by the user.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        matrix storage of BSR blocks.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  transX      matrix \f$X\f$ operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  nrhs        number of columns of the dense matrix \f$op(X)\f$.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnzb containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*              \ref hipsparseSbsrsm2_analysis "hipsparseXbsrsm2_analysis()" and
*              \ref hipsparseSbsrsm2_solve "hipsparseXbsrsm2_solve()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nrhs, \p nnzb, \p blockDim,
*              \p descrA, \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, \p info or
*              \p pBufferSizeInBytes is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
*              \p transX == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsm2_analysis performs the analysis step for \ref hipsparseSbsrsm2_solve
*  "hipsparseXbsrsm2_solve()". It is expected that this function will be executed only once
*  for a given matrix and particular operation type.
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        matrix storage of BSR blocks.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  transX      matrix \f$X\f$ operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  nrhs        number of columns of the dense matrix \f$op(X)\f$.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsrSortedColIndA array of \p nnzb containing the block column indices of the sparse
*              BSR matrix \f$A\f$.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix \f$A\f$.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nrhs, \p nnzb or
*              \p blockDim, \p descrA, \p bsrSortedValA, \p bsrSortedRowPtrA,
*              \p bsrSortedColIndA, \p info or \p pBuffer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
*              \p transX == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsm2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a column-oriented dense solution matrix
*  \f$X\f$ and the column-oriented dense right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$,
*  such that
*  \f[
*    op(A) \cdot op(X) = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if transA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if transA == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  ,
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transX == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transX == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transX == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if transX == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        X^T, & \text{if transX == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        X^H, & \text{if transX == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and where \f$m = blockDim \times mb\f$.
*
*  Note that as indicated above, the operation type of both \f$op(B)\f$ and \f$op(X)\f$ is specified by the
*  \p transX parameter and that the operation type of \f$B\f$ and \f$X\f$ must match. For example, if \f$op(B)=B\f$ then
*  \f$op(X)=X\f$. Likewise, if \f$op(B)=B^T\f$ then \f$op(X)=X^T\f$.
*
*  Given that the sparse matrix \f$A\f$ is a square matrix, its size is \f$m \times m\f$ regardless of
*  whether \f$A\f$ is transposed or not. The size of the column-oriented dense matrices \f$B\f$ and \f$X\f$ have
*  size that depends on the value of \p transX:
*
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        ldb \times nrhs, \text{  } ldb \ge m, & \text{if transX == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        ldb \times m, \text{  } ldb \ge nrhs,  & \text{if transX == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        ldb \times m, \text{  } ldb \ge nrhs, & \text{if transX == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        ldb \times nrhs, \text{  } ldb \ge m, & \text{if transX == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        ldb \times m, \text{  } ldb \ge nrhs,  & \text{if transX == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        ldb \times m, \text{  } ldb \ge nrhs, & \text{if transX == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXbsrsm2_solve requires a user allocated temporary buffer. Its size is returned by
*  \ref hipsparseSbsrsm2_bufferSize "hipsparseXbsrsm2_bufferSize()". The size of the required buffer is larger
*  when  \p transA equals \ref HIPSPARSE_OPERATION_TRANSPOSE or \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
*  when \p transX is \ref HIPSPARSE_OPERATION_NON_TRANSPOSE. The subsequent solve will also be faster when \f$A\f$ is
*  non-transposed and \f$B\f$ is transposed (or conjugate transposed). For example, instead of solving:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       a_{00} & a_{01} \\
*       a_{10} & a_{11}
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       a_{20} & a_{21} \\
*       a_{30} & a_{31}
*      \end{array} &
*      \begin{array}{c c}
*       a_{22} & a_{23} \\
*       a_{32} & a_{33}
*      \end{array} \\
*    \end{array}
*   \right]
*    \cdot
*    \begin{bmatrix}
*    x_{00} & x_{01} \\
*    x_{10} & x_{11} \\
*    x_{20} & x_{21} \\
*    x_{30} & x_{31} \\
*    \end{bmatrix}
*    =
*    \begin{bmatrix}
*    b_{00} & b_{01} \\
*    b_{10} & b_{11} \\
*    b_{20} & b_{21} \\
*    b_{30} & b_{31} \\
*    \end{bmatrix}
*  \f]
*
*  Consider solving:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       a_{00} & a_{01} \\
*       a_{10} & a_{11}
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       a_{20} & a_{21} \\
*       a_{30} & a_{31}
*      \end{array} &
*      \begin{array}{c c}
*       a_{22} & a_{23} \\
*       a_{32} & a_{33}
*      \end{array} \\
*    \end{array}
*   \right]
*    \cdot
*    \begin{bmatrix}
*    x_{00} & x_{10} & x_{20} & x_{30} \\
*    x_{01} & x_{11} & x_{21} & x_{31}
*    \end{bmatrix}^{T}
*    =
*    \begin{bmatrix}
*    b_{00} & b_{10} & b_{20} & b_{30} \\
*    b_{01} & b_{11} & b_{21} & b_{31}
*    \end{bmatrix}^{T}
*  \f]
*
*  Once the temporary storage buffer has been allocated, analysis meta data is required. It can be obtained
*  by hipsparseSbsrsm2_analysis "hipsparseXbsrsm2_analysis()". The triangular solve is completed by calling
*  \p hipsparseXbsrsm2_solve and once all solves are performed, the temporary storage buffer allocated by the
*  user can be freed.
*
*  Solving a triangular system involves inverting the diagonal blocks. This means that if the sparse matrix is
*  missing the diagonal block (referred to as a structural zero) or the diagonal block is not invertible (referred
*  to as a numerical zero) then a solution is not possible. \p hipsparseXbsrsm2_solve tracks the location of the first
*  zero pivot (either numerical or structural zero). The zero pivot status can be checked calling \ref hipsparseXbsrsm2_zeroPivot().
*  If \ref hipsparseXbsrsm2_zeroPivot() returns \ref HIPSPARSE_STATUS_SUCCESS, then no zero pivot was found and therefore
*  the matrix does not have a structural or numerical zero.
*
*  The user can specify that the sparse matrix should be interpreted as having identity blocks on the diagonal by setting the diagonal
*  type on the descriptor \p descrA to \ref HIPSPARSE_DIAG_TYPE_UNIT using \ref hipsparseSetMatDiagType. If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be reported, even if the diagonal block \f$A_{j,j}\f$
*  for some \f$j\f$ is not invertible.
*
*  The sparse CSR matrix passed to \p hipsparseXbsrsm2_solve does not actually have to be a triangular matrix. Instead the
*  triangular upper or lower part of the sparse matrix is solved based on \ref hipsparseFillMode_t set on the descriptor
*  \p descrA. If the fill mode is set to \ref HIPSPARSE_FILL_MODE_LOWER, then the lower triangular matrix is solved. If the
*  fill mode is set to \ref HIPSPARSE_FILL_MODE_UPPER then the upper triangular matrix is solved.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  Operation type of B and X must match, if \f$op(B)=B, op(X)=X\f$.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p transA != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
*  \p transX != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  dirA             matrix storage of BSR blocks.
*  @param[in]
*  transA           matrix \f$A\f$ operation type.
*  @param[in]
*  transX           matrix \f$X\f$ operation type.
*  @param[in]
*  mb               number of block rows of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  nrhs             number of columns of the dense matrix \f$op(X)\f$.
*  @param[in]
*  nnzb             number of non-zero blocks of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  descrA           descriptor of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsrSortedValA    array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*                   the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnzb containing the block column indices of the sparse
*                   BSR matrix.
*  @param[in]
*  blockDim         block dimension of the sparse BSR matrix.
*  @param[in]
*  info             structure that holds the information collected during the analysis step.
*  @param[in]
*  B                rhs matrix B with leading dimension \p ldb.
*  @param[in]
*  ldb              leading dimension of rhs matrix \f$B\f$.
*  @param[out]
*  X                solution matrix X with leading dimension \p ldx.
*  @param[in]
*  ldx              leading dimension of solution matrix \f$X\f$.
*  @param[in]
*  policy           \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer          temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nrhs, \p nnzb, \p blockDim,
*              \p alpha, \p descrA, \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA,
*              \p B, \p X \p info or \p pBuffer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
*              \p transX == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_bsrsm.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const float*              B,
                                         int                       ldb,
                                         float*                    X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const double*             B,
                                         int                       ldb,
                                         double*                   X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipComplex*         B,
                                         int                       ldb,
                                         hipComplex*               X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipDoubleComplex*   B,
                                         int                       ldb,
                                         hipDoubleComplex*         X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSRSM_H */
