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
#ifndef HIPSPARSE_BSRSV_H
#define HIPSPARSE_BSRSV_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \details
*  \p hipsparseXbsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during \ref hipsparseSbsrsv2_analysis 
*  "hipsparseXbsrsv2_analysis()" or \ref hipsparseSbsrsv2_solve "hipsparseXbsrsv2_solve()" 
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, 
*  using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXbsrsv2_zeroPivot is a blocking function. It might influence
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
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \details
*  \p hipsparseXbsrsv2_bufferSize returns the size of the temporary storage buffer in bytes 
*  that is required by \ref hipsparseSbsrsv2_analysis "hipsparseXbsrsv2_analysis()" and 
*  \ref hipsparseSbsrsv2_solve "hipsparseXbsrsv2_solve()". The temporary storage buffer must 
*  be allocated by the user.
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
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*              \ref hipsparseSbsrsv2_analysis "hipsparseXbsrsv2_analysis()" and
*              \ref hipsparseSbsrsv2_solve "hipsparseXbsrsv2_solve()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb or \p blockDim, 
*              \p descr, \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, 
*              \p info or \p pBufferSizeInBytes is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup level2_module
*  \details
*  \p hipsparseXbsrsv2_bufferSizeExt returns the size of the temporary storage buffer in bytes 
*  that is required by \ref hipsparseSbsrsv2_analysis "hipsparseXbsrsv2_analysis()" and 
*  \ref hipsparseSbsrsv2_solve "hipsparseXbsrsv2_solve()". The temporary storage buffer must be 
*  allocated by the user.
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
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*              \ref hipsparseSbsrsv2_analysis "hipsparseXbsrsv2_analysis()" and
*              \ref hipsparseSbsrsv2_solve "hipsparseXbsrsv2_solve()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb or \p blockDim, 
*              \p descr, \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, 
*              \p info or \p pBufferSizeInBytes is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \details
*  \p hipsparseXbsrsv2_analysis performs the analysis step for \ref hipsparseSbsrsv2_solve 
*  "hipsparseXbsrsv2_solve()". It is expected that this function will be executed only once 
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
*  transA      matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix.
*  @param[out]
*  info        structure that holds the information collected during
*              the analysis step.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, 
*              \p descrA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, \p info or 
*              \p pBuffer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  Performing the above operation requires three steps. First, the user calls 
*  \ref hipsparseSbsrsv2_bufferSize "hipsparseXbsrsv2_bufferSize()" which will determine the size of the required 
*  temporary storage buffer. The user then allocates this buffer and calls 
*  \ref hipsparseSbsrsv2_analysis "hipsparseXbsrsv2_analysis()" which will perform analysis on the sparse matrix 
*  \f$op(A)\f$. Finally, the user completes the computation by calling \p hipsparseXbsrsv2_solve. The buffer size, 
*  buffer allocation, and analysis only need to be called once for a given sparse matrix \f$op(A)\f$ while the 
*  computation stage can be repeatedly used with different \f$x\f$ and \f$y\f$ vectors. Once all calls to 
*  \p hipsparseXbsrsv2_solve are complete, the temporary buffer can be deallocated. 
*
*  Solving a triangular system involves inverting the diagonal blocks. This means that if the sparse matrix is 
*  missing the diagonal block (referred to as a structural zero) or the diagonal block is not invertible (referred 
*  to as a numerical zero) then a solution is not possible. \p hipsparseXbsrsv2_solve tracks the location of the first 
*  zero pivot (either numerical or structural zero). The zero pivot status can be checked calling \ref hipsparseXbsrsv2_zeroPivot(). 
*  If \ref hipsparseXbsrsv2_zeroPivot() returns \ref HIPSPARSE_STATUS_SUCCESS, then no zero pivot was found and therefore 
*  the matrix does not have a structural or numerical zero.
*
*  The user can specify that the sparse matrix should be interpreted as having identity blocks on the diagonal by setting the diagonal 
*  type on the descriptor \p descrA to \ref HIPSPARSE_DIAG_TYPE_UNIT using \ref hipsparseSetMatDiagType. If 
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be reported, even if the diagonal block \f$A_{j,j}\f$ 
*  for some \f$j\f$ is not invertible.
*
*  The sparse CSR matrix passed to \p hipsparseXbsrsv2_solve does not actually have to be a triangular matrix. Instead the 
*  triangular upper or lower part of the sparse matrix is solved based on \ref hipsparseFillMode_t set on the descriptor 
*  \p descrA. If the fill mode is set to \ref HIPSPARSE_FILL_MODE_LOWER, then the lower triangular matrix is solved. If the 
*  fill mode is set to \ref HIPSPARSE_FILL_MODE_UPPER then the upper triangular matrix is solved.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
*  \p transA == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
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
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix.
*  @param[in]
*  bsrSortedValA array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrSortedColIndA array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  f           array of \p m elements, holding the right-hand side.
*  @param[out]
*  x           array of \p m elements, holding the solution.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, 
*              \p descrA, \p alpha, \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, 
*              \p f or \p x is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // A = ( 1.0  0.0  0.0  0.0 )
*      //     ( 2.0  3.0  0.0  0.0 )
*      //     ( 4.0  5.0  6.0  0.0 )
*      //     ( 7.0  0.0  8.0  9.0 )
*      //
*      // with bsr_dim = 2
*      //
*      //      -------------------
*      //   = | 1.0 0.0 | 0.0 0.0 |
*      //     | 2.0 3.0 | 0.0 0.0 |
*      //      -------------------
*      //     | 4.0 5.0 | 6.0 0.0 |
*      //     | 7.0 0.0 | 8.0 9.0 |
*      //      -------------------
*
*      // Number of rows and columns
*      int m = 4;
*
*      // Number of block rows and block columns
*      int mb = 2;
*      int nb = 2;
*
*      // BSR block dimension
*      int bsr_dim = 2;
*
*      // Number of non-zero blocks
*      int nnzb = 3;
*
*      // BSR row pointers
*      int hbsrRowPtr[3] = {0, 1, 3};
*
*      // BSR column indices
*      int hbsrColInd[3] = {0, 0, 1};
*
*      // BSR values
*      double hbsrVal[12] = {1.0, 2.0, 0.0, 3.0, 4.0, 7.0, 5.0, 0.0, 6.0, 8.0, 0.0, 9.0};
*
*      // Storage scheme of the BSR blocks
*      hipsparseDirection_t dir = HIPSPARSE_DIRECTION_COLUMN;
*
*      // Transposition of the matrix and rhs matrix
*      hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*      // Solve policy
*      hipsparseSolvePolicy_t solve_policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*
*      double hx[4] = {1, 2, 3, 4};
*      double hy[4];
*
*      // Offload data to device
*      int* dbsrRowPtr;
*      int* dbsrColInd;
*      double* dbsrVal;
*      double* dx;
*      double* dy;
*
*      hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*      hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*      hipMalloc((void**)&dbsrVal, sizeof(double) * nnzb * bsr_dim * bsr_dim);
*      hipMalloc((void**)&dx, sizeof(double) * nb * bsr_dim);
*      hipMalloc((void**)&dy, sizeof(double) * mb * bsr_dim);
*
*      hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice);
*      hipMemcpy(dbsrVal, hbsrVal, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * nb * bsr_dim, hipMemcpyHostToDevice);
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*
*      // Matrix fill mode
*      hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER);
*
*      // Matrix diagonal type
*      hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT);
*
*      // Matrix info structure
*      bsrsv2Info_t info;
*      hipsparseCreateBsrsv2Info(&info);
*
*      // Obtain required buffer size
*      int buffer_size;
*      hipsparseDbsrsv2_bufferSize(handle,
*                                  dir,
*                                  trans,
*                                  mb,
*                                  nnzb,
*                                  descr,
*                                  dbsrVal,
*                                  dbsrRowPtr,
*                                  dbsrColInd,
*                                  bsr_dim,
*                                  info,
*                                  &buffer_size);
*
*      // Allocate temporary buffer
*      void* dbuffer;
*      hipMalloc(&dbuffer, buffer_size);
*
*      // Perform analysis step
*      hipsparseDbsrsv2_analysis(handle,
*                                dir,
*                                trans,
*                                mb,
*                                nnzb,
*                                descr,
*                                dbsrVal,
*                                dbsrRowPtr,
*                                dbsrColInd,
*                                bsr_dim,
*                                info,
*                                solve_policy,
*                                dbuffer);
*
*      // Call dbsrsm to perform lower triangular solve LX = B
*      hipsparseDbsrsv2_solve(handle,
*                             dir,
*                             trans,
*                             mb,
*                             nnzb,
*                             &alpha,
*                             descr,
*                             dbsrVal,
*                             dbsrRowPtr,
*                             dbsrColInd,
*                             bsr_dim,
*                             info,
*                             dx,
*                             dy,
*                             solve_policy,
*                             dbuffer);
*
*      // Check for zero pivots
*      int    pivot;
*      hipsparseStatus_t status = hipsparseXbsrsv2_zeroPivot(handle, info, &pivot);
*
*      if(status == HIPSPARSE_STATUS_ZERO_PIVOT)
*      {
*          std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
*      }
*
*      // Copy results back to the host
*      hipMemcpy(hy, dy, sizeof(double) * mb * bsr_dim, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyBsrsv2Info(info);
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dbsrRowPtr);
*      hipFree(dbsrColInd);
*      hipFree(dbsrVal);
*      hipFree(dx);
*      hipFree(dy);
*      hipFree(dbuffer);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSRSV_H */
