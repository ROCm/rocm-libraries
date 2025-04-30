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
*  \brief Sparse triangular system solve using BSR storage format
*
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
*  \brief Sparse triangular system solve using BSR storage format
*
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
*              hipsparseSbsrsm2_analysis(), hipsparseDbsrsm2_analysis(),
*              hipsparseCbsrsm2_analysis(), hipsparseZbsrsm2_analysis(),
*              hipsparseSbsrsm2_solve(), hipsparseDbsrsm2_solve(),
*              hipsparseCbsrsm2_solve() and hipsparseZbsrsm2_solve().
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
*  "hipsparseXbsrsm2_solve()".
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
*  \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution matrix
*  \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
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
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if transX == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        X^T, & \text{if transX == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        X^H, & \text{if transX == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXbsrsm2_solve requires a user allocated temporary buffer. Its size is
*  returned by \ref hipsparseSbsrsm2_bufferSize "hipsparseXbsrsm2_bufferSize()". Furthermore, 
*  analysis meta data is required. It can be obtained by \ref hipsparseSbsrsm2_analysis 
*  "hipsparseXbsrsm2_analysis()". \p hipsparseXbsrsm2_solve reports the first zero pivot 
*  (either numerical or structural zero). The zero pivot status can be checked calling 
*  \ref hipsparseXbsrsm2_zeroPivot(). If \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, 
*  no zero pivot will be reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
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
*      // Number of right-hand-sides
*      int nrhs = 4;
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
*      hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*      hipsparseOperation_t transX = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*      // Solve policy
*      hipsparseSolvePolicy_t solve_policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;
*
*      // Scalar alpha and beta
*      double alpha = 1.0;
*
*      // rhs and solution matrix
*      int ldb = nb * bsr_dim;
*      int ldx = mb * bsr_dim;
*
*      double hB[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
*      double hX[16];
*
*      // Offload data to device
*      int* dbsrRowPtr;
*      int* dbsrColInd;
*      double*        dbsrVal;
*      double*        dB;
*      double*        dX;
*
*      hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*      hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*      hipMalloc((void**)&dbsrVal, sizeof(double) * nnzb * bsr_dim * bsr_dim);
*      hipMalloc((void**)&dB, sizeof(double) * nb * bsr_dim * nrhs);
*      hipMalloc((void**)&dX, sizeof(double) * mb * bsr_dim * nrhs);
*
*      hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice);
*      hipMemcpy(dbsrVal, hbsrVal, sizeof(double) * nnzb * bsr_dim * bsr_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dB, hB, sizeof(double) * nb * bsr_dim * nrhs, hipMemcpyHostToDevice);
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*
*      // Matrix fill mode
*      hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER);
*
*      // Matrix diagonal type
*      hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_NON_UNIT);
*
*      // Matrix info structure
*      bsrsm2Info_t info;
*      hipsparseCreateBsrsm2Info(&info);
*
*      // Obtain required buffer size
*      int buffer_size;
*      hipsparseDbsrsm2_bufferSize(handle,
*                                  dir,
*                                  transA,
*                                  transX,
*                                  mb,
*                                  nrhs,
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
*      hipsparseDbsrsm2_analysis(handle,
*                                dir,
*                                transA,
*                                transX,
*                                mb,
*                                nrhs,
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
*      hipsparseDbsrsm2_solve(handle,
*                             dir,
*                             transA,
*                             transX,
*                             mb,
*                             nrhs,
*                             nnzb,
*                             &alpha,
*                             descr,
*                             dbsrVal,
*                             dbsrRowPtr,
*                             dbsrColInd,
*                             bsr_dim,
*                             info,
*                             dB,
*                             ldb,
*                             dX,
*                             ldx,
*                             solve_policy,
*                             dbuffer);
*
*      // Check for zero pivots
*      int    pivot;
*      hipsparseStatus_t status = hipsparseXbsrsm2_zeroPivot(handle, info, &pivot);
*
*      if(status == HIPSPARSE_STATUS_ZERO_PIVOT)
*      {
*          std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
*      }
*
*      // Copy result back to host
*      hipMemcpy(hX, dX, sizeof(double) * mb * bsr_dim * nrhs, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyBsrsm2Info(info);
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dbsrRowPtr);
*      hipFree(dbsrColInd);
*      hipFree(dbsrVal);
*      hipFree(dB);
*      hipFree(dX);
*      hipFree(dbuffer);
*  \endcode
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
