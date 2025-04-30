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
#ifndef HIPSPARSE_CSRSM_H
#define HIPSPARSE_CSRSM_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during \ref hipsparseScsrsm2_analysis 
*  "hipsparseXcsrsm2_analysis()" or \ref hipsparseScsrsm2_solve "hipsparseXcsrsm2_solve()" 
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, 
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrsm2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseScsrsm2_analysis "hipsparseXcsrsm2_analysis()" 
*  and \ref hipsparseScsrsm2_solve "hipsparseXcsrsm2_solve()". The temporary storage buffer 
*  must be allocated by the user.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  algo        algorithm to use.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  transB      matrix \f$B\f$ operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  nrhs        number of columns of the dense matrix \f$op(B)\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix \f$A\f$.
*  @param[in]
*  B           array of \p m \f$\times\f$ \p nrhs elements of the rhs matrix \f$B\f$.
*  @param[in]
*  ldb         leading dimension of rhs matrix \f$B\f$.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*              hipsparseScsrsm2_analysis(), hipsparseDcsrsm2_analysis(),
*              hipsparseCcsrsm2_analysis(), hipsparseZcsrsm2_analysis(),
*              hipsparseScsrsm2_solve(), hipsparseDcsrsm2_solve(),
*              hipsparseCcsrsm2_solve() and hipsparseZcsrsm2_solve().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nrhs, \p nnz, \p alpha, 
*              \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p B, 
*              \p info or \p pBufferSizeInBytes is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
*              \p transB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const float*              alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const float*              B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const double*             alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const double*             B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipComplex*         alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipComplex*         B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipDoubleComplex*   alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipDoubleComplex*   B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_analysis performs the analysis step for \ref hipsparseScsrsm2_solve 
*  "hipsparseXcsrsm2_solve()".
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
*  algo        algorithm to use.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  transB      matrix \f$B\f$ operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  nrhs        number of columns of the dense matrix \f$op(B)\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix \f$A\f$.
*  @param[in]
*  B           array of \p m \f$\times\f$ \p nrhs elements of the rhs matrix \f$B\f$.
*  @param[in]
*  ldb         leading dimension of rhs matrix \f$B\f$.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nrhs, \p nnz, \p alpha, 
*              \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p B, 
*              \p info or \p pBuffer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
*              \p transB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const float*              alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const float*              B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const double*             alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const double*             B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipComplex*         alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipComplex*         B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipDoubleComplex*   alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipDoubleComplex*   B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution matrix
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
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        X^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        X^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXcsrsm2_solve requires a user allocated temporary buffer. Its size is
*  returned by \ref hipsparseScsrsm2_bufferSizeExt "hipsparseXcsrsm2_bufferSizeExt()". 
*  Furthermore, analysis meta data is required. It can be obtained by 
*  \ref hipsparseScsrsm2_analysis "hipsparseXcsrsm2_analysis()". \p hipsparseXcsrsm2_solve 
*  reports the first zero pivot (either numerical or structural zero). The zero pivot 
*  status can be checked calling \ref hipsparseXcsrsm2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p transA != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
*  \p transB != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  algo        algorithm to use.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  transB      matrix \f$B\f$ operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  nrhs        number of columns of the dense matrix \f$op(B)\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix \f$A\f$.
*  @param[inout]
*  B           array of \p m \f$\times\f$ \p nrhs elements of the rhs matrix \f$B\f$.
*  @param[in]
*  ldb         leading dimension of rhs matrix \f$B\f$.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nrhs, \p nnz, \p alpha, 
*              \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p B, 
*              \p info or \p pBuffer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
*              \p transB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
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
*
*      // Number of rows and columns
*      int m = 4;
*      int n = 4;
*
*      // Number of right-hand-sides
*      int nrhs = 4;
*
*      // Number of non-zeros
*      int nnz = 9;
*
*      // CSR row pointers
*      int hcsrRowPtr[5] = {0, 1, 3, 6, 9};
*
*      // CSR column indices
*      int hcsrColInd[9] = {0, 0, 1, 0, 1, 2, 0, 2, 3};
*
*      // CSR values
*      double hcsrVal[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
*
*      // Transposition of the matrix and rhs matrix
*      hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*      hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*      // Solve policy
*      hipsparseSolvePolicy_t solve_policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;
*
*      // Scalar alpha and beta
*      double alpha = 1.0;
*
*      // rhs and solution matrix
*      int ldb = n;
*
*      double hB[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
*
*      // Offload data to device
*      int* dcsrRowPtr;
*      int* dcsrColInd;
*      double*        dcsrVal;
*      double*        dB;
*
*      hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*      hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*      hipMalloc((void**)&dcsrVal, sizeof(double) * nnz);
*      hipMalloc((void**)&dB, sizeof(double) * n * nrhs);
*
*      hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dcsrVal, hcsrVal, sizeof(double) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dB, hB, sizeof(double) * n * nrhs, hipMemcpyHostToDevice);
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
*      csrsm2Info_t info;
*      hipsparseCreateCsrsm2Info(&info);
*
*      // Obtain required buffer size
*      size_t buffer_size;
*      hipsparseDcsrsm2_bufferSizeExt(handle,
*                                     0,
*                                     transA,
*                                     transB,
*                                     m,
*                                     nrhs,
*                                     nnz,
*                                     &alpha,
*                                     descr,
*                                     dcsrVal,
*                                     dcsrRowPtr,
*                                     dcsrColInd,
*                                     dB,
*                                     ldb,
*                                     info,
*                                     solve_policy,
*                                     &buffer_size);
*
*      // Allocate temporary buffer
*      void* dbuffer;
*      hipMalloc(&dbuffer, buffer_size);
*
*      // Perform analysis step
*      hipsparseDcsrsm2_analysis(handle,
*                                0,
*                                transA,
*                                transB,
*                                m,
*                                nrhs,
*                                nnz,
*                                &alpha,
*                                descr,
*                                dcsrVal,
*                                dcsrRowPtr,
*                                dcsrColInd,
*                                dB,
*                                ldb,
*                                info,
*                                solve_policy,
*                                dbuffer);
*
*      // Call dcsrsm to perform lower triangular solve LB = B
*      hipsparseDcsrsm2_solve(handle,
*                             0,
*                             transA,
*                             transB,
*                             m,
*                             nrhs,
*                             nnz,
*                             &alpha,
*                             descr,
*                             dcsrVal,
*                             dcsrRowPtr,
*                             dcsrColInd,
*                             dB,
*                             ldb,
*                             info,
*                             solve_policy,
*                             dbuffer);
*
*      // Check for zero pivots
*      int    pivot;
*      hipsparseStatus_t status = hipsparseXcsrsm2_zeroPivot(handle, info, &pivot);
*
*      if(status == HIPSPARSE_STATUS_ZERO_PIVOT)
*      {
*          std::cout << "Found zero pivot in matrix row " << pivot << std::endl;
*      }
*
*      // Copy result back to host
*      hipMemcpy(hB, dB, sizeof(double) * m * nrhs, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyCsrsm2Info(info);
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dcsrRowPtr);
*      hipFree(dcsrColInd);
*      hipFree(dcsrVal);
*      hipFree(dB);
*      hipFree(dbuffer);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         float*                    B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         double*                   B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipComplex*               B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipDoubleComplex*         B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRSM_H */
