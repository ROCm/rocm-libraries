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
#ifndef HIPSPARSE_CSRIC0_H
#define HIPSPARSE_CSRIC0_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \details
*  \p hipsparseXcsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during \ref hipsparseScsric02_analysis
*  "hipsparseXcsric02_analysis()" or \ref hipsparseScsric02 "hipsparseXcsric02()"
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsric02_zeroPivot is a blocking function. It might influence
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
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position pointer is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \details
*  \p hipsparseXcsric02_bufferSize returns the size of the temporary storage buffer in bytes
*  that is required by \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" and
*  \ref hipsparseScsric02 "hipsparseXcsric02()". The temporary storage buffer must be allocated
*  by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[out]
*  info               structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" and
*                     \ref hipsparseScsric02 "hipsparseXcsric02()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA,
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes pointer is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup precond_module
*  \details
*  \p hipsparseXcsric02_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()"
*  and \ref hipsparseScsric02 "hipsparseXcsric02()". The temporary storage buffer must be
*  allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[out]
*  info               structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" and
*                     \ref hipsparseScsric02 "hipsparseXcsric02()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA,
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes pointer is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \details
*  \p hipsparseXcsric02_analysis performs the analysis step for \ref hipsparseScsric02
*  "hipsparseXcsric02()".
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz              number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*                   CSR matrix.
*  @param[out]
*  info             structure that holds the information collected during
*                   the analysis step.
*  @param[in]
*  policy           \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer          temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA,
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02 computes the incomplete Cholesky factorization with 0 fill-ins
*  and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LL^T
*  \f]
*  where the lower triangular matrix \f$L\f$ is computed using:
*  \f[
*    L_{ij} = \left\{
*    \begin{array}{ll}
*        \sqrt{A_{jj} - \sum_{k=0}^{j-1}(L_{jk})^{2}},   & \text{if i == j} \\
*        \frac{1}{L_{jj}}(A_{jj} - \sum_{k=0}^{j-1}L_{ik} \times L_{jk}), & \text{if i > j}
*    \end{array}
*    \right.
*  \f]
*  for each entry found in the CSR matrix \f$A\f$.
*
*  Computing the above incomplete Cholesky factorization requires three steps to complete. First,
*  the user determines the size of the required temporary storage buffer by calling
*  \ref hipsparseScsric02_bufferSize "hipsparseXcsric02_bufferSize()". Once this buffer size has been determined,
*  the user allocates the buffer and passes it to \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()".
*  This will perform analysis on the sparsity pattern of the matrix. Finally, the user calls \p hipsparseScsric02,
*  \p hipsparseDcsric02, \p hipsparseCcsric02, or \p hipsparseZcsric02 to perform the actual factorization. The calculation
*  of the buffer size and the analysis of the sparse matrix only need to be performed once for a given sparsity pattern
*  while the factorization can be repeatedly applied to multiple matrices having the same sparsity pattern. Once all calls
*  to \ref hipsparseScsric02 "hipsparseXcsric02()" are complete, the temporary buffer can be deallocated.
*
*  When computing the Cholesky factorization, it is possible that \f$L_{jj} == 0\f$ which would result in a division by zero.
*  This could occur from either \f$A_{jj}\f$ not existing in the sparse CSR matrix (referred to as a structural zero) or because
*  \f$A_{jj} - \sum_{k=0}^{j-1}(L_{jk})^{2} == 0\f$ (referred to as a numerical zero). For example, running the Cholesky
*  factorization on the following matrix:
*  \f[
*    \begin{bmatrix}
*    2 & 1 & 0 \\
*    1 & 2 & 1 \\
*    0 & 1 & 2
*    \end{bmatrix}
*  \f]
*  results in a successful Cholesky factorization, however running with the matrix:
*  \f[
*    \begin{bmatrix}
*    2 & 1 & 0 \\
*    1 & 1/2 & 1 \\
*    0 & 1 & 2
*    \end{bmatrix}
*  \f]
*  results in a numerical zero because:
*  \f[
*    \begin{array}{ll}
*        L_{00} &= \sqrt{2} \\
*        L_{10} &= \frac{1}{\sqrt{2}} \\
*        L_{11} &= \sqrt{\frac{1}{2} - (\frac{1}{\sqrt{2}})^2}
*               &= 0
*    \end{array}
*  \f]
*  The user can detect the presence of a structural zero by calling \ref hipsparseXcsric02_zeroPivot() after
*  \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" and/or the presence of a structural or
*  numerical zero by calling \ref hipsparseXcsric02_zeroPivot() after \ref hipsparseScsric02 "hipsparseXcsric02()":
*  \code{.c}
*  hipsparseDcsric02(handle,
*                    m,
*                    nnz,
*                    descrM,
*                    csrVal,
*                    csrRowPtr,
*                    csrColInd,
*                    info,
*                    HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
*                    buffer);
*
*  // Check for zero pivot
*  if(CUSPARSE_STATUS_ZERO_PIVOT == hipsparseXcsric02_zeroPivot(handle, info, &position))
*  {
*      printf("L has structural and/or numerical zero at L(%d,%d)\n", position, position);
*  }
*  \endcode
*  In both cases, \ref hipsparseXcsric02_zeroPivot() will report the first zero pivot (either numerical or structural)
*  found. See full example below. The user can also set the diagonal type to be \f$1\f$ using \ref hipsparseSetMatDiagType()
*  which will interpret the matrix \f$A\f$ as having ones on its diagonal (even if no nonzero exists in the sparsity pattern).
*
*  \p hipsparseXcsric02 computes the Cholesky factorization inplace meaning that the values array \p csrSortedValA_valM of the \f$A\f$
*  matrix is overwritten with the \f$L\f$ matrix stored in the lower triangular part of \f$A\f$:
*
*  \f[
*    \begin{align}
*    \begin{bmatrix}
*    a_{00} & a_{01} & a_{02} \\
*    a_{10} & a_{11} & a_{12} \\
*    a_{20} & a_{21} & a_{22}
*    \end{bmatrix}
*    \rightarrow
*    \begin{bmatrix}
*    l_{00} & a_{01} & a_{02} \\
*    l_{10} & l_{11} & a_{12} \\
*    l_{20} & l_{21} & l_{22}
*    \end{bmatrix}
*    \end{align}
*  \f]
*  The row pointer array \p csrSortedRowPtrA and the column indices array \p csrSortedColIndA remain the same for \f$A\f$ and the
*  output as the incomplete factorization does not generate new nonzeros in the output which do not already exist in \f$A\f$.
*
*  The performance of computing Cholesky factorization with hipSPARSE greatly depends on the sparisty pattern
*  the the matrix \f$A\f$ as this is what determines the amount of parallelism available.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  \ref hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[inout]
*  csrSortedValA_valM array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start
*                     of every row of the sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[in]
*  info               structure that holds the information collected during the analysis step.
*  @param[in]
*  policy             \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer            temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA_valM,
*              \p csrSortedRowPtrA or \p csrSortedColIndA pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_csric02.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRIC0_H */
