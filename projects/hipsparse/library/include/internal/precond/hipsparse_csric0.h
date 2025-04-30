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
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
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
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_bufferSize returns the size of the temporary storage buffer in bytes
*  that is required by \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" and 
*  \ref hipsparseScsric02 "hipsparseXcsric02()".
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
*                     hipsparseScsric02_analysis(), hipsparseDcsric02_analysis(),
*                     hipsparseCcsric02_analysis(), hipsparseZcsric02_analysis(),
*                     hipsparseScsric02(), hipsparseDcsric02(), hipsparseCcsric02()
*                     and hipsparseZcsric02().
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
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" 
*  and \ref hipsparseScsric02 "hipsparseXcsric02()".
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
*                     hipsparseScsric02_analysis(), hipsparseDcsric02_analysis(),
*                     hipsparseCcsric02_analysis(), hipsparseZcsric02_analysis(),
*                     hipsparseScsric02(), hipsparseDcsric02(), hipsparseCcsric02()
*                     and hipsparseZcsric02().
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
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
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
*
*  \p hipsparseXcsric02 requires a user allocated temporary buffer. Its size is returned
*  by \ref hipsparseScsric02_bufferSize "hipsparseXcsric02_bufferSize()" or 
*  \ref hipsparseScsric02_bufferSizeExt "hipsparseXcsric02_bufferSizeExt()". Furthermore,
*  analysis meta data is required. It can be obtained by \ref hipsparseScsric02_analysis 
*  "hipsparseXcsric02_analysis()". \p hipsparseXcsric02 reports the first zero pivot 
*  (either numerical or structural zero). The zero pivot status can be obtained by calling 
*  \ref hipsparseXcsric02_zeroPivot().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
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
