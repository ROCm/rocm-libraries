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
#ifndef HIPSPARSE_CSRGEAM_H
#define HIPSPARSE_CSRGEAM_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \details
*  \p hipsparseXcsrgeamNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix \f$C\f$. It is assumed that \p csrRowPtrC has been allocated with
*  size \p m+1. The desired index base in the output CSR matrix is set in the
*  \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase().
*
*  For full code example, see \ref hipsparseScsrgeam().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$. \p nnzTotalDevHostPtr can be a host or device pointer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB, \p descrA,
*          \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, \p csrColIndB, \p descrC,
*          \p csrRowPtrC or \p nnzTotalDevHostPtr is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t         handle,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  This computation involves a multi step process. First the user must allocate \p csrRowPtrC
*  to have size \p m+1. The user then calls \ref hipsparseXcsrgeamNnz which fills in the \p csrRowPtrC
*  array as well as computes the total number of nonzeros in \f$C\f$, \p nnzC. The user then allocates both
*  arrays \p csrColIndC and \p csrValC to have size \p nnzC and calls \p hipsparseXcsrgeam to complete
*  the computation. The desired index base in the output CSR matrix \f$C\f$ is set in the
*  \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrValA         array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrValB         array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC         array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrColIndC      array of elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB,
*          \p alpha, \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p beta,
*          \p descrB, \p csrValB, \p csrRowPtrB, \p csrColIndB, \p descrC, \p csrValC,
*          \p csrRowPtrC or \p csrColIndC is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_csrgeam.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const float*              alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const float*              beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const double*             alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const double*             beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipComplex*         alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipComplex*         beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipDoubleComplex*   alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipDoubleComplex*   beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

/*! \ingroup extra_module
*  \details
*  \p hipsparseXcsrgeam2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseXcsrgeam2Nnz() and \ref hipsparseScsrgeam2
*  "hipsparseXcsrgeam2()". The temporary storage buffer must be allocated by the user.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha              scalar \f$\alpha\f$.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA               number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA      array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA   array of \p nnzA elements containing the column indices of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta               scalar \f$\beta\f$.
*  @param[in]
*  descrB             descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB               number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedValB      array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedRowPtrB   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedColIndB   array of \p nnzB elements containing the column indices of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC             descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrSortedValC      array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrSortedRowPtrC   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrSortedColIndC   array of elements containing the column indices of the
*                     sparse CSR matrix \f$C\f$.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseXcsrgeam2Nnz() and \ref hipsparseScsrgeam2 "hipsparseXcsrgeam2()".
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB,
*          \p alpha, \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA,
*          \p beta, \p descrB, \p csrSortedValB, \p csrSortedRowPtrB, \p csrSortedColIndB,
*          \p descrC, \p csrSortedValC, \p csrSortedRowPtrC, \p csrSortedColIndC, or
*          \p pBufferSizeInBytes is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const float*              csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const float*              csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const float*              csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const double*             csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const double*             csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const double*             csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipComplex*         csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipComplex*         csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipDoubleComplex*   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipDoubleComplex*   csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipDoubleComplex*   csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup extra_module
*  \details
*  \p hipsparseXcsrgeam2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix \f$C\f$. It is assumed that \p csrRowPtrC has been allocated with
*  size \p m+1. The required buffer size can be obtained by
*  \ref hipsparseScsrgeam2_bufferSizeExt "hipsparseXcsrgeam2_bufferSizeExt()". The
*  desired index base in the output CSR matrix \f$C\f$ is set in the \ref hipsparseMatDescr_t
*  \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA               number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA   array of \p nnzA elements containing the column indices of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB             descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB               number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedRowPtrB   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedColIndB   array of \p nnzB elements containing the column indices of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC             descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedRowPtrC   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$. \p nnzTotalDevHostPtr can be a host or device pointer.
*  @param[in]
*  workspace          temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB,
*          \p descrA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p descrB, \p csrSortedRowPtrB,
*          \p csrSortedColIndB, \p descrC, \p csrSortedRowPtrC or \p nnzTotalDevHostPtr is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        void*                     workspace);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  This computation involves a multi step process. First the user must call
*  \ref hipsparseScsrgeam2_bufferSizeExt "hipsparseXcsrgeam2_bufferSizeExt()" in order to determine the
*  required user allocated temporary buffer size. The user then allocates this buffer and also allocates
*  \p csrRowPtrC to have size \p m+1. Both the temporary storage buffer and \p csrRowPtrC array are then
*  passed to \ref hipsparseXcsrgeam2Nnz which fills in the \p csrRowPtrC array as well as computes the total
*  number of nonzeros in C, \p nnzC. The user then allocates both arrays \p csrColIndC and \p csrValC to have
*  size \p nnzC and calls \p hipsparseXcsrgeam2 to complete the computation. The desired index base in
*  the output CSR matrix C is set in the \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  m                number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n                number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                   \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA             number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA    array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnzA elements containing the column indices of the
*                   sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[in]
*  descrB           descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                   \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB             number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedValB    array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedRowPtrB array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedColIndB array of \p nnzB elements containing the column indices of the
*                   sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC           descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                   \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrSortedValC    array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrSortedRowPtrC array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrSortedColIndC array of elements containing the column indices of the
*                   sparse CSR matrix \f$C\f$.
*  @param[in]
*  pBuffer          temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB,
*          \p alpha, \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p beta,
*          \p descrB, \p csrSortedValB, \p csrSortedRowPtrB, \p csrSortedColIndB, \p descrC, \p csrSortedValC,
*          \p csrSortedRowPtrC, \p csrSortedColIndC or \p pBuffer is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_csrgeam2.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRGEAM_H */
