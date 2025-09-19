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
#ifndef HIPSPARSE_CSRGEMM_H
#define HIPSPARSE_CSRGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \details
*  \p hipsparseXcsrgemmNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix \f$C\f$. It is assumed that \p csrRowPtrC has been allocated
*  with size \p m+1. The desired index base in the output CSR matrix \f$C\f$ is set in the
*  \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \p transA == \p transB == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is
*  supported.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  transA          matrix \f$A\f$ operation type.
*  @param[in]
*  transB          matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[inout]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$. \p nnzTotalDevHostPtr can be a host or device pointer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, \p nnzC,
*          \p descrA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, \p csrColIndB,
*          \p descrC, \p csrRowPtrC or \p nnzTotalDevHostPtr is invalid.
*  \retval HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
*          \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE,
*          \p transB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE, or
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
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
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm multiplies the sparse \f$m \times k\f$ matrix \f$op(A)\f$, defined in
*  CSR storage format with the sparse \f$k \times n\f$ matrix \f$op(B)\f$, defined in CSR
*  storage format, and stores the result in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  defined in CSR storage format, such that
*  \f[
*    C := op(A) \cdot op(B),
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
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  This computation involves a multi step process. First the user must allocate \p csrRowPtrC
*  to have size \p m+1. The user then calls \ref hipsparseXcsrgemmNnz which fills in the \p csrRowPtrC
*  array as well as computes the total number of nonzeros in C, \p nnzC. The user then allocates both
*  arrays \p csrColIndC and \p csrValC to have size \p nnzC and calls \p hipsparseXcsrgemm to complete
*  the computation. The desired index base in the output CSR matrix C is set in the
*  \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \p transB == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  transA          matrix \f$A\f$ operation type.
*  @param[in]
*  transB          matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrValA         array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrValB         array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC         array of \p nnzC elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrColIndC      array of \p nnzC elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB,
*          \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrValB,
*          \p csrRowPtrB, \p csrColIndB, \p descrC, \p csrValC, \p csrRowPtrC, \p csrColIndC
*          is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE,
*          \p transB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE, or
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_csrgemm.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \details
*  \p hipsparseXcsrgemm2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseXcsrgemm2Nnz() and \ref hipsparseScsrgemm2
*  "hipsparseXcsrgemm2()". The temporary storage buffer must be allocated by the user.
*
*  \note
*  Please note, that for matrix products with more than 4096 non-zero entries per row,
*  additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descrD          descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzD            number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrRowPtrD      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrColIndD      array of \p nnzD elements containing the column indices of the sparse
*                  CSR matrix \f$D\f$.
*  @param[inout]
*  info            structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseXcsrgemm2Nnz(), hipsparseScsrgemm2(), hipsparseDcsrgemm2(),
*                     hipsparseCcsrgemm2() and hipsparseZcsrgemm2().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, \p nnz_D,
*          \p alpha, \p beta, \p descrA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB,
*          \p csrColIndB, \p descrD, \p csrRowPtrD, \p csrColIndD, \p info or \p pBufferSizeInBytes
*          is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \details
*  \p hipsparseXcsrgemm2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix \f$C\f$. It is assumed that \p csrRowPtrC has been allocated
*  with size \p m+1. The required buffer size can be obtained by
*  \ref hipsparseScsrgemm2_bufferSizeExt "hipsparseXcsrgemm2_bufferSizeExt()". The desired
*  index base in the output CSR matrix \f$C\f$ is set in the \ref hipsparseMatDescr_t \p descrC.
*  See \ref hipsparseSetMatIndexBase().
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
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrD          descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzD            number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrRowPtrD      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrColIndD      array of \p nnzD elements containing the column indices of the sparse
*                  CSR matrix \f$D\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$.
*  @param[in]
*  info            structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned
*                  by hipsparseScsrgemm2_bufferSizeExt(), hipsparseDcsrgemm2_bufferSizeExt(),
*                  hipsparseZcsrgemm2_bufferSizeExt() or hipsparseZcsrgemm2_bufferSizeExt().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, \p nnzD,
*          \p descrA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, \p csrColIndB,
*          \p descrD, \p csrRowPtrD, \p csrColIndD, \p descrC, \p csrRowPtrC, \p nnzTotalDevHostPtr,
*          \p info or \p pBuffer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrD,
                                        int                       nnzD,
                                        const int*                csrRowPtrD,
                                        const int*                csrColIndD,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        const csrgemm2Info_t      info,
                                        void*                     pBuffer);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
*  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
*  storage format, such
*  that
*  \f[
*    C := \alpha \cdot A \cdot B + \beta \cdot D
*  \f]
*
*  This computation involves a multi step process. First the user must call
*  \ref hipsparseScsrgemm2_bufferSizeExt "hipsparseXcsrgemm2_bufferSizeExt()" in order to
*  determine the required user allocated temporary buffer size. The user then allocates this
*  buffer and also allocates \p csrRowPtrC to have size \p m+1. Both the temporary storage
*  buffer and \p csrRowPtrC array are then passed to \ref hipsparseXcsrgemm2Nnz which fills
*  in the \p csrRowPtrC array as well as computes the total number of nonzeros in C, \p nnzC.
*  The user then allocates both arrays \p csrColIndC and \p csrValC to have size \p nnzC and
*  calls \p hipsparseXcsrgemm2 to complete the computation. The desired index base in the output
*  CSR matrix C is set in the \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot A \cdot B\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
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
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrValB         array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descrD          descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzD            number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrValD         array of \p nnzD elements of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrRowPtrD      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrColIndD      array of \p nnzD elements containing the column indices of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC         array of \p nnzC elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrColIndC      array of \p nnzC elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*  @param[in]
*  info            structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned
*                  by hipsparseScsrgemm2_bufferSizeExt(), hipsparseDcsrgemm2_bufferSizeExt(),
*                  hipsparseCcsrgemm2_bufferSizeExt() or hipsparseZcsrgemm2_bufferSizeExt().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB,
*          \p nnzD, \p alpha, \p beta, \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA,
*          \p descrB, \p csrValB, \p csrRowPtrB, \p csrColIndB, \p descrD, \p csrValD,
*          \p csrRowPtrD, \p csrColIndD, \p csrValC, \p csrRowPtrC, \p csrColIndC, \p info
*          or \p pBuffer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_csrgemm2.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const float*              csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const double*             csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipComplex*         csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipDoubleComplex*   csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRGEMM_H */
