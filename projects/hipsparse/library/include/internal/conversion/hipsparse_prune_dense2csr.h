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
#ifndef HIPSPARSE_PRUNE_DENSE2CSR_H
#define HIPSPARSE_PRUNE_DENSE2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csr_bufferSize computes the the size of the user allocated temporary storage buffer
*  used when converting a dense matrix to a pruned CSR matrix.
*
*  \details
*  Specifically given an input dense column ordered matrix A, with leading dimension \p lda where \p lda>=m,
*  the resulting pruned sparse CSR matrix C is computed using:
*  \f[
*   |C(i,j)| = A(i, j) \text{  if |A(i, j)| > threshold}
*  \f]
*
*  The first step in this conversion is to determine the required user allocated buffer size
*  using \p hipsparseXpruneDense2csr_bufferSize() that will be passed to the subsequent steps of the conversion.
*  Once the buffer size has been determined the user must allocate it. This user allocated buffer is then passed
*  to \ref hipsparseSpruneDense2csrNnz "hipsparseXpruneDense2csrNnz()" and \ref hipsparseSpruneDense2csr
*  "hipsparseXpruneDense2csr()" to complete the conversion. The user is responsible to then free the buffer once
*  the conversion has been completed.
*
*  See hipsparseSpruneDense2csr() for a full code example.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  threshold          pointer to the pruning non-negative threshold which can exist in either host or device memory.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL
*                     and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal             array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd          integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSpruneDense2csrNnz(), hipsparseDpruneDense2csrNnz(),
*                     hipsparseSpruneDense2csr() and hipsparseDpruneDense2csr().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const float*              A,
                                                      int                       lda,
                                                      const float*              threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const float*              csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const double*             A,
                                                      int                       lda,
                                                      const double*             threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const double*             csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const float*              A,
                                                         int                       lda,
                                                         const float*              threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const float*              csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const double*             A,
                                                         int                       lda,
                                                         const double*             threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const double*             csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csrNnz function computes the number of nonzero elements per row and the total
*  number of nonzero elements in a dense matrix once the elements less than the (non-negative) threshold are
*  pruned from the matrix.
*
*  \details
*  Specifically given an input dense column ordered matrix A, with leading dimension \p lda where \p lda>=m,
*  the resulting pruned sparse CSR matrix C is computed using:
*  \f[
*   |C(i,j)| = A(i, j) \text{  if |A(i, j)| > threshold}
*  \f]
*
*  First the user must determine the size of the required temporary buffer using the routine
*  \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" and then allocate it. Next
*  the user allocates \p csrRowPtr with size \p m+1. Then the passes both the temporary storage buffer as well
*  as \p csrRowPtr to \p hipsparseXpruneDense2csrNnz in order to determine the total number of non-zeros that
*  will exist in the sparse CSR matrix C (after pruning has been performed on A) as well as fill the output CSR
*  row pointer array \p csrRowPtr.
*
*  For example, given the dense matrix:
*
*  \f[
*    \begin{bmatrix}
*    6 & 2 & 3 & 7 \\
*    5 & 6 & 7 & 8 \\
*    5 & 4 & 8 & 1
*    \end{bmatrix}
*  \f]
*
*  and the \p threshold value 5, the resulting matrix after pruning is:
*
*  \f[
*    \begin{bmatrix}
*    6 & 0 & 0 & 7 \\
*    0 & 6 & 7 & 8 \\
*    0 & 0 & 8 & 0
*    \end{bmatrix}
*  \f]
*
*  and corresponding row pointer array and non-zero count:
*
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 5 & 6 \end{bmatrix} \\
*    \text{nnzTotalDevHostPtr} &= 6
*    \end{align}
*  \f]
*
*  The above example assumes a zero index base for the output CSR matrix. We can set the desired index base
*  in the output CSR matrix by setting it in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase().
*
*  For a full code example on how to use this routine, see hipsparseSpruneDense2csr().
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  threshold          pointer to the pruning non-negative threshold which can exist in either host or device memory.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A.
*  @param[out]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
*  @param[out]
*  buffer             buffer allocated by the user whose size is determined by calling
*                     \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" or
*                     \ref hipsparseSpruneDense2csr_bufferSizeExt "hipsparseXpruneDense2csr_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p A, \p threshold, \p descr, \p csrRowPtr
*              \p nnzTotalDevHostPtr or \p buffer pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csr converts the matrix A in dense format into a sparse matrix in CSR format
*  while pruning values that are less than the (non-negative) threshold. All the parameters are assumed
*  to have been pre-allocated by the user.
*
*  \details
*  Specifically given an input dense column ordered matrix A, with leading dimension \p lda where \p lda>=m,
*  the resulting pruned sparse CSR matrix C is computed using:
*  \f[
*   |C(i,j)| = A(i, j) \text{  if |A(i, j)| > threshold}
*  \f]
*
*  The user first calls \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" to
*  determine the size of the required user allocate temporary storage buffer. The user then allocates this
*  buffer. Next, the user allocates \p csrRowPtr to have \p m+1 elements and then calls
*  \ref hipsparseSpruneDense2csrNnz "hipsparseXpruneDense2csrNnz()" which fills in the \p csrRowPtr array
*  and stores the number of elements that are larger than the pruning \p threshold in \p nnzTotalDevHostPtr.
*  The user then allocates \p csrColInd and \p csrVal to have size \p nnzTotalDevHostPtr and completes the
*  conversion by calling \p hipsparseXpruneDense2csr().
*
*  For example, performing these steps with the dense input matrix A:
*  \f[
*    \begin{bmatrix}
*    6 & 2 & 3 & 7 \\
*    5 & 6 & 7 & 8 \\
*    5 & 4 & 8 & 1
*    \end{bmatrix}
*  \f]
*
*  and the \p threshold value 5, results in the pruned matrix C:
*
*  \f[
*    \begin{bmatrix}
*    6 & 0 & 0 & 7 \\
*    0 & 6 & 7 & 8 \\
*    0 & 0 & 8 & 0
*    \end{bmatrix}
*  \f]
*
*  and corresponding CSR row, column, and values arrays:
*
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 5 & 6 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 1 & 2 & 3 & 2 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 6 & 7 & 6 & 7 & 8 & 8 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \note
*  The routine \p hipsparseXpruneDense2csr() is executed asynchronously with respect to the host and may
*  return control to the application on the host before the entire result is ready.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*  @param[in]
*  lda         leading dimension of dense array \p A.
*  @param[in]
*  threshold   pointer to the non-negative pruning threshold which can exist in either host or device memory.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL
*              and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[out]
*  csrVal      array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr  integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csrColInd  integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[in]
*  buffer     temporary storage buffer allocated by the user, size is returned by
*             \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" or
*             \ref hipsparseSpruneDense2csr_bufferSizeExt "hipsparseXpruneDense2csr_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p A, \p descr, \p threshold, \p csrVal
*              \p csrRowPtr, \p csrColInd, \p buffer pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_prune_dense2csr.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const float*              A,
                                           int                       lda,
                                           const float*              threshold,
                                           const hipsparseMatDescr_t descr,
                                           float*                    csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const double*             A,
                                           int                       lda,
                                           const double*             threshold,
                                           const hipsparseMatDescr_t descr,
                                           double*                   csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_PRUNE_DENSE2CSR_H */
