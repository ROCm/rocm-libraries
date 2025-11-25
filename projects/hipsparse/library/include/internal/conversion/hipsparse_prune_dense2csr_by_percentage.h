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
#ifndef HIPSPARSE_PRUNE_DENSE2CSR_BY_PRECENTAGE_H
#define HIPSPARSE_PRUNE_DENSE2CSR_BY_PRECENTAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize computes the size of the user allocated temporary
*  storage buffer used when converting a dense matrix to a pruned CSR matrix where the pruning is done
*  based on a percantage.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \ref hipsparseSpruneDense2csrNnzByPercentage "hipsparseXpruneDense2csrNnzByPercentage()".
*  Finally the user finishes the conversion by allocating the \p csrColInd and \p csrVal arrays
*  (whose size is determined by the value at \p nnzTotalDevHostPtr) and calling
*  \ref hipsparseSpruneDense2csrByPercentage "hipsparseXpruneDense2csrByPercentage()".
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()".
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
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
*  percentage         \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also
*                     any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal             array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd          integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info               prune information structure
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSpruneDense2csrNnzByPercentage(), hipsparseDpruneDense2csrNnzByPercentage().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE the \p handle or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const float*      A,
                                                                  int               lda,
                                                                  float             percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const float*              csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const double*     A,
                                                                  int               lda,
                                                                  double            percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const double*             csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSizeExt which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \ref hipsparseSpruneDense2csrNnzByPercentage "hipsparseXpruneDense2csrNnzByPercentage()".
*  Finally the user finishes the conversion by allocating the \p csrColInd and \p csrVal arrays
*  (whos size is determined by the value at \p nnzTotalDevHostPtr) and calling
*  \ref hipsparseSpruneDense2csrByPercentage "hipsparseXpruneDense2csrByPercentage()".
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()".
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
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
*  percentage         \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also
*                     any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal             array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd          integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info               prune information structure
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSpruneDense2csrNnzByPercentage(), hipsparseDpruneDense2csrNnzByPercentage().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE the \p handle or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const float*              csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t* pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const double*             csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \ref hipsparseSpruneDense2csrByPercentage_bufferSize "hipsparseXpruneDense2csrByPercentage_bufferSize()"
*  which determines the size of the temporary storage buffer. Once determined, this buffer must be allocated
*  by the user. Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the \p csrColInd and \p csrVal arrays (whos size is determined by the value
*  at \p nnzTotalDevHostPtr) and calling \ref hipsparseSpruneDense2csrByPercentage
*  "hipsparseXpruneDense2csrByPercentage()".
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()".
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
*  percentage         \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A.
*  @param[out]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
*  @param[in]
*  info               prune information structure
*  @param[out]
*  buffer             buffer allocated by the user whose size is determined by calling
*                     \ref hipsparseSpruneDense2csrByPercentage_bufferSize "hipsparseXpruneDense2csrByPercentage_bufferSize()"
*                     or \ref hipsparseSpruneDense2csrByPercentage_bufferSizeExt "hipsparseXpruneDense2csrByPercentage_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p percentage, \p A, \p descr, \p info, \p csrRowPtr
*              \p nnzTotalDevHostPtr or \p buffer pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          float                     percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          double                    percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls \ref hipsparseSpruneDense2csrByPercentage_bufferSize
*  "hipsparseXpruneDense2csrByPercentage_bufferSize()" which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \ref hipsparseSpruneDense2csrNnzByPercentage "hipsparseXpruneDense2csrNnzByPercentage()". Finally the
*  user finishes the conversion by allocating the \p csrColInd and \p csrVal arrays (whos size is
*  determined by the value at \p nnzTotalDevHostPtr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \ cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()".
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
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
*  percentage  \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and
*              also any valid value of the \ref hipsparseIndexBase_t.
*  @param[out]
*  csrVal      array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr   integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csrColInd   integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info prune  information structure
*  @param[in]
*  buffer      temporary storage buffer allocated by the user, size is returned by
*              \ref hipsparseSpruneDense2csrByPercentage_bufferSize "hipsparseXpruneDense2csrByPercentage_bufferSize()" or
*              \ref hipsparseSpruneDense2csrByPercentage_bufferSizeExt "hipsparseXpruneDense2csrByPercentage_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p percentage, \p A, \p descr, \p info, \p csrVal
*              \p csrRowPtr, \p csrColInd or \p buffer pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_prune_dense2csr_by_percentage.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       float*                    csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       double*                   csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_PRUNE_DENSE2CSR_BY_PRECENTAGE_H */
