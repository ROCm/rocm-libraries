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
#ifndef HIPSPARSE_CSR2GEBSR_H
#define HIPSPARSE_CSR2GEBSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEBSR matrix
*
*  \details
*  \p hipsparseXcsr2gebsr_bufferSize returns the size of the temporary buffer that
*  is required by \ref hipsparseXcsr2gebsrNnz and \ref hipsparseScsr2gebsr "hipsparseXcsr2gebsr()".
*  Once the temporary buffer size has been determined, it must be allocated by the user prior
*  to calling \ref hipsparseXcsr2gebsrNnz and \ref hipsparseScsr2gebsr "hipsparseXcsr2gebsr()".
*
*  See hipsparseScsr2gebsr() for complete code example.
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  dir                direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW
*                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix.
*  @param[in]
*  csr_descr          descriptor of the sparse CSR matrix. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrVal             array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr          integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*  @param[in]
*  csrColInd          integer array of the column indices for each non-zero element in the CSR matrix
*  @param[in]
*  rowBlockDim        the row block dimension of the GEneral BSR matrix. Between 1 and \p m
*  @param[in]
*  colBlockDim        the col block dimension of the GEneral BSR matrix. Between 1 and \p n
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by \ref hipsparseXcsr2gebsrNnz()
*                     and \ref hipsparseScsr2gebsr "hipsparseXcsr2gebsr()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p rowBlockDim, \p colBlockDim, \p csrVal,
*              \p csrRowPtr, \p csrColInd or \p pBufferSizeInBytes pointer is invalid.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const float*              csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const double*             csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipComplex*         csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipDoubleComplex*   csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  GEBSR matrix given a sparse CSR matrix as input.
*
*  \details
*  This is the second step in conveting a CSR matrix to a GEBSR matrix. The user must first call
*  \ref hipsparseScsr2gebsr_bufferSize "hipsparseXcsr2gebsr_bufferSize()" to determine the size of
*  the required temporary storage buffer. The user then allocates this buffer as well as the
*  \p bsrRowPtr array ( size \p mb+1 ) and passes both to \p hipsparseXcsr2gebsrNnz(). This second
*  step then computes the number of nonzero block columns per row and the total number of nonzero blocks.
*
*  In general, when converting a CSR matrix of size \p m x \p n to a GEBSR matrix, the resulting GEBSR matrix will have size
*  \p mb x \p nb where \p mb and \p nb equal:
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / rowBlockDim + 1} \\
*    \text{nb} &= \text{(n - 1) / colBlockDim + 1}
*    \end{align}
*  \f]
*
*  For example given a matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 & 4 & 0 \\
*    3 & 4 & 0 & 0 & 5 & 1 \\
*    5 & 0 & 6 & 7 & 6 & 2
*    \end{bmatrix}
*  \f]
*
*  represented in CSR format with the arrays:
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 3 & 7 & 12 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 4 & 0 & 1 & 4 & 5 & 0 & 2 & 3 & 4 & 5 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 4 & 3 & 4 & 5 & 1 & 5 & 6 & 7 & 6 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  the \p bsrRowPtr array and total nonzero block count will be filled with:
*  \f[
*    \begin{align}
*    \text{bsrRowPtr} &= \begin{bmatrix} 0 & 3 \end{bmatrix} \\
*    \text{*bsrNnzDevhost} &= 3
*    \end{align}
*  \f]
*
*  after calling \p hipsparseXcsr2gebsrNnz with \p rowBlockDim=3 and \p colBlockDim=2.
*
*  \note
*  As indicated, \p bsrNnzDevhost can point either to host or device memory. This is controlled
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  It may be the case that \p rowBlockDim does not divide evenly into \p m and/or that \p colBlockDim does not divide
*  evenly into \p n. In these cases, the CSR matrix is expanded in size in order to fit full GEBSR blocks. For example,
*  using the original CSR matrix but this time with \p rowBlockDim=2 and \p colBlockDim=3, the function
*  \p hipsparseXcsr2gebsrNnz computes the GEBSR row pointer array and total number of non-zero blocks for the GEBSR matrix:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c c}
*       1 & 0 & 0 \\
*       3 & 4 & 0
*      \end{array} &
*      \begin{array}{c c c}
*       2 & 4 & 0 \\
*       0 & 5 & 1
*      \end{array} \\
*    \hline
*      \begin{array}{c c c}
*       5 & 0 & 6 \\
*       0 & 0 & 0
*      \end{array} &
*      \begin{array}{c c c}
*       7 & 6 & 2 \\
*       0 & 0 & 0
*      \end{array}
*   \end{array}
*  \right]
*  \f]
*
*  See hipsparseScsr2gebsr() for full code example.
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  dir           direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW or by
*                \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*  @param[in]
*  n             number of columns of the sparse CSR matrix.
*  @param[in]
*  csr_descr     descriptor of the sparse CSR matrix. Currently, only
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtr     integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*  @param[in]
*  csrColInd     integer array of the column indices for each non-zero element in the CSR matrix
*  @param[in]
*  bsr_descr     descriptor of the sparse GEneral BSR matrix. Currently, only
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrRowPtr     integer array containing \p mb+1 elements that point to the start of each block row of the General BSR matrix
*
*  @param[in]
*  rowBlockDim   the row block dimension of the GEneral BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*
*  @param[in]
*  colBlockDim   the col block dimension of the GEneral BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*
*  @param[out]
*  bsrNnzDevhost total number of nonzero elements in device or host memory.
*
*  @param[in]
*  pbuffer       buffer allocated by the user whose size is determined by calling \ref hipsparseScsr2gebsr_bufferSize
*                "hipsparseXcsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p rowBlockDim, \p colBlockDim, \p csrRowPtr,
*              \p csrColInd, \p bsrRowPtr or \p bsrNnzDevhost pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2gebsrNnz(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         int                       m,
                                         int                       n,
                                         const hipsparseMatDescr_t csr_descr,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipsparseMatDescr_t bsr_descr,
                                         int*                      bsrRowPtr,
                                         int                       rowBlockDim,
                                         int                       colBlockDim,
                                         int*                      bsrNnzDevhost,
                                         void*                     pbuffer);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEBSR matrix
*
*  \details
*  \p hipsparseXcsr2gebsr converts a CSR matrix into a GEBSR matrix. It is assumed,
*  that \p bsrVal, \p bsrColInd and \p bsrRowPtr are allocated. Allocation size
*  for \p bsrRowPtr is computed as \p mb+1 where \p mb is the number of block rows in
*  the GEBSR matrix. The number of nonzero blocks in the resulting GEBSR matrix
*  is computed using \ref hipsparseXcsr2gebsrNnz which also fills in \p bsrRowPtr.
*
*  In more detail, \p hipsparseXcsr2gebsr is the third and final step on the conversion from CSR to GEBSR.
*  The user first determines the size of the required user allocated temporary storage buffer using
*  \ref hipsparseScsr2gebsr_bufferSize "hipsparseXcsr2gebsr_bufferSize()". The user then allocates this buffer
*  as well as the row pointer array \p bsrRowPtr with size \p mb+1, where \p mb is the number of block rows
*  in the GEBSR matrix and \p nb is the number of block columns in GEBSR matrix:
*
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / rowBlockDim + 1} \\
*    \text{nb} &= \text{(n - 1) / colBlockDim + 1}
*    \end{align}
*  \f]
*
*  Both the temporary storage buffer and the GEBSR row pointer array are then passed to \ref hipsparseXcsr2gebsrNnz
*  which fills the GEBSR row pointer array \p bsrRowPtr and also computes the number of nonzero blocks,
*  \p bsrNnzDevhost, that will exist in the GEBSR matrix. The user then allocates both the GEBSR column indices array
*  \p bsrColInd with size \p bsrNnzDevhost as well as the GEBSR values array \p bsrVal with size
*  \p bsrNnzDevhost*rowBlockDim*colBlockDim. Finally, with all arrays allocated, the conversion is completed by calling
*  \p hipsparseXcsr2gebsr.
*
*  For example, assuming the matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 & 4 & 0 \\
*    3 & 4 & 0 & 0 & 5 & 1 \\
*    5 & 0 & 6 & 7 & 6 & 2
*    \end{bmatrix}
*  \f]
*
*  represented in CSR format with the arrays:
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 3 & 7 & 12 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 4 & 0 & 1 & 4 & 5 & 0 & 2 & 3 & 4 & 5 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 4 & 3 & 4 & 5 & 1 & 5 & 6 & 7 & 6 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  then using \p rowBlockDim=3 and \p colBlockDim=2, the final GEBSR matrix is:
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       1 & 0 \\
*       3 & 4 \\
*       3 & 0
*      \end{array} &
*      \begin{array}{c c}
*       0 & 2 \\
*       0 & 0 \\
*       6 & 7
*      \end{array} &
*      \begin{array}{c c}
*       4 & 0 \\
*       5 & 1 \\
*       6 & 2
*      \end{array}
*   \end{array}
*  \right]
*  \f]
*
*  and is represented with the arrays:
*  \f[
*    \begin{align}
*    \text{bsrRowPtr} &= \begin{bmatrix} 0 & 3 \end{bmatrix} \\
*    \text{bsrColInd} &= \begin{bmatrix} 0 & 1 & 2 \end{bmatrix} \\
*    \text{bsrVal} &= \begin{bmatrix} 1 & 0 & 3 & 4 & 3 & 0 & 0 & 2 & 0 & 0 & 6 & 7 & 4 & 0 & 5 & 1 & 6 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The above example assumes that the blocks are row ordered. If instead the blocks are column ordered, the \p bsrVal arrays
*  becomes:
*  \f[
*    \begin{align}
*    \text{bsrVal} &= \begin{bmatrix} 1 & 3 & 3 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 & 4 & 5 & 6 & 0 & 1 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The block order direction is determined by \p dir.
*
*  It may be the case that \p rowBlockDim does not divide evenly into \p m and/or that \p colBlockDim does not divide
*  evenly into \p n. In these cases, the CSR matrix is expanded in size in order to fit full GEBSR blocks. For example,
*  using the original CSR matrix but this time with \p rowBlockDim=2 and \p colBlockDim=3, the resulting GEBSR matrix
*  would looks like:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c c}
*       1 & 0 & 0 \\
*       3 & 4 & 0
*      \end{array} &
*      \begin{array}{c c c}
*       2 & 4 & 0 \\
*       0 & 5 & 1
*      \end{array} \\
*    \hline
*      \begin{array}{c c c}
*       5 & 0 & 6 \\
*       0 & 0 & 0
*      \end{array} &
*      \begin{array}{c c c}
*       7 & 6 & 2 \\
*       0 & 0 & 0
*      \end{array}
*   \end{array}
*  \right]
*  \f]
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  dir          the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrVal       array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr    array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csrColInd    array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrVal       array of \p nnzb* \p rowBlockDim* \p colBlockDim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsrRowPtr    array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsrColInd    array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  rowBlockDim  row size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  colBlockDim  col size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  pbuffer      buffer allocated by the user whose size is determined by calling \ref hipsparseScsr2gebsr_bufferSize
*               "hipsparseXcsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p rowBlockDim, \p colBlockDim, \p bsrVal,
*              \p bsrRowPtr, \p bsrColInd, \p csrVal, \p csrRowPtr or \p csrColInd pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_csr2gebsr.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      float*                    bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      double*                   bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipComplex*               bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipDoubleComplex*         bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSR2GEBSR_H */
