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
#ifndef HIPSPARSE_GEBSR2CSR_H
#define HIPSPARSE_GEBSR2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse GEBSR matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXgebsr2csr converts a GEBSR matrix into a CSR matrix. It is assumed,
*  that \p csrValC, \p csrColIndC and \p csrRowPtrC are already allocated prior to
*  calling \p hipsparseXgebsr2csr. Allocation size for \p csrRowPtrC equals
*  \p m+1 where:
*
*  \f[
*    \begin{align}
*    \text{m} &= \text{mb * rowBlockDim} \\
*    \text{n} &= \text{nb * colBlockDim}
*    \end{align}
*  \f]
*
*  Allocation size for \p csrValC and \p csrColIndC is computed by the the number of blocks in the GEBSR
*  matrix, \p nnzb, multiplied by the product of the block dimensions, i.e. \p nnz=nnzb*rocBlockDim*colBlockDim.
*
*  For example, given the GEBSR matrix:
*  \f[
*   \left[
*    \begin{array}{c | c | c}
*      \begin{array}{c c}
*       6 & 2 \\
*       1 & 4 \\
*       5 & 4
*      \end{array} &
*      \begin{array}{c c}
*       0 & 3 \\
*       5 & 0 \\
*       0 & 7
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0 \\
*       0 & 0
*      \end{array} &
*      \begin{array}{c c}
*       3 & 0 \\
*       0 & 0 \\
*       0 & 7
*      \end{array} &
*      \begin{array}{c c}
*       2 & 2 \\
*       4 & 3 \\
*       1 & 4
*      \end{array} \\
*   \end{array}
*  \right]
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb          number of block rows in the sparse general BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse general BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse general BSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  bsrValA      array of \p nnzb*rowBlockDim*colBlockDim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  rowBlockDim row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  colBlockDim column size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  descrC      descriptor of the sparse CSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC     array of \p nnzb*rowBlockDim*colBlockDim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csrRowPtrC  array of \p m+1 where \p m=mb*rowBlockDim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csrColIndC  array of \p nnzb*block_dim*block_dim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p block_dim, \p bsrValA,
*              \p bsrRowPtrA, \p bsrColIndA, \p csrValC, \p csrRowPtrC or \p csrColIndC pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_gebsr2csr.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const float*              bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      float*                    csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const double*             bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      double*                   csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipComplex*         bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipComplex*               csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipDoubleComplex*   bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipDoubleComplex*         csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GEBSR2CSR_H */
