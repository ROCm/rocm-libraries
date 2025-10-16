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
#ifndef HIPSPARSE_BSR2CSR_H
#define HIPSPARSE_BSR2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse BSR matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXbsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csrValC, \p csrColIndC and \p csrRowPtrC are allocated. Allocation size
*  for \p csrRowPtrC is computed by the number of block rows multiplied by the block
*  dimension plus one. Allocation for \p csrValC and \p csrColInd is computed by the
*  the number of blocks in the BSR matrix multiplied by the block dimension squared.
*
*  For example, given the BSR matrix using block dimension 2:
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       1 & 0 \\
*       3 & 4
*      \end{array} &
*      \begin{array}{c c}
*       0 & 2 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       5 & 0 \\
*       1 & 2
*      \end{array} &
*      \begin{array}{c c}
*       6 & 7 \\
*       3 & 4
*      \end{array} \\
*   \end{array}
*  \right]
*  \f]
*
*  The resulting CSR matrix row pointer, column indices, and values arrays are:
*  \f[
*    \begin{align}
*    \text{csrRowPtrC} &= \begin{bmatrix} 0 & 4 & 8 & 12 & 16 \end{bmatrix} \\
*    \text{csrColIndC} &= \begin{bmatrix} 0 & 1 & 2 & 3 & 0 & 1 & 2 & 3 & 0 & 1 & 2 & 3 & 0 & 1 & 2 & 3 \end{bmatrix} \\
*    \text{csrValC} &= \begin{bmatrix} 1 & 0 & 0 & 2 & 3 & 4 & 0 & 0 & 5 & 0 & 6 & 7 & 1 & 2 & 3 & 4 \end{bmatrix} \\
*    \end{align}
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
*  mb          number of block rows in the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  bsrValA     array of \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  blockDim    size of the blocks in the sparse BSR matrix.
*  @param[in]
*  descrC      descriptor of the sparse CSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC     array of \p nnzb*blockDim*blockDim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csrRowPtrC  array of \p m+1 where \p m=mb*blockDim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csrColIndC  array of \p nnzb*blockDim*blockDim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p blockDim, \p bsrValA,
*              \p bsrRowPtrA, \p bsrColIndA, \p csrValC, \p csrRowPtrC or \p csrColIndC pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_bsr2csr.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSR2CSR_H */
