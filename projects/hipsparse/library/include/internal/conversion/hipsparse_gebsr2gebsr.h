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
#ifndef HIPSPARSE_GEBSR2GEBSR_H
#define HIPSPARSE_GEBSR2GEBSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
*  GEBSR matrix to another sparse GEBSR matrix.
*
*  \details
*  \p hipsparseXgebsr2gebsr_bufferSize returns the size of the temporary storage buffer that is required by
*  \ref hipsparseXgebsr2gebsrNnz() and \ref hipsparseSgebsr2gebsr "hipsparseXgebsr2gebsr()". The temporary storage
*  buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  dirA               the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb                 number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb                 number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb               number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descrA             the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrValA            array of \p nnzb*rowBlockDimA*colBlockDimA containing the values of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrRowPtrA         array of \p mb+1 elements that point to the start of every block row of the
*                     sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrColIndA         array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  rowBlockDimA       row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  colBlockDimA       column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  rowBlockDimC       row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  colBlockDimC       column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by hipsparseXgebsr2gebsrNnz(),
*                     hipsparseSgebsr2gebsr(), hipsparseDgebsr2gebsr(), hipsparseCgebsr2gebsr(), and
*                     hipsparseZgebsr2gebsr().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p rowBlockDimA, \p colBlockDimA,
*              \p rowBlockDimC, \p colBlockDimC, \p bsrRowPtrA, \p bsrColIndA, \p descrA or \p pBufferSizeInBytes pointer
*              is invalid.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const float*              bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const double*             bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipComplex*         bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipDoubleComplex*   bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief This function is used when converting a GEBSR sparse matrix \f$A\f$ to another GEBSR sparse matrix \f$C\f$.
*  Specifically, this function determines the number of non-zero blocks that will exist in \f$C\f$ (stored using either a host
*  or device pointer), and computes the row pointer array for \f$C\f$.
*
*  \details
*  The routine does support asynchronous execution.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  dirA               the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb                 number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb                 number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb               number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descrA             the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrRowPtrA         array of \p mb+1 elements that point to the start of every block row of the
*                     sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrColIndA         array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \p A.
*  @param[in]
*  rowBlockDimA       row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  colBlockDimA       column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  descrC             the descriptor of the general BSR sparse matrix \f$C\f$, the supported matrix type is
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrRowPtrC         array of \p mbC+1 elements that point to the start of every block row of the
*                     sparse general BSR matrix \f$C\f$ where \p mbC = ( \p m+rowBlockDimC-1 ) / \p rowBlockDimC.
*  @param[in]
*  rowBlockDimC       row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  colBlockDimC       column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero blocks in general BSR sparse matrix \f$C\f$ stored using device or host memory.
*  @param[out]
*  buffer             buffer allocated by the user whose size is determined by calling \ref hipsparseSgebsr2gebsr_bufferSize
*                     "hipsparseXgebsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p rowBlockDimA, \p colBlockDimA, \p rowBlockDimC,
*              \p colBlockDimC, \p bsrRowPtrA, \p bsrColIndA, \p bsrRowPtrC, \p descrA, \p descrC, \p buffer pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXgebsr2gebsrNnz(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                       mb,
                                           int                       nb,
                                           int                       nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const int*                bsrRowPtrA,
                                           const int*                bsrColIndA,
                                           int                       rowBlockDimA,
                                           int                       colBlockDimA,
                                           const hipsparseMatDescr_t descrC,
                                           int*                      bsrRowPtrC,
                                           int                       rowBlockDimC,
                                           int                       colBlockDimC,
                                           int*                      nnzTotalDevHostPtr,
                                           void*                     buffer);

/*! \ingroup conv_module
*  \brief
*  This function converts the GEBSR sparse matrix \f$A\f$ to another GEBSR sparse matrix \f$C\f$.
*
*  \details
*  The conversion uses three steps. First, the user calls \ref hipsparseSgebsr2gebsr_bufferSize
*  "hipsparseXgebsr2gebsr_bufferSize()" to determine the size of the required temporary storage buffer.
*  The user then allocates this buffer. Secondly, the user then allocates \p mbC+1 integers for the row
*  pointer array for \f$C\f$ where:
*  \f[
*    \begin{align}
*    \text{mbC} &= \text{(m - 1) / rowBlockDimC + 1} \\
*    \text{nbC} &= \text{(n - 1) / colBlockDimC + 1}
*    \end{align}
*  \f]
*  The user then calls hipsparseXgebsr2gebsrNnz() to fill in the row pointer array for \f$C\f$ ( \p bsrRowPtrC ) and
*  determine the number of non-zero blocks that will exist in \f$C\f$. Finally, the user allocates space for the column
*  indices array of \f$C\f$ to have \p nnzbC elements and space for the values array of \f$C\f$ to have
*  \p nnzbC*rowBlockDimC*colBlockDimC and then calls \p hipsparseXgebsr2gebsr to complete the conversion.
*
*  It may be the case that \p rowBlockDimC does not divide evenly into \p m and/or \p colBlockDim does not divide evenly
*  into \p n. In these cases, the GEBSR matrix is expanded in size in order to fit full GEBSR blocks. For example, if
*  the original GEBSR matrix A (using \p rowBlockDimA=2, \p colBlockDimA=3) looks like:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c c}
*       1 & 0 & 0 \\
*       3 & 4 & 0
*      \end{array} &
*      \begin{array}{c c c}
*       2 & 0 & 0 \\
*       4 & 5 & 6
*      \end{array} \\
*    \hline
*      \begin{array}{c c c}
*       1 & 2 & 3 \\
*       1 & 2 & 0
*      \end{array} &
*      \begin{array}{c c c}
*       4 & 0 & 0 \\
*       3 & 0 & 1
*      \end{array} \\
*   \end{array}
*  \right]
*  \f]
*
*  then if we specify \p rowBlockDimC=3 and \p colBlockDimC=2, our output GEBSR matrix C would be:
*
*  \f[
*   \left[
*    \begin{array}{c | c | c}
*      \begin{array}{c c}
*       1 & 0 \\
*       3 & 4 \\
*       1 & 2
*      \end{array} &
*      \begin{array}{c c}
*       0 & 2 \\
*       0 & 4 \\
*       3 & 4
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       5 & 6 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       1 & 2 \\
*       0 & 0 \\
*       0 & 0
*      \end{array} &
*      \begin{array}{c c}
*       0 & 3 \\
*       0 & 0 \\
*       0 & 0
*      \end{array} &
*      \begin{array}{c c}
*       0 & 1 \\
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*   \end{array}
*  \right]
*  \f]
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  dirA          the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb            number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb            number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb          number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descrA        the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrValA       array of \p nnzb*rowBlockDimA*colBlockDimA containing the values of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrRowPtrA    array of \p mb+1 elements that point to the start of every block row of the
*                sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrColIndA    array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  rowBlockDimA  row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  colBlockDimA  column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  descrC        the descriptor of the general BSR sparse matrix \f$C\f$, the supported matrix type is
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrValC       array of \p nnzbC*rowBlockDimC*colBlockDimC containing the values of the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  bsrRowPtrC    array of \p mbC+1 elements that point to the start of every block row of the
*                sparse general BSR matrix \f$C\f$.
*  @param[in]
*  bsrColIndC    array of \p nnzbC elements containing the block column indices of the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  rowBlockDimC  row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  colBlockDimC  column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  buffer        buffer allocated by the user whose size is determined by calling \ref hipsparseSgebsr2gebsr_bufferSize
*                "hipsparseXgebsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p rowBlockDimA, \p colBlockDimA,
*              \p rowBlockDimC, \p colBlockDimC, \p bsrRowPtrA, \p bsrColIndA, \p bsrValA, \p bsrRowPtrC, \p bsrColIndC,
*              \p bsrValC, \p descrA, \p descrC or \p buffer pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_gebsr2gebsr.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GEBSR2GEBSR_H */
