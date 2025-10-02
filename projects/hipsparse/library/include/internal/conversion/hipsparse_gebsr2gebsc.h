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
#ifndef HIPSPARSE_GEBSR2GEBSC_H
#define HIPSPARSE_GEBSR2GEBSC_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse GEBSR matrix into a sparse GEBSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc_bufferSize returns the size of the temporary storage buffer
*  required by \ref hipsparseSgebsr2gebsc "hipsparseXgebsr2gebsc()" and is the first step
*  in converting a sparse matrix in GEBSR format to a sparse matrix in GEBSC format. Once
*  the size of the temporary storage buffer has been determined, it must be allocated by the user.
*
*  See hipsparseSgebsr2gebsc() for a complete code example.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  mb                 number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb                 number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb               number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsrVal             array of \p nnzb*rowBlockDim*colBlockDim containing the values of the sparse GEneral
*                     BSR matrix.
*  @param[in]
*  bsrRowPtr          array of \p mb+1 elements that point to the start of every row of the
*                     sparse GEneral BSR matrix.
*  @param[in]
*  bsrColInd          array of \p nnzb elements containing the column indices of the sparse
*                     GEneral BSR matrix.
*  @param[in]
*  rowBlockDim        row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  colBlockDim        col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSgebsr2gebsc(), hipsparseDgebsr2gebsc(), hipsparseCgebsr2gebsc() and
*                     hipsparseZgebsr2gebsc().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p bsrRowPtr, \p bsrColInd
*              or \p pBufferSizeInBytes pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsrVal,
                                                   const int*              bsrRowPtr,
                                                   const int*              bsrColInd,
                                                   int                     rowBlockDim,
                                                   int                     colBlockDim,
                                                   size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEBSR matrix into a sparse GEBSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc converts a GEBSR matrix into a GEBSC matrix. \p hipsparseXgebsr2gebsc
*  can also be used to convert a GEBSC matrix into a GEBSR matrix. \p copyValues decides
*  whether \p bscVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \p hipsparseXgebsr2gebsc requires extra temporary storage buffer that has to be allocated
*  by the user. Storage buffer size can be determined by \ref hipsparseSgebsr2gebsc_bufferSize
*  "hipsparseXgebsr2gebsc_bufferSize()".
*
*  For example, given the GEBSR matrix:
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       1 & 2 \\
*       3 & 4 \\
*       6 & 0
*      \end{array} &
*      \begin{array}{c c}
*       0 & 2 \\
*       0 & 0 \\
*       3 & 4
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       5 & 0 \\
*       1 & 2 \\
*       3 & 4
*      \end{array} &
*      \begin{array}{c c}
*       6 & 7 \\
*       3 & 4 \\
*       3 & 4
*      \end{array} \\
*   \end{array}
*  \right]
*  \f]
*
*  represented with the arrays:
*  \f[
*    \begin{align}
*    \text{bsrRowPtr} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \text{bsrColInd} &= \begin{bmatrix} 0 & 1 & 0 & 1 \end{bmatrix} \\
*    \text{bsrVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 6 & 0 & 0 & 2 & 0 & 0 & 3 & 4 & 5 & 0 & 1 & 2 & 3 & 4 & 6 & 7 & 3 & 4 & 3 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  this function converts the matrix to GEBSC format:
*  \f[
*    \begin{align}
*    \text{bscRowInd} &= \begin{bmatrix} 0 & 1 & 0 & 1 \end{bmatrix} \\
*    \text{bscColPtr} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \text{bscVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 6 & 0 & 5 & 0 & 1 & 2 & 3 & 4 & 0 & 2 & 0 & 0 & 3 & 4 & 6 & 7 & 3 & 4 & 3 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The GEBSC arrays, \p bscRowInd, \p bscColPtr, and \p bscVal must be allocated by the user prior
*  to calling \p hipsparseXgebsr2gebsc(). The \p bscRowInd array has size \p nnzb, the \p bscColPtr
*  array has size \p nb+1, and the \p bscVal array has size \p nnzb*rowBlockDim*colBlockDim.
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  mb          number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb          number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb        number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsrVal      array of \p nnzb * \p rowBlockDim * \p colBlockDim  elements of the sparse GEneral BSR matrix.
*  @param[in]
*  bsrRowPtr   array of \p m+1 elements that point to the start of every row of the
*              sparse GEneral BSR matrix.
*  @param[in]
*  bsrColInd   array of \p nnz elements containing the column indices of the sparse
*              GEneral BSR matrix.
*  @param[in]
*  rowBlockDim row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  colBlockDim col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  bscVal      array of \p nnz elements of the sparse BSC matrix.
*  @param[out]
*  bscRowInd   array of \p nnz elements containing the row indices of the sparse BSC
*              matrix.
*  @param[out]
*  bscColPtr   array of \p n+1 elements that point to the start of every column of the
*              sparse BSC matrix.
*  @param[in]
*  copyValues  \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase     \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              \ref hipsparseSgebsr2gebsc_bufferSize "hipsparseXgebsr2gebsc_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p bsrVal,
*              \p bsrRowPtr, \p bsrColInd, \p bscVal, \p bscRowInd, \p bscColPtr or
*              \p temp_buffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \snippet example_hipsparse_gebsr2gebsc.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const float*         bsrVal,
                                        const int*           bsrRowPtr,
                                        const int*           bsrColInd,
                                        int                  rowBlockDim,
                                        int                  colBlockDim,
                                        float*               bscVal,
                                        int*                 bscRowInd,
                                        int*                 bscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const double*        bsrVal,
                                        const int*           bsrRowPtr,
                                        const int*           bsrColInd,
                                        int                  rowBlockDim,
                                        int                  colBlockDim,
                                        double*              bscVal,
                                        int*                 bscRowInd,
                                        int*                 bscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const hipComplex*    bsrVal,
                                        const int*           bsrRowPtr,
                                        const int*           bsrColInd,
                                        int                  rowBlockDim,
                                        int                  colBlockDim,
                                        hipComplex*          bscVal,
                                        int*                 bscRowInd,
                                        int*                 bscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc(hipsparseHandle_t       handle,
                                        int                     mb,
                                        int                     nb,
                                        int                     nnzb,
                                        const hipDoubleComplex* bsrVal,
                                        const int*              bsrRowPtr,
                                        const int*              bsrColInd,
                                        int                     rowBlockDim,
                                        int                     colBlockDim,
                                        hipDoubleComplex*       bscVal,
                                        int*                    bscRowInd,
                                        int*                    bscColPtr,
                                        hipsparseAction_t       copyValues,
                                        hipsparseIndexBase_t    idxBase,
                                        void*                   temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GEBSR2GEBSC_H */
