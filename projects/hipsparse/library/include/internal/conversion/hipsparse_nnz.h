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
#ifndef HIPSPARSE_NNZ_H
#define HIPSPARSE_NNZ_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  \p hipsparseXnnz computes the number of nonzero elements per row or column and the total
*  number of nonzero elements in a dense matrix.
*
*  \details
*  For example, given the dense matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  then using \p dirA == \ref HIPSPARSE_DIRECTION_ROW results in:
*  \f[
*    \begin{align}
*    \text{nnzPerRowColumn} &= \begin{bmatrix} 2 & 2 & 3 \end{bmatrix} \\
*    \text{nnzTotalDevHostPtr} &= 7
*    \end{align}
*  \f]
*
*  while using \p dirA == \ref HIPSPARSE_DIRECTION_COLUMN results in:
*  \f[
*    \begin{align}
*    \text{nnzPerRowColumn} &= \begin{bmatrix} 3 & 1 & 1 & 2 \end{bmatrix} \\
*    \text{nnzTotalDevHostPtr} &= 7
*    \end{align}
*  \f]
*
*  The array \p nnzPerRowColumn must be allocated by the user before calling \p hipsparseXnnz and
*  has length equal to \p m if \p dirA == \ref HIPSPARSE_DIRECTION_ROW or \p n if
*  \p dirA == \ref HIPSPARSE_DIRECTION_COLUMN.
*
*  For a complete code example on its usage, see the example found with hipsparseSdense2csr().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the rocsparse library context queue.
*  @param[in]
*  dirA               direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW
*                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  descrA             the descriptor of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[out]
*  nnzPerRowColumn    array of size \p m or \p n containing the number of nonzero elements per row or column, respectively.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p A or \p nnzPerRowColumn or
*              \p nnzTotalDevHostPtr pointer is invalid.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const float*              A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const double*             A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipComplex*         A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipDoubleComplex*   A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_NNZ_H */
