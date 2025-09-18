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
#ifndef HIPSPARSE_CSR2COO_H
#define HIPSPARSE_CSR2COO_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse COO matrix
*
*  \details
*  \p hipsparseXcsr2coo converts the CSR array containing the row offsets, that point
*  to the start of every row, into a COO array of row indices. All arrays are assumed
*  to be allocated by the user prior to calling \p hipsparseXcsr2coo.
*
*  For example, given the CSR row pointer array (assuming zero index base):
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 1 & 3 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  Calling \p hipsparseXcsr2coo() results in the COO row indices array:
*  \f[
*    \begin{align}
*    \text{cooRowInd} &= \begin{bmatrix} 0 & 1 & 1 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  \note
*  It can also be used to convert a CSC array containing the column offsets into a COO
*  array of column indices.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  csrRowPtr   array of \p m+1 elements that point to the start of every row
*              of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[out]
*  cooRowInd   array of \p nnz elements containing the row indices of the sparse COO
*              matrix.
*  @param[in]
*  idxBase    \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p csrRowPtr or \p cooRowInd
*              pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t    handle,
                                    const int*           csrRowPtr,
                                    int                  nnz,
                                    int                  m,
                                    int*                 cooRowInd,
                                    hipsparseIndexBase_t idxBase);

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSR2COO_H */
