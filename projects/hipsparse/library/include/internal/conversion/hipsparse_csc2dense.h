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
#ifndef HIPSPARSE_CSC2DENSE_H
#define HIPSPARSE_CSC2DENSE_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXcsc2dense function converts the sparse matrix in CSC format into a dense matrix.
*
*  \details
*  Given the input CSC matrix of size \p mxn, the routine writes the matrix to the dense array \p A such 
*  that \p A has leading dimension \p ld and is column ordered. This means that \p A has size \p ldxn where
*  \p ld>=m. All the parameters are assumed to have been pre-allocated by the user. If the input CSC matrix 
*  has index base of one, it must be set in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase() 
*  prior to calling \p hipsparseXcsc2dense.
*
*  For example, consider the sparse CSC matrix:
*  \f[
*    \begin{align}
*    \text{cscRowInd} &= \begin{bmatrix} 0 & 1 & 2 & 1 & 2 & 0 & 2 \end{bmatrix} \\
*    \text{cscColPtr} &= \begin{bmatrix} 0 & 3 & 4 & 5 & 7 \end{bmatrix} \\
*    \text{cscVal} &= \begin{bmatrix} 1 & 3 & 5 & 4 & 6 & 2 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \p hipsparseXcsc2dense is used to convert to the dense matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  where the values in the \p A array are column ordered:
*  \f[
*    \text{A} &= \begin{bmatrix} 1 & 3 & 5 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 \end{bmatrix} \\
*  \f]
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the application 
*  on the host before the entire result is ready.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also 
*              any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  cscVal      array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  cscRowInd   integer array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  cscColPtr   integer array of \p n+1 elements that contains the start of every column and the end of the last column plus one.
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*  @param[out]
*  ld          leading dimension of dense array \p A.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p cscVal \p cscColPtr 
*              or \p cscRowInd pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      float*                    A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      double*                   A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipComplex*               A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipDoubleComplex*         A,
                                      int                       ld);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSC2DENSE_H */
