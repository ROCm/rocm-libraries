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
#ifndef HIPSPARSE_DENSE2CSC_H
#define HIPSPARSE_DENSE2CSC_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXdense2csc converts the matrix A in dense format into a sparse matrix in CSC format.
*
*  \details
*  Given a dense, column ordered, matrix \p A with leading dimension \p ld where \p ld>=m,
*  \p hipsparseXdense2csc converts the matrix to a sparse CSC format matrix.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays
*  are filled in based on number of nonzeros per row, which can be pre-computed with
*  \ref hipsparseSnnz "hipsparseXnnz()". We can set the desired index base in the output CSC
*  matrix by setting it in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase().
*
*  As an example, if using index base zero (i.e. the default) and the dense
*  matrix:
*
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  where the \p A values have column ordering with leading dimension \p ld=m:
*  \f[
*    \text{A} &= \begin{bmatrix} 1 & 3 & 5 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 \end{bmatrix} \\
*  \f]
*
*  the conversion results in the CSC arrays:
*
*  \f[
*    \begin{align}
*    \text{cscRowInd} &= \begin{bmatrix} 0 & 1 & 2 & 1 & 2 & 0 & 2 \end{bmatrix} \\
*    \text{cscColPtr} &= \begin{bmatrix} 0 & 3 & 4 & 5 & 7 \end{bmatrix} \\
*    \text{cscVal} &= \begin{bmatrix} 1 & 3 & 5 & 4 & 6 & 2 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  This function works very similar to \ref hipsparseSdense2csr "hipsparseXdense2csr()".
&  See hipsparseSdense2csr() for a code example.
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  m            number of rows of the dense matrix \p A.
*  @param[in]
*  n            number of columns of the dense matrix \p A.
*  @param[in]
*  descr        the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also
*               any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  A            array of dimensions (\p ld, \p n)
*  @param[in]
*  ld           leading dimension of dense array \p A.
*  @param[in]
*  nnzPerColumn array of size \p n containing the number of non-zero elements per column.
*  @param[out]
*  cscVal       array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  cscRowInd    integer array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  cscColPtr    integer array of \p n+1 elements that contains the start of every column and the end of the last column plus one.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p nnzPerColumn or \p cscVal \p cscColPtr
*              or \p cscRowInd pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      float*                    cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      double*                   cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      hipComplex*               cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      hipDoubleComplex*         cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_DENSE2CSC_H */
