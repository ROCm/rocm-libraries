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
#ifndef HIPSPARSE_DENSE2CSR_H
#define HIPSPARSE_DENSE2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXdense2csr converts the matrix A in dense format into a sparse matrix in CSR format.
*
*  \details
*  Given a dense, column ordered, matrix \p A with leading dimension \p ld where \p ld>=m, 
*  \p hipsparseXdense2csr converts the matrix to a sparse CSR format matrix. All the parameters 
*  are assumed to have been pre-allocated by the user and the arrays are filled in based on number 
*  of nonzeros per row, which can be pre-computed with \ref hipsparseSnnz "hipsparseXnnz()". The 
*  desired index base in the output CSR matrix is set in the \ref hipsparseMatDescr_t. See 
*  \ref hipsparseSetMatIndexBase(). 
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
*  The conversion results in the CSR arrays:
*
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 4 & 7 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 0 & 1 & 0 & 2 & 3 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
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
*  nnzPerRow    array of size \p n containing the number of non-zero elements per row.
*  @param[out]
*  csrVal       array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  csrRowPtr    integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csrColInd    integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p nnzPerRow, \p csrVal \p csrRowPtr or \p csrColInd
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Dense matrix in column order
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    float hdense_A[15] = {1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};
*
*    int m         = 3;
*    int n         = 5;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*    float* ddense_A = nullptr;
*    hipMalloc((void**)&ddense_A, sizeof(float) * m * n);
*    hipMemcpy(ddense_A, hdense_A, sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*    // Allocate memory for the nnz_per_row_columns array
*    int* dnnz_per_row;
*    hipMalloc((void**)&dnnz_per_row, sizeof(int) * m);
*
*    int nnz_A;
*    hipsparseSnnz(handle, dir, m, n, descr, ddense_A, m, dnnz_per_row, &nnz_A);
*
*    // Allocate sparse CSR matrix
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipsparseSdense2csr(handle, m, n, descr, ddense_A, m, dnnz_per_row, dcsrVal, dcsrRowPtr, dcsrColInd);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dnnz_per_row);
*    hipFree(ddense_A);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      float*                    csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      double*                   csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipComplex*               csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipDoubleComplex*         csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_DENSE2CSR_H */
