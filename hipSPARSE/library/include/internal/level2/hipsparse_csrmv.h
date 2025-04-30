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
#ifndef HIPSPARSE_CSRMV_H
#define HIPSPARSE_CSRMV_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if transA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if transA == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      for(i = 0; i < m; ++i)
*      {
*          y[i] = beta * y[i];
*
*          for(j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j)
*          {
*              y[i] = y[i] + alpha * csrVal[j] * x[csrColInd[j]];
*          }
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  transA      matrix operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
*              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
*              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n or \p nnz, \p descr, 
*              \p alpha, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p x, 
*              \p beta or \p y is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // alpha * ( 1.0  0.0  2.0 ) * ( 1.0 ) + beta * ( 4.0 ) = (  31.1 )
*      //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
*      //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
*      //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )
*
*      int m = 4;
*      int n = 3;
*      int nnz = 8;
*
*      // CSR row pointers
*      int hcsrRowPtr[5] = {0, 2, 4, 6, 8};
*
*      // CSR column indices
*      int hcsrColInd[8] = {0, 2, 0, 2, 0, 1, 0, 2};
*
*      // CSR values
*      double hcsrVal[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
*
*      // Transposition of the matrix
*      hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*      double beta  = 1.3;
*
*      // x and y
*      double hx[3] = {1.0, 2.0, 3.0};
*      double hy[4] = {4.0, 5.0, 6.0, 7.0};
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*
*      // Offload data to device
*      int* dcsrRowPtr;
*      int* dcsrColInd;
*      double*        dcsrVal;
*      double*        dx;
*      double*        dy;
*
*      hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*      hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*      hipMalloc((void**)&dcsrVal, sizeof(double) * nnz);
*      hipMalloc((void**)&dx, sizeof(double) * n);
*      hipMalloc((void**)&dy, sizeof(double) * m);
*
*      hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dcsrVal, hcsrVal, sizeof(double) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(double) * m, hipMemcpyHostToDevice);
*
*      // Call dcsrmv to perform y = alpha * A x + beta * y
*      hipsparseDcsrmv(handle,
*                      trans,
*                      m,
*                      n,
*                      nnz,
*                      &alpha,
*                      descr,
*                      dcsrVal,
*                      dcsrRowPtr,
*                      dcsrColInd,
*                      dx,
*                      &beta,
*                      dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroyMatDescr(descr);
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dcsrRowPtr);
*      hipFree(dcsrColInd);
*      hipFree(dcsrVal);
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRMV_H */
