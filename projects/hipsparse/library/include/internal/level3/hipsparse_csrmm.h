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
#ifndef HIPSPARSE_CSRMM_H
#define HIPSPARSE_CSRMM_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level3_module
*  \brief Sparse matrix dense matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR storage format, and the column-oriented dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the column-oriented dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot B + \beta \cdot C,
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
*      for(i = 0; i < ldc; ++i)
*      {
*          for(j = 0; j < n; ++j)
*          {
*              C[i][j] = beta * C[i][j];
*
*              for(k = csrRowPtr[i]; k < csrRowPtr[i + 1]; ++k)
*              {
*                  C[i][j] += alpha * csrVal[k] * B[csrColInd[k]][j];
*              }
*          }
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
*  @param[in]
*  k           number of columns of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix \f$A\f$.
*  @param[in]
*  B           array of dimension \p ldb*n (\f$op(B) == B\f$),
*              \p ldb*k otherwise.
*  @param[in]
*  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
*              (\f$op(B) == B\f$), \f$\max{(1, n)}\f$ otherwise.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  C           array of dimension \p ldc*n.
*  @param[in]
*  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
*              (\f$op(A) == A\f$), \f$\max{(1, k)}\f$ otherwise.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnz, \p ldb, \p ldc
*              \p descrA, \p alpha, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, 
*              \p B, \p beta or \p C is invalid.
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
*      //     1 2 0 3 0 0
*      // A = 0 4 5 0 0 0
*      //     0 0 0 7 8 0
*      //     0 0 1 2 4 1
*
*      int m   = 4;
*      int k   = 6;
*      int nnz = 11;
*      hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*      int hcsrRowPtr[4 + 1] = {0, 3, 5, 7, 11};
*      int hcsrColInd[11]    = {0, 1, 3, 1, 2, 3, 4, 2, 3, 4, 5};
*      float hcsrVal[11]      = {1, 2, 3, 4, 5, 7, 8, 1, 2, 4, 1};
*
*      // Set dimension n of B
*      int n = 3;
*
*      // Allocate and generate dense matrix B (k x n)
*      float hB[6 * 3] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 
*                         11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
*
*      int* dcsrRowPtr = NULL;
*      int* dcsrColInd = NULL;
*      float* dcsrVal = NULL;
*      hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*      hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*      hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*      hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*      // Copy B to the device
*      float* dB;
*      hipMalloc((void**)&dB, sizeof(float) * k * n);
*      hipMemcpy(dB, hB, sizeof(float) * k * n, hipMemcpyHostToDevice);
*
*      // alpha and beta
*      float alpha = 1.0f;
*      float beta  = 0.0f;
*
*      // Allocate memory for the resulting matrix C
*      float* dC;
*      hipMalloc((void**)&dC, sizeof(float) * m * n);
*
*      // Matrix descriptor
*      hipsparseMatDescr_t descr;
*      hipsparseCreateMatDescr(&descr);
*
*      // Perform the matrix multiplication
*      hipsparseScsrmm(handle,
*                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                      m,
*                      n,
*                      k,
*                      nnz,
*                      &alpha,
*                      descr,
*                      dcsrVal,
*                      dcsrRowPtr,
*                      dcsrColInd,
*                      dB,
*                      k,
*                      &beta,
*                      dC,
*                      m);
*
*      // Copy results to host
*      float hC[6 * 3];
*      hipMemcpy(hC, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost);
*
*      hipFree(dcsrRowPtr);
*      hipFree(dcsrColInd);
*      hipFree(dcsrVal);
*      hipFree(dB);
*      hipFree(dC);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level3_module
*  \brief Sparse matrix dense matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR storage format, and the column-oriented dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the column-oriented dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
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
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      for(i = 0; i < ldc; ++i)
*      {
*          for(j = 0; j < n; ++j)
*          {
*              C[i][j] = beta * C[i][j];
*
*              for(k = csrRowPtr[i]; k < csrRowPtr[i + 1]; ++k)
*              {
*                  C[i][j] += alpha * csrVal[k] * B[csrColInd[k]][j];
*              }
*          }
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  transA      matrix \f$A\f$ operation type.
*  @param[in]
*  transB      matrix \f$B\f$ operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
*  @param[in]
*  k           number of columns of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix \f$A\f$.
*  @param[in]
*  B           array of dimension \p ldb*n (\f$op(B) == B\f$),
*              \p ldb*k otherwise.
*  @param[in]
*  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
*              (\f$op(B) == B\f$), \f$\max{(1, n)}\f$ otherwise.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  C           array of dimension \p ldc*n.
*  @param[in]
*  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
*              (\f$op(A) == A\f$), \f$\max{(1, k)}\f$ otherwise.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnz, \p ldb, \p ldc
*              \p descrA, \p alpha, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, 
*              \p B, \p beta or \p C is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const float*              csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const float*              B,
                                   int                       ldb,
                                   const float*              beta,
                                   float*                    C,
                                   int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const double*             csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const double*             B,
                                   int                       ldb,
                                   const double*             beta,
                                   double*                   C,
                                   int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipComplex*         csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipComplex*         B,
                                   int                       ldb,
                                   const hipComplex*         beta,
                                   hipComplex*               C,
                                   int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipDoubleComplex*   csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipDoubleComplex*   B,
                                   int                       ldb,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         C,
                                   int                       ldc);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRMM_H */
