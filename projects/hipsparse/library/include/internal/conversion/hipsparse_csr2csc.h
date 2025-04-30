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
#ifndef HIPSPARSE_CSR2CSC_H
#define HIPSPARSE_CSR2CSC_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p hipsparseXcsr2csc converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2csc
*  can also be used to convert a CSC matrix into a CSR matrix. \p copyValues decides
*  whether \p cscSortedVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  For example given the matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  Represented using the sparse CSR format as:
*  \f[
*    \begin{align}
*    \text{csrSortedRowPtr} &= \begin{bmatrix} 0 & 2 & 4 & 7 \end{bmatrix} \\
*    \text{csrSortedColInd} &= \begin{bmatrix} 0 & 3 & 0 & 1 & 0 & 2 & 3 \end{bmatrix} \\
*    \text{csrSortedVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix}
*    \end{align}
*  \f]
*
*  this function converts to sparse CSC format:
*  \f[
*    \begin{align}
*    \text{cscSortedRowInd} &= \begin{bmatrix} 0 & 1 & 2 & 1 & 2 & 0 & 2 \end{bmatrix} \\
*    \text{cscSortedColPtr} &= \begin{bmatrix} 0 & 3 & 4 & 5 & 7 \end{bmatrix} \\
*    \text{cscSortedVal} &= \begin{bmatrix} 1 & 3 & 5 & 4 & 6 & 2 & 7 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The CSC arrays, \p cscSortedRowInd, \p cscSortedColPtr, and \p cscSortedVal must be allocated by the 
*  user prior to calling \p hipsparseXcsr2csc().
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix.
*  @param[in]
*  n               number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrSortedVal    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtr array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[in]
*  csrSortedColInd array of \p nnz elements containing the column indices of the sparse
*                  CSR matrix.
*  @param[out]
*  cscSortedVal    array of \p nnz elements of the sparse CSC matrix.
*  @param[out]
*  cscSortedRowInd array of \p nnz elements containing the row indices of the sparse CSC
*                  matrix.
*  @param[out]
*  cscSortedColPtr array of \p n+1 elements that point to the start of every column of the
*                  sparse CSC matrix.
*  @param[in]
*  copyValues      \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase         \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrSortedVal, \p csrSortedRowPtr,
*              \p csrSortedColInd, \p cscSortedVal, \p cscSortedRowInd or \p cscSortedColPtr pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*    hipsparseAction_t action = HIPSPARSE_ACTION_NUMERIC;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dcscRowInd = nullptr;
*    int* dcscColPtr = nullptr;
*    float* dcsc_val   = nullptr;
*    hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1));
*    hipMalloc((void**)&dcsc_val, sizeof(float) * nnz);
*
*    hipsparseScsr2csc(handle, m, n, nnz, dcsrVal, dcsrRowPtr, dcsrColInd, dcsc_val, dcscRowInd, dcscColPtr, action, base);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    
*    hipFree(dcscRowInd);
*    hipFree(dcscColPtr);
*    hipFree(dcsc_val);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const float*         csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    float*               cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const double*        csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    double*              cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const hipComplex*    csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    hipComplex*          cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     nnz,
                                    const hipDoubleComplex* csrSortedVal,
                                    const int*              csrSortedRowPtr,
                                    const int*              csrSortedColInd,
                                    hipDoubleComplex*       cscSortedVal,
                                    int*                    cscSortedRowInd,
                                    int*                    cscSortedColPtr,
                                    hipsparseAction_t       copyValues,
                                    hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
/*! \ingroup conv_module
*  \brief This function computes the size of the user allocated temporary storage buffer used
*  when converting a sparse CSR matrix into a sparse CSC matrix.
*
*  \details
*  \p hipsparseCsr2cscEx2_bufferSize calculates the required user allocated temporary buffer needed 
*  by \ref hipsparseCsr2cscEx2 to convert a CSR matrix into a CSC matrix. \ref hipsparseCsr2cscEx2
*  can also be used to convert a CSC matrix into a CSR matrix. \p copyValues decides
*  whether \p cscVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrVal             array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr          array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrColInd          array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[in]
*  cscVal             array of \p nnz elements of the sparse CSC matrix.
*  @param[in]
*  cscColPtr          array of \p n+1 elements that point to the start of every column of the
*                     sparse CSC matrix.
*  @param[in]
*  cscRowInd          array of \p nnz elements containing the row indices of the sparse
*                     CSC matrix.
*  @param[in]
*  valType            The data type of the values arrays \p csrVal and \p cscVal. Can be HIP_R_32F, 
*                     HIP_R_64F, HIP_C_32F or HIP_C_64F
*  @param[in]
*  copyValues         \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase            \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*  @param[in]
*  alg                HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1 or HIPSPARSE_CSR2CSC_ALG2.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseCsr2cscEx2().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrRowPtr, \p csrColInd or
*              \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(hipsparseHandle_t     handle,
                                                 int                   m,
                                                 int                   n,
                                                 int                   nnz,
                                                 const void*           csrVal,
                                                 const int*            csrRowPtr,
                                                 const int*            csrColInd,
                                                 void*                 cscVal,
                                                 int*                  cscColPtr,
                                                 int*                  cscRowInd,
                                                 hipDataType           valType,
                                                 hipsparseAction_t     copyValues,
                                                 hipsparseIndexBase_t  idxBase,
                                                 hipsparseCsr2CscAlg_t alg,
                                                 size_t*               pBufferSizeInBytes);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p hipsparseCsr2cscEx2 converts a CSR matrix into a CSC matrix. \p hipsparseCsr2cscEx2
*  can also be used to convert a CSC matrix into a CSR matrix. \p copyValues decides
*  whether \p cscVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
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
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrVal      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr   array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csrColInd   array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  cscVal      array of \p nnz elements of the sparse CSC matrix.
*  @param[in]
*  cscColPtr   array of \p n+1 elements that point to the start of every column of the
*              sparse CSC matrix.
*  @param[in]
*  cscRowInd   array of \p nnz elements containing the row indices of the sparse
*              CSC matrix.
*  @param[in]
*  valType     The data type of the values arrays \p csrVal and \p cscVal. Can be HIP_R_32F, 
*              HIP_R_64F, HIP_C_32F or HIP_C_64F
*  @param[in]
*  copyValues  \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase     \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*  @param[in]
*  alg         HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1 or HIPSPARSE_CSR2CSC_ALG2.
*  @param[in]
*  buffer      temporary storage buffer allocated by the user, size is returned by
*              hipsparseCsr2cscEx2_bufferSize().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrRowPtr, \p csrColInd or
*              \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*    hipsparseAction_t action = HIPSPARSE_ACTION_NUMERIC;
*    hipsparseCsr2CscAlg_t alg = HIPSPARSE_CSR2CSC_ALG_DEFAULT;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dcscRowInd = nullptr;
*    int* dcscColPtr = nullptr;
*    float* dcsc_val   = nullptr;
*    hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1));
*    hipMalloc((void**)&dcsc_val, sizeof(float) * nnz);
*
*    size_t bufferSize;
*    hipsparseCsr2cscEx2_bufferSize(handle, 
*                                   m, 
*                                   n, 
*                                   nnz, 
*                                   dcsrVal, 
*                                   dcsrRowPtr, 
*                                   dcsrColInd, 
*                                   dcsc_val, 
*                                   dcscColPtr, 
*                                   dcscRowInd,
*                                   HIP_R_32F, 
*                                   action, 
*                                   base, 
*                                   alg, 
*                                   &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    hipsparseCsr2cscEx2(handle, 
*                        m, 
*                        n, 
*                        nnz, 
*                        dcsrVal, 
*                        dcsrRowPtr, 
*                        dcsrColInd, 
*                        dcsc_val, 
*                        dcscColPtr, 
*                        dcscRowInd, 
*                        HIP_R_32F, 
*                        action, 
*                        base, 
*                        alg, 
*                        dbuffer);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    
*    hipFree(dcscRowInd);
*    hipFree(dcscColPtr);
*    hipFree(dcsc_val);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsr2cscEx2(hipsparseHandle_t     handle,
                                      int                   m,
                                      int                   n,
                                      int                   nnz,
                                      const void*           csrVal,
                                      const int*            csrRowPtr,
                                      const int*            csrColInd,
                                      void*                 cscVal,
                                      int*                  cscColPtr,
                                      int*                  cscRowInd,
                                      hipDataType           valType,
                                      hipsparseAction_t     copyValues,
                                      hipsparseIndexBase_t  idxBase,
                                      hipsparseCsr2CscAlg_t alg,
                                      void*                 buffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSR2CSC_H */
