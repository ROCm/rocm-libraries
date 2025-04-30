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
#ifndef HIPSPARSE_CSCSORT_H
#define HIPSPARSE_CSCSORT_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes required by hipsparseXcscsort(). The temporary storage buffer must be 
*  allocated by the user.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  m                   number of rows of the sparse CSC matrix.
*  @param[in]
*  n                   number of columns of the sparse CSC matrix.
*  @param[in]
*  nnz                 number of non-zero entries of the sparse CSC matrix.
*  @param[in]
*  cscColPtr           array of \p n+1 elements that point to the start of every column of
*                      the sparse CSC matrix.
*  @param[in]
*  cscRowInd           array of \p nnz elements containing the row indices of the sparse
*                      CSC matrix.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by
*                      \ref hipsparseXcscsort().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cscColPtr, \p cscRowInd or
*              \p pBufferSizeInBytes pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cscColPtr,
                                                  const int*        cscRowInd,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort sorts a matrix in CSC format. The sorted permutation vector
*  \p P can be used to obtain sorted \p cscVal array. In this case, \p P must be
*  initialized as the identity permutation, see \ref hipsparseCreateIdentityPermutation(). To 
*  apply the permutation vector to the CSC values, see hipsparse \ref hipsparseSgthr 
*  "hipsparseXgthr()".
*
*  \p hipsparseXcscsort requires extra temporary storage buffer that has to be allocated by
*  the user. Storage buffer size can be determined by \ref hipsparseXcscsort_bufferSizeExt().
*
*  \note
*  \p P can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSC matrix.
*  @param[in]
*  n               number of columns of the sparse CSC matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse CSC matrix.
*  @param[in]
*  descrA          descriptor of the sparse CSC matrix. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  cscColPtr       array of \p n+1 elements that point to the start of every column of
*                  the sparse CSC matrix.
*  @param[inout]
*  cscRowInd       array of \p nnz elements containing the row indices of the sparse
*                  CSC matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcscsort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p descrA, \p cscColPtr, 
*              \p cscRowInd or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Sparse matrix in CSC format (unsorted row indices)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcscRowInd[8] = {2, 0, 1, 0, 1, 2, 0, 2};
*    int hcscColPtr[6] = {0, 2, 4, 5, 7, 8};
*    float hcscVal[8]  = {6.0f, 1.0f, 4.0f, 2.0f, 5.0f, 7.0f, 3.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*
*    int* dcscRowInd = nullptr;
*    int* dcscColPtr = nullptr;
*    float* dcscVal = nullptr;
*    hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1));
*    hipMalloc((void**)&dcscVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcscRowInd, hcscRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcscColPtr, hcscColPtr, sizeof(int) * (n + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcscVal, hcscVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcscsort_bufferSizeExt(handle,
*                                    m,
*                                    n,
*                                    nnz,
*                                    dcscColPtr,
*                                    dcscRowInd,
*                                    &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcscsort(handle,
*                      m,
*                      n,
*                      nnz,
*                      descr,
*                      dcscColPtr,
*                      dcscRowInd,
*                      dperm,
*                      dbuffer);
*
*    float* dcscValSorted = nullptr;
*    hipMalloc((void**)&dcscValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcscVal, dcscValSorted, dperm, HIPSPARSE_INDEX_BASE_ZERO);
*
*    hipFree(dcscRowInd);
*    hipFree(dcscColPtr);
*    hipFree(dcscVal);
*    hipFree(dcscValSorted);
*
*    hipFree(dbuffer);
*    hipFree(dperm);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                cscColPtr,
                                    int*                      cscRowInd,
                                    int*                      P,
                                    void*                     pBuffer);

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSCSORT_H */
