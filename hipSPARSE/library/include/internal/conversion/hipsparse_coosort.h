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
#ifndef HIPSPARSE_COOSORT_H
#define HIPSPARSE_COOSORT_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix
*
*  \details
*  \p hipsparseXcoosort_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes required by \ref hipsparseXcoosortByRow() and \ref hipsparseXcoosortByColumn(). 
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  m                   number of rows of the sparse COO matrix.
*  @param[in]
*  n                   number of columns of the sparse COO matrix.
*  @param[in]
*  nnz                 number of non-zero entries of the sparse COO matrix.
*  @param[in]
*  cooRows             array of \p nnz elements containing the row indices of the sparse
*                      COO matrix.
*  @param[in]
*  cooCols             array of \p nnz elements containing the column indices of the sparse
*                      COO matrix.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by
*                      hipsparseXcoosortByRow() and hipsparseXcoosortByColumn().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cooRows, 
*              \p cooCols or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cooRows,
                                                  const int*        cooCols,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by row
*
*  \details
*  \p hipsparseXcoosortByRow sorts a matrix in COO format by row. The sorted
*  permutation vector \p P can be used to obtain sorted \p cooVal array. In this
*  case, \p P must be initialized as the identity permutation, see
*  \ref hipsparseCreateIdentityPermutation(). To apply the permutation vector to the COO 
*  values, see hipsparse \ref hipsparseSgthr "hipsparseXgthr()".
*
*  \p hipsparseXcoosortByRow requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  \ref hipsparseXcoosort_bufferSizeExt().
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
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[inout]
*  cooRows         array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[inout]
*  cooCols         array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcoosort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cooRows, 
*              \p cooCols or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in COO format (with unsorted row indices)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcooRowInd[8] = {0, 2, 0, 1, 1, 0, 2, 2};
*    int hcooColInd[8] = {0, 0, 1, 1, 2, 3, 3, 4};
*    float hcooVal[8]   = {1.0f, 6.0f, 2.0f, 4.0f, 5.0f, 3.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int* dcooRowInd = nullptr;
*    int* dcooColInd = nullptr;
*    float* dcooVal = nullptr;
*    hipMalloc((void**)&dcooRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcooRowInd, hcooRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooColInd, hcooColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooVal, hcooVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, dcooRowInd, dcooColInd, &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcoosortByRow(handle, m, n, nnz, dcooRowInd, dcooColInd, dperm, dbuffer);
*
*    float* dcooValSorted = nullptr;
*    hipMalloc((void**)&dcooValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcooVal, dcooValSorted, dperm, base);
*
*    hipFree(dcooRowInd);
*    hipFree(dcooColInd);
*    hipFree(dcooVal);
*    hipFree(dcooValSorted);
*    hipFree(dperm);
*   
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle,
                                         int               m,
                                         int               n,
                                         int               nnz,
                                         int*              cooRows,
                                         int*              cooCols,
                                         int*              P,
                                         void*             pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by column
*
*  \details
*  \p hipsparseXcoosortByColumn sorts a matrix in COO format by column. The sorted
*  permutation vector \p P can be used to obtain sorted \p cooVal array. In this
*  case, \p P must be initialized as the identity permutation, see
*  \ref hipsparseCreateIdentityPermutation(). To apply the permutation vector to the COO 
*  values, see hipsparse \ref hipsparseSgthr "hipsparseXgthr()".
*
*  \p hipsparseXcoosortByColumn requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  \ref hipsparseXcoosort_bufferSizeExt().
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
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[inout]
*  cooRows         array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[inout]
*  cooCols         array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcoosort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cooRows, 
*              \p cooCols or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in COO format (with unsorted column indices)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcooRowInd[8] = {0, 0, 0, 1, 1, 2, 2, 2};
*    int hcooColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcooVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int* dcooRowInd = nullptr;
*    int* dcooColInd = nullptr;
*    float* dcooVal = nullptr;
*    hipMalloc((void**)&dcooRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcooRowInd, hcooRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooColInd, hcooColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooVal, hcooVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, dcooRowInd, dcooColInd, &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcoosortByColumn(handle, m, n, nnz, dcooRowInd, dcooColInd, dperm, dbuffer);
*
*    float* dcooValSorted = nullptr;
*    hipMalloc((void**)&dcooValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcooVal, dcooValSorted, dperm, base);
*
*    hipFree(dcooRowInd);
*    hipFree(dcooColInd);
*    hipFree(dcooVal);
*    hipFree(dcooValSorted);
*    hipFree(dperm);
*   
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               nnz,
                                            int*              cooRows,
                                            int*              cooCols,
                                            int*              P,
                                            void*             pBuffer);

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_COOSORT_H */
