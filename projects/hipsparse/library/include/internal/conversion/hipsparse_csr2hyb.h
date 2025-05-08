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
#ifndef HIPSPARSE_CSR2HYB_H
#define HIPSPARSE_CSR2HYB_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse HYB matrix
*
*  \details
*  \p hipsparseXcsr2hyb converts a CSR matrix into a HYB matrix. It is assumed
*  that \p hyb has been initialized with \ref hipsparseCreateHybMat().
*
*  \note
*  This function requires a significant amount of storage for the HYB matrix,
*  depending on the matrix structure.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle            handle to the hipsparse library context queue.
*  @param[in]
*  m                 number of rows of the sparse CSR matrix.
*  @param[in]
*  n                 number of columns of the sparse CSR matrix.
*  @param[in]
*  descrA            descriptor of the sparse CSR matrix. Currently, only
*                    \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedValA     array containing the values of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA  array of \p m+1 elements that point to the start of every row of the
*                    sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA  array containing the column indices of the sparse CSR matrix.
*  @param[out]
*  hybA              sparse matrix in HYB format.
*  @param[in]
*  userEllWidth      width of the ELL part of the HYB matrix (only required if
*                    \p partitionType == \ref HIPSPARSE_HYB_PARTITION_USER).
*  @param[in]
*  partitionType     \ref HIPSPARSE_HYB_PARTITION_AUTO (recommended),
*                    \ref HIPSPARSE_HYB_PARTITION_USER or
*                    \ref HIPSPARSE_HYB_PARTITION_MAX.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p userEllWidth, \p partitionType, 
*              \p descrA, \p hybA, \p csrSortedValA, \p csrSortedRowPtrA or \p csrSortedColIndA pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ALLOC_FAILED the buffer for the HYB matrix could not be
*              allocated.
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
*    int userEllWidth = 2;
*    hipsparseHybPartition_t partitionType = HIPSPARSE_HYB_PARTITION_AUTO;
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
*    hipsparseHybMat_t hyb;
*    hipsparseCreateHybMat(&hyb);
*
*    hipsparseScsr2hyb(handle, m, n, descr, dcsrVal, dcsrRowPtr, dcsrColInd, hyb, userEllWidth, partitionType);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*
*    hipsparseDestroyHybMat(hyb);    
*    hipsparseDestroyMatDescr(descr);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSR2HYB_H */
