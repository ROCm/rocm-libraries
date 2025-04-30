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
#ifndef HIPSPARSE_CSR2CSR_COMPRESS_H
#define HIPSPARSE_CSR2CSR_COMPRESS_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
*
*  \details
*  \p hipsparseXcsr2csr_compress converts a CSR matrix into a compressed CSR matrix by
*  removing entries in the input CSR matrix that are below a non-negative threshold \p tol:
*
*  \f[ 
*   C(i,j) = A(i, j) \text{  if |A(i, j)| > tol}
*  \f]
*
*  The user must first call \ref hipsparseSnnz_compress "hipsparseXnnz_compress()" to determine the number 
*  of nonzeros per row as well as the total number of nonzeros that will exist in resulting compressed CSR 
*  matrix. The user then uses this information to allocate the column indices array \p csrColIndC and the 
*  values array \p csrValC. The user then calls \p hipsparseXcsr2csr_compress to complete the conversion.
*
*  \note
*  In the case of complex matrices only the magnitude of the real part of \p tol is used.
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*  @param[in]
*  n             number of columns of the sparse CSR matrix.
*  @param[in]
*  descrA        matrix descriptor for the CSR matrix
*  @param[in]
*  csrValA       array of \p nnzA elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
*                uncompressed sparse CSR matrix.
*  @param[in]
*  csrColIndA    array of \p nnzA elements containing the column indices of the uncompressed
*                sparse CSR matrix.
*  @param[in]
*  nnzA          number of elements in the column indices and values arrays of the uncompressed
*                sparse CSR matrix.
*  @param[in]
*  nnzPerRow     array of length \p m containing the number of entries that will be kept per row in
*                the final compressed CSR matrix.
*  @param[out]
*  csrValC       array of \p nnzC elements of the compressed sparse CSC matrix.
*  @param[out]
*  csrRowPtrC    array of \p m+1 elements that point to the start of every column of the compressed
*                sparse CSR matrix.
*  @param[out]
*  csrColIndC    array of \p nnzC elements containing the row indices of the compressed
*                sparse CSR matrix.
*  @param[in]
*  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
*                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
*                are removed in output compressed CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p tol, \p csrValA, \p csrRowPtrA,
*              \p csrColIndA, \p csrValC, \p csrRowPtrC, \p csrColIndC or \p nnzPerRow pointer is invalid.
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
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtrA[4] = {0, 3, 5, 8};
*    int hcsrColIndA[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrValA[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m    = 3;
*    int n    = 5;
*    int nnzA = 8;
*
*    float tol = 5.9f;
*    
*    int* dcsrRowPtrA = nullptr;
*    int* dcsrColIndA = nullptr;
*    float* dcsrValA = nullptr;
*    hipMalloc((void**)&dcsrRowPtrA, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColIndA, sizeof(int) * nnzA);
*    hipMalloc((void**)&dcsrValA, sizeof(float) * nnzA);
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA, sizeof(int) * nnzA, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA, sizeof(float) * nnzA, hipMemcpyHostToDevice);
*
*    // Allocate memory for the nnz_per_row array
*    int* dnnz_per_row;
*    hipMalloc((void**)&dnnz_per_row, sizeof(int) * m);
*
*    // Call snnz_compress() which fills in nnz_per_row array and finds the number
*    // of entries that will be in the compressed CSR matrix
*    int nnzC;
*    hipsparseSnnz_compress(handle, m, descr, dcsrValA, dcsrRowPtrA, dnnz_per_row, &nnzC, tol);
*
*    int* dcsrRowPtrC = nullptr;
*    int* dcsrColIndC = nullptr;
*    float* dcsrValC = nullptr;
*    hipMalloc((void**)&dcsrRowPtrC, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC);
*
*    hipsparseScsr2csr_compress(handle,
*                               m,
*                               n,
*                               descr,
*                               dcsrValA,
*                               dcsrColIndA,
*                               dcsrRowPtrA,
*                               nnzA,
*                               dnnz_per_row,
*                               dcsrValC,
*                               dcsrColIndC,
*                               dcsrRowPtrC,
*                               tol);
*
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*
*    hipFree(dnnz_per_row);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             float*                    csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             float                     tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             double*                   csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             double                    tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipComplex*               csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipComplex                tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipDoubleComplex*         csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipDoubleComplex          tol);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSR2CSR_COMPRESS_H */
