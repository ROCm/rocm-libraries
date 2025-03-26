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
#ifndef HIPSPARSE_NNZ_COMPRESS_H
#define HIPSPARSE_NNZ_COMPRESS_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  This function is used as the first step in converting a CSR matrix to a compressed CSR matrix.
*
*  \details
*  Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
*  in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
*  of remaining elements in the matrix. 
*
*  Specifically given an input sparse matrix A in CSR format, the resulting compressed sparse CSR matrix C is 
*  computed using:
*  \f[ 
*   C(i,j) = A(i, j) \text{  if |A(i, j)| > tol}
*  \f]
*
*  The user first allocates \p nnzPerRow with size \p m elements. Then calling \p hipsparseXnnz_compress, 
*  the function fills in the \p nnzPerRow array and sets the total number of nonzeros found in \p nnzC.
*
*  See hipsparseScsr2csr_compress() for full code example.
*
*  \note
*  In the case of complex matrices only the magnitude of the real part of \p tol is used.
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*  @param[in]
*  descrA        the descriptor of the sparse CSR matrix.
*  @param[in]
*  csrValA       array of \p nnzA elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
*                uncompressed sparse CSR matrix.
*  @param[out]
*  nnzPerRow     array of length \p m containing the number of entries that will be kept per row in
*                the final compressed CSR matrix.
*  @param[out]
*  nnzC          number of elements in the column indices and values arrays of the compressed
*                sparse CSR matrix. Can be either host or device pointer.
*  @param[in]
*  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
*                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
*                are removed in output compressed CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p tol, \p csrValA, \p csrRowPtrA, \p nnzPerRow or \p nnzC
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr_A;
*    hipsparseCreateMatDescr(&descr_A);
*
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    float tol = 4.2f;
*
*    int m     = 3;
*    int n     = 5;
*    int nnz_A = 8;
*
*    int hcsrRowPtr_A[4] = {0, 3, 5, 8};             
*    float hcsrVal_A[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
*
*    int* dcsrRowPtr_A = nullptr;
*    float* dcsrVal_A = nullptr;
*    hipMalloc((void**)&dcsrRowPtr_A, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrVal_A, sizeof(float) * nnz_A);
*
*    hipMemcpy(dcsrRowPtr_A, hcsrRowPtr_A, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal_A, hcsrVal_A, sizeof(float) * nnz_A, hipMemcpyHostToDevice);
*
*    // Allocate memory for the nnz_per_row array
*    int* dnnz_per_row;
*    hipMalloc((void**)&dnnz_per_row, sizeof(int) * m);
*
*    // Call snnz_compress() which fills in nnz_per_row array and finds the number
*    // of entries that will be in the compressed CSR matrix
*    int nnz_C;
*    hipsparseSnnz_compress(handle, m, descr_A, dcsrVal_A, dcsrRowPtr_A, dnnz_per_row, &nnz_C, tol);
*
*    hipFree(dcsrRowPtr_A);
*    hipFree(dcsrVal_A);
*    hipFree(dnnz_per_row);
*
*    hipsparseDestroyMatDescr(descr_A);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_NNZ_COMPRESS_H */
