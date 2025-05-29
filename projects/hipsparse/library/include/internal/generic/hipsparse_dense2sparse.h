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
#ifndef HIPSPARSE_DENSE2SPARSE_H
#define HIPSPARSE_DENSE2SPARSE_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseDenseToSparse_bufferSize computes the required user allocated buffer size needed when converting 
*  a dense matrix to a sparse matrix. This routine currently accepts the sparse matrix descriptor \p matB in CSR, 
*  CSC, or COO format. This routine is used to determine the size of the buffer 
*  needed in \ref hipsparseDenseToSparse_analysis and \ref hipsparseDenseToSparse_convert.
*
*  \p hipsparseDenseToSparse_bufferSize supports different data types for the dense and sparse matrices. See 
*  \ref hipsparseDenseToSparse_convert for a complete listing of all the data types available.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  matA                dense matrix descriptor.
*  @param[in]
*  matB                sparse matrix descriptor.
*  @param[in]
*  alg                 algorithm for the dense to sparse computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p pBufferSizeInBytes
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseConstDnMatDescr_t  matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseDnMatDescr_t       matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseDenseToSparse_analysis performs analysis that is later used in \ref hipsparseDenseToSparse_convert when 
*  converting a dense matrix to sparse matrix. This routine currently accepts the sparse matrix descriptor \p matB in CSR, 
*  CSC, or COO format. This routine takes a user allocated buffer whose size must first be computed 
*  using \ref hipsparseDenseToSparse_bufferSize.
*
*  \p hipsparseDenseToSparse_analysis supports different data types for the dense and sparse matrices. See 
*  \ref hipsparseDenseToSparse_convert for a complete listing of all the data types available.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  matA            dense matrix descriptor.
*  @param[in]
*  matB            sparse matrix descriptor.
*  @param[in]
*  alg             algorithm for the dense to sparse computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p externalBuffer
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseConstDnMatDescr_t  matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_convert converts a dense matrix to a sparse matrix. This routine currently accepts 
*  the sparse matrix descriptor \p matB in CSR, CSC, or COO format. This routine requires a user allocated buffer
*  whose size must be determined by first calling \ref hipsparseDenseToSparse_bufferSize.
*
*  The conversion of a dense matrix into a sparse one involves three steps. First, the user creates the dense and 
*  sparse matrix descriptors. Because the number of non-zeros that will exist in the sparse matrix is not known apriori, 
*  when creating the sparse matrix descriptor, the user simply sets the arrays to \p NULL and the non-zero count to zero.
*  For example, in the case of a CSR sparse matrix, this would look like:  
*  \code{.c}
*  hipsparseCreateCsr(&matB,
*                     m,
*                     n,
*                     0,
*                     dcsrRowPtrB, // This array can be allocated as its size (i.e. m + 1) is known
*                     NULL,        // Column indices array size is not yet known, pass NULL for now
*                     NULL,        // Values array size is not yet known, pass NULL for now
*                     rowIdxTypeB,
*                     colIdxTypeB,
*                     idxBaseB,
*                     dataTypeB);
*  \endcode
*  In the case of a COO sparse matrix, this would look like:
*  \code{.c}
*  hipsparseCreateCoo(&matB,
*                     m,
*                     n,
*                     0,
*                     NULL,  // Row indices array size is not yet known, pass NULL for now
*                     NULL,  // Column indices array size is not yet known, pass NULL for now
*                     NULL,  // Values array size is not yet known, pass NULL for now
*                     rowIdxTypeB,
*                     colIdxTypeB,
*                     idxBaseB,
*                     dataTypeB);
*  \endcode
*  Once the descriptors have been created, the user calls \ref hipsparseDenseToSparse_bufferSize. This routine will 
*  determine the size of the required temporary storage buffer. The user then allocates this buffer and passes it to
*  \ref hipsparseDenseToSparse_analysis which will perform analysis on the dense matrix in order to determine the number 
*  of non-zeros that will exist in the sparse matrix. Once this \ref hipsparseDenseToSparse_analysis routine has been 
*  called, the non-zero count is stored in the sparse matrix descriptor \p matB. In order to allocate our remaining sparse 
*  matrix arrays, we query the sparse matrix descriptor \p matB for this non-zero count:
*  \code{.c}
*    // Grab the non-zero count from the B matrix decriptor
*    int64_t rows;
*    int64_t cols;
*    int64_t nnz;
*    hipsparseSpMatGetSize(matB, &rows, &cols, &nnz);
*  \endcode
*  The remaining arrays are then allocated and set on the sparse matrix descriptor \p matB. Finally, we complete the 
*  conversion by calling \ref hipsparseDenseToSparse_convert. Once the conversion is complete, the user is free to deallocate 
*  the storage buffer. See full example below for details.
*
*  \p hipsparseDenseToSparse_convert supports the following uniform precision data types for the dense and sparse matrices \f$A\f$ 
*  and \f$B\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="dense2sparse_uniform">Uniform Precisions</caption>
*  <tr><th>A / B
*  <tr><td>HIP_R_16F
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \note Currently only the sparse matrix formats CSR, CSC, and COO are supported when converting a dense matrix to a sparse matrix.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  matA            dense matrix descriptor.
*  @param[in]
*  matB            sparse matrix descriptor.
*  @param[in]
*  alg             algorithm for the dense to sparse computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p externalBuffer
*               pointer is invalid.
*
*  \par Example
*  \code{.c}
*    //     1 0 0 0
*    // A = 4 2 0 4
*    //     0 3 7 0
*    //     9 0 0 1
*    int m   = 4;
*    int n   = 4;
*
*    std::vector<float> hdenseA = {1.0f, 4.0f, 0.0f, 9.0f, 
*                                  0.0f, 2.0f, 3.0f, 0.0f, 
*                                  0.0f, 0.0f, 7.0f, 0.0f, 
*                                  0.0f, 4.0f, 0.0f, 1.0f};
*
*    float* ddenseA;
*    hipMalloc((void**)&ddenseA, sizeof(float) * m * n);
*    hipMemcpy(ddenseA, hdenseA.data(), sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*    int* dcsrRowPtrB;
*    hipMalloc((void**)&dcsrRowPtrB, sizeof(int) * (m + 1));
*
*    hipsparseHandle_t     handle;
*    hipsparseDnMatDescr_t matA;
*    hipsparseSpMatDescr_t matB;
*
*    hipsparseCreate(&handle);
*
*    // Create dense matrix A
*    hipsparseCreateDnMat(&matA,
*                        m,
*                        n,
*                        m,
*                        ddenseA,
*                        HIP_R_32F,
*                        HIPSPARSE_ORDER_COL);
*
*    hipsparseIndexType_t rowIdxTypeB = HIPSPARSE_INDEX_32I;
*    hipsparseIndexType_t colIdxTypeB = HIPSPARSE_INDEX_32I;
*    hipDataType  dataTypeB = HIP_R_32F;
*    hipsparseIndexBase_t idxBaseB = HIPSPARSE_INDEX_BASE_ZERO;
*
*    // Create sparse matrix B
*    hipsparseCreateCsr(&matB,
*                        m,
*                        n,
*                        0,
*                        dcsrRowPtrB,
*                        NULL,
*                        NULL,
*                        rowIdxTypeB,
*                        colIdxTypeB,
*                        idxBaseB,
*                        dataTypeB);
*
*    hipsparseDenseToSparseAlg_t alg = HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
*
*    size_t bufferSize;
*    hipsparseDenseToSparse_bufferSize(handle, matA, matB, alg, &bufferSize);
*
*    void* tempBuffer;
*    hipMalloc((void**)&tempBuffer, bufferSize);
*
*    // Perform analysis which will determine the number of non-zeros in the CSR matrix
*    hipsparseDenseToSparse_analysis(handle, matA, matB, alg, tempBuffer);
*
*    // Grab the non-zero count from the B matrix decriptor
*    int64_t rows;
*    int64_t cols;
*    int64_t nnz;
*    hipsparseSpMatGetSize(matB, &rows, &cols, &nnz);
*
*    // Allocate the column indices and values arrays
*    int* dcsrColIndB;
*    float* dcsrValB;
*    hipMalloc((void**)&dcsrColIndB, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrValB, sizeof(float) * nnz);
*
*    // Set the newly allocated arrays on the sparse matrix descriptor
*    hipsparseCsrSetPointers(matB, dcsrRowPtrB, dcsrColIndB, dcsrValB);
*
*    // Complete the conversion
*    hipsparseDenseToSparse_convert(handle, matA, matB, alg, tempBuffer);
*
*    // Copy result back to host
*    std::vector<int> hcsrRowPtrB(m + 1);
*    std::vector<int> hcsrColIndB(nnz);
*    std::vector<float> hcsrValB(nnz);
*    hipMemcpy(hcsrRowPtrB.data(), dcsrRowPtrB, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
*    hipMemcpy(hcsrColIndB.data(), dcsrColIndB, sizeof(int) * nnz, hipMemcpyDeviceToHost);
*    hipMemcpy(hcsrValB.data(), dcsrValB, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*
*    // Clear hipSPARSE
*    hipsparseDestroyMatDescr(matA);
*    hipsparseDestroyMatDescr(matB);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(ddenseA);
*    hipFree(dcsrRowPtrB);
*    hipFree(dcsrColIndB);
*    hipFree(dcsrValB);
*    hipFree(tempBuffer);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseConstDnMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseDnMatDescr_t       matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_DENSE2SPARSE_H */
