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
*  <tr><td>HIP_R_16BF
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
*  \snippet example_hipsparse_dense_to_sparse.cpp doc example
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
