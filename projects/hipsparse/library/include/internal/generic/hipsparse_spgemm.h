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
#ifndef HIPSPARSE_SPGEMM_H
#define HIPSPARSE_SPGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpGEMM_createDescr creates a sparse matrix sparse matrix product descriptor. It should be
*  destroyed at the end using \ref hipsparseSpGEMM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpGEMM_destroyDescr destroys a sparse matrix sparse matrix product descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Work estimation step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_workEstimation is called twice. We call it to compute the size of the first required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_workEstimation
*  a second time with the newly allocated buffer passed in. This second call inspects the matrices \f$A\f$ and \f$B\f$ to 
*  determine the number of intermediate products that will result from multipltying \f$A\f$ and \f$B\f$ together.
*
*  \p hipsparseSpGEMM_workEstimation supports multiple combinations of data types and compute types. See \ref hipsparseSpGEMM_copy 
*  for a complete listing of all the data type and compute type combinations available.
*  
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize1      number of bytes of the temporary storage buffer. 
*  @param[in]
*  externalBuffer1  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC 
*                                         or \p bufferSize1 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer1  = NULL; 
*    size_t bufferSize1 = 0;
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    size_t bufferSize1 = 0;
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                  &alpha, matA, matB, &beta, matC,
*                                  computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                  spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Determine number of intermediate product when computing A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                    &alpha, matA, matB, &beta, matC,
*                                    computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc, &bufferSize1, dBuffer1);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t          handle,
                                                 hipsparseOperation_t       opA,
                                                 hipsparseOperation_t       opB,
                                                 const void*                alpha,
                                                 hipsparseConstSpMatDescr_t matA,
                                                 hipsparseConstSpMatDescr_t matB,
                                                 const void*                beta,
                                                 hipsparseSpMatDescr_t      matC,
                                                 hipDataType                computeType,
                                                 hipsparseSpGEMMAlg_t       alg,
                                                 hipsparseSpGEMMDescr_t     spgemmDescr,
                                                 size_t*                    bufferSize1,
                                                 void*                      externalBuffer1);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t      handle,
                                                 hipsparseOperation_t   opA,
                                                 hipsparseOperation_t   opB,
                                                 const void*            alpha,
                                                 hipsparseSpMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t  matB,
                                                 const void*            beta,
                                                 hipsparseSpMatDescr_t  matC,
                                                 hipDataType            computeType,
                                                 hipsparseSpGEMMAlg_t   alg,
                                                 hipsparseSpGEMMDescr_t spgemmDescr,
                                                 size_t*                bufferSize1,
                                                 void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Compute step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_compute is called twice. First to compute the size of the second required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_compute
*  a second time with the newly allocated buffer passed in. This second call performs the actual computation 
*  of \f$C' = \alpha \cdot A \cdot B\f$ (the result is stored in the temporary buffers).
*
*  \p hipsparseSpGEMM_compute supports multiple combinations of data types and compute types. See \ref hipsparseSpGEMM_copy 
*  for a complete listing of all the data type and compute type combinations available.
*  
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize2      number of bytes of the temporary storage buffer. 
*  @param[in]
*  externalBuffer2  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC 
*                                         or \p bufferSize2 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer2  = NULL; 
*    size_t bufferSize2 = 0;
*
*    size_t bufferSize2 = 0;
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                            spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // compute the intermediate product of A * B
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                            spgemmDesc, &bufferSize2, dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t          handle,
                                          hipsparseOperation_t       opA,
                                          hipsparseOperation_t       opB,
                                          const void*                alpha,
                                          hipsparseConstSpMatDescr_t matA,
                                          hipsparseConstSpMatDescr_t matB,
                                          const void*                beta,
                                          hipsparseSpMatDescr_t      matC,
                                          hipDataType                computeType,
                                          hipsparseSpGEMMAlg_t       alg,
                                          hipsparseSpGEMMDescr_t     spgemmDescr,
                                          size_t*                    bufferSize2,
                                          void*                      externalBuffer2);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t      handle,
                                          hipsparseOperation_t   opA,
                                          hipsparseOperation_t   opB,
                                          const void*            alpha,
                                          hipsparseSpMatDescr_t  matA,
                                          hipsparseSpMatDescr_t  matB,
                                          const void*            beta,
                                          hipsparseSpMatDescr_t  matC,
                                          hipDataType            computeType,
                                          hipsparseSpGEMMAlg_t   alg,
                                          hipsparseSpGEMMDescr_t spgemmDescr,
                                          size_t*                bufferSize2,
                                          void*                  externalBuffer2);
#endif

/*! \ingroup generic_module
*  \brief Copy step of the sparse matrix sparse matrix product: 
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_copy is called once to copy the results (that are currently stored in the temporary arrays) 
*  to the output sparse matrix. If \f$\beta != 0\f$, then the \f$beta \cdot C\f$ portion of the computation: 
*  \f$C' = \alpha \cdot A \cdot B + \beta * C\f$ is handled. This is possible because \f$C'\f$ and \f$C\f$ must have 
*  the same sparsity pattern.
*
*  \p hipsparseSpGEMM_copy supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrices \f$op(A)\f$, \f$op(B)\f$, \f$C\f$, and \f$C'\f$
*  and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spgemm_copy_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / C' / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \p hipsparseSpGEMM_copy supports \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I index precisions 
*  for storing the row pointer and row/column indices arrays of the sparse matrices.
*
*  \note The two user allocated temporary buffers can only be freed after the call to \p hipsparseSpGEMM_copy
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*  
*  \par Example (Full example)
*  \code{.c}
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                                        dcsrRowPtrA, dcsrColIndA, dcsrValA,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                                        dcsrRowPtrB, dcsrColIndB, dcsrValB,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                                        dcsrRowPtrC, NULL, NULL,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize1, dBuffer1);
*
*    // Determine size of second user allocated buffer
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                &alpha, matA, matB, &beta, matC,
*                                computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // Compute C = alpha * A * B and store result in temporary buffers
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize2, dBuffer2);
*
*    // Get matrix C non-zero entries C_nnz1
*    int64_t C_num_rows1, C_num_cols1, C_nnz1;
*    hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
*
*    // Allocate the CSR structures for the matrix C
*    hipMalloc((void**) &dcsrColIndC, C_nnz1 * sizeof(int));
*    hipMalloc((void**) &dcsrValC,  C_nnz1 * sizeof(float));
*
*    // Update matC with the new pointers
*    hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC);
*
*    // Copy the final products to the matrix C
*    hipsparseSpGEMM_copy(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc);
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
* 
*    // Free device memory
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t          handle,
                                       hipsparseOperation_t       opA,
                                       hipsparseOperation_t       opB,
                                       const void*                alpha,
                                       hipsparseConstSpMatDescr_t matA,
                                       hipsparseConstSpMatDescr_t matB,
                                       const void*                beta,
                                       hipsparseSpMatDescr_t      matC,
                                       hipDataType                computeType,
                                       hipsparseSpGEMMAlg_t       alg,
                                       hipsparseSpGEMMDescr_t     spgemmDescr);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t      handle,
                                       hipsparseOperation_t   opA,
                                       hipsparseOperation_t   opB,
                                       const void*            alpha,
                                       hipsparseSpMatDescr_t  matA,
                                       hipsparseSpMatDescr_t  matB,
                                       const void*            beta,
                                       hipsparseSpMatDescr_t  matC,
                                       hipDataType            computeType,
                                       hipsparseSpGEMMAlg_t   alg,
                                       hipsparseSpGEMMDescr_t spgemmDescr);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPGEMM_H */
