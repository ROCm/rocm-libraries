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
#ifndef HIPSPARSE_SPMM_H
#define HIPSPARSE_SPMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Calculate the buffer size required for the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR format, \f$B\f$ is a dense matrix of size \f$k \times n\f$ and 
*  \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \ref hipsparseSpMM_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpMM for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMM for full code example.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  opB                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  matB                matrix descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[in]
*  matC                matrix descriptor.
*  @param[in]
*  computeType         floating point precision for the SpMM computation.
*  @param[in]
*  alg                 SpMM algorithm for the SpMM computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Preprocess step of the sparse matrix multiplication with a dense matrix.
*
*  \details
*  \p hipsparseSpMM_preprocess performs the required preprocessing used when computing the 
*  sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR format, \f$B\f$ is a dense matrix of size \f$k \times n\f$ and 
*  \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \ref hipsparseSpMM_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSpMM for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMM for full code example.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  opB             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  matB            matrix descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  matC            matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMM computation.
*  @param[in]
*  alg             SpMM algorithm for the SpMM computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM computes sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR format, \f$B\f$ is a dense matrix of size \f$k \times n\f$ and 
*  \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \ref hipsparseSpMM supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix A and the dense matrices B and C and the compute
*  type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth and storage
*  when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmm_mixed">Mixed Precisions</caption>
*  <tr><th>A / B    <th>C         <th>compute_type
*  <tr><td>HIP_R_8I <td>HIP_R_32I <td>HIP_R_32I
*  <tr><td>HIP_R_8I <td>HIP_R_32F <td>HIP_R_32F
*  </table>
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  opB             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  matB            matrix descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  matC            matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMM computation.
*  @param[in]
*  alg             SpMM algorithm for the SpMM computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \code{.c}
*    // A, B, and C are m×k, k×n, and m×n
*    int m = 3, n = 5, k = 4;
*    int ldb = n, ldc = n;
*    int nnz_A = 8, nnz_B = 20, nnz_C = 15;
*    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transC = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOrder_t order = HIPSPARSE_ORDER_ROW;
*
*    // alpha and beta
*    float alpha = 0.5f;
*    float beta  = 0.25f;
*
*    std::vector<int> hcsrRowPtr = {0, 3, 5, 8};
*    std::vector<int> hcsrColInd = {0, 1, 3, 1, 2, 0, 2, 3}; 
*    std::vector<float> hcsrVal     = {1, 2, 3, 4, 5, 6, 7, 8}; 
*
*    std::vector<float> hB(nnz_B, 1.0f);
*    std::vector<float> hC(nnz_C, 1.0f);
*
*    int *dcsrRowPtr;
*    int *dcsrColInd;
*    float *dcsrVal;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseSpMatDescr_t matA;
*    hipsparseCreateCsr(&matA, m, k, nnz_A,
*                        dcsrRowPtr, dcsrColInd, dcsrVal,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    // Allocate memory for the matrix B
*    float* dB;
*    hipMalloc((void**)&dB, sizeof(float) * nnz_B);
*    hipMemcpy(dB, hB.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t matB;
*    hipsparseCreateDnMat(&matB, k, n, ldb, dB, HIP_R_32F, order);
*
*    // Allocate memory for the resulting matrix C
*    float* dC;
*    hipMalloc((void**)&dC, sizeof(float) * nnz_C);
*    hipMemcpy(dC, hC.data(), sizeof(float) * nnz_C, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t matC;
*    hipsparseCreateDnMat(&matC, m, n, ldc, dC, HIP_R_32F, HIPSPARSE_ORDER_ROW);
*
*    // Compute buffersize
*    size_t bufferSize;
*    hipsparseSpMM_bufferSize(handle,
*                             transA,
*                             transB,
*                             &alpha,
*                             matA,
*                             matB,
*                             &beta,
*                             matC,
*                             HIP_R_32F,
*                             HIPSPARSE_MM_ALG_DEFAULT,
*                             &bufferSize);
*
*    void* buffer;
*    hipMalloc(&buffer, bufferSize);
*
*    // Preprocess operation (Optional)
*    hipsparseSpMM_preprocess(handle,
*                            transA,
*                            transB,
*                            &alpha,
*                            matA,
*                            matB,
*                            &beta,
*                            matC,
*                            HIP_R_32F,
*                            HIPSPARSE_MM_ALG_DEFAULT,
*                            &buffer);
*
*    // Perform operation
*    hipsparseSpMM(handle,
*                 transA,
*                 transB,
*                 &alpha,
*                 matA,
*                 matB,
*                 &beta,
*                 matC,
*                 HIP_R_32F,
*                 HIPSPARSE_MM_ALG_DEFAULT,
*                 &buffer);
*
*    // Copy device to host
*    hipMemcpy(hC.data(), dC, sizeof(float) * nnz_C, hipMemcpyDeviceToHost);
*
*    // Destroy matrix descriptors and handles
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroyDnMat(matB);
*    hipsparseDestroyDnMat(matC);
*    hipsparseDestroy(handle);
*
*    hipFree(buffer);
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dB);
*    hipFree(dC);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnMatDescr_t  matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPMM_H */
