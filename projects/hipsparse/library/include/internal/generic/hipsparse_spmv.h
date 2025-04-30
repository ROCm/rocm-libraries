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
#ifndef HIPSPARSE_SPMV_H
#define HIPSPARSE_SPMV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR format, \f$x\f$ is a dense vector of length \f$n\f$ and 
*  \f$y\f$ is a dense vector of length \f$m\f$.
*
*  \ref hipsparseSpMV_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpMV for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMV for full code example.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  vecX                vector descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[inout]
*  vecY                vector descriptor.
*  @param[in]
*  computeType         floating point precision for the SpMV computation.
*  @param[in]
*  alg                 SpMV algorithm for the SpMV computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p pBufferSizeInBytes pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Preprocess step of the sparse matrix multiplication with a dense vector (optional)
*
*  \details
*  \p hipsparseSpMV_preprocess performs analysis on the sparse matrix \f$A\f$ when computing the 
*  sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR format, \f$x\f$ is a dense vector of length \f$n\f$ and 
*  \f$y\f$ is a dense vector of length \f$m\f$.
*
*  This step is optional but if used may results in better performance.
*
*  \ref hipsparseSpMV_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSpMV for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMV for full code example.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  vecX            vector descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[inout]
*  vecY            vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMV computation.
*  @param[in]
*  alg             SpMV algorithm for the SpMV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p externalBuffer pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV computes sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR format, \f$x\f$ is a dense vector of length \f$n\f$ and 
*  \f$y\f$ is a dense vector of length \f$m\f$.
*
*  \ref hipsparseSpMV supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix A and the dense vectors X and Y and the compute
*  type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth and storage
*  when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmv_mixed">Mixed Precisions</caption>
*  <tr><th>A / X    <th>Y         <th>compute_type
*  <tr><td>HIP_R_8I <td>HIP_R_32I <td>HIP_R_32I
*  <tr><td>HIP_R_8I <td>HIP_R_32F <td>HIP_R_32F
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="spmv_mixed_regular_real">Mixed-regular real precisions</caption>
*  <tr><th>A         <th>X / Y / compute_type
*  <tr><td>HIP_R_32F <td>HIP_R_64F
*  <tr><td>HIP_C_32F <td>HIP_C_64F
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="spmv_mixed_regular_complex">Mixed-regular Complex precisions</caption>
*  <tr><th>A         <th>X / Y / compute_type
*  <tr><td>HIP_R_32F <td>HIP_C_32F
*  <tr><td>HIP_R_64F <td>HIP_C_64F
*  </table>
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  vecX            vector descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[inout]
*  vecY            vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMV computation.
*  @param[in]
*  alg             SpMV algorithm for the SpMV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p externalBuffer pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \code{.c}
*    // A, x, and y are m×k, k×1, and m×1
*    int m = 3, k = 4;
*    int nnz_A = 8;
*    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*    // alpha and beta
*    float alpha = 0.5f;
*    float beta  = 0.25f;
*
*    std::vector<int> hcsrRowPtr = {0, 3, 5, 8};
*    std::vector<int> hcsrColInd = {0, 1, 3, 1, 2, 0, 2, 3}; 
*    std::vector<float> hcsrVal     = {1, 2, 3, 4, 5, 6, 7, 8}; 
*
*    std::vector<float> hx(k, 1.0f);
*    std::vector<float> hy(m, 1.0f);
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
*    // Allocate memory for the vector x
*    float* dx;
*    hipMalloc((void**)&dx, sizeof(float) * k);
*    hipMemcpy(dx, hx.data(), sizeof(float) * k, hipMemcpyHostToDevice);
*
*    hipsparseDnVecDescr_t vecX;
*    hipsparseCreateDnVec(&vecX, k, dx, HIP_R_32F);
*
*    // Allocate memory for the resulting vector y
*    float* dy;
*    hipMalloc((void**)&dy, sizeof(float) * m);
*    hipMemcpy(dy, hy.data(), sizeof(float) * m, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, m, dy, HIP_R_32F);
*
*    // Compute buffersize
*    size_t bufferSize;
*    hipsparseSpMV_bufferSize(handle,
*                             transA,
*                             &alpha,
*                             matA,
*                             vecX,
*                             &beta,
*                             vecY,
*                             HIP_R_32F,
*                             HIPSPARSE_MV_ALG_DEFAULT,
*                             &bufferSize);
*
*    void* buffer;
*    hipMalloc(&buffer, bufferSize);
*
*    // Preprocess operation (Optional)
*    hipsparseSpMV_preprocess(handle,
*                            transA,
*                            &alpha,
*                            matA,
*                            vecX,
*                            &beta,
*                            vecY,
*                            HIP_R_32F,
*                            HIPSPARSE_MV_ALG_DEFAULT,
*                            &buffer);
*
*    // Perform operation
*    hipsparseSpMV(handle,
*                 transA,
*                 &alpha,
*                 matA,
*                 vecX,
*                 &beta,
*                 vecY,
*                 HIP_R_32F,
*                 HIPSPARSE_MV_ALG_DEFAULT,
*                 &buffer);
*
*    // Copy device to host
*    hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost);
*
*    // Destroy matrix descriptors and handles
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroyDnVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    hipFree(buffer);
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dx);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnVecDescr_t  vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPMV_H */
