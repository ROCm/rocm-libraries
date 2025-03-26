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
#ifndef HIPSPARSE_SPVV_H
#define HIPSPARSE_SPVV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV_bufferSize computes the required user allocated buffer size needed when computing the 
*  inner dot product of a sparse vector with a dense vector
*
*  See \ref hipsparseSpVV for full code example.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opX                 sparse vector operation type.
*  @param[in]
*  vecX                sparse vector descriptor.
*  @param[in]
*  vecY                dense vector descriptor.
*  @param[out]
*  result              pointer to the result, can be host or device memory
*  @param[in]
*  computeType         floating point precision for the SpVV computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX, \p vecY, \p result or \p pBufferSizeInBytes
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType is currently not
*               supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opX,
                                           hipsparseConstSpVecDescr_t vecX,
                                           hipsparseConstDnVecDescr_t vecY,
                                           void*                      result,
                                           hipDataType                computeType,
                                           size_t*                    pBufferSizeInBytes);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV computes the inner dot product of a sparse vector with a dense vector. This routine takes a user 
*  allocated buffer whose size must first be computed by calling \ref hipsparseSpVV_bufferSize
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opX             sparse vector operation type.
*  @param[in]
*  vecX            sparse vector descriptor.
*  @param[in]
*  vecY            dense vector descriptor.
*  @param[out]
*  result          pointer to the result, can be host or device memory
*  @param[in]
*  computeType     floating point precision for the SpVV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX, \p vecY, \p result or \p externalBuffer
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType is currently not
*               supported.
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hxInd = {0, 3, 5};
*
*    // Sparse value vector
*    std::vector<float> hxVal = {1.0f, 2.0f, 3.0f};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Offload data to device
*    int* dxInd;
*    float* dxVal;
*    float* dy;
*    hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dxVal, hxVal.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                        size,
*                        nnz,
*                        dxInd,
*                        dxVal,
*                        HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO,
*                        HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Obtain buffer size
*    float hresult = 0.0f;
*    size_t buffer_size;
*    hipsparseSpVV_bufferSize(handle,
*                HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                vecX,
*                vecY,
*                &hresult,
*                HIP_R_32F,
*                &buffer_size);
*
*    void* temp_buffer;
*    hipMalloc(&temp_buffer, buffer_size);
*
*    // SpVV
*    hipsparseSpVV(handle,
*                HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                vecX,
*                vecY,
*                &hresult,
*                HIP_R_32F,
*                temp_buffer);
*
*    hipDeviceSynchronize();
*
*    std::cout << "hresult: " << hresult << std::endl;
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dxInd);
*    hipFree(dxVal);
*    hipFree(dy);
*    hipFree(temp_buffer);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t          handle,
                                hipsparseOperation_t       opX,
                                hipsparseConstSpVecDescr_t vecX,
                                hipsparseConstDnVecDescr_t vecY,
                                void*                      result,
                                hipDataType                computeType,
                                void*                      externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPVV_H */
