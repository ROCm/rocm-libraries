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
#ifndef HIPSPARSE_ROT_H
#define HIPSPARSE_ROT_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \ref hipsparseRot applies the Givens rotation matrix \f$G\f$ to the sparse vector
*  \f$x\f$ and the dense vector \f$y\f$, where
*  \f[
*    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_tmp = xVal[i];
*          y_tmp = y[xInd[i]];
*
*          xVal[i]    = c * x_tmp + s * y_tmp;
*          y[xInd[i]] = c * y_tmp - s * x_tmp;
*      }
*  \endcode
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  c_coeff     pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s_coeff     pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[inout]
*  vecX        sparse vector descriptor \f$x\f$.
*  @param[inout]
*  vecY        dense vector descriptor \f$y\f$.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p c_coeff, \p s_coeff, \p vecX or \p vecY pointer is
*              invalid.
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
*    // Scalar c
*    float c = 3.7f;
*
*    // Scalar s
*    float s = 1.2f;
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
*                                size,
*                                nnz,
*                                dxInd,
*                                dxVal,
*                                HIPSPARSE_INDEX_32I,
*                                HIPSPARSE_INDEX_BASE_ZERO,
*                                HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Call rot
*    hipsparseRot(handle, (void*)&c, (void*)&s, vecX, vecY);
*
*    hipsparseSpVecGetValues(vecX, (void**)&dxVal);
*    hipsparseDnVecGetValues(vecY, (void**)&dy);
*
*    // Copy result back to host
*    hipMemcpy(hxVal.data(), dxVal, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*    hipMemcpy(hy.data(), dy, sizeof(float) * size, hipMemcpyDeviceToHost);
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
*  \endcode
*/
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseRot(hipsparseHandle_t     handle,
                               const void*           c_coeff,
                               const void*           s_coeff,
                               hipsparseSpVecDescr_t vecX,
                               hipsparseDnVecDescr_t vecY);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_ROT_H */
