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
#ifndef HIPSPARSE_ROTI_H
#define HIPSPARSE_ROTI_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \p hipsparseXroti applies the Givens rotation matrix \f$G\f$ to the sparse vector
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
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  nnz         number of non-zero entries of \f$x\f$.
*  @param[inout]
*  xVal       array of \p nnz elements containing the non-zero values of \f$x\f$.
*  @param[in]
*  xInd       array of \p nnz elements containing the indices of the non-zero
*              values of \f$x\f$.
*  @param[inout]
*  y           array of values in dense format.
*  @param[in]
*  c           pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s           pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[in]
*  idxBase    \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p idxBase, \p nnz, \p c, \p s, \p xVal, \p xInd 
*              or \p y is invalid.
*
*  \par Example
*  \code{.c}
*      // Number of non-zeros of the sparse vector
*      int nnz = 3;
*
*      // Sparse index vector
*      int hxInd[3] = {0, 3, 5};
*
*      // Sparse value vector
*      float hxVal[3] = {1.0f, 2.0f, 3.0f};
*
*      // Dense vector
*      float hy[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*      // c and s
*      float c = 3.7;
*      float s = 1.3;
*
*      // Index base
*      hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;
*
*      // Offload data to device
*      int* dxInd;
*      float*        dxVal;
*      float*        dy;
*
*      hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*      hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*      hipMalloc((void**)&dy, sizeof(float) * 9);
*
*      hipMemcpy(dxInd, hxInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dxVal, hxVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(float) * 9, hipMemcpyHostToDevice);
*
*      // hipSPARSE handle
*      hipsparseHandle_t handle;
*      hipsparseCreate(&handle);
*
*      // Call sroti
*      hipsparseSroti(handle, nnz, dxVal, dxInd, dy, &c, &s, idxBase);
*
*      // Copy result back to host
*      hipMemcpy(hxVal, dxVal, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*      hipMemcpy(hy, dy, sizeof(float) * 9, hipMemcpyDeviceToHost);
*
*      // Clear hipSPARSE
*      hipsparseDestroy(handle);
*
*      // Clear device memory
*      hipFree(dxInd);
*      hipFree(dxVal);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 float*               xVal,
                                 const int*           xInd,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 double*              xVal,
                                 const int*           xInd,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 hipsparseIndexBase_t idxBase);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_ROTI_H */
