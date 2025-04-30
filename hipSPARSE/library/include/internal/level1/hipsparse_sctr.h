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
#ifndef HIPSPARSE_SCTR_H
#define HIPSPARSE_SCTR_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Scatter elements from a dense vector across a sparse vector.
*
*  \details
*  \p hipsparseXsctr scatters the elements that are listed in \p xInd from the sparse
*  vector \f$x\f$ into the dense vector \f$y\f$. Indices of \f$y\f$ that are not listed
*  in \p xInd remain unchanged.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[xInd[i]] = xVal[i];
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
*  @param[in]
*  xVal       array of \p nnz elements containing the non-zero values of \f$x\f$.
*  @param[in]
*  xInd       array of \p nnz elements containing the indices of the non-zero
*              values of x.
*  @param[inout]
*  y           array of values in dense format.
*  @param[in]
*  idxBase    \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p idxBase, \p nnz, \p xVal, \p xInd 
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
*      float hxVal[3] = {9.0, 2.0, 3.0};
*
*      // Dense vector
*      float hy[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
*
*      // Index base
*      hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;
*
*      // Offload data to device
*      int* dxInd;
*      float*         dxVal;
*      float*         dy;
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
*      // Call ssctr
*      hipsparseSsctr(handle, nnz, dxVal, dxInd, dy, idxBase);
*
*      // Copy result back to host
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
hipsparseStatus_t hipsparseSsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 float*               y,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 double*              y,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 hipComplex*          y,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZsctr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 hipDoubleComplex*       y,
                                 hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SCTR_H */
