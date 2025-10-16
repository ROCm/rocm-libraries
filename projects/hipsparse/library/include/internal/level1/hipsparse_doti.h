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
#ifndef HIPSPARSE_DOTI_H
#define HIPSPARSE_DOTI_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level1_module
*  \brief Compute the dot product of a sparse vector with a dense vector.
*
*  \details
*  \p hipsparseXdoti computes the dot product of the sparse vector \f$x\f$ with the
*  dense vector \f$y\f$, such that
*  \f[
*    result := y^T x
*  \f]
*
*  \code{.c}
*      result = 0
*      for(i = 0; i < nnz; ++i)
*      {
*          result += xVal[i] * y[xInd[i]];
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
*  nnz         number of non-zero entries of vector \f$x\f$.
*  @param[in]
*  xVal       array of \p nnz values.
*  @param[in]
*  xInd       array of \p nnz elements containing the indices of the non-zero
*              values of \f$x\f$.
*  @param[in]
*  y           array of values in dense format.
*  @param[out]
*  result      pointer to the result, can be host or device memory
*  @param[in]
*  idxBase    \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p idxBase, \p nnz, \p xVal,
*          \p xInd, \p y or \p result is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED the buffer for the dot product reduction
*          could not be allocated.
*  \retval HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \snippet example_hipsparse_doti.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 const float*         y,
                                 float*               result,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 const double*        y,
                                 double*              result,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 const hipComplex*    y,
                                 hipComplex*          result,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdoti(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       result,
                                 hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_DOTI_H */
