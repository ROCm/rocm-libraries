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
#ifndef HIPSPARSE_GTSV_H
#define HIPSPARSE_GTSV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p hipsparseXgtsv2_bufferSize returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseSgtsv2 "hipsparseXgtsv2()". The 
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const float*      dl,
                                                const float*      d,
                                                const float*      du,
                                                const float*      B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const double*     dl,
                                                const double*     d,
                                                const double*     du,
                                                const double*     B,
                                                int               db,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const hipComplex* dl,
                                                const hipComplex* d,
                                                const hipComplex* du,
                                                const hipComplex* B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* dl,
                                                const hipDoubleComplex* d,
                                                const hipDoubleComplex* du,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p hipsparseXgtsv2 solves a tridiagonal system for multiple right hand sides using pivoting.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const float*      dl,
                                  const float*      d,
                                  const float*      du,
                                  float*            B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const double*     dl,
                                  const double*     d,
                                  const double*     du,
                                  double*           B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const hipComplex* dl,
                                  const hipComplex* d,
                                  const hipComplex* du,
                                  hipComplex*       B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* dl,
                                  const hipDoubleComplex* d,
                                  const hipDoubleComplex* du,
                                  hipDoubleComplex*       B,
                                  int                     ldb,
                                  void*                   pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GTSV_H */
