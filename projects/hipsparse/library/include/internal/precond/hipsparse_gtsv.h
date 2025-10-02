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
*  \details
*  \p hipsparseSgtsv2_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by \ref hipsparseSgtsv2 "hipsparseXgtsv2()". The temporary
*  storage buffer must be allocated by the user.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle             handle to the hipSPARSE library context queue.
*  @param[in]
*  m                  size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n                  number of columns in the dense matrix B.
*  @param[in]
*  dl                 lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d                  main diagonal of tri-diagonal system.
*  @param[in]
*  du                 upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[in]
*  B                  Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb                Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     \ref hipsparseSgtsv2 "hipsparseXgtsv2()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ldb, \p dl, \p d, \p du,
*              \p B or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
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
                                                int               ldb,
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
*  \p hipsparseXgtsv2 solves a tridiagonal system for multiple right hand sides using pivoting
*  \f[
*    T*B = B
*  \f]
*  where \f$T\f$ is a sparse tridiagonal matrix and \f$B\f$ is a dense \f$ldb \times n\f$ matrix storing the
*  right-hand side vectors in column order. The tridiagonal matrix \f$T\f$ is defined by three vectors: \p dl
*  for the lower diagonal, \p d for the main diagonal and \p du for the upper diagonal.
*
*  Solving the tridiagonal system involves two steps. First, the user calls
*  \ref hipsparseSgtsv2_bufferSizeExt "hipsparseXgtsv2_bufferSizeExt()" in order to determine the size of the required
*  temporary storage buffer. Once determined, the user allocates this buffer and passes it to
*  \ref hipsparseSgtsv2 "hipsparseXgtsv2()" to perform the actual solve. The \f$B\f$ dense matrix, which initially
*  stores the \p n right-hand side vectors, is overwritten with the \p n solution vectors after the call to
*  \ref hipsparseSgtsv2 "hipsparseXgtsv2()".
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the hipSPARSE library context queue.
*  @param[in]
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n           number of columns in the dense matrix B.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  B           Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb         Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ldb, \p dl, \p d,
*              \p du, \p B or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \snippet example_hipsparse_gtsv.cpp doc example
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
