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
#ifndef HIPSPARSE_GTSV_STRIDED_BATCH_H
#define HIPSPARSE_GTSV_STRIDED_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p hipsparseXgtsv2StridedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgtsv2StridedBatch "hipsparseXgtsv2StridedBatch()".
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  size of the tri-diagonal linear system.
*  @param[in]
*  dl                 lower diagonal of tri-diagonal system where the ith system lower diagonal starts at
*                     \p dl+batchStride*i.
*  @param[in]
*  d                  main diagonal of tri-diagonal system where the ith system diagonal starts at
*                     \p d+batchStride*i.
*  @param[in]
*  du                 upper diagonal of tri-diagonal system where the ith system upper diagonal starts at
*                     \p du+batchStride*i.
*  @param[inout]
*  x                  Dense array of righthand-sides where the ith righthand-side starts at \p x+batchStride*i.
*  @param[in]
*  batchCount         The number of systems to solve.
*  @param[in]
*  batchStride        The number of elements that separate each system. Must satisfy \p batchStride >= m.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSgtsv2StridedBatch "hipsparseXgtsv2StridedBatch()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p batchStride, \p dl,
*              \p d, \p du, \p x or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch solves a batched tridiagonal linear system
*  \f[
*    T^{i}*x^{i} = x^{i}
*  \f]
*  where for each batch \f$i=0\ldots\f$ \p batchCount, \f$T^{i}\f$ is a sparse tridiagonal matrix and
*  \f$x^{i}\f$ is a dense right-hand side vector. All of the tridiagonal matrices, \f$T^{i}\f$, are
*  packed one after the other into three vectors: \p dl for the lower diagonals, \p d for the main
*  diagonals and \p du for the upper diagonals. See below for a description of what this strided
*  memory pattern looks like.
*
*  Solving the batched tridiagonal system involves two steps. First, the user calls
*  \ref hipsparseSgtsv2StridedBatch_bufferSizeExt "hipsparseXgtsv2StridedBatch_bufferSizeExt()"
*  in order to determine the size of the required temporary storage buffer. Once determined, the user allocates
*  this buffer and passes it to \ref hipsparseSgtsv2StridedBatch "hipsparseXgtsv2StridedBatch()"
*  to perform the actual solve. The \f$x^{i}\f$ vectors, which initially stores the right-hand side values, are
*  overwritten with the solution after the call to
*  \ref hipsparseSgtsv2StridedBatch "hipsparseXgtsv2StridedBatch()".
*
*  The strided batch routines write each batch matrix one after the other in memory. For example, consider
*  the following batch matrices:
*
*  \f[
*    \begin{bmatrix}
*    t^{0}_{00} & t^{0}_{01} & 0 \\
*    t^{0}_{10} & t^{0}_{11} & t^{0}_{12} \\
*    0 & t^{0}_{21} & t^{0}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{1}_{00} & t^{1}_{01} & 0 \\
*    t^{1}_{10} & t^{1}_{11} & t^{1}_{12} \\
*    0 & t^{1}_{21} & t^{1}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{2}_{00} & t^{2}_{01} & 0 \\
*    t^{2}_{10} & t^{2}_{11} & t^{2}_{12} \\
*    0 & t^{2}_{21} & t^{2}_{22}
*    \end{bmatrix}
*  \f]
*
*  In strided format, the upper, lower, and diagonal arrays would look like:
*  \f[
*    \begin{align}
*    \text{lower} &= \begin{bmatrix} 0 & t^{0}_{10} & t^{0}_{21} & 0 & t^{1}_{10} & t^{1}_{21} & 0 & t^{2}_{10} & t^{2}_{21} \end{bmatrix} \\
*    \text{diagonal} &= \begin{bmatrix} t^{0}_{00} & t^{0}_{11} & t^{0}_{22} & t^{1}_{00} & t^{1}_{11} & t^{1}_{22} & t^{2}_{00} & t^{2}_{11} & t^{2}_{22} \end{bmatrix} \\
*    \text{upper} &= \begin{bmatrix} t^{0}_{01} & t^{0}_{12} & 0 & t^{1}_{01} & t^{1}_{12} & 0 & t^{2}_{01} & t^{2}_{12} & 0 \end{bmatrix} \\
*    \end{align}
*  \f]
*  For the lower array, for each batch \p i, the \p i*batchStride entries are zero and for the upper array the
*  \p i*batchStride+batchStride-1 entries are zero.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides where the ith righthand-side starts at \p x+batchStride*i.
*  @param[in]
*  batchCount  The number of systems to solve.
*  @param[in]
*  batchStride The number of elements that separate each system. Must satisfy \p batchStride >= m.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p batchStride, \p dl, \p d,
*              \p du, \p x or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GTSV_STRIDED_BATCH_H */
