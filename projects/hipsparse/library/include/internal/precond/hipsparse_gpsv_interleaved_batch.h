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
#ifndef HIPSPARSE_GPSV_INTERLEAVED_BATCH_H
#define HIPSPARSE_GPSV_INTERLEAVED_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p hipsparseXgpsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgpsvInterleavedBatch "hipsparseXgpsvInterleavedBatch()".
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  algo               algorithm to solve the linear system.
*  @param[in]
*  m                  size of the pentadiagonal linear system.
*  @param[in]
*  ds                 lower diagonal (distance 2) of pentadiagonal system. First two entries
*                     must be zero.
*  @param[in]
*  dl                 lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[in]
*  d                  main diagonal of pentadiagonal system.
*  @param[in]
*  du                 upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[in]
*  dw                 upper diagonal (distance 2) of pentadiagonal system. Last two entries
*                     must be zero.
*  @param[in]
*  x                  Dense array of right-hand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount         The number of systems to solve.
*  @param[out]
*  pBufferSizeInBytes Number of bytes of the temporary storage buffer required.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p alg, \p batchCount, \p ds, \p dl,
*              \p d, \p du, \p dw, \p x or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      ds,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      dw,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     ds,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     dw,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* ds,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* dw,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* ds,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* dw,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes);

/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch pentadiagonal solver
*
*  \details
*  \p hipsparseXgpsvInterleavedBatch solves a batch of pentadiagonal linear systems
*  \f[
*    P^{i}*x^{i} = x^{i}
*  \f]
*  where for each batch \f$i=0\ldots\f$ \p batchCount, \f$P^{i}\f$ is a sparse pentadiagonal matrix and
*  \f$x^{i}\f$ is a dense right-hand side vector. All of the pentadiagonal matrices, \f$P^{i}\f$, are
*  packed in an interleaved fashion into five vectors: \p ds for the lowest diagonals, \p dl for the lower
*  diagonals, \p d for the main diagonals, \p du for the upper diagonals, and \p dw for the highest digaonals.
*  See below for a description of what this interleaved memory pattern looks like.
*
*  Solving the batched pentadiagonal system involves two steps. First, the user calls
*  \ref hipsparseSgpsvInterleavedBatch_bufferSizeExt "hipsparseSgpsvInterleavedBatch_bufferSizeExt()"
*  in order to determine the size of the required temporary storage buffer. Once determined, the user allocates
*  this buffer and passes it to \ref hipsparseSgpsvInterleavedBatch "hipsparseXgpsvInterleavedBatch()"
*  to perform the actual solve. The \f$x^{i}\f$ vectors, which initially stores the right-hand side values, are
*  overwritten with the solution after the call to
*  \ref hipsparseSgpsvInterleavedBatch "hipsparseXgpsvInterleavedBatch()".
*
*  Unlike the strided batch routines which write each batch matrix one after the other in memory, the interleaved
*  routines write the batch matrices such that each element from each matrix is written consecutively one after
*  the other. For example, consider the following batch matrices:
*
*  \f[
*    \begin{bmatrix}
*    t^{0}_{00} & t^{0}_{01} & t^{0}_{02} \\
*    t^{0}_{10} & t^{0}_{11} & t^{0}_{12} \\
*    t^{0}_{20} & t^{0}_{21} & t^{0}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{1}_{00} & t^{1}_{01} & t^{1}_{02} \\
*    t^{1}_{10} & t^{1}_{11} & t^{1}_{12} \\
*    t^{1}_{20} & t^{1}_{21} & t^{1}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{2}_{00} & t^{2}_{01} & t^{2}_{02} \\
*    t^{2}_{10} & t^{2}_{11} & t^{2}_{12} \\
*    t^{2}_{20} & t^{2}_{21} & t^{2}_{22}
*    \end{bmatrix}
*  \f]
*
*  In interleaved format, the highest, higher, lowest, lower, and diagonal arrays would look like:
*  \f[
*    \begin{align}
*    \text{lowest} &= \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & t^{0}_{20} & t^{1}_{20} & t^{2}_{20} \end{bmatrix} \\
*    \text{lower} &= \begin{bmatrix} 0 & 0 & 0 & t^{0}_{10} & t^{1}_{10} & t^{1}_{10} & t^{0}_{21} & t^{1}_{21} & t^{2}_{21} \end{bmatrix} \\
*    \text{diagonal} &= \begin{bmatrix} t^{0}_{00} & t^{1}_{00} & t^{2}_{00} & t^{0}_{11} & t^{1}_{11} & t^{2}_{11} & t^{0}_{22} & t^{1}_{22} & t^{2}_{22} \end{bmatrix} \\
*    \text{higher} &= \begin{bmatrix} t^{0}_{01} & t^{1}_{01} & t^{2}_{01} & t^{0}_{12} & t^{1}_{12} & t^{2}_{12} & 0 & 0 & 0 \end{bmatrix} \\
*    \text{highest} &= \begin{bmatrix} t^{0}_{02} & t^{1}_{02} & t^{2}_{02} & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix} \\
*    \end{align}
*  \f]
*  For the lowest array, the first \p 2*batchCount entries are zero, for the lower array, the first \p batchCount entries are zero,
*  for the upper array the last \p batchCount entries are zero, and for the highest array, the last \p 2*batchCount entries are zero.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  algo        algorithm to solve the linear system.
*  @param[in]
*  m           size of the pentadiagonal linear system.
*  @param[inout]
*  ds          lower diagonal (distance 2) of pentadiagonal system. First two entries
*              must be zero.
*  @param[inout]
*  dl          lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[inout]
*  d           main diagonal of pentadiagonal system.
*  @param[inout]
*  du          upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[inout]
*  dw          upper diagonal (distance 2) of pentadiagonal system. Last two entries
*              must be zero.
*  @param[inout]
*  x           Dense array of right-hand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount  The number of systems to solve.
*  @param[in]
*  pBuffer     Temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p alg, \p batchCount, \p ds,
*              \p dl, \p d, \p du, \p dw, \p x or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            ds,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            dw,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           ds,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           dw,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       ds,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       dw,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* ds,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* dw,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GPSV_INTERLEAVED_BATCH_H */
