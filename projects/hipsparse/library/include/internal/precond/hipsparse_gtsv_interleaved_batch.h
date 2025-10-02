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
#ifndef HIPSPARSE_GTSV_INTERLEAVED_BATCH_H
#define HIPSPARSE_GTSV_INTERLEAVED_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p hipsparseXgtsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgtsvInterleavedBatch "hipsparseXgtsvInterleavedBatch()".
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  algo               Algorithm to use when solving tridiagonal systems. Options are thomas ( \p algo=0 ),
*                     LU ( \p algo=1 ), or QR ( \p algo=2 ). Thomas algorithm is the fastest but is not
*                     stable while LU and QR are slower but are stable.
*  @param[in]
*  m                  size of the tri-diagonal linear system.
*  @param[in]
*  dl                 lower diagonal of tri-diagonal system. The first element of the lower diagonal must be zero.
*  @param[in]
*  d                  main diagonal of tri-diagonal system.
*  @param[in]
*  du                 upper diagonal of tri-diagonal system. The last element of the upper diagonal must be zero.
*  @param[inout]
*  x                  Dense array of righthand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount         The number of systems to solve.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     \ref hipsparseSgtsvInterleavedBatch "hipsparseSgtsvInterleavedBatch()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p dl, \p d, \p du,
*              \p x or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch tridiagonal solver
*
*  \details
*  \p hipsparseXgtsvInterleavedBatch solves a batched tridiagonal linear system
*  \f[
*    T^{i}*x^{i} = x^{i}
*  \f]
*  where for each batch \f$i=0\ldots\f$ \p batchCount, \f$T^{i}\f$ is a sparse tridiagonal matrix and
*  \f$x^{i}\f$ is a dense right-hand side vector. All of the tridiagonal matrices, \f$T^{i}\f$, are
*  packed in an interleaved fashion into three vectors: \p dl for the lower diagonals, \p d for the main
*  diagonals and \p du for the upper diagonals. See below for a description of what this interleaved
*  memory pattern looks like.
*
*  Solving the batched tridiagonal system involves two steps. First, the user calls
*  \ref hipsparseSgtsvInterleavedBatch_bufferSizeExt "hipsparseXgtsvInterleavedBatch_bufferSizeExt()"
*  in order to determine the size of the required temporary storage buffer. Once determined, the user allocates
*  this buffer and passes it to \ref hipsparseSgtsvInterleavedBatch "hipsparseXgtsvInterleavedBatch()"
*  to perform the actual solve. The \f$x^{i}\f$ vectors, which initially stores the right-hand side values, are
*  overwritten with the solution after the call to
*  \ref hipsparseSgtsvInterleavedBatch "hipsparseXgtsvInterleavedBatch()".
*
*  The user can specify different algorithms for \p hipsparseXgtsvInterleavedBatch
*  to use. Options are thomas ( \p algo=0 ),
*  LU ( \p algo=1 ), or QR ( \p algo=2 ).
*
*  Unlike the strided batch routines which write each batch matrix one after the other in memory, the interleaved
*  routines write the batch matrices such that each element from each matrix is written consecutively one after
*  the other. For example, consider the following batch matrices:
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
*  In interleaved format, the upper, lower, and diagonal arrays would look like:
*  \f[
*    \begin{align}
*    \text{lower} &= \begin{bmatrix} 0 & 0 & 0 & t^{0}_{10} & t^{1}_{10} & t^{1}_{10} & t^{0}_{21} & t^{1}_{21} & t^{2}_{21} \end{bmatrix} \\
*    \text{diagonal} &= \begin{bmatrix} t^{0}_{00} & t^{1}_{00} & t^{2}_{00} & t^{0}_{11} & t^{1}_{11} & t^{2}_{11} & t^{0}_{22} & t^{1}_{22} & t^{2}_{22} \end{bmatrix} \\
*    \text{upper} &= \begin{bmatrix} t^{0}_{01} & t^{1}_{01} & t^{2}_{01} & t^{0}_{12} & t^{1}_{12} & t^{2}_{12} & 0 & 0 & 0 \end{bmatrix} \\
*    \end{align}
*  \f]
*  For the lower array, the first \p batchCount entries are zero and for the upper array the last \p batchCount
*  entries are zero.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  algo        Algorithm to use when solving tridiagonal systems. Options are thomas ( \p algo=0 ),
*              LU ( \p algo=1 ), or QR ( \p algo=2 ). Thomas algorithm is the fastest but is not
*              stable while LU and QR are slower but are stable.
*  @param[in]
*  m           size of the tri-diagonal linear system.
*  @param[inout]
*  dl          lower diagonal of tri-diagonal system. The first element of the lower diagonal must be zero.
*  @param[inout]
*  d           main diagonal of tri-diagonal system.
*  @param[inout]
*  du          upper diagonal of tri-diagonal system. The last element of the upper diagonal must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount  The number of systems to solve.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p dl, \p d,
*              \p du, \p x or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \snippet example_hipsparse_gtsv_interleaved_batch.cpp doc example
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GTSV_INTERLEAVED_BATCH_H */
