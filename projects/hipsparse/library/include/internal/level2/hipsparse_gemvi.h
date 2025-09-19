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
#ifndef HIPSPARSE_GEMVI_H
#define HIPSPARSE_GEMVI_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
 *  \details
 *  \p hipsparseXgemvi_bufferSize returns the size of the temporary storage buffer in bytes
 *  required by \ref hipsparseSgemvi "hipsparseXgemvi()". The temporary storage buffer must
 *  be allocated by the user.
 *
 *  @param[in]
 *  handle      handle to the hipsparse library context queue.
 *  @param[in]
 *  transA      matrix operation type.
 *  @param[in]
 *  m           number of rows of the dense matrix.
 *  @param[in]
 *  n           number of columns of the dense matrix.
 *  @param[in]
 *  nnz         number of non-zero entries in the sparse vector.
 *  @param[out]
 *  pBufferSizeInBytes temporary storage buffer size.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz or
 *              \p pBufferSizeInBytes is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
 *  \brief Dense matrix sparse vector multiplication
 *
 *  \details
 *  \p hipsparseXgemvi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times n\f$
 *  matrix \f$A\f$ and the sparse vector \f$x\f$ and adds the result to the dense vector
 *  \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if transA == HIPSPARSE_OPERATION_NON_TRANSPOSE}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  Performing the above operation involves two steps. First, the user calls
 *  \ref hipsparseSgemvi_bufferSize "hipsparseXgemvi_bufferSize()" in order to determine the size of
 *  the temporary storage buffer. Next, the user allocates this temporary buffer and passes it to
 *  \p hipsparseXgemvi to complete the computation. Once all calls to \p hipsparseXgemvi are complete the
 *  temporary storage buffer can be freed.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
 *
 *  @param[in]
 *  handle      handle to the hipsparse library context queue.
 *  @param[in]
 *  transA      matrix operation type.
 *  @param[in]
 *  m           number of rows of the dense matrix.
 *  @param[in]
 *  n           number of columns of the dense matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  A           pointer to the dense matrix.
 *  @param[in]
 *  lda         leading dimension of the dense matrix
 *  @param[in]
 *  nnz         number of non-zero entries in the sparse vector
 *  @param[in]
 *  x           array of \p nnz elements containing the values of the sparse vector
 *  @param[in]
 *  xInd        array of \p nnz elements containing the indices of the sparse vector
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  idxBase     \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
 *  @param[in]
 *  pBuffer     temporary storage buffer
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p nnz, \p alpha,
 *              \p A, \p x, \p xInd, \p beta, \p y or \p pBuffer is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 *
 *  \par Example
 *  \snippet example_hipsparse_gemvi.cpp doc example
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const float*         alpha,
                                  const float*         A,
                                  int                  lda,
                                  int                  nnz,
                                  const float*         x,
                                  const int*           xInd,
                                  const float*         beta,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const double*        alpha,
                                  const double*        A,
                                  int                  lda,
                                  int                  nnz,
                                  const double*        x,
                                  const int*           xInd,
                                  const double*        beta,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const hipComplex*    alpha,
                                  const hipComplex*    A,
                                  int                  lda,
                                  int                  nnz,
                                  const hipComplex*    x,
                                  const int*           xInd,
                                  const hipComplex*    beta,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t       handle,
                                  hipsparseOperation_t    transA,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  int                     nnz,
                                  const hipDoubleComplex* x,
                                  const int*              xInd,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase,
                                  void*                   pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GEMVI_H */
