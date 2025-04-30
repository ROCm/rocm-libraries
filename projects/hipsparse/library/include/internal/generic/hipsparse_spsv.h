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
#ifndef HIPSPARSE_SPSV_H
#define HIPSPARSE_SPSV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Create sparse matrix triangular solve descriptor
*  \details
*  \p hipsparseSpSV_createDescr creates a sparse matrix triangular solve descriptor. It should be
*  destroyed at the end using hipsparseSpSV_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Destroy sparse matrix triangular solve descriptor
*  \details
*  \p hipsparseSpSV_destroyDescr destroys a sparse matrix triangular solve descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of solution of triangular linear system: 
*  \f[
*    op(A) \cdot y := \alpha \cdot x,
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  \details
*  \p hipsparseSpSV_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system \f$op(A) \cdot y = \alpha \cdot x\f$, where \f$A\f$ is a sparse matrix in CSR storage 
*  format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  x                   vector descriptor.
*  @param[inout]
*  y                   vector descriptor.
*  @param[in]
*  computeType         floating point precision for the SpSV computation.
*  @param[in]
*  alg                 SpSV algorithm for the SpSV computation.
*  @param[in]
*  spsvDescr           SpSV descriptor.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p y, \p spsvDescr or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Analysis step of solution of triangular linear system:
*  \f[
*    op(A) \cdot y := \alpha \cdot x,
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  \details
*  \p hipsparseSpSV_analysis performs the required analysis used when computing the 
*  solution of triangular linear system \f$op(A) \cdot y = \alpha \cdot x\f$,
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  x               vector descriptor.
*  @param[inout]
*  y               vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSV computation.
*  @param[in]
*  alg             SpSV algorithm for the SpSV computation.
*  @param[in]
*  spsvDescr       SpSV descriptor.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p y, \p spsvDescr or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnVecDescr_t  x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnVecDescr_t x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular solve
*
*  \details
*  \p hipsparseSpSV_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  x               vector descriptor.
*  @param[inout]
*  y               vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSV computation.
*  @param[in]
*  alg             SpSV algorithm for the SpSV computation.
*  @param[in]
*  spsvDescr       SpSV descriptor.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p y, or \p spsvDescr 
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnVecDescr_t  x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnVecDescr_t x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPSV_H */
