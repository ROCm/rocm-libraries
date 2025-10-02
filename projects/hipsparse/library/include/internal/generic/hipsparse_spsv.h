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
*  \details
*  \p hipsparseSpSV_createDescr creates a sparse matrix triangular solve descriptor. It should be
*  destroyed at the end using \ref hipsparseSpSV_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpSV_destroyDescr destroys a sparse matrix triangular solve descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpSV_bufferSize computes the size of the required user allocated buffer needed when solving the
*  triangular linear system:
*  \f[
*    op(A) \cdot y := \alpha \cdot x,
*  \f]
*  where \f$op(A)\f$ is a sparse matrix in CSR or COO storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  \p hipsparseSpSV_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpSV_solve
*  for a complete listing of all the data type and compute type combinations available.
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
*  \details
*  \p hipsparseSpSV_analysis performs the required analysis needed when solving the
*  triangular linear system:
*  \f[
*    op(A) \cdot y := \alpha \cdot x,
*  \f]
*  where \f$op(A)\f$ is a sparse matrix in CSR or COO storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  \p hipsparseSpSV_analysis supports multiple combinations of data types and compute types. See \ref hipsparseSpSV_solve
*  for a complete listing of all the data type and compute type combinations available.
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
*  \p hipsparseSpSV_solve solves a triangular linear system of equations defined by a sparse \f$m \times m\f$ square matrix
*  \f$op(A)\f$, given in CSR or COO storage format, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and where \f$y\f$ is the dense solution vector and \f$x\f$ is the dense right-hand side vector.
*
*  Performing the above operation requires three steps. First, \ref hipsparseSpSV_bufferSize must be called which will
*  determine the size of the required temporary storage buffer. The user then allocates this buffer and calls
*  \ref hipsparseSpSV_analysis which will perform analysis on the sparse matrix \f$op(A)\f$. Finally, the user completes
*  the computation by calling \p hipsparseSpSV_solve. The buffer size and preprecess routines only need to be called once
*  for a given sparse matrix \f$op(A)\f$ while the computation can be repeatedly used with different \f$x\f$ and \f$y\f$
*  vectors. Once all calls to \p hipsparseSpSV_solve are complete, the temporary buffer can be deallocated.
*
*  \p hipsparseSpSV_solve supports \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I index types for
*  storing the row pointer and column indices arrays of the sparse matrices. \p hipsparseSpSV_solve supports the following
*  data types for \f$op(A)\f$, \f$x\f$, \f$y\f$ and compute types for \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spsv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
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
*
*  \par Example
*  \snippet example_hipsparse_spsv.cpp doc example
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
