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
#ifndef HIPSPARSE_SPSM_H
#define HIPSPARSE_SPSM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpSM_createDescr creates a sparse matrix triangular solve with multiple rhs descriptor. It should be
*  destroyed at the end using \ref hipsparseSpSM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpSM_destroyDescr destroys a sparse matrix triangular solve with multiple rhs descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpSM_bufferSize computes the size of the required user allocated buffer needed when solving the
*  triangular linear system:
*  \f[
*    op(A) \cdot C := \alpha \cdot op(B),
*  \f]
*  where \f$op(A)\f$ is a square sparse matrix in CSR or COO storage format, \f$B\f$ and \f$C\f$ are dense matrices.
*
*  \p hipsparseSpSM_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpSM_solve
*  for a complete listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type for the sparse matrix \f$A\f$.
*  @param[in]
*  opB                 matrix operation type for the dense matrix \f$B\f$.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                sparse matrix descriptor.
*  @param[in]
*  matB                dense matrix descriptor.
*  @param[inout]
*  matC                dense matrix descriptor.
*  @param[in]
*  computeType         floating point precision for the SpSM computation.
*  @param[in]
*  alg                 SpSM algorithm for the SpSM computation.
*  @param[in]
*  spsmDescr           SpSM descriptor.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p spsmDescr or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpSM_analysis performs the required analysis needed when solving the
*  triangular linear system:
*  \f[
*    op(A) \cdot C := \alpha \cdot op(B),
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR or COO storage format, \f$B\f$ and \f$C\f$ are dense vectors.
*
*  \p hipsparseSpSM_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpSM_solve
*  for a complete listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type for the sparse matrix \f$A\f$.
*  @param[in]
*  opB             matrix operation type for the dense matrix \f$B\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            sparse matrix descriptor.
*  @param[in]
*  matB            dense matrix descriptor.
*  @param[inout]
*  matC            dense matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSM computation.
*  @param[in]
*  alg             SpSM algorithm for the SpSM computation.
*  @param[in]
*  spsmDescr       SpSM descriptor.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p spsmDescr or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnMatDescr_t  matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnMatDescr_t matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular system solve
*
*  \details
*  \p hipsparseSpSM_solve solves a triangular linear system of equations defined by a sparse \f$m \times m\f$ square matrix \f$op(A)\f$,
*  given in CSR or COO storage format, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if opA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if opB == HIPSPARSE_OPERATION_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if opA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if opB == HIPSPARSE_OPERATION_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and where \f$C\f$ is the dense solution matrix and \f$B\f$ is the dense right-hand side matrix. Both \f$B\f$
*  and \f$C\f$ can be in row or column order.
*
*  Performing the above operation requires three steps. First, the user calls \ref hipsparseSpSM_bufferSize in order to
*  determine the size of the required temporary storage buffer. The user then allocates this buffer and calls
*  \ref hipsparseSpSM_analysis which will perform analysis on the sparse matrix \f$op(A)\f$. Finally, the user completes
*  the computation by calling \p hipsparseSpSM_solve. The buffer size and analysis routines only need to be called once
*  for a given sparse matrix \f$op(A)\f$ while the computation can be called repeatedly with different \f$B\f$ and \f$C\f$
*  matrices. Once all calls to \p hipsparseSpSM_solve are complete, the temporary buffer can be deallocated.
*
*  As noted above, both \f$B\f$ and \f$C\f$ can be in row or column order (this includes mixing the order so that \f$B\f$ is
*  row order and \f$C\f$ is column order and vice versa). When running on an AMD system with the rocSPARSE backend, the kernels
*  solve the system assuming the matrices \f$B\f$ and \f$C\f$ are in row order as this provides the best memory access. This
*  means that if the matrix \f$C\f$ is not in row order and/or the matrix \f$B\f$ is not row order (or \f$B^{T}\f$ is not column
*  order as this is equivalent to being in row order), then internally memory copies and/or transposing of data may be performed
*  to get them into the correct order (possbily using extra buffer size). Once computation is completed, additional memory copies
*  and/or transposing of data may be performed to get them back into the user arrays. For best performance and smallest required
*  temporary storage buffers on an AMD system, use row order for the matrix \f$C\f$ and row order for the matrix \f$B\f$ (or column
*  order if \f$B\f$ is being transposed).
*
*  \p hipsparseSpSM_solve supports \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I index precisions for storing the
*  row pointer and column indices arrays of the sparse matrices. \p hipsparseSpSM_solve supports the following data types for
*  \f$op(A)\f$, \f$op(B)\f$, \f$C\f$ and compute types for \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spsm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type for the sparse matrix \f$A\f$.
*  @param[in]
*  opB             matrix operation type for the dense matrix \f$B\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            sparse matrix descriptor.
*  @param[in]
*  matB            dense matrix descriptor.
*  @param[inout]
*  matC            dense matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSM computation.
*  @param[in]
*  alg             SpSM algorithm for the SpSM computation.
*  @param[in]
*  spsmDescr       SpSM descriptor.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p spsmDescr or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \snippet example_hipsparse_spsm.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnMatDescr_t  matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPSM_H */
