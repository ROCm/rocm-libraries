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
#ifndef HIPSPARSE_SPMV_H
#define HIPSPARSE_SPMV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpMV_bufferSize computes the required user allocated buffer size needed when computing the
*  sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR, CSC, COO, or COO (AoS) format, \f$x\f$ is
*  a dense vector of length \f$n\f$ and \f$y\f$ is a dense vector of length \f$m\f$.
*
*  \p hipsparseSpMV_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpMV for a complete
*  listing of all the data type and compute type combinations available.
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
*  vecX                vector descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[inout]
*  vecY                vector descriptor.
*  @param[in]
*  computeType         floating point precision for the SpMV computation.
*  @param[in]
*  alg                 SpMV algorithm for the SpMV computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p pBufferSizeInBytes pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpMV_preprocess performs analysis on the sparse matrix \f$op(A)\f$ when computing the
*  sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR, CSC, COO, or COO (AoS) format, \f$x\f$
*  is a dense vector of length \f$n\f$ and \f$y\f$ is a dense vector of length \f$m\f$. This step is
*  optional but if used may results in better performance.
*
*  \p hipsparseSpMV_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSpMV for
*  a complete listing of all the data type and compute type combinations available.
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
*  vecX            vector descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[inout]
*  vecY            vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMV computation.
*  @param[in]
*  alg             SpMV algorithm for the SpMV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p externalBuffer pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$ matrix \f$op(A)\f$, defined in CSR,
*  CSC, COO, or COO (AoS) format, with the dense vector \f$x\f$ and adds the result to the dense vector \f$y\f$
*  that is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
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
*  Performing the above operation involves multiple steps. First the user calls \ref hipsparseSpMV_bufferSize to determine the
*  size of the required temporary storage buffer. The user then allocates this buffer and calls \ref hipsparseSpMV_preprocess.
*  Depending on the algorithm and sparse matrix format, this will perform analysis on the sparsity pattern of \f$op(A)\f$. Finally
*  the user completes the operation by calling \p hipsparseSpMV. The buffer size and preprecess routines only need to be called
*  once for a given sparse matrix \f$op(A)\f$ while the computation can be repeatedly used with different \f$x\f$ and \f$y\f$
*  vectors. Once all calls to \p hipsparseSpMV are complete, the temporary buffer can be deallocated.
*
*  \p hipsparseSpMV supports multiple different algorithms. These algorithms have different trade offs depending on the sparsity
*  pattern of the matrix, whether or not the results need to be deterministic, and how many times the sparse-vector product will
*  be performed.
*
*  <table>
*  <caption id="spmv_csr_algorithms">CSR/CSC Algorithms</caption>
*  <tr><th>CSR Algorithms
*  <tr><td>HIPSPARSE_SPMV_CSR_ALG1</td>
*  <tr><td>HIPSPARSE_SPMV_CSR_ALG2</td>
*  </table>
*
*  <table>
*  <caption id="spmv_coo_algorithms">COO Algorithms</caption>
*  <tr><th>COO Algorithms
*  <tr><td>HIPSPARSE_SPMV_COO_ALG1</td>
*  <tr><td>HIPSPARSE_SPMV_COO_ALG2</td>
*  </table>
*
*  \p hipsparseSpMV supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported data types that can be used for the sparse matrix \f$op(A)\f$ and the dense vectors \f$x\f$ and \f$y\f$ and the
*  compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth
*  and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmv_mixed">Mixed Precisions</caption>
*  <tr><th>A / X      <th>Y         <th>compute_type
*  <tr><td>HIP_R_8I   <td>HIP_R_32I <td>HIP_R_32I
*  <tr><td>HIP_R_8I   <td>HIP_R_32F <td>HIP_R_32F
*  <tr><td>HIP_R_16F  <td>HIP_R_32F <td>HIP_R_32F
*  <tr><td>HIP_R_16BF <td>HIP_R_32F <td>HIP_R_32F
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="spmv_mixed_regular_real">Mixed-regular real precisions</caption>
*  <tr><th>A         <th>X / Y / compute_type
*  <tr><td>HIP_R_32F <td>HIP_R_64F
*  <tr><td>HIP_C_32F <td>HIP_C_64F
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="spmv_mixed_regular_complex">Mixed-regular Complex precisions</caption>
*  <tr><th>A         <th>X / Y / compute_type
*  <tr><td>HIP_R_32F <td>HIP_C_32F
*  <tr><td>HIP_R_64F <td>HIP_C_64F
*  </table>
*
*  \p hipsparseSpMV supports \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I index precisions
*  for storing the row pointer and row/column indices arrays of the sparse matrices.
*
*  \note
*  None of the algorithms above are deterministic when \f$A\f$ is transposed.
*
*  \note
*  The sparse matrix formats currently supported are: \ref HIPSPARSE_FORMAT_COO, \ref HIPSPARSE_FORMAT_COO_AOS,
*  \ref HIPSPARSE_FORMAT_CSR, and \ref HIPSPARSE_FORMAT_CSC.
*
*  \note
*  Only the \ref hipsparseSpMV_bufferSize and \ref hipsparseSpMV routines are non blocking and executed asynchronously
*  with respect to the host. They may return before the actual computation has finished. The \ref hipsparseSpMV_preprocess
*  routine is blocking with respect to the host.
*
*  \note
*  Only the \ref hipsparseSpMV_bufferSize and the \ref hipsparseSpMV routines support execution in a hipGraph context.
*  The \ref hipsparseSpMV_preprocess stage does not support hipGraph.
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
*  vecX            vector descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[inout]
*  vecY            vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMV computation.
*  @param[in]
*  alg             SpMV algorithm for the SpMV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p externalBuffer pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \snippet example_hipsparse_spmv.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnVecDescr_t  vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPMV_H */
