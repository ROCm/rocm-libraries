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
#ifndef HIPSPARSE_SPMM_H
#define HIPSPARSE_SPMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpMM_bufferSize computes the required user allocated buffer size needed when computing the
*  sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR, COO, BSR or Blocked ELL storage format,
*  \f$B\f$ is a dense matrix of size \f$k \times n\f$ and \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \p hipsparseSpMM_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpMM
*  for a complete listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  opB                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  matB                matrix descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[in]
*  matC                matrix descriptor.
*  @param[in]
*  computeType         floating point precision for the SpMM computation.
*  @param[in]
*  alg                 SpMM algorithm for the SpMM computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpMM_preprocess performs the required preprocessing used when computing the
*  sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR, COO, BSR or Blocked ELL storage format,
*  \f$B\f$ is a dense matrix of size \f$k \times n\f$ and \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \p hipsparseSpMM_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSpMM for a complete
*  listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  opB             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  matB            matrix descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  matC            matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMM computation.
*  @param[in]
*  alg             SpMM algorithm for the SpMM computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$ matrix \f$op(A)\f$,
*  defined in CSR, COO, BSR or Blocked ELL storage format, and the dense \f$k \times n\f$ matrix \f$op(B)\f$
*  and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that is multiplied by the scalar
*  \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  Both \f$B\f$ and \f$C\f$ can be in row or column order.
*
*  \p hipsparseSpMM requires three stages to complete. First, the user calls \ref hipsparseSpMM_bufferSize to determine
*  the size of the required temporary storage buffer. Next, the user allocates this buffer and calls
*  \ref hipsparseSpMM_preprocess which will perform analysis on the sparse matrix \f$op(A)\f$. Finally, the user calls
*  \p hipsparseSpMM to perform the actual computation. The buffer size and preprecess routines only need to be called once for a given
*  sparse matrix \f$op(A)\f$ while the computation routine can be repeatedly used with different \f$B\f$ and \f$C\f$ matrices.
*  Once all calls to \p hipsparseSpMM are complete, the temporary buffer can be deallocated.
*
*  As noted above, both \f$B\f$ and \f$C\f$ can be in row or column order (this includes mixing the order so that \f$B\f$ is
*  row order and \f$C\f$ is column order and vice versa). For best performance, use row order for both \f$B\f$ and \f$C\f$ as
*  this provides the best memory access.
*
*  \p hipsparseSpMM supports multiple different algorithms. These algorithms have different trade offs depending on the sparsity
*  pattern of the matrix, whether or not the results need to be deterministic, and how many times the sparse-matrix product will
*  be performed.
*
*  <table>
*  <caption id="spmm_csr_algorithms">CSR Algorithms</caption>
*  <tr><th>CSR Algorithms
*  <tr><td>HIPSPARSE_SPMM_CSR_ALG1</td>
*  <tr><td>HIPSPARSE_SPMM_CSR_ALG2</td>
*  <tr><td>HIPSPARSE_SPMM_CSR_ALG3</td>
*  </table>
*
*  <table>
*  <caption id="spmm_coo_algorithms">COO Algorithms</caption>
*  <tr><th>COO Algorithms
*  <tr><td>HIPSPARSE_SPMM_COO_ALG1</td>
*  <tr><td>HIPSPARSE_SPMM_COO_ALG2</td>
*  <tr><td>HIPSPARSE_SPMM_COO_ALG3</td>
*  <tr><td>HIPSPARSE_SPMM_COO_ALG4</td>
*  </table>
*
*  <table>
*  <caption id="spmm_bell_algorithms">Blocked-ELL Algorithms</caption>
*  <tr><th>ELL Algorithms
*  <tr><td>HIPSPARSE_SPMM_BLOCKED_ELL_ALG1</td>
*  </table>
*
*  <table>
*  <caption id="spmm_bsr_algorithms">BSR Algorithms</caption>
*  <tr><th>BSR Algorithms
*  <tr><td>CUSPARSE_SPMM_BSR_ALG1</td>
*  </table>
*
*  One can also pass \ref HIPSPARSE_SPMM_ALG_DEFAULT which will automatically select from the algorithms listed above
*  based on the sparse matrix format.
*
*  When A is transposed, \p hipsparseSpMM will revert to using \ref HIPSPARSE_SPMM_CSR_ALG2
*  for CSR format and \ref HIPSPARSE_SPMM_COO_ALG1 for COO format regardless of algorithm selected.
*
*  \p hipsparseSpMM supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix \f$op(A)\f$ and the dense matrices \f$op(B)\f$ and
*  \f$C\f$ and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmm_mixed">Mixed Precisions</caption>
*  <tr><th>A / B      <th>C         <th>compute_type
*  <tr><td>HIP_R_8I   <td>HIP_R_32I <td>HIP_R_32I
*  <tr><td>HIP_R_8I   <td>HIP_R_32F <td>HIP_R_32F
*  <tr><td>HIP_R_16F  <td>HIP_R_32F <td>HIP_R_32F
*  <tr><td>HIP_R_16BF <td>HIP_R_32F <td>HIP_R_32F
*  </table>
*
*  \p hipsparseSpMM supports \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I index precisions
*  for storing the row pointer and column indices arrays of the sparse matrices.
*
*  \p hipsparseSpMM also supports batched computation for CSR and COO matrices. There are three supported batch modes:
*  \f[
*      C_i = A \times B_i \\
*      C_i = A_i \times B \\
*      C_i = A_i \times B_i
*  \f]
*
*  The batch mode is determined by the batch count and stride passed for each matrix. For example
*  to use the first batch mode (\f$C_i = A \times B_i\f$) with 100 batches for non-transposed \f$A\f$,
*  \f$B\f$, and \f$C\f$, one passes:
*  \f[
*      batchCountA=1 \\
*      batchCountB=100 \\
*      batchCountC=100 \\
*      offsetsBatchStrideA=0 \\
*      columnsValuesBatchStrideA=0 \\
*      batchStrideB=k*n \\
*      batchStrideC=m*n
*  \f]
*  To use the second batch mode (\f$C_i = A_i \times B\f$) one could use:
*  \f[
*      batchCountA=100 \\
*      batchCountB=1 \\
*      batchCountC=100 \\
*      offsetsBatchStrideA=m+1 \\
*      columnsValuesBatchStrideA=nnz \\
*      batchStrideB=0 \\
*      batchStrideC=m*n
*  \f]
*  And to use the third batch mode (\f$C_i = A_i \times B_i\f$) one could use:
*  \f[
*      batchCountA=100 \\
*      batchCountB=100 \\
*      batchCountC=100 \\
*      offsetsBatchStrideA=m+1 \\
*      columnsValuesBatchStrideA=nnz \\
*      batchStrideB=k*n \\
*      batchStrideC=m*n
*  \f]
*  See examples below.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  opB             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  matB            matrix descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  matC            matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMM computation.
*  @param[in]
*  alg             SpMM algorithm for the SpMM computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \snippet example_hipsparse_spmm.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnMatDescr_t  matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPMM_H */
