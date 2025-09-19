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
#ifndef HIPSPARSE_SDDMM_H
#define HIPSPARSE_SDDMM_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSDDMM_bufferSize returns the size of the required buffer needed when computing the sampled
*  dense-dense matrix multiplication:
*  \f[
*    C := \alpha (op(A) \cdot op(B)) \circ spy(C) + \beta \cdot C,
*  \f]
*  where \f$C\f$ is a sparse matrix and \f$A\f$ and \f$B\f$ are dense matrices. This routine is used in
*  conjunction with \ref hipsparseSDDMM_preprocess() and \ref hipsparseSDDMM().
*
*  \p hipsparseSDDMM_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSDDMM
*  for a complete listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB                 dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  A                   dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B                   dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[inout]
*  C                   sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType         floating point precision for the SDDMM computation.
*  @param[in]
*  alg                 specification of the algorithm to use.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p A, \p B, \p D, \p C or
*          \p pBufferSizeInBytes pointer is invalid or the value of \p opA or \p opB is incorrect
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*          \p opB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t A,
                                            hipsparseConstDnMatDescr_t B,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      C,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            size_t*                    pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSDDMM_preprocess performs the required preprocessing used when computing the
*  sampled dense dense matrix multiplication:
*  \f[
*    C := \alpha (op(A) \cdot op(B)) \circ spy(C) + \beta \cdot C,
*  \f]
*  where \f$C\f$ is a sparse matrix and \f$A\f$ and \f$B\f$ are dense matrices. This routine is
*  used in conjunction with \ref hipsparseSDDMM().
*
*  \p hipsparseSDDMM_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSDDMM
*  for a complete listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType  floating point precision for the SDDMM computation.
*  @param[in]
*  alg          specification of the algorithm to use.
*  @param[in]
*  tempBuffer   temporary storage buffer allocated by the user. The size must be greater or equal to
*               the size obtained with \ref hipsparseSDDMM_bufferSize.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p A, \p B, \p C or
*          \p tempBuffer pointer is invalid or the value of \p opA or \p opB is incorrect.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*          \p opB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t A,
                                            hipsparseConstDnMatDescr_t B,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      C,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            void*                      tempBuffer);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            void*                       tempBuffer);
#endif

/*! \ingroup generic_module
*  \brief Sampled Dense-Dense Matrix Multiplication.
*
*  \details
*  \p hipsparseSDDMM multiplies the scalar \f$\alpha\f$ with the dense
*  \f$m \times k\f$ matrix \f$op(A)\f$, the dense \f$k \times n\f$ matrix \f$op(B)\f$, filtered by the sparsity pattern
*  of the \f$m \times n\f$ sparse matrix \f$C\f$ and adds the result to \f$C\f$ scaled by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha ( op(A) \cdot op(B) ) \circ spy(C) + \beta C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if op(A) == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if op(A) == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f],
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if op(B) == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if op(B) == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f]
*   and
*  \f[
*    spy(C)_{ij} = \left\{
*    \begin{array}{ll}
*        1, & \text{ if C_{ij} != 0} \\
*        0, & \text{ otherwise} \\
*    \end{array}
*    \right.
*  \f]
*
*  Computing the above sampled dense-dense multiplication requires three steps to complete. First, the user calls
*  \ref hipsparseSDDMM_bufferSize to determine the size of the required temporary storage buffer. Next, the user
*  allocates this buffer and calls \ref hipsparseSDDMM_preprocess which performs any analysis of the input matrices
*  that may be required. Finally, the user calls \p hipsparseSDDMM to complete the computation. Once all calls to
*  \p hipsparseSDDMM are complete, the temporary buffer can be deallocated.
*
*  \p hipsparseSDDMM supports different algorithms which can provide better performance for different matrices.
*
*  <table>
*  <caption id="sddmm_algorithms">Algorithms</caption>
*  <tr><th>CSR/CSC Algorithms
*  <tr><td>HIPSPARSE_SDDMM_ALG_DEFAULT</td>
*  </table>
*
*  Currently, \p hipsparseSDDMM only supports the uniform precisions indicated in the table below. For the sparse matrix
*  \f$C\f$, \p hipsparseSDDMM supports the index types \ref HIPSPARSE_INDEX_32I and \ref HIPSPARSE_INDEX_64I.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="sddmm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>HIP_R_16F
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="sddmm_mixed">Mixed Precisions</caption>
*  <tr><th>A / B     <th>C         <th>compute_type
*  <tr><td>HIP_R_16F <td>HIP_R_32F <td>HIP_R_32F
*  <tr><td>HIP_R_16F <td>HIP_R_16F <td>HIP_R_32F
*  </table>
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType  floating point precision for the SDDMM computation.
*  @param[in]
*  alg          specification of the algorithm to use.
*  @param[in]
*  tempBuffer   temporary storage buffer allocated by the user. The size must be greater or equal to
*               the size obtained with \ref hipsparseSDDMM_bufferSize.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p A, \p B, \p C or
*          \p tempBuffer pointer is invalid or the value of \p opA or \p opB is incorrect.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*          \p opB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
*
*  \par Example
*  This example performs sampled dense-dense matrix product, \f$C := \alpha ( A \cdot B ) \circ spy(C) + \beta C\f$
*  where \f$\circ\f$ is the hadamard product
*  \snippet example_hipsparse_sddmm.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t          handle,
                                 hipsparseOperation_t       opA,
                                 hipsparseOperation_t       opB,
                                 const void*                alpha,
                                 hipsparseConstDnMatDescr_t A,
                                 hipsparseConstDnMatDescr_t B,
                                 const void*                beta,
                                 hipsparseSpMatDescr_t      C,
                                 hipDataType                computeType,
                                 hipsparseSDDMMAlg_t        alg,
                                 void*                      tempBuffer);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t           handle,
                                 hipsparseOperation_t        opA,
                                 hipsparseOperation_t        opB,
                                 const void*                 alpha,
                                 const hipsparseDnMatDescr_t A,
                                 const hipsparseDnMatDescr_t B,
                                 const void*                 beta,
                                 hipsparseSpMatDescr_t       C,
                                 hipDataType                 computeType,
                                 hipsparseSDDMMAlg_t         alg,
                                 void*                       tempBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SDDMM_H */
