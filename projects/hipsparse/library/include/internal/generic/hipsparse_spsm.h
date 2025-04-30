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
*  \brief Create sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_createDescr creates a sparse matrix triangular solve with multiple rhs descriptor. It should be
*  destroyed at the end using hipsparseSpSM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Destroy sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_destroyDescr destroys a sparse matrix triangular solve with multiple rhs descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of solution of triangular linear system:
*  \f[
*    op(A) \cdot C := \alpha \cdot op(B),
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$B\f$ and \f$C\f$ are dense matrices.
*
*  \details
*  \p hipsparseSpSM_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system \f$op(A) \cdot C = \alpha \cdot op(B)\f$, where \f$A\f$ is a sparse matrix in CSR storage 
*  format, \f$B\f$ and \f$C\f$ are dense matrices.
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
*  \brief Analysis step of solution of triangular linear system:
*  \f[
*    op(A) \cdot C := \alpha \cdot op(B),
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$B\f$ and \f$C\f$ are dense vectors.
*
*  \details
*  \p hipsparseSpSM_analysis performs the required analysis used when computing the 
*  solution of triangular linear system \f$op(A) \cdot C = \alpha \cdot op(B)\f$,
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$B\f$ and \f$C\f$ are dense vectors.
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
*  \p hipsparseSpSM_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution matrix
*  \f$C\f$ and the right-hand side \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
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
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
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
