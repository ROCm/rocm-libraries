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
#ifndef HIPSPARSE_CSRSV_H
#define HIPSPARSE_CSRSV_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \details
*  \p hipsparseXcsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() or hipsparseZcsrsv2_solve()
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrsv2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle \p info or \p position is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \details
*  \p hipsparseXcsrsv2_bufferSize returns the size of the temporary storage buffer in bytes
*  that is required by \ref hipsparseScsrsv2_analysis "hipsparseScsrsv2_analysis()" and
*  \ref hipsparseScsrsv2_solve "hipsparseXcsrsv2_solve()". The temporary storage buffer must
*  be allocated by the user.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  transA           matrix operation type.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz              number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*                   CSR matrix.
*  @param[out]
*  info             structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     \ref hipsparseScsrsv2_analysis "hipsparseXcsrsv2_analysis()" and
*                     \ref hipsparseScsrsv2_solve "hipsparseXcsrsv2_solve()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA,
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup level2_module
*  \details
*  \p hipsparseXcsrsv2_bufferSizeExt returns the size of the temporary storage buffer in bytes
*  that is required by \ref hipsparseScsrsv2_analysis "hipsparseXcsrsv2_analysis()" and
*  \ref hipsparseScsrsv2_solve "hipsparseScsrsv2_solve()". The temporary storage buffer must be
*  allocated by the user.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  transA           matrix operation type.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz              number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*                   CSR matrix.
*  @param[out]
*  info             structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     \ref hipsparseScsrsv2_analysis "hipsparseXcsrsv2_analysis()" and
*                     \ref hipsparseScsrsv2_solve "hipsparseXcsrsv2_solve()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA,
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \details
*  \p hipsparseXcsrsv2_analysis performs the analysis step for
*  \ref hipsparseScsrsv2_solve "hipsparseXcsrsv2_solve()". It is expected that this
*  function will be executed only once for a given matrix and particular operation
*  type.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  transA           matrix operation type.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz              number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*                   CSR matrix.
*  @param[out]
*  info             structure that holds the information collected during
*                   the analysis step.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descr,
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBuffer is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
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
*  Performing the above operation requires three steps. First, the user calls
*  \ref hipsparseScsrsv2_bufferSize "hipsparseXcsrsv2_bufferSize()" (or
*  \ref hipsparseScsrsv2_bufferSizeExt "hipsparseXcsrsv2_bufferSizeExt()") which will determine the size of the
*  required temporary storage buffer. The user then allocates this buffer and calls
*  \ref hipsparseScsrsv2_analysis "hipsparseXcsrsv2_analysis()" which will perform analysis on the sparse matrix
*  \f$op(A)\f$. Finally, the user completes the computation by calling \p hipsparseXcsrsv2_solve. The buffer size,
*  buffer allocation, and analysis only need to be called once for a given sparse matrix \f$op(A)\f$ while the
*  computation stage can be repeatedly used with different \f$x\f$ and \f$y\f$ vectors. Once all calls to
*  \p hipsparseXcsrsv2_solve are complete, the temporary buffer can be deallocated.
*
*  Solving a triangular system involves division by the diagonal elements. This means that if the sparse matrix is
*  missing the diagonal entry (referred to as a structural zero) or the diagonal entry is zero (referred to as a numerical zero)
*  then a division by zero would occur. \p hipsparseXcsrsv2_solve tracks the location of the first zero pivot (either numerical
*  or structural zero). The zero pivot status can be checked calling \ref hipsparseXcsrsv2_zeroPivot(). If
*  \ref hipsparseXcsrsv2_zeroPivot() returns \ref HIPSPARSE_STATUS_SUCCESS, then no zero pivot was found and therefore
*  the matrix does not have a structural or numerical zero.
*
*  The user can specify that the sparse matrix should be interpreted as having ones on the diagonal by setting the diagonal type
*  on the descriptor \p descrA to \ref HIPSPARSE_DIAG_TYPE_UNIT using \ref hipsparseSetMatDiagType. If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be reported, even if \f$A_{j,j} = 0\f$ for
*  some \f$j\f$.
*
*  The sparse CSR matrix passed to \p hipsparseXcsrsv2_solve does not actually have to be a triangular matrix. Instead the
*  triangular upper or lower part of the sparse matrix is solved based on \ref hipsparseFillMode_t set on the descriptor
*  \p descrA. If the fill mode is set to \ref HIPSPARSE_FILL_MODE_LOWER, then the lower triangular matrix is solved. If the
*  fill mode is set to \ref HIPSPARSE_FILL_MODE_UPPER then the upper triangular matrix is solved.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
*  \p transA == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  transA      matrix operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  f           array of \p m elements, holding the right-hand side.
*  @param[out]
*  x           array of \p m elements, holding the solution.
*  @param[in]
*  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or
*              \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA,
*              \p alpha, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA,
*              \p f or \p x is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p transA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \snippet example_hipsparse_csrsv2.cpp doc example
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRSV_H */
