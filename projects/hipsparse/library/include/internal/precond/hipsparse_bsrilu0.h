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
#ifndef HIPSPARSE_BSRILU0_H
#define HIPSPARSE_BSRILU0_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \details
 *  \p hipsparseXbsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
 *  structural or numerical zero has been found during \ref hipsparseSbsrilu02_analysis
 *  "hipsparseXbsrilu02_analysis()" or \ref hipsparseSbsrilu02 "hipsparseXbsrilu02()"
 *  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
 *  using same index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
 *  \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
 *  invertible (numerical zero).
 *
 *  \note \p hipsparseXbsrilu02_zeroPivot is a blocking function. It might influence
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
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position pointer is
 *              invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \details
 *  \p hipsparseXbsrilu02_numericBoost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  @param[in]
 *  handle        handle to the hipsparse library context queue.
 *  @param[in]
 *  info          structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost  enable/disable numeric boost.
 *  @param[in]
 *  tol           tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val     boost value to replace a numerical value.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  double*           boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \details
 *  \p hipsparseXbsrilu02_bufferSize returns the size of the temporary storage buffer
 *  in bytes that is required by \ref hipsparseSbsrilu02_analysis "hipsparseXbsrilu02_analysis()"
 *  and \ref hipsparseSbsrilu02 "hipsparseXbsrilu02()". The temporary storage buffer must be
 *  allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA               direction that specifies whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW
 *                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb                 number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb               number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA             descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedValA      array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedRowPtrA   array of \p mb+1 elements that point to the start of every block row of the
 *                     sparse BSR matrix.
 *  @param[in]
 *  bsrSortedColIndA   array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim           the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[out]
 *  info               structure that holds the information collected during the analysis step.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
 *                     hipsparseSbsrilu02_analysis(), hipsparseDbsrilu02_analysis(),
 *                     hipsparseCbsrilu02_analysis(), hipsparseZbsrilu02_analysis(),
 *                     hipsparseSbsrilu02(), hipsparseDbsrilu02(), hipsparseCbsrilu02()
 *                     and hipsparseZbsrilu02().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA,
 *              \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, \p info or
 *              \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \details
 *  \p hipsparseXbsrilu02_analysis performs the analysis step for \ref hipsparseSbsrilu02
 *  "hipsparseXbsrilu02()". It is expected that this function will be executed only once
 *  for a given matrix.
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle           handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA             direction that specified whether to count nonzero elements by
 *                   \ref HIPSPARSE_DIRECTION_ROW or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb               number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb             number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA           descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedValA    array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of the
 *                   sparse BSR matrix.
 *  @param[in]
 *  bsrSortedColIndA array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim         the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[out]
 *  info             structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy           \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
 *  @param[in]
 *  pBuffer          temporary storage buffer allocated by the user.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA,
 *              \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, \p info or \p pBuffer pointer
 *              is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
 *  pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LU
 *  \f]
 *
 *  Computing the above incomplete LU factorization requires three steps to complete. First,
 *  the user determines the size of the required temporary storage buffer by calling
 *  \ref hipsparseSbsrilu02_bufferSize "hipsparseXbsrilu02_bufferSize()". Once this buffer size
 *  has been determined, the user allocates the buffer and passes it to
 *  \ref hipsparseSbsrilu02_analysis "hipsparseXbsrilu02_analysis()". This will perform analysis on
 *  the sparsity pattern of the matrix. Finally, the user calls \p hipsparseXbsrilu02 to perform the
 *  actual factorization. The calculation of the buffer size and the analysis of the sparse matrix
 *  only need to be performed once for a given sparsity pattern while the factorization can be
 *  repeatedly applied to multiple matrices having the same sparsity pattern. Once all calls to
 *  \ref hipsparseSbsrilu02 "hipsparseXbsrilu02()" are complete, the temporary buffer can be deallocated.
 *
 *  \p hipsparseXbsrilu02 reports the first zero pivot (either numerical or structural zero).
 *  The zero pivot status can be obtained by calling \ref hipsparseXbsrilu02_zeroPivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA               direction that specified whether to count nonzero elements by
 *                     \ref HIPSPARSE_DIRECTION_ROW or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb                 number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb               number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA             descriptor of the sparse BSR matrix.
 *  @param[inout]
 *  bsrSortedValA_valM array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedRowPtrA   array of \p mb+1 elements that point to the start of every block row of the
 *                     sparse BSR matrix.
 *  @param[in]
 *  bsrSortedColIndA   array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim           the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[in]
 *  info               structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy             \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
 *  @param[in]
 *  pBuffer            temporary storage buffer allocated by the user.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA,
 *              \p bsrSortedValA_valM, \p bsrSortedRowPtrA or \p bsrSortedColIndA pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 * 
 *  \par Example
 *  \snippet example_hipsparse_bsrilu02.cpp doc example
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSRILU0_H */
