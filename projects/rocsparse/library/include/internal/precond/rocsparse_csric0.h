/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_CSRIC0_H
#define ROCSPARSE_CSRIC0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during \ref rocsparse_scsric0 "rocsparse_Xcsric0()"
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using
*  same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csric0_zero_pivot is a blocking function. It might influence
*  performance negatively.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_zero_pivot zero pivot has been found.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csric0_zero_pivot(rocsparse_handle   handle,
                                             rocsparse_mat_info info,
                                             rocsparse_int*     position);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric0_singular_pivot() returns the position of a
*  numerical singular pivot (where \f$|L_{j,j}| \leq \text{tolerance}\f$)
*  that has been found during \ref rocsparse_scsric0 "rocsparse_Xcsric0()" computation.
*  The first singular pivot \f$j\f$ at \f$L_{j,j}\f$ is stored in \p position, using
*  same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no singular pivot has been found,
*  \p position is set to -1.
*
*  \note \p rocsparse_csric0_singular_pivot() is a blocking function. It might influence
*  performance negatively.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to singular pivot \f$k\f$, can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csric0_singular_pivot(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 rocsparse_int*     position);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric0_set_tolerance()  sets the numerical tolerance for detecting a
*  numerical singular pivot (where \f$|L_{j,j}|  \leq \text{tolerance}\f$)
*  that might be found during \ref rocsparse_scsric0 "rocsparse_Xcsric0()" computation.
*
*
*  \note \p rocsparse_csric0_set_tolerance() is a blocking function. It might influence
*  performance negatively.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  tolerance    tolerance for detecting singular pivot (\f$|L_{j,j}|  \leq \text{tolerance}\f$)
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer if \p info tolerance pointer is
*              invalid
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csric0_set_tolerance(rocsparse_handle   handle,
                                                rocsparse_mat_info info,
                                                double             tolerance);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric0_get_tolerance() returns the numerical tolerance for detecting a
*  numerical singular pivot (where \f$|L_{j,j}|  \leq \text{tolerance}\f$)
*  that might be found during \ref rocsparse_scsric0 "rocsparse_Xcsric0()" computation.
*
*
*  \note \p rocsparse_csric0_get_tolerance() is a blocking function. It might influence
*  performance negatively.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  tolerance    obtain tolerance for detecting singular pivot (\f$|L_{j,j}|  \leq \text{tolerance}\f$)
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer if \p info or \p tolerance pointer is
*              invalid
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csric0_get_tolerance(rocsparse_handle   handle,
                                                rocsparse_mat_info info,
                                                double*            tolerance);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric0_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()".
*  The temporary storage buffer must be allocated by the user. The size of the temporary
*  storage buffer is identical to the size returned by
*  \ref rocsparse_scsrsv_buffer_size "rocsparse_Xcsrsv_buffer_size()" and
*  \ref rocsparse_scsrilu0_buffer_size "rocsparse_Xcsrilu0_buffer_size()" if the matrix
*  sparsity pattern is identical. The user allocated buffer can thus be shared between
*  subsequent calls to those functions.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()" and
*              \ref rocsparse_scsric0 "rocsparse_Xcsric0()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p info or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsric0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsric0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsric0_buffer_size(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  nnz,
                                               const rocsparse_mat_descr      descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               rocsparse_mat_info             info,
                                               size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsric0_buffer_size(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   nnz,
                                               const rocsparse_mat_descr       descr,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               rocsparse_mat_info              info,
                                               size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric0_analysis performs the analysis step for
*  \ref rocsparse_scsric0 "rocsparse_Xcsric0()". It is expected that this function will be
*  executed only once for a given matrix and particular operation type. The analysis meta
*  data can be cleared by \ref rocsparse_csric0_clear().
*
*  \p rocsparse_csric0_analysis can share its meta data with
*  \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()",
*  \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()", and
*  \ref rocsparse_scsrsm_analysis "rocsparse_Xcsrsm_analysis()". Selecting
*  \ref rocsparse_analysis_policy_reuse policy can greatly improve computation
*  performance of meta data. However, the user needs to make sure that the sparsity
*  pattern remains unchanged. If this cannot be assured,
*  \ref rocsparse_analysis_policy_force has to be used.
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[out]
*  info        structure that holds the information collected during
*              the analysis step.
*  @param[in]
*  analysis    \ref rocsparse_analysis_policy_reuse or
*              \ref rocsparse_analysis_policy_force.
*  @param[in]
*  solve       \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p info or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsric0_analysis(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             nnz,
                                            const rocsparse_mat_descr descr,
                                            const float*              csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsric0_analysis(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             nnz,
                                            const rocsparse_mat_descr descr,
                                            const double*             csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsric0_analysis(rocsparse_handle               handle,
                                            rocsparse_int                  m,
                                            rocsparse_int                  nnz,
                                            const rocsparse_mat_descr      descr,
                                            const rocsparse_float_complex* csr_val,
                                            const rocsparse_int*           csr_row_ptr,
                                            const rocsparse_int*           csr_col_ind,
                                            rocsparse_mat_info             info,
                                            rocsparse_analysis_policy      analysis,
                                            rocsparse_solve_policy         solve,
                                            void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsric0_analysis(rocsparse_handle                handle,
                                            rocsparse_int                   m,
                                            rocsparse_int                   nnz,
                                            const rocsparse_mat_descr       descr,
                                            const rocsparse_double_complex* csr_val,
                                            const rocsparse_int*            csr_row_ptr,
                                            const rocsparse_int*            csr_col_ind,
                                            rocsparse_mat_info              info,
                                            rocsparse_analysis_policy       analysis,
                                            rocsparse_solve_policy          solve,
                                            void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csric0_clear deallocates all memory that was allocated by
*  \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()". This is especially
*  useful, if memory is an issue and the analysis data is not required for further
*  computation.
*
*  \note
*  Calling \p rocsparse_csric0_clear is optional. All allocated resources will be
*  cleared, when the opaque \ref rocsparse_mat_info struct is destroyed using
*  \ref rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[inout]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer holding the meta data could not
*              be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csric0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csric0 computes the incomplete Cholesky factorization with 0 fill-ins
*  and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LL^T
*  \f]
*  where the lower triangular matrix \f$L\f$ is computed using:
*  \f[
*    L_{ij} = \left\{
*    \begin{array}{ll}
*        \sqrt{A_{jj} - \sum_{k=0}^{j-1}(L_{jk})^{2}},   & \text{if i == j} \\
*        \frac{1}{L_{jj}}(A_{jj} - \sum_{k=0}^{j-1}L_{ik} \times L_{jk}), & \text{if i > j}
*    \end{array}
*    \right.
*  \f]
*  for each entry found in the CSR matrix \f$A\f$.
*
*  Computing the above incomplete Cholesky factorization requires three steps to complete. First,
*  the user determines the size of the required temporary storage buffer by calling
*  \ref rocsparse_scsric0_buffer_size "rocsparse_Xcsric0_buffer_size()". Once this buffer size has been determined,
*  the user allocates the buffer and passes it to \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()".
*  This will perform analysis on the sparsity pattern of the matrix. Finally, the user calls \p rocsparse_scsric0,
*  \p rocsparse_dcsric0, \p rocsparse_ccsric0, or \p rocsparse_zcsric0 to perform the actual factorization. The calculation
*  of the buffer size and the analysis of the sparse matrix only need to be performed once for a given sparsity pattern
*  while the factorization can be repeatedly applied to multiple matrices having the same sparsity pattern. Once all calls
*  to \ref rocsparse_scsric0 "rocsparse_Xcsric0()" are complete, the temporary buffer can be deallocated.
*
*  When computing the Cholesky factorization, it is possible that \f$L_{jj} == 0\f$ which would result in a division by zero.
*  This could occur from either \f$A_{jj}\f$ not existing in the sparse CSR matrix (referred to as a structural zero) or because
*  \f$A_{jj} - \sum_{k=0}^{j-1}(L_{jk})^{2} == 0\f$ (referred to as a numerical zero). For example, running the Cholesky
*  factorization on the following matrix:
*  \f[
*    \begin{bmatrix}
*    2 & 1 & 0 \\
*    1 & 2 & 1 \\
*    0 & 1 & 2
*    \end{bmatrix}
*  \f]
*  results in a successful Cholesky factorization, however running with the matrix:
*  \f[
*    \begin{bmatrix}
*    2 & 1 & 0 \\
*    1 & 1/2 & 1 \\
*    0 & 1 & 2
*    \end{bmatrix}
*  \f]
*  results in a numerical zero because:
*  \f[
*    \begin{array}{ll}
*        L_{00} &= \sqrt{2} \\
*        L_{10} &= \frac{1}{\sqrt{2}} \\
*        L_{11} &= \sqrt{\frac{1}{2} - (\frac{1}{\sqrt{2}})^2}
*               &= 0
*    \end{array}
*  \f]
*  The user can detect the presence of a structural zero by calling \ref rocsparse_csric0_zero_pivot() after
*  \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()" and/or the presence of a structural or
*  numerical zero by calling \ref rocsparse_csric0_zero_pivot() after \ref rocsparse_scsric0 "rocsparse_Xcsric0()":
*  \code{.c}
*  rocsparse_dcsric0(handle,
*                  m,
*                  nnz,
*                  descr_M,
*                  csr_val,
*                  csr_row_ptr,
*                  csr_col_ind,
*                  info,
*                  rocsparse_solve_policy_auto,
*                  temp_buffer);
*
*  // Check for zero pivot
*  if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle,
*                                                                info,
*                                                                &position))
*  {
*      printf("L has structural and/or numerical zero at L(%d,%d)\n", position, position);
*  }
*  \endcode
*  In both cases, \ref rocsparse_csric0_zero_pivot() will report the first zero pivot (either numerical or structural)
*  found. See full example below. The user can also set the diagonal type to be \f$1\f$ using \ref rocsparse_set_mat_diag_type()
*  which will interpret the matrix \f$A\f$ as having ones on its diagonal (even if no nonzero exists in the sparsity pattern).
*
*  \p rocsparse_csric0 computes the Cholesky factorization inplace meaning that the values array \p csr_val of the \f$A\f$
*  matrix is overwritten with the \f$L\f$ matrix stored in the lower triangular part of \f$A\f$:
*
*  \f[
*    \begin{align}
*    \begin{bmatrix}
*    a_{00} & a_{01} & a_{02} \\
*    a_{10} & a_{11} & a_{12} \\
*    a_{20} & a_{21} & a_{22}
*    \end{bmatrix}
*    \rightarrow
*    \begin{bmatrix}
*    l_{00} & a_{01} & a_{02} \\
*    l_{10} & l_{11} & a_{12} \\
*    l_{20} & l_{21} & l_{22}
*    \end{bmatrix}
*    \end{align}
*  \f]
*  The row pointer array \p csr_row_ptr and the column indices array \p csr_col_ind remain the same for \f$A\f$ and the output as
*  the incomplete factorization does not generate new nonzeros in the output which do not already exist in \f$A\f$.
*
*  The performance of computing Cholesky factorization with rocSPARSE greatly depends on the sparisty pattern
*  the the matrix \f$A\f$ as this is what determines the amount of parallelism available.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[inout]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the sparse \f$m \times m\f$ matrix \f$A\f$, stored in CSR
*  storage format. The following example computes the incomplete Cholesky factorization
*  \f$M \approx LL^T\f$ and solves the preconditioned system \f$My = x\f$.
*  \snippet example_rocsparse_csric0.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsric0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   float*                    csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsric0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   double*                   csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsric0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_float_complex*  csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsric0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_double_complex* csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRIC0_H */
