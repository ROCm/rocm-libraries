/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_CSRSV_H
#define ROCSPARSE_CSRSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \details
*  \p rocsparse_csrsv_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during \ref rocsparse_scsrsv_solve "rocsparse_Xcsrsv_solve()"
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csrsv_zero_pivot is a blocking function. It might influence
*  performance negatively.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
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
rocsparse_status rocsparse_csrsv_zero_pivot(rocsparse_handle          handle,
                                            const rocsparse_mat_descr descr,
                                            rocsparse_mat_info        info,
                                            rocsparse_int*            position);

/*! \ingroup level2_module
*  \details
*  \p rocsparse_csrsv_buffer_size returns the size of the temporary storage buffer that
*  is required by \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()" and
*  \ref rocsparse_scsrsv_solve "rocsparse_Xcsrsv_solve()". The temporary storage buffer
*  must be allocated by the user. The size of the temporary storage buffer is identical
*  to the size returned by \ref rocsparse_scsrilu0_buffer_size "rocsparse_Xcsrilu0_buffer_size()"
*  if the matrix sparsity pattern is identical. The user allocated buffer can thus be shared
*  between subsequent calls to those functions.
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
*  trans       matrix operation type.
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
*              \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()" and
*              \ref rocsparse_scsrsv_solve "rocsparse_Xcsrsv_solve()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p info or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsv_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsv_buffer_size(rocsparse_handle               handle,
                                              rocsparse_operation            trans,
                                              rocsparse_int                  m,
                                              rocsparse_int                  nnz,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* csr_val,
                                              const rocsparse_int*           csr_row_ptr,
                                              const rocsparse_int*           csr_col_ind,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsv_buffer_size(rocsparse_handle                handle,
                                              rocsparse_operation             trans,
                                              rocsparse_int                   m,
                                              rocsparse_int                   nnz,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* csr_val,
                                              const rocsparse_int*            csr_row_ptr,
                                              const rocsparse_int*            csr_col_ind,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size);

/**@}*/
/*! \ingroup level2_module
*  \details
*  \p rocsparse_csrsv_analysis performs the analysis step for
*  \ref rocsparse_scsrsv_solve "rocsparse_Xcsrsv_solve()". It is expected that this
*  function will be executed only once for a given matrix and particular operation
*  type. The analysis meta data can be cleared by \ref rocsparse_csrsv_clear().
*
*  If the matrix sparsity pattern changes, the gathered information will become invalid. In order to perform another
*  sparse triangular solve with a matrix having a different sparsity pattern, the user would need to either destroy
*  the old \p info object and create a new one or the user would need to clear the existing \p info object using
*  \ref rocsparse_csrsv_clear(). In both cases, the analysis will need to be called again.
*
*  \p rocsparse_csrsv_analysis can share its meta data with
*  \ref rocsparse_scsrsm_analysis "rocsparse_Xcsrsm_analysis()",
*  \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()" and
*  \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()". Selecting
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
*  trans       matrix operation type.
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
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_row_ptr,
*              \p csr_col_ind, \p info or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
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
rocsparse_status rocsparse_dcsrsv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
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
rocsparse_status rocsparse_ccsrsv_analysis(rocsparse_handle               handle,
                                           rocsparse_operation            trans,
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
rocsparse_status rocsparse_zcsrsv_analysis(rocsparse_handle                handle,
                                           rocsparse_operation             trans,
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

/*! \ingroup level2_module
*  \details
*  \p rocsparse_csrsv_clear deallocates all memory that was allocated by
*  \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()". This is especially useful,
*  if memory is an issue and the analysis data is not required for further computation, e.g.
*  when switching to another sparse matrix format.
*
*  Calling \p rocsparse_csrsv_clear is optional. All allocated resources will be cleared when the
*  opaque \ref rocsparse_mat_info struct is destroyed using \ref rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
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
rocsparse_status rocsparse_csrsv_clear(rocsparse_handle          handle,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_mat_info        info);

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p rocsparse_csrsv_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  Performing the above operation requires three steps. First, the user calls
*  \ref rocsparse_scsrsv_buffer_size "rocsparse_Xcsrsv_buffer_size()" which will determine the size of the required
*  temporary storage buffer. The user then allocates this buffer and calls
*  \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()" which will perform analysis on the sparse matrix
*  \f$op(A)\f$. Finally, the user completes the computation by calling \p rocsparse_csrsv_solve. The buffer size,
*  buffer allocation, and analysis only need to be called once for a given sparse matrix \f$op(A)\f$ while the
*  computation stage can be repeatedly used with different \f$x\f$ and \f$y\f$ vectors. Once all calls to
*  \p rocsparse_csrsv_solve are complete, the temporary buffer can be deallocated.
*
*  Solving a triangular system involves division by the diagonal elements. This means that if the sparse matrix is
*  missing the diagonal entry (referred to as a structural zero) or the diagonal entry is zero (referred to as a numerical zero)
*  then a division by zero would occur. \p rocsparse_csrsv_solve tracks the location of the first zero pivot (either numerical
*  or structural zero). The zero pivot status can be checked calling \ref rocsparse_csrsv_zero_pivot(). If
*  \ref rocsparse_csrsv_zero_pivot() returns \ref rocsparse_status_success, then no zero pivot was found and therefore
*  the matrix does not have a structural or numerical zero.
*
*  The user can specify that the sparse matrix should be interpreted as having ones on the diagonal by setting the diagonal type
*  on the descriptor \p descr to \ref rocsparse_diag_type_unit using \ref rocsparse_set_mat_diag_type. If
*  \ref rocsparse_diag_type == \ref rocsparse_diag_type_unit, no zero pivot will be reported, even if \f$A_{j,j} = 0\f$ for
*  some \f$j\f$.
*
*  The sparse CSR matrix passed to \p rocsparse_csrsv_solve does not actually have to be a triangular matrix. Instead the
*  triangular upper or lower part of the sparse matrix is solved based on \ref rocsparse_fill_mode set on the descriptor
*  \p descr. If the fill mode is set to \ref rocsparse_fill_mode_lower, then the lower triangular matrix is solved. If the
*  fill mode is set to \ref rocsparse_fill_mode_upper then the upper triangular matrix is solved.
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
*  Currently, only \p trans == \ref rocsparse_operation_none and
*  \p trans == \ref rocsparse_operation_transpose is supported.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[in]
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
*  x           array of \p m elements, holding the right-hand side.
*  @param[out]
*  y           array of \p m elements, holding the solution.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p x or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the lower triangular \f$m \times m\f$ matrix \f$L\f$, stored in CSR
*  storage format with unit diagonal. The following example solves \f$L \cdot y = x\f$.
*  \snippet example_rocsparse_csrsv.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        rocsparse_int             m,
                                        rocsparse_int             nnz,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        rocsparse_mat_info        info,
                                        const float*              x,
                                        float*                    y,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsv_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        rocsparse_int             m,
                                        rocsparse_int             nnz,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        rocsparse_mat_info        info,
                                        const double*             x,
                                        double*                   y,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsv_solve(rocsparse_handle               handle,
                                        rocsparse_operation            trans,
                                        rocsparse_int                  m,
                                        rocsparse_int                  nnz,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* csr_val,
                                        const rocsparse_int*           csr_row_ptr,
                                        const rocsparse_int*           csr_col_ind,
                                        rocsparse_mat_info             info,
                                        const rocsparse_float_complex* x,
                                        rocsparse_float_complex*       y,
                                        rocsparse_solve_policy         policy,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsv_solve(rocsparse_handle                handle,
                                        rocsparse_operation             trans,
                                        rocsparse_int                   m,
                                        rocsparse_int                   nnz,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* csr_val,
                                        const rocsparse_int*            csr_row_ptr,
                                        const rocsparse_int*            csr_col_ind,
                                        rocsparse_mat_info              info,
                                        const rocsparse_double_complex* x,
                                        rocsparse_double_complex*       y,
                                        rocsparse_solve_policy          policy,
                                        void*                           temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRSV_H */
