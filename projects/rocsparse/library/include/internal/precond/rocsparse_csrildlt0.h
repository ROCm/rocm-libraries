/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_CSRILDLT0_H
#define ROCSPARSE_CSRILDLT0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrildlt0_numeric_boost enables/disables numeric boost for the diagonal
*  entries of \f$D\f$ computed during \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()".
*
*  If \p enable_boost is non-zero, any diagonal entry \f$|D_i| \leq \text{boost\_tol}\f$ is
*  replaced by \p boost_val before it is stored and used to update later entries.
*  This prevents near-zero or negative diagonal values from degrading the factorization.
*
*  Call this function after \ref rocsparse_scsrildlt0_analysis "rocsparse_Xcsrildlt0_analysis()" and before
*  \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()" to activate the boost; call it with \p enable_boost = 0
*  to deactivate.
*
*  \note \p boost_tol and \p boost_val may reside in host or device memory depending
*        on the pointer mode set via \ref rocsparse_set_pointer_mode().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle        handle to the rocSPARSE library context queue.
*  @param[in]
*  info          structure that holds the information collected during the analysis step.
*  @param[in]
*  enable_boost  enable (non-zero) or disable (0) numeric boost.
*  @param[in]
*  boost_tol     threshold tolerance; entries with \f$|D_i| \leq \text{boost\_tol}\f$ are
*                boosted. For \p s and \p c variants this is \p float*; for \p d and \p z
*                variants this is \p double*.
*  @param[in]
*  boost_val     replacement value for small diagonal entries. Always real: \p float* for
*                \p s and \p c variants, \p double* for \p d and \p z variants.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info, \p boost_tol, or \p boost_val
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrildlt0_numeric_boost(rocsparse_handle   handle,
                                                    rocsparse_mat_info info,
                                                    int                enable_boost,
                                                    const float*       boost_tol,
                                                    const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrildlt0_numeric_boost(rocsparse_handle   handle,
                                                    rocsparse_mat_info info,
                                                    int                enable_boost,
                                                    const double*      boost_tol,
                                                    const double*      boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrildlt0_numeric_boost(rocsparse_handle   handle,
                                                    rocsparse_mat_info info,
                                                    int                enable_boost,
                                                    const float*       boost_tol,
                                                    const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrildlt0_numeric_boost(rocsparse_handle   handle,
                                                    rocsparse_mat_info info,
                                                    int                enable_boost,
                                                    const double*      boost_tol,
                                                    const double*      boost_val);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrildlt0_zero_pivot returns \ref rocsparse_status_zero_pivot if either a
*  structural or numerical zero has been found during \ref rocsparse_scsrildlt0
*  "rocsparse_Xcsrildlt0()" computation. The first zero pivot \f$j\f$ at \f$D_{j}\f$ is
*  stored in \p position, using the same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csrildlt0_zero_pivot is a blocking function. It might negatively
*  influence performance.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, which can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_zero_pivot zero pivot has been found.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrildlt0_zero_pivot(rocsparse_handle   handle,
                                               rocsparse_mat_info info,
                                               rocsparse_int*     position);

/*! \ingroup precond_module
*  \details
*  rocsparse_csrildlt0_singular_pivot() returns the position of a
*  numerical singular pivot (where \f$|D_{j}| \leq \text{tolerance}\f$)
*  that has been found during \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()" computation.
*  The first singular pivot \f$j\f$ at \f$D_{j}\f$ is stored in \p position, using the
*  same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no singular pivot has been found,
*  \p position is set to -1.
*
*  \note rocsparse_csrildlt0_singular_pivot() is a blocking function. It might negatively
*  influence performance.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to singular pivot \f$k\f$, which can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrildlt0_singular_pivot(rocsparse_handle   handle,
                                                   rocsparse_mat_info info,
                                                   rocsparse_int*     position);

/*! \ingroup precond_module
*  \details
*  rocsparse_csrildlt0_set_tolerance() sets the numerical tolerance for detecting a
*  numerical singular pivot (where \f$|D_{j}| \leq \text{tolerance}\f$)
*  that might be found during \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()" computation.
*
*  \note rocsparse_csrildlt0_set_tolerance() is a blocking function. It might negatively
*  influence performance.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  tolerance   tolerance for detecting singular pivot (\f$|D_{j}| \leq \text{tolerance}\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer if \p info pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrildlt0_set_tolerance(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  double             tolerance);

/*! \ingroup precond_module
*  \details
*  rocsparse_csrildlt0_get_tolerance() returns the numerical tolerance for detecting a
*  numerical singular pivot (where \f$|D_{j}| \leq \text{tolerance}\f$)
*  that might be found during \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()" computation.
*
*  \note rocsparse_csrildlt0_get_tolerance() is a blocking function. It might negatively
*  influence performance.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  tolerance   obtained tolerance for detecting singular pivot.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer if \p info or \p tolerance pointer is
*              invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrildlt0_get_tolerance(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  double*            tolerance);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrildlt0_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_scsrildlt0_analysis "rocsparse_Xcsrildlt0_analysis()".
*  The temporary storage buffer must be allocated by the user. The size of the temporary
*  storage buffer is identical to the size returned by
*  \ref rocsparse_scsrsv_buffer_size "rocsparse_Xcsrsv_buffer_size()" and
*  \ref rocsparse_scsric0_buffer_size "rocsparse_Xcsric0_buffer_size()" if the matrix
*  sparsity pattern is identical. The user-allocated buffer can therefore be shared between
*  subsequent calls to those functions.
*
*  \note
*  This function is non-blocking and executed asynchronously with respect to the host.
*  It can return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
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
*              \ref rocsparse_scsrildlt0_analysis "rocsparse_Xcsrildlt0_analysis()" and
*              \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p info, or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrildlt0_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrildlt0_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrildlt0_buffer_size(rocsparse_handle               handle,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  nnz,
                                                 const rocsparse_mat_descr      descr,
                                                 const rocsparse_float_complex* csr_val,
                                                 const rocsparse_int*           csr_row_ptr,
                                                 const rocsparse_int*           csr_col_ind,
                                                 rocsparse_mat_info             info,
                                                 size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrildlt0_buffer_size(rocsparse_handle                handle,
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
*  \p rocsparse_csrildlt0_analysis performs the analysis step for
*  \ref rocsparse_scsrildlt0 "rocsparse_Xcsrildlt0()". It is expected that this function will
*  be executed only once for a given matrix and particular operation type. The analysis
*  metadata can be cleared by \ref rocsparse_csrildlt0_clear().
*
*  \p rocsparse_csrildlt0_analysis can share its metadata with
*  \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()",
*  \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()",
*  \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()", and
*  \ref rocsparse_scsrsm_analysis "rocsparse_Xcsrsm_analysis()". Selecting
*  \ref rocsparse_analysis_policy_reuse policy can greatly improve the computation
*  performance of metadata. However, the user needs to ensure that the sparsity
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
*  handle      handle to the rocSPARSE library context queue.
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
*              \p csr_col_ind, \p info, or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrildlt0_analysis(rocsparse_handle          handle,
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
rocsparse_status rocsparse_dcsrildlt0_analysis(rocsparse_handle          handle,
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
rocsparse_status rocsparse_ccsrildlt0_analysis(rocsparse_handle               handle,
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
rocsparse_status rocsparse_zcsrildlt0_analysis(rocsparse_handle                handle,
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
*  \p rocsparse_csrildlt0_clear deallocates all memory that was allocated by
*  \ref rocsparse_scsrildlt0_analysis "rocsparse_Xcsrildlt0_analysis()". This is especially
*  useful if memory is an issue and the analysis data is not required for further
*  computation.
*
*  \note
*  Calling \p rocsparse_csrildlt0_clear is optional. All allocated resources will be
*  cleared when the opaque \ref rocsparse_mat_info struct is destroyed using
*  \ref rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[inout]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer holding the metadata could not
*              be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrildlt0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
*  \brief Incomplete \f$LDL^H\f$ factorization with 0 fill-ins and no pivoting using the
*  CSR storage format.
*
*  \details
*  \p rocsparse_csrildlt0 computes the incomplete \f$LDL^H\f$ factorization with 0 fill-ins
*  and no pivoting of a sparse \f$m \times m\f$ Hermitian (or symmetric for real types)
*  CSR matrix \f$A\f$, such that
*  \f[
*    A \approx L D L^H
*  \f]
*  where \f$L\f$ is unit lower triangular and \f$D\f$ is a real diagonal (since \f$A\f$ is
*  Hermitian, the diagonal of \f$D\f$ is always real).
*
*  The diagonal entries are computed as:
*  \f[
*    D_i = \mathrm{real}(A_{ii}) - \sum_{k<i} |L_{ik}|^2 D_k
*  \f]
*  and the off-diagonal entries as:
*  \f[
*    L_{ij} = \frac{1}{D_j} \left( A_{ij} - \sum_{k<j} L_{ik} D_k \overline{L_{jk}} \right)
*  \f]
*  for each entry found in the lower triangular part of the CSR matrix \f$A\f$.
*
*  For real types (\p s, \p d), \f$L^H = L^T\f$ and \f$D_i = A_{ii} - \sum_{k<i} L_{ik}^2 D_k\f$,
*  so the real and complex APIs are consistent.
*
*  The factorization is performed in-place: \p csr_val stores the strictly
*  lower-triangular entries of \f$L\f$ (the unit diagonal of \f$L\f$ is implicit),
*  and \p diag stores the diagonal entries of \f$D\f$.
*
*  The preconditioner \f$M = L D L^H\f$ is applied through:
*  - one lower-triangular SpSV solve with unit diagonal (\f$L^{-1}\f$),
*  - one diagonal scaling by \f$D^{-1}\f$,
*  - one conjugate-transpose triangular SpSV solve (\f$L^{-H}\f$).
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is non-blocking and executed asynchronously with respect to the host.
*  It can return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix.
*  @param[inout]
*  csr_val     array of \p nnz elements of the sparse CSR matrix. On output stores
*              the strictly lower-triangular entries of \f$L\f$ (unit diagonal implicit).
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[out]
*  diag        dense array of \p m real-valued elements storing the diagonal of \f$D\f$.
*              For \p s and \p d variants this is \p float* / \p double*; for \p c and \p z
*              variants this is also \p float* / \p double* since \f$D\f$ is real for
*              Hermitian matrices.
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
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p diag, or \p info pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrildlt0(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     float*                    csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     float*                    diag,
                                     rocsparse_mat_info        info,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrildlt0(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     double*                   csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     double*                   diag,
                                     rocsparse_mat_info        info,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrildlt0(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  nnz,
                                     const rocsparse_mat_descr      descr,
                                     rocsparse_float_complex*       csr_val,
                                     const rocsparse_int*           csr_row_ptr,
                                     const rocsparse_int*           csr_col_ind,
                                     float*                         diag,
                                     rocsparse_mat_info             info,
                                     rocsparse_solve_policy         policy,
                                     void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrildlt0(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   nnz,
                                     const rocsparse_mat_descr       descr,
                                     rocsparse_double_complex*       csr_val,
                                     const rocsparse_int*            csr_row_ptr,
                                     const rocsparse_int*            csr_col_ind,
                                     double*                         diag,
                                     rocsparse_mat_info              info,
                                     rocsparse_solve_policy          policy,
                                     void*                           temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRILDLT0_H */
