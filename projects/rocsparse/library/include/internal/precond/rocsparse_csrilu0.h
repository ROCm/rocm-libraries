/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_CSRILU0_H
#define ROCSPARSE_CSRILU0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrilu0_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()"
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same index
*  base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csrilu0_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle   handle,
                                              rocsparse_mat_info info,
                                              rocsparse_int*     position);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrilu0_set_tolerance() sets the numerical tolerance for detecting a
*  near numerical zero entry during \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()"
*  computation. The first singular pivot \f$j\f$ at \f$|A_{j,j}| \leq \text{tolerance}\f$.
*
*
*  \note \p rocsparse_csrilu0_set_tolerance() is a blocking function. It might influence
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
*  tolerance    tolerance value to determine singular pivot \f$|A_{j,j}| \leq \text{tolerance}\f$,
*               where variable tolerance is in host memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_set_tolerance(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 double             tolerance);

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrilu0_get_tolerance() returns the numerical tolerance for detecing
*  a near numerical zero entry during \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()"
*  computation. The first singular pivot \f$j\f$ at \f$|A_{j,j}| \leq \text{tolerance}\f$.
*
*  \note \p rocsparse_csrilu0_get_tolerance() is a blocking function. It might influence
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
*  tolerance   obtain tolerance value to determine singular pivot \f$|A_{j,j}| \leq \text{tolerance}\f$,
*              where variable tolerance is in host memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or tolerance pointer is invalid..
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_get_tolerance(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 double*            tolerance);
/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrilu0_singular_pivot() returns the position of a
*  near numerical zero entry that has been found during \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()"
*  computation. The first singular pivot \f$j\f$ at \f$|A_{j,j}| \leq \text{tolerance}\f$  is stored
*  in \p position, using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no singular pivot has been found,
*  \p position is set to -1.
*
*  \note \p rocsparse_csrilu0_singular_pivot() is a blocking function. It might influence
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
*  position    pointer to singular pivot \f$j\f$, can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_singular_pivot(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  rocsparse_int*     position);

/*! \ingroup precond_module
 *  \details
 *  \p rocsparse_csrilu0_numeric_boost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  info            structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost    enable/disable numeric boost.
 *  @param[in]
 *  boost_tol       tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val       boost value to replace a numerical value.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const float*       boost_tol,
                                                  const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const double*      boost_tol,
                                                  const double*      boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_numeric_boost(rocsparse_handle               handle,
                                                  rocsparse_mat_info             info,
                                                  int                            enable_boost,
                                                  const float*                   boost_tol,
                                                  const rocsparse_float_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_numeric_boost(rocsparse_handle                handle,
                                                  rocsparse_mat_info              info,
                                                  int                             enable_boost,
                                                  const double*                   boost_tol,
                                                  const rocsparse_double_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dscsrilu0_numeric_boost(rocsparse_handle   handle,
                                                   rocsparse_mat_info info,
                                                   int                enable_boost,
                                                   const double*      boost_tol,
                                                   const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dccsrilu0_numeric_boost(rocsparse_handle               handle,
                                                   rocsparse_mat_info             info,
                                                   int                            enable_boost,
                                                   const double*                  boost_tol,
                                                   const rocsparse_float_complex* boost_val);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_csrilu0_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()" and
*  \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()". The temporary storage buffer must be allocated
*  by the user. The size of the temporary storage buffer is identical to the size returned by
*  \ref rocsparse_scsrsv_buffer_size "rocsparse_Xcsrsv_buffer_size()" if the matrix sparsity pattern
*  is identical. The user allocated buffer can thus be shared between subsequent calls to those functions.
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
*              \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()" and
*              \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()".
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
rocsparse_status rocsparse_scsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const float*              csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const double*             csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_buffer_size(rocsparse_handle               handle,
                                                rocsparse_int                  m,
                                                rocsparse_int                  nnz,
                                                const rocsparse_mat_descr      descr,
                                                const rocsparse_float_complex* csr_val,
                                                const rocsparse_int*           csr_row_ptr,
                                                const rocsparse_int*           csr_col_ind,
                                                rocsparse_mat_info             info,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_buffer_size(rocsparse_handle                handle,
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
*  \p rocsparse_csrilu0_analysis performs the analysis step for \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()".
*  It is expected that this function will be executed only once for a given matrix and particular
*  operation type. The analysis meta data can be cleared by \ref rocsparse_csrilu0_clear().
*
*  \p rocsparse_csrilu0_analysis can share its meta data with
*  \ref rocsparse_scsric0_analysis "rocsparse_Xcsric0_analysis()",
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
rocsparse_status rocsparse_scsrilu0_analysis(rocsparse_handle          handle,
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
rocsparse_status rocsparse_dcsrilu0_analysis(rocsparse_handle          handle,
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
rocsparse_status rocsparse_ccsrilu0_analysis(rocsparse_handle               handle,
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
rocsparse_status rocsparse_zcsrilu0_analysis(rocsparse_handle                handle,
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
*  \p rocsparse_csrilu0_clear deallocates all memory that was allocated by
*  \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()". This is especially
*  useful, if memory is an issue and the analysis data is not required for further
*  computation.
*
*  \note
*  Calling \p rocsparse_csrilu0_clear is optional. All allocated resources will be
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
rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csrilu0 computes the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LU
*  \f]
*  where the lower triangular matrix \f$L\f$ and the upper triangular matrix \f$U\f$ are computed using:
*  \f[
*    \begin{array}{ll}
*        L_{ij} = \frac{1}{U_{jj}}(A_{ij} - \sum_{k=0}^{j-1}L_{ik} \times U_{kj}), & \text{if i > j} \\
*        U_{ij} = (A_{ij} - \sum_{k=0}^{j-1}L_{ik} \times U_{kj}), & \text{if i <= j}
*    \end{array}
*  \f]
*  for each entry found in the CSR matrix \f$A\f$.
*
*  Computing the above incomplete \f$LU\f$ factorization requires three steps to complete. First,
*  the user determines the size of the required temporary storage buffer by calling
*  \ref rocsparse_scsrilu0_buffer_size "rocsparse_Xcsrilu0_buffer_size()". Once this buffer size has been determined,
*  the user allocates the buffer and passes it to \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()".
*  This will perform analysis on the sparsity pattern of the matrix. Finally, the user calls \p rocsparse_scsrilu0,
*  \p rocsparse_dcsrilu0, \p rocsparse_ccsrilu0, or \p rocsparse_zcsrilu0 to perform the actual factorization. The calculation
*  of the buffer size and the analysis of the sparse matrix only need to be performed once for a given sparsity pattern
*  while the factorization can be repeatedly applied to multiple matrices having the same sparsity pattern. Once all calls
*  to \ref rocsparse_scsrilu0 "rocsparse_Xcsrilu0()" are complete, the temporary buffer can be deallocated.
*
*  When computing the \f$LU\f$ factorization, it is possible that \f$U_{jj} == 0\f$ which would result in a division by zero.
*  This could occur from either \f$A_{jj}\f$ not existing in the sparse CSR matrix (referred to as a structural zero) or because
*  \f$A_{ij} - \sum_{k=0}^{j-1}L_{ik} \times U_{kj} == 0\f$ (referred to as a numerical zero). For example, running the
*  \f$LU\f$ factorization on the following matrix:
*  \f[
*    \begin{bmatrix}
*    2 & 1 & 0 \\
*    1 & 2 & 1 \\
*    0 & 1 & 2
*    \end{bmatrix}
*  \f]
*  results in a successful \f$LU\f$ factorization, however running with the matrix:
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
*        U_{00} &= 2 \\
*        U_{01} &= 1 \\
*        L_{10} &= \frac{1}{2} \\
*        U_{11} &= \frac{1}{2} - \frac{1}{2}
*               &= 0
*    \end{array}
*  \f]
*  The user can detect the presence of a structural zero by calling \ref rocsparse_csrilu0_zero_pivot() after
*  \ref rocsparse_scsrilu0_analysis "rocsparse_Xcsrilu0_analysis()" and/or the presence of a structural or
*  numerical zero by calling \ref rocsparse_csrilu0_zero_pivot() after \ref rocsparse_scsrilu0 "rocsparse_Xcsric0()".
*  In both cases, \ref rocsparse_csrilu0_zero_pivot() will report the first zero pivot (either numerical or structural)
*  found. See example below. The user can also set the diagonal type to be \f$1\f$ using \ref rocsparse_set_mat_diag_type()
*  which will interpret the matrix \f$A\f$ as having ones on its diagonal (even if no nonzero exists in the sparsity pattern).
*
*  \p rocsparse_csrilu0 computes the \f$LU\f$ factorization inplace meaning that the values array \p csr_val of the \f$A\f$
*  matrix is overwritten with the \f$L\f$ matrix stored in the strictly lower triangular part of \f$A\f$ and the \f$U\f$ matrix
*  stored in the upper part of \f$A\f$:
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
*    u_{00} & u_{01} & u_{02} \\
*    l_{10} & u_{11} & u_{12} \\
*    l_{20} & l_{21} & u_{22}
*    \end{bmatrix}
*    \end{align}
*  \f]
*  The row pointer array \p csr_row_ptr and the column indices array \p csr_col_ind remain the same for \f$A\f$ and \f$LU\f$ as
*  the incomplete factorization does not generate new nonzeros in \f$LU\f$ which do not already exist in \f$A\f$.
*
*  The performance of computing \f$LU\f$ factorization with rocSPARSE greatly depends on the sparisty pattern
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
*  storage format. The following example computes the incomplete LU factorization
*  \f$M \approx LU\f$ and solves the preconditioned system \f$My = x\f$.
*  \snippet example_rocsparse_csrilu0.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0(rocsparse_handle          handle,
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
rocsparse_status rocsparse_dcsrilu0(rocsparse_handle          handle,
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
rocsparse_status rocsparse_ccsrilu0(rocsparse_handle          handle,
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
rocsparse_status rocsparse_zcsrilu0(rocsparse_handle          handle,
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

#endif /* ROCSPARSE_CSRILU0_H */
