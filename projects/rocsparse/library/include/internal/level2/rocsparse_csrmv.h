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

#ifndef ROCSPARSE_CSRMV_H
#define ROCSPARSE_CSRMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \details
*  \p rocsparse_csrmv_analysis performs the analysis step for \ref rocsparse_scsrmv "rocsparse_Xcsrmv()".
*  It is expected that this function will be executed only once for a given sparsity pattern and particular operation
*  type. The gathered analysis meta data is stored in the \ref rocsparse_mat_info object and can be cleared by
*  \ref rocsparse_csrmv_clear().
*
*  If the matrix sparsity pattern changes, the gathered information will become invalid. In order to perform another
*  sparse matrix multiplication with a matrix having a different sparsity pattern, the user would need to either destroy
*  the old \p info object and create a new one or the user would need to clear the existing info object using
*  \ref rocsparse_csrmv_clear(). In both cases, the analysis will need to be called again.
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
*  n           number of columns of the sparse CSR matrix.
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
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind or \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer for the gathered information
*              could not be allocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented if \ref rocsparse_matrix_type is not one of
*              \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric, or
*              \ref rocsparse_matrix_type_triangular.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const float*              csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const double*             csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmv_analysis(rocsparse_handle               handle,
                                           rocsparse_operation            trans,
                                           rocsparse_int                  m,
                                           rocsparse_int                  n,
                                           rocsparse_int                  nnz,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* csr_val,
                                           const rocsparse_int*           csr_row_ptr,
                                           const rocsparse_int*           csr_col_ind,
                                           rocsparse_mat_info             info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmv_analysis(rocsparse_handle                handle,
                                           rocsparse_operation             trans,
                                           rocsparse_int                   m,
                                           rocsparse_int                   n,
                                           rocsparse_int                   nnz,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* csr_val,
                                           const rocsparse_int*            csr_row_ptr,
                                           const rocsparse_int*            csr_col_ind,
                                           rocsparse_mat_info              info);
/**@}*/

/*! \ingroup level2_module
*  \details
*  \p rocsparse_csrmv_clear deallocates all memory that was allocated by
*  \ref rocsparse_scsrmv_analysis "rocsparse_Xcsrmv_analysis()". This is especially useful,
*  if memory is an issue and the analysis data is not required anymore for further
*  computation, e.g. when switching to another sparse matrix format.
*
*  \note
*  Calling \p rocsparse_csrmv_clear is optional. All allocated resources will be
*  cleared, when the opaque \ref rocsparse_mat_info object is destroyed using
*  \ref rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[inout]
*  info        structure that holds the information collected during analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer for the gathered information
*              could not be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
* */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using CSR storage format
*
*  \details
*  \p rocsparse_csrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
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
*  The \p info parameter is optional and contains information collected by
*  \ref rocsparse_scsrmv_analysis "rocsparse_Xcsrmv_analysis()". If present, the
*  information will be used to speed up the \p csrmv computation. If
*  \p info == \p NULL, a general \p csrmv routine will be used instead. Running with
*  analysis may result in better performance when computing the matrix vector product
*  but will also incur a performance cost attributed to the additional analysis step.
*  For this reason, running with analysis makes sense when a user plans on computing
*  the matrix vector product many times and therefore can amortize the analysis cost.
*
*  \code{.c}
*      for(i = 0; i < m; ++i)
*      {
*          y[i] = beta * y[i];
*
*          for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
*          {
*              y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
*          }
*      }
*  \endcode
*
*  To run without analysis, performing the above operation involves simply calling the \p rocsparse_csrmv routine while passing
*  \p NULL for the \p info parameter.
*
*  To run with analysis, completing the sparse matrix vector multiplication involves two steps. First,
*  the user creates a \ref rocsparse_mat_info object by calling \ref rocsparse_create_mat_info and then passes this to
*  \ref rocsparse_scsrmv_analysis "rocsparse_Xcsrmv_analysis()" which will perform analysis on the sparsity pattern of the
*  matrix \f$op(A)\f$. The user then completes the operation by calling \p rocsparse_csrmv. The creation of the \p info object
*  and the call to the analysis routine only need to be performed once for a given sparsity pattern while the computation
*  can be performed repeatedly as long as the sparsity pattern has not changed. Once all calls to \p rocsparse_csrmv have
*  been made, the \p info object can be destroyed with a call to \ref rocsparse_destroy_mat_info.
*
*  When running with analysis, a user may find themselves in the situation where they wish to perform multiple sparse matrix
*  multiplications with each sparse matrix having a different sparsity pattern. Instead of creating and destroying multiple
*  \ref rocsparse_mat_info objects for each unique sparsity pattern, the user can instead create the \p info object once and
*  then call \ref rocsparse_csrmv_clear followed by re-running the analysis in between each sparse matrix multiplication.
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
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  info        information collected by \ref rocsparse_scsrmv_analysis "rocsparse_Xcsrmv_analysis()",
*              can be \p NULL if no information is available.
*  @param[in]
*  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
*              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
*              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p x, \p beta or \p y pointer is
*              invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example performs a sparse matrix vector multiplication in CSR format
*  using additional meta data to improve performance.
*  \snippet example_rocsparse_csrmv.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int*           csr_row_ptr,
                                  const rocsparse_int*           csr_col_ind,
                                  rocsparse_mat_info             info,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int*            csr_row_ptr,
                                  const rocsparse_int*            csr_col_ind,
                                  rocsparse_mat_info              info,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRMV_H */
