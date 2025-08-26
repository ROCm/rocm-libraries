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

#ifndef ROCSPARSE_COOMV_H
#define ROCSPARSE_COOMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using COO storage format
*
*  \details
*  \p rocsparse_coomv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in COO storage format, and the dense vector \f$x\f$ and adds the
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
*  The COO matrix has to be sorted by row indices. This can be achieved by using
*  rocsparse_coosort_by_row().
*
*  \code{.c}
*      for(i = 0; i < m; ++i)
*      {
*          y[i] = beta * y[i];
*      }
*
*      for(i = 0; i < nnz; ++i)
*      {
*          y[coo_row_ind[i]] += alpha * coo_val[i] * x[coo_col_ind[i]];
*      }
*  \endcode
*
*  \note
*  This function does not produce deterministic results when A is transposed.
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
*  m           number of rows of the sparse COO matrix.
*  @param[in]
*  n           number of columns of the sparse COO matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse COO matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse COO matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  coo_val     array of \p nnz elements of the sparse COO matrix.
*  @param[in]
*  coo_row_ind array of \p nnz elements containing the row indices of the sparse COO
*              matrix.
*  @param[in]
*  coo_col_ind array of \p nnz elements containing the column indices of the sparse
*              COO matrix.
*  @param[in]
*  x           array of \p n elements (\f$op(A) = A\f$) or \p m elements
*              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p m elements (\f$op(A) = A\f$) or \p n elements
*              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p coo_val,
*              \p coo_row_ind, \p coo_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example performs a sparse matrix vector multiplication in COO format.
*  \snippet example_rocsparse_coomv.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scoomv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              coo_val,
                                  const rocsparse_int*      coo_row_ind,
                                  const rocsparse_int*      coo_col_ind,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcoomv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             coo_val,
                                  const rocsparse_int*      coo_row_ind,
                                  const rocsparse_int*      coo_col_ind,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccoomv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* coo_val,
                                  const rocsparse_int*           coo_row_ind,
                                  const rocsparse_int*           coo_col_ind,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcoomv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* coo_val,
                                  const rocsparse_int*            coo_row_ind,
                                  const rocsparse_int*            coo_col_ind,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_COOMV_H */
