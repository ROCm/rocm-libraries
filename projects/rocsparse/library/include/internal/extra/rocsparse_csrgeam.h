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

#ifndef ROCSPARSE_CSRGEAM_H
#define ROCSPARSE_CSRGEAM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup extra_module
*  \details
*  \p rocsparse_csrgeam_nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
*  size \p m+1.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  Currently, only \ref rocsparse_matrix_type_general is supported.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr_A   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnz_C           pointer to the number of non-zero entries of the sparse CSR
*                  matrix \f$C\f$. \p nnz_C can be a host or device pointer.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p nnz_A or \p nnz_B is invalid.
*  \retval rocsparse_status_invalid_pointer \p descr_A, \p csr_row_ptr_A,
*          \p csr_col_ind_A, \p descr_B, \p csr_row_ptr_B, \p csr_col_ind_B,
*          \p descr_C, \p csr_row_ptr_C or \p nnz_C is invalid.
*  \retval rocsparse_status_not_implemented
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr descr_A,
                                       rocsparse_int             nnz_A,
                                       const rocsparse_int*      csr_row_ptr_A,
                                       const rocsparse_int*      csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       rocsparse_int             nnz_B,
                                       const rocsparse_int*      csr_row_ptr_B,
                                       const rocsparse_int*      csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       rocsparse_int*            csr_row_ptr_C,
                                       rocsparse_int*            nnz_C);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p rocsparse_csrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by \ref rocsparse_csrgeam_nnz().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*
*  \note Currently, only \ref rocsparse_matrix_type_general is supported.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_val_A       array of \p nnz_A elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr_A   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_val_B       array of \p nnz_B elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val_C       array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csr_col_ind_C   array of elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p nnz_A or \p nnz_B is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p descr_A, \p csr_val_A,
*          \p csr_row_ptr_A, \p csr_col_ind_A, \p beta, \p descr_B, \p csr_val_B,
*          \p csr_row_ptr_B, \p csr_col_ind_B, \p descr_C, \p csr_val_C,
*          \p csr_row_ptr_C or \p csr_col_ind_C is invalid.
*  \retval rocsparse_status_not_implemented
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example adds two CSR matrices.
*  \snippet example_rocsparse_csrgeam.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgeam(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const float*              csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const float*              beta,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const float*              csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const rocsparse_mat_descr descr_C,
                                    float*                    csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgeam(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const double*             csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const double*             beta,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const double*             csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const rocsparse_mat_descr descr_C,
                                    double*                   csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgeam(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr_A,
                                    rocsparse_int                  nnz_A,
                                    const rocsparse_float_complex* csr_val_A,
                                    const rocsparse_int*           csr_row_ptr_A,
                                    const rocsparse_int*           csr_col_ind_A,
                                    const rocsparse_float_complex* beta,
                                    const rocsparse_mat_descr      descr_B,
                                    rocsparse_int                  nnz_B,
                                    const rocsparse_float_complex* csr_val_B,
                                    const rocsparse_int*           csr_row_ptr_B,
                                    const rocsparse_int*           csr_col_ind_B,
                                    const rocsparse_mat_descr      descr_C,
                                    rocsparse_float_complex*       csr_val_C,
                                    const rocsparse_int*           csr_row_ptr_C,
                                    rocsparse_int*                 csr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgeam(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr_A,
                                    rocsparse_int                   nnz_A,
                                    const rocsparse_double_complex* csr_val_A,
                                    const rocsparse_int*            csr_row_ptr_A,
                                    const rocsparse_int*            csr_col_ind_A,
                                    const rocsparse_double_complex* beta,
                                    const rocsparse_mat_descr       descr_B,
                                    rocsparse_int                   nnz_B,
                                    const rocsparse_double_complex* csr_val_B,
                                    const rocsparse_int*            csr_row_ptr_B,
                                    const rocsparse_int*            csr_col_ind_B,
                                    const rocsparse_mat_descr       descr_C,
                                    rocsparse_double_complex*       csr_val_C,
                                    const rocsparse_int*            csr_row_ptr_C,
                                    rocsparse_int*                  csr_col_ind_C);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRGEAM_H */
