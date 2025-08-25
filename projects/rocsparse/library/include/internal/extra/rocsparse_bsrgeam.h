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

#ifndef ROCSPARSE_BSRGEAM_H
#define ROCSPARSE_BSRGEAM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup extra_module
*  \details
*  \p rocsparse_bsrgeam_nnzb computes the total BSR non-zero elements and the BSR row
*  offsets, that point to the start of every row of the sparse BSR matrix, of the
*  resulting matrix C. It is assumed that \p bsr_row_ptr_C has been allocated with
*  size \p mb+1.
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
*  dir             direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
*                  \ref rocsparse_direction_column in the BSR matrices \f$A\f$, \f$B\f$, and \f$C\f$.
*  @param[in]
*  mb              number of block rows in the sparse BSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  nb              number of block columns of the sparse BSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  block_dim       the block dimension of the BSR matrix \f$A\f$. Between 1 and m where \p m=mb*block_dim.
*  @param[in]
*  descr_A         descriptor of the sparse BSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_A          number of non-zero block entries of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A   array of \p mb+1 elements that point to the start of every block row of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_col_ind_A   array of \p nnzb_A elements containing the column indices of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse BSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_B          number of non-zero block entries of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_row_ptr_B   array of \p mb+1 elements that point to the start of every block row of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_col_ind_B   array of \p nnzb_B elements containing the block column indices of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  descr_C         descriptor of the sparse BSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_row_ptr_C   array of \p mb+1 elements that point to the start of every block row of the
*                  sparse BSR matrix \f$C\f$.
*  @param[out]
*  nnzb_C          pointer to the number of non-zero block entries of the sparse BSR
*                  matrix \f$C\f$. \p nnzb_C can be a host or device pointer.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p mb, \p nb, \p kb, \p nnzb_A or \p nnzb_B is invalid.
*  \retval rocsparse_status_invalid_pointer \p descr_A, \p bsr_row_ptr_A,
*          \p bsr_col_ind_A, \p descr_B, \p bsr_row_ptr_B, \p bsr_col_ind_B,
*          \p descr_C, \p bsr_row_ptr_C or \p nnzb_C is invalid.
*  \retval rocsparse_status_not_implemented
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrgeam_nnzb(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             block_dim,
                                        const rocsparse_mat_descr descr_A,
                                        rocsparse_int             nnzb_A,
                                        const rocsparse_int*      bsr_row_ptr_A,
                                        const rocsparse_int*      bsr_col_ind_A,
                                        const rocsparse_mat_descr descr_B,
                                        rocsparse_int             nnzb_B,
                                        const rocsparse_int*      bsr_row_ptr_B,
                                        const rocsparse_int*      bsr_col_ind_B,
                                        const rocsparse_mat_descr descr_C,
                                        rocsparse_int*            bsr_row_ptr_C,
                                        rocsparse_int*            nnzb_C);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using BSR storage format
*
*  \details
*  \p rocsparse_bsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in BSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$mb \times nb\f$ matrix \f$B\f$, defined in BSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$mb \times nb\f$ matrix \f$C\f$, defined in BSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p bsr_row_ptr_C has already been filled and that \p bsr_val_C and
*  \p bsr_col_ind_C are allocated by the user. \p bsr_row_ptr_C and allocation size of
*  \p bsr_col_ind_C and \p bsr_val_C is defined by the number of non-zero block elements of
*  the sparse BSR matrix C. Both can be obtained by \ref rocsparse_bsrgeam_nnzb().
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
*  dir             direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
*                  \ref rocsparse_direction_column in the BSR matrices \f$A\f$, \f$B\f$, and \f$C\f$.
*  @param[in]
*  mb               number of rows of the sparse BSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  nb               number of columns of the sparse BSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  block_dim       the block dimension of the BSR matrix \f$A\f$. Between 1 and m where \p m=mb*block_dim.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_A           number of non-zero block entries of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_val_A       array of \p nnzb_A block elements of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_col_ind_A   array of \p nnzb_A block elements containing the block column indices of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_B         descriptor of the sparse BSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_B          number of non-zero block entries of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_val_B       array of \p nnzb_B block elements of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_row_ptr_B   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_col_ind_B   array of \p nnzb_B block elements containing the block column indices of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  descr_C         descriptor of the sparse BSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_val_C       array of block elements of the sparse BSR matrix \f$C\f$.
*  @param[in]
*  bsr_row_ptr_C   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$C\f$.
*  @param[out]
*  bsr_col_ind_C   array of block elements containing the block column indices of the
*                  sparse BSR matrix \f$C\f$.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p mb, \p nb, \p nnzb_A or \p nnzb_B is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p descr_A, \p bsr_val_A,
*          \p bsr_row_ptr_A, \p bsr_col_ind_A, \p beta, \p descr_B, \p bsr_val_B,
*          \p bsr_row_ptr_B, \p bsr_col_ind_B, \p descr_C, \p csr_val_C,
*          \p bsr_row_ptr_C or \p bsr_col_ind_C is invalid.
*  \retval rocsparse_status_not_implemented
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example adds two CSR matrices.
*  \snippet example_rocsparse_bsrgeam.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrgeam(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             block_dim,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnzb_A,
                                    const float*              bsr_val_A,
                                    const rocsparse_int*      bsr_row_ptr_A,
                                    const rocsparse_int*      bsr_col_ind_A,
                                    const float*              beta,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnzb_B,
                                    const float*              bsr_val_B,
                                    const rocsparse_int*      bsr_row_ptr_B,
                                    const rocsparse_int*      bsr_col_ind_B,
                                    const rocsparse_mat_descr descr_C,
                                    float*                    bsr_val_C,
                                    const rocsparse_int*      bsr_row_ptr_C,
                                    rocsparse_int*            bsr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrgeam(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             block_dim,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnzb_A,
                                    const double*             bsr_val_A,
                                    const rocsparse_int*      bsr_row_ptr_A,
                                    const rocsparse_int*      bsr_col_ind_A,
                                    const double*             beta,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnzb_B,
                                    const double*             bsr_val_B,
                                    const rocsparse_int*      bsr_row_ptr_B,
                                    const rocsparse_int*      bsr_col_ind_B,
                                    const rocsparse_mat_descr descr_C,
                                    double*                   bsr_val_C,
                                    const rocsparse_int*      bsr_row_ptr_C,
                                    rocsparse_int*            bsr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrgeam(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr_A,
                                    rocsparse_int                  nnzb_A,
                                    const rocsparse_float_complex* bsr_val_A,
                                    const rocsparse_int*           bsr_row_ptr_A,
                                    const rocsparse_int*           bsr_col_ind_A,
                                    const rocsparse_float_complex* beta,
                                    const rocsparse_mat_descr      descr_B,
                                    rocsparse_int                  nnzb_B,
                                    const rocsparse_float_complex* bsr_val_B,
                                    const rocsparse_int*           bsr_row_ptr_B,
                                    const rocsparse_int*           bsr_col_ind_B,
                                    const rocsparse_mat_descr      descr_C,
                                    rocsparse_float_complex*       bsr_val_C,
                                    const rocsparse_int*           bsr_row_ptr_C,
                                    rocsparse_int*                 bsr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrgeam(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr_A,
                                    rocsparse_int                   nnzb_A,
                                    const rocsparse_double_complex* bsr_val_A,
                                    const rocsparse_int*            bsr_row_ptr_A,
                                    const rocsparse_int*            bsr_col_ind_A,
                                    const rocsparse_double_complex* beta,
                                    const rocsparse_mat_descr       descr_B,
                                    rocsparse_int                   nnzb_B,
                                    const rocsparse_double_complex* bsr_val_B,
                                    const rocsparse_int*            bsr_row_ptr_B,
                                    const rocsparse_int*            bsr_col_ind_B,
                                    const rocsparse_mat_descr       descr_C,
                                    rocsparse_double_complex*       bsr_val_C,
                                    const rocsparse_int*            bsr_row_ptr_C,
                                    rocsparse_int*                  bsr_col_ind_C);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRGEAM_H */
