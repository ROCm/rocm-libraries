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

#ifndef ROCSPARSE_BSRGEMM_H
#define ROCSPARSE_BSRGEMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup extra_module
*  \details
*  \p rocsparse_bsrgemm_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_bsrgemm_nnzb() and \ref rocsparse_sbsrgemm "rocsparse_Xbsrgemm()".
*  The temporary storage buffer must be allocated by the user.
*
*  \note
*  This function is blocking with respect to the host.
*  \note
*  Currently, only \p trans_A == \p trans_B == \ref rocsparse_operation_none is
*  supported.
*  \note
*  Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  dir             direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
*                  \ref rocsparse_direction_column in the BSR matrices \f$A\f$, \f$B\f$, \f$C\f$, and \f$D\f$.
*  @param[in]
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  mb              number of block rows in the sparse BSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  nb              number of block columns of the sparse BSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  kb              number of block columns of the sparse BSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse BSR matrix \f$op(B)\f$.
*  @param[in]
*  block_dim       the block dimension of the BSR matrix \f$A\f$, \f$B\f$, \f$C\f$, and \f$D\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descr_A         descriptor of the sparse BSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_A          number of non-zero block entries of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A   array of \p mb+1 elements (\f$op(A) == A\f$, \p kb+1 otherwise)
*                  that point to the start of every block row of the sparse BSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  bsr_col_ind_A   array of \p nnzb_A elements containing the block column indices of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse BSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_B          number of non-zero block entries of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_row_ptr_B   array of \p kb+1 elements (\f$op(B) == B\f$, \p mb+1 otherwise)
*                  that point to the start of every block row of the sparse BSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  bsr_col_ind_B   array of \p nnzb_B elements containing the block column indices of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_D         descriptor of the sparse BSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_D          number of non-zero block entries of the sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_row_ptr_D   array of \p mb+1 elements that point to the start of every block row of the
*                  sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_col_ind_D   array of \p nnzb_D elements containing the block column indices of the sparse
*                  BSR matrix \f$D\f$.
*  @param[inout]
*  info_C          structure that holds meta data for the sparse BSR matrix \f$C\f$.
*  @param[out]
*  buffer_size     number of bytes of the temporary storage buffer required by
*                  rocsparse_bsrgemm_nnzb(), rocsparse_sbsrgemm(), rocsparse_dbsrgemm(),
*                  rocsparse_cbsrgemm() and rocsparse_zbsrgemm().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p mb, \p nb, \p kb, \p block_dim, \p nnzb_A, \p nnzb_B or
*          \p nnzb_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p descr_A, \p bsr_row_ptr_A, \p bsr_col_ind_A, \p descr_B,
*          \p bsr_row_ptr_B or \p bsr_col_ind_B are invalid if \p alpha is valid,
*          \p descr_D, \p bsr_row_ptr_D or \p bsr_col_ind_D is invalid if \p beta is
*          valid, \p info_C or \p buffer_size is invalid.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrgemm_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             mb,
                                                rocsparse_int             nb,
                                                rocsparse_int             kb,
                                                rocsparse_int             block_dim,
                                                const float*              alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnzb_A,
                                                const rocsparse_int*      bsr_row_ptr_A,
                                                const rocsparse_int*      bsr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnzb_B,
                                                const rocsparse_int*      bsr_row_ptr_B,
                                                const rocsparse_int*      bsr_col_ind_B,
                                                const float*              beta,
                                                const rocsparse_mat_descr descr_D,
                                                rocsparse_int             nnzb_D,
                                                const rocsparse_int*      bsr_row_ptr_D,
                                                const rocsparse_int*      bsr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrgemm_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             mb,
                                                rocsparse_int             nb,
                                                rocsparse_int             kb,
                                                rocsparse_int             block_dim,
                                                const double*             alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnzb_A,
                                                const rocsparse_int*      bsr_row_ptr_A,
                                                const rocsparse_int*      bsr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnzb_B,
                                                const rocsparse_int*      bsr_row_ptr_B,
                                                const rocsparse_int*      bsr_col_ind_B,
                                                const double*             beta,
                                                const rocsparse_mat_descr descr_D,
                                                rocsparse_int             nnzb_D,
                                                const rocsparse_int*      bsr_row_ptr_D,
                                                const rocsparse_int*      bsr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrgemm_buffer_size(rocsparse_handle               handle,
                                                rocsparse_direction            dir,
                                                rocsparse_operation            trans_A,
                                                rocsparse_operation            trans_B,
                                                rocsparse_int                  mb,
                                                rocsparse_int                  nb,
                                                rocsparse_int                  kb,
                                                rocsparse_int                  block_dim,
                                                const rocsparse_float_complex* alpha,
                                                const rocsparse_mat_descr      descr_A,
                                                rocsparse_int                  nnzb_A,
                                                const rocsparse_int*           bsr_row_ptr_A,
                                                const rocsparse_int*           bsr_col_ind_A,
                                                const rocsparse_mat_descr      descr_B,
                                                rocsparse_int                  nnzb_B,
                                                const rocsparse_int*           bsr_row_ptr_B,
                                                const rocsparse_int*           bsr_col_ind_B,
                                                const rocsparse_float_complex* beta,
                                                const rocsparse_mat_descr      descr_D,
                                                rocsparse_int                  nnzb_D,
                                                const rocsparse_int*           bsr_row_ptr_D,
                                                const rocsparse_int*           bsr_col_ind_D,
                                                rocsparse_mat_info             info_C,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrgemm_buffer_size(rocsparse_handle                handle,
                                                rocsparse_direction             dir,
                                                rocsparse_operation             trans_A,
                                                rocsparse_operation             trans_B,
                                                rocsparse_int                   mb,
                                                rocsparse_int                   nb,
                                                rocsparse_int                   kb,
                                                rocsparse_int                   block_dim,
                                                const rocsparse_double_complex* alpha,
                                                const rocsparse_mat_descr       descr_A,
                                                rocsparse_int                   nnzb_A,
                                                const rocsparse_int*            bsr_row_ptr_A,
                                                const rocsparse_int*            bsr_col_ind_A,
                                                const rocsparse_mat_descr       descr_B,
                                                rocsparse_int                   nnzb_B,
                                                const rocsparse_int*            bsr_row_ptr_B,
                                                const rocsparse_int*            bsr_col_ind_B,
                                                const rocsparse_double_complex* beta,
                                                const rocsparse_mat_descr       descr_D,
                                                rocsparse_int                   nnzb_D,
                                                const rocsparse_int*            bsr_row_ptr_D,
                                                const rocsparse_int*            bsr_col_ind_D,
                                                rocsparse_mat_info              info_C,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup extra_module
*  \details
*  \p rocsparse_bsrgemm_nnzb computes the total BSR non-zero block elements and the BSR block row
*  offsets, that point to the start of every block row of the sparse BSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p bsr_row_ptr_C has been allocated
*  with size \p mb+1.
*  The required buffer size can be obtained by
*  \ref rocsparse_sbsrgemm_buffer_size "rocsparse_Xbsrgemm_buffer_size()".
*
*  \note
*  This function is blocking with respect to the host.
*  \note
*  Currently, only \p trans_A == \p trans_B == \ref rocsparse_operation_none is
*  supported.
*  \note
*  Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  dir             direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
*                  \ref rocsparse_direction_column in the BSR matrices \f$A\f$, \f$B\f$, \f$C\f$, and \f$D\f$.
*  @param[in]
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  mb              number of block rows in the sparse BSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  nb              number of block columns of the sparse BSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  kb              number of block columns of the sparse BSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse BSR matrix \f$op(B)\f$.
*  @param[in]
*  block_dim       the block dimension of the BSR matrix \f$A\f$, \f$B\f$, \f$C\f$, and \f$D\f$.
*  @param[in]
*  descr_A         descriptor of the sparse BSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_A          number of non-zero block entries of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A   array of \p mb+1 block elements (\f$op(A) == A\f$, \p kb+1 otherwise)
*                  that point to the start of every row of the sparse BSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  bsr_col_ind_A   array of \p nnzb_A block elements containing the block column indices of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse BSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_B          number of non-zero block entries of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_row_ptr_B   array of \p kb+1 block elements (\f$op(B) == B\f$, \p mb+1 otherwise)
*                  that point to the start of every block row of the sparse BSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  bsr_col_ind_B   array of \p nnzb_B block elements containing the block column indices of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  descr_D         descriptor of the sparse BSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_D          number of non-zero block entries of the sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_row_ptr_D   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_col_ind_D   array of \p nnzb_D block elements containing the block column indices of the sparse
*                  BSR matrix \f$D\f$.
*  @param[in]
*  descr_C         descriptor of the sparse BSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_row_ptr_C   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$C\f$.
*  @param[out]
*  nnzb_C          pointer to the number of non-zero block entries of the sparse BSR
*                  matrix \f$C\f$.
*  @param[in]
*  info_C          structure that holds meta data for the sparse BSR matrix \f$C\f$.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned
*                  by rocsparse_sbsrgemm_buffer_size(),
*                  rocsparse_dbsrgemm_buffer_size(), rocsparse_cbsrgemm_buffer_size() or
*                  rocsparse_zbsrgemm_buffer_size().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p mb, \p nb, \p kb, \p block_dim, \p nnzb_A, \p nnzb_B or
*          \p nnzb_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p descr_A, \p bsr_row_ptr_A,
*          \p bsr_col_ind_A, \p descr_B, \p bsr_row_ptr_B, \p bsr_col_ind_B,
*          \p descr_D, \p bsr_row_ptr_D, \p bsr_col_ind_D, \p descr_C,
*          \p bsr_row_ptr_C, \p nnzb_C, \p info_C or \p temp_buffer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrgemm_nnzb(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             kb,
                                        rocsparse_int             block_dim,
                                        const rocsparse_mat_descr descr_A,
                                        rocsparse_int             nnzb_A,
                                        const rocsparse_int*      bsr_row_ptr_A,
                                        const rocsparse_int*      bsr_col_ind_A,
                                        const rocsparse_mat_descr descr_B,
                                        rocsparse_int             nnzb_B,
                                        const rocsparse_int*      bsr_row_ptr_B,
                                        const rocsparse_int*      bsr_col_ind_B,
                                        const rocsparse_mat_descr descr_D,
                                        rocsparse_int             nnzb_D,
                                        const rocsparse_int*      bsr_row_ptr_D,
                                        const rocsparse_int*      bsr_col_ind_D,
                                        const rocsparse_mat_descr descr_C,
                                        rocsparse_int*            bsr_row_ptr_C,
                                        rocsparse_int*            nnzb_C,
                                        const rocsparse_mat_info  info_C,
                                        void*                     temp_buffer);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrgemm multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$mb \times kb\f$ matrix \f$A\f$, defined in BSR storage format, and the sparse
*  \f$kb \times nb\f$ matrix \f$B\f$, defined in BSR storage format, and adds the result
*  to the sparse \f$mb \times nb\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$mb \times nb\f$ matrix \f$C\f$, defined in BSR
*  storage format, such
*  that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot D,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function does not produce deterministic results.
*
*  It is assumed that \p bsr_row_ptr_C has already been filled and that \p bsr_val_C and
*  \p bsr_col_ind_C are allocated by the user. \p bsr_row_ptr_C and allocation size of
*  \p bsr_col_ind_C and \p bsr_val_C is defined by the number of non-zero elements of
*  the sparse BSR matrix C. Both can be obtained by \ref rocsparse_bsrgemm_nnzb(). The
*  required buffer size for the computation can be obtained by
*  \ref rocsparse_sbsrgemm_buffer_size "rocsparse_Xbsrgemm_buffer_size()".
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \cdot op(B)\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*  \note Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note This function is blocking with respect to the host.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  dir             direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
*                  \ref rocsparse_direction_column in the BSR matrices \f$A\f$, \f$B\f$, \f$C\f$, and \f$D\f$.
*  @param[in]
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  mb              number of block rows of the sparse BSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  nb              number of block columns of the sparse BSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  kb              number of block columns of the sparse BSR matrix \f$op(A)\f$ and number of
*                  block rows of the sparse BSR matrix \f$op(B)\f$.
*  @param[in]
*  block_dim       the block dimension of the BSR matrix \f$A\f$, \f$B\f$, \f$C\f$, and \f$D\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descr_A         descriptor of the sparse BSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_A          number of non-zero block entries of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_val_A       array of \p nnzb_A block elements of the sparse BSR matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A   array of \p mb+1 block elements (\f$op(A) == A\f$, \p kb+1 otherwise)
*                  that point to the start of every block row of the sparse BSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  bsr_col_ind_A   array of \p nnzb_A block elements containing the block column indices of the
*                  sparse BSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse BSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_B          number of non-zero block entries of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_val_B       array of \p nnzb_B block elements of the sparse BSR matrix \f$B\f$.
*  @param[in]
*  bsr_row_ptr_B   array of \p kb+1 block elements (\f$op(B) == B\f$, \p mb+1 otherwise)
*                  that point to the start of every block row of the sparse BSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  bsr_col_ind_B   array of \p nnzb_B block elements containing the block column indices of the
*                  sparse BSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_D         descriptor of the sparse BSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnzb_D          number of non-zero block entries of the sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_val_D       array of \p nnzb_D block elements of the sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_row_ptr_D   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$D\f$.
*  @param[in]
*  bsr_col_ind_D   array of \p nnzb_D block elements containing the block column indices of the
*                  sparse BSR matrix \f$D\f$.
*  @param[in]
*  descr_C         descriptor of the sparse BSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_val_C       array of \p nnzb_C elements of the sparse BSR matrix \f$C\f$.
*  @param[in]
*  bsr_row_ptr_C   array of \p mb+1 block elements that point to the start of every block row of the
*                  sparse BSR matrix \f$C\f$.
*  @param[out]
*  bsr_col_ind_C   array of \p nnzb_C block elements containing the block column indices of the
*                  sparse BSR matrix \f$C\f$.
*  @param[in]
*  info_C          structure that holds meta data for the sparse BSR matrix \f$C\f$.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned
*                  by rocsparse_sbsrgemm_buffer_size(),
*                  rocsparse_dbsrgemm_buffer_size(), rocsparse_cbsrgemm_buffer_size() or
*                  rocsparse_zbsrgemm_buffer_size().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p mb, \p nb, \p kb, \p block_dim, \p nnzb_A, \p nnzb_B or
*          \p nnzb_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p descr_A, \p bsr_val_A, \p bsr_row_ptr_A, \p bsr_col_ind_A, \p descr_B,
*          \p bsr_val_B, \p bsr_row_ptr_B or \p bsr_col_ind_B are invalid if \p alpha
*          is valid, \p descr_D, \p bsr_val_D, \p bsr_row_ptr_D or \p bsr_col_ind_D is
*          invalid if \p beta is valid, \p bsr_val_C, \p bsr_row_ptr_C,
*          \p bsr_col_ind_C, \p info_C or \p temp_buffer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example multiplies two BSR matrices with a scalar alpha and adds the result to
*  another BSR matrix.
*  \snippet example_rocsparse_bsrgemm.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrgemm(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             kb,
                                    rocsparse_int             block_dim,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnzb_A,
                                    const float*              bsr_val_A,
                                    const rocsparse_int*      bsr_row_ptr_A,
                                    const rocsparse_int*      bsr_col_ind_A,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnzb_B,
                                    const float*              bsr_val_B,
                                    const rocsparse_int*      bsr_row_ptr_B,
                                    const rocsparse_int*      bsr_col_ind_B,
                                    const float*              beta,
                                    const rocsparse_mat_descr descr_D,
                                    rocsparse_int             nnzb_D,
                                    const float*              bsr_val_D,
                                    const rocsparse_int*      bsr_row_ptr_D,
                                    const rocsparse_int*      bsr_col_ind_D,
                                    const rocsparse_mat_descr descr_C,
                                    float*                    bsr_val_C,
                                    const rocsparse_int*      bsr_row_ptr_C,
                                    rocsparse_int*            bsr_col_ind_C,
                                    const rocsparse_mat_info  info_C,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrgemm(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             kb,
                                    rocsparse_int             block_dim,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnzb_A,
                                    const double*             bsr_val_A,
                                    const rocsparse_int*      bsr_row_ptr_A,
                                    const rocsparse_int*      bsr_col_ind_A,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnzb_B,
                                    const double*             bsr_val_B,
                                    const rocsparse_int*      bsr_row_ptr_B,
                                    const rocsparse_int*      bsr_col_ind_B,
                                    const double*             beta,
                                    const rocsparse_mat_descr descr_D,
                                    rocsparse_int             nnzb_D,
                                    const double*             bsr_val_D,
                                    const rocsparse_int*      bsr_row_ptr_D,
                                    const rocsparse_int*      bsr_col_ind_D,
                                    const rocsparse_mat_descr descr_C,
                                    double*                   bsr_val_C,
                                    const rocsparse_int*      bsr_row_ptr_C,
                                    rocsparse_int*            bsr_col_ind_C,
                                    const rocsparse_mat_info  info_C,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrgemm(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_operation            trans_A,
                                    rocsparse_operation            trans_B,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    rocsparse_int                  kb,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr_A,
                                    rocsparse_int                  nnzb_A,
                                    const rocsparse_float_complex* bsr_val_A,
                                    const rocsparse_int*           bsr_row_ptr_A,
                                    const rocsparse_int*           bsr_col_ind_A,
                                    const rocsparse_mat_descr      descr_B,
                                    rocsparse_int                  nnzb_B,
                                    const rocsparse_float_complex* bsr_val_B,
                                    const rocsparse_int*           bsr_row_ptr_B,
                                    const rocsparse_int*           bsr_col_ind_B,
                                    const rocsparse_float_complex* beta,
                                    const rocsparse_mat_descr      descr_D,
                                    rocsparse_int                  nnzb_D,
                                    const rocsparse_float_complex* bsr_val_D,
                                    const rocsparse_int*           bsr_row_ptr_D,
                                    const rocsparse_int*           bsr_col_ind_D,
                                    const rocsparse_mat_descr      descr_C,
                                    rocsparse_float_complex*       bsr_val_C,
                                    const rocsparse_int*           bsr_row_ptr_C,
                                    rocsparse_int*                 bsr_col_ind_C,
                                    const rocsparse_mat_info       info_C,
                                    void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrgemm(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_operation             trans_A,
                                    rocsparse_operation             trans_B,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    rocsparse_int                   kb,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr_A,
                                    rocsparse_int                   nnzb_A,
                                    const rocsparse_double_complex* bsr_val_A,
                                    const rocsparse_int*            bsr_row_ptr_A,
                                    const rocsparse_int*            bsr_col_ind_A,
                                    const rocsparse_mat_descr       descr_B,
                                    rocsparse_int                   nnzb_B,
                                    const rocsparse_double_complex* bsr_val_B,
                                    const rocsparse_int*            bsr_row_ptr_B,
                                    const rocsparse_int*            bsr_col_ind_B,
                                    const rocsparse_double_complex* beta,
                                    const rocsparse_mat_descr       descr_D,
                                    rocsparse_int                   nnzb_D,
                                    const rocsparse_double_complex* bsr_val_D,
                                    const rocsparse_int*            bsr_row_ptr_D,
                                    const rocsparse_int*            bsr_col_ind_D,
                                    const rocsparse_mat_descr       descr_C,
                                    rocsparse_double_complex*       bsr_val_C,
                                    const rocsparse_int*            bsr_row_ptr_C,
                                    rocsparse_int*                  bsr_col_ind_C,
                                    const rocsparse_mat_info        info_C,
                                    void*                           temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRGEMM_H */
