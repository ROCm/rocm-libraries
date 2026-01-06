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

#ifndef ROCSPARSE_CSRGEMM_H
#define ROCSPARSE_CSRGEMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup extra_module
*  \details
*  \p rocsparse_csrgemm_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_csrgemm_nnz() and \ref rocsparse_scsrgemm "rocsparse_Xcsrgemm()".
*  The temporary storage buffer must be allocated by the user.
*
*  \note
*  Please note, that for matrix products with more than 4096 non-zero entries per row,
*  additional temporary storage buffer is allocated by the algorithm.
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
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
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the sparse
*                  CSR matrix \f$D\f$.
*  @param[inout]
*  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[out]
*  buffer_size     number of bytes of the temporary storage buffer required by
*                  \ref rocsparse_csrgemm_nnz() and \ref rocsparse_scsrgemm "rocsparse_Xcsrgemm()".
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
*          \p nnz_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p descr_A, \p csr_row_ptr_A, \p csr_col_ind_A, \p descr_B,
*          \p csr_row_ptr_B or \p csr_col_ind_B are invalid if \p alpha is valid,
*          \p descr_D, \p csr_row_ptr_D or \p csr_col_ind_D is invalid if \p beta is
*          valid, \p info_C or \p buffer_size is invalid.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgemm_buffer_size(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                rocsparse_int             k,
                                                const float*              alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnz_B,
                                                const rocsparse_int*      csr_row_ptr_B,
                                                const rocsparse_int*      csr_col_ind_B,
                                                const float*              beta,
                                                const rocsparse_mat_descr descr_D,
                                                rocsparse_int             nnz_D,
                                                const rocsparse_int*      csr_row_ptr_D,
                                                const rocsparse_int*      csr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgemm_buffer_size(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                rocsparse_int             k,
                                                const double*             alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnz_B,
                                                const rocsparse_int*      csr_row_ptr_B,
                                                const rocsparse_int*      csr_col_ind_B,
                                                const double*             beta,
                                                const rocsparse_mat_descr descr_D,
                                                rocsparse_int             nnz_D,
                                                const rocsparse_int*      csr_row_ptr_D,
                                                const rocsparse_int*      csr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgemm_buffer_size(rocsparse_handle               handle,
                                                rocsparse_operation            trans_A,
                                                rocsparse_operation            trans_B,
                                                rocsparse_int                  m,
                                                rocsparse_int                  n,
                                                rocsparse_int                  k,
                                                const rocsparse_float_complex* alpha,
                                                const rocsparse_mat_descr      descr_A,
                                                rocsparse_int                  nnz_A,
                                                const rocsparse_int*           csr_row_ptr_A,
                                                const rocsparse_int*           csr_col_ind_A,
                                                const rocsparse_mat_descr      descr_B,
                                                rocsparse_int                  nnz_B,
                                                const rocsparse_int*           csr_row_ptr_B,
                                                const rocsparse_int*           csr_col_ind_B,
                                                const rocsparse_float_complex* beta,
                                                const rocsparse_mat_descr      descr_D,
                                                rocsparse_int                  nnz_D,
                                                const rocsparse_int*           csr_row_ptr_D,
                                                const rocsparse_int*           csr_col_ind_D,
                                                rocsparse_mat_info             info_C,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgemm_buffer_size(rocsparse_handle                handle,
                                                rocsparse_operation             trans_A,
                                                rocsparse_operation             trans_B,
                                                rocsparse_int                   m,
                                                rocsparse_int                   n,
                                                rocsparse_int                   k,
                                                const rocsparse_double_complex* alpha,
                                                const rocsparse_mat_descr       descr_A,
                                                rocsparse_int                   nnz_A,
                                                const rocsparse_int*            csr_row_ptr_A,
                                                const rocsparse_int*            csr_col_ind_A,
                                                const rocsparse_mat_descr       descr_B,
                                                rocsparse_int                   nnz_B,
                                                const rocsparse_int*            csr_row_ptr_B,
                                                const rocsparse_int*            csr_col_ind_B,
                                                const rocsparse_double_complex* beta,
                                                const rocsparse_mat_descr       descr_D,
                                                rocsparse_int                   nnz_D,
                                                const rocsparse_int*            csr_row_ptr_D,
                                                const rocsparse_int*            csr_col_ind_D,
                                                rocsparse_mat_info              info_C,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p rocsparse_csrgemm_nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
*  with size \p m+1.
*  The required buffer size can be obtained by rocsparse_scsrgemm_buffer_size(),
*  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() and
*  rocsparse_zcsrgemm_buffer_size(), respectively.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*  \note
*  This function supports unsorted CSR matrices as input, while output will be sorted.
*  Please note that matrices B and D can only be unsorted up to 8192 intermediate
*  products per row. If this number is exceeded, \ref rocsparse_status_requires_sorted_storage
*  will be returned.
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
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the sparse
*                  CSR matrix \f$D\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnz_C           pointer to the number of non-zero entries of the sparse CSR
*                  matrix \f$C\f$.
*  @param[in]
*  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned
*                  by rocsparse_scsrgemm_buffer_size(),
*                  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() or
*                  rocsparse_zcsrgemm_buffer_size().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
*          \p nnz_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p descr_A, \p csr_row_ptr_A,
*          \p csr_col_ind_A, \p descr_B, \p csr_row_ptr_B, \p csr_col_ind_B,
*          \p descr_D, \p csr_row_ptr_D, \p csr_col_ind_D, \p descr_C,
*          \p csr_row_ptr_C, \p nnz_C, \p info_C or \p temp_buffer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             k,
                                       const rocsparse_mat_descr descr_A,
                                       rocsparse_int             nnz_A,
                                       const rocsparse_int*      csr_row_ptr_A,
                                       const rocsparse_int*      csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       rocsparse_int             nnz_B,
                                       const rocsparse_int*      csr_row_ptr_B,
                                       const rocsparse_int*      csr_col_ind_B,
                                       const rocsparse_mat_descr descr_D,
                                       rocsparse_int             nnz_D,
                                       const rocsparse_int*      csr_row_ptr_D,
                                       const rocsparse_int*      csr_col_ind_D,
                                       const rocsparse_mat_descr descr_C,
                                       rocsparse_int*            csr_row_ptr_C,
                                       rocsparse_int*            nnz_C,
                                       const rocsparse_mat_info  info_C,
                                       void*                     temp_buffer);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p rocsparse_csrgemm multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
*  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
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
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by rocsparse_csrgemm_nnz(). The
*  required buffer size for the computation can be obtained by
*  rocsparse_scsrgemm_buffer_size(), rocsparse_dcsrgemm_buffer_size(),
*  rocsparse_ccsrgemm_buffer_size() and rocsparse_zcsrgemm_buffer_size(), respectively.
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \cdot op(B)\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*  \note Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*  \note
*  This function supports unsorted CSR matrices as input, while output will be sorted.
*  Please note that matrices B and D can only be unsorted up to 4096 non-zero entries
*  per row. If this number is exceeded, \ref rocsparse_status_requires_sorted_storage
*  will be returned.
*  \note
*  This function is blocking with respect to the host.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
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
*  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_val_B       array of \p nnz_B elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_val_D       array of \p nnz_D elements of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val_C       array of \p nnz_C elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csr_col_ind_C   array of \p nnz_C elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*  @param[in]
*  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned
*                  by rocsparse_scsrgemm_buffer_size(),
*                  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() or
*                  rocsparse_zcsrgemm_buffer_size().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
*          \p nnz_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p descr_A, \p csr_val_A, \p csr_row_ptr_A, \p csr_col_ind_A, \p descr_B,
*          \p csr_val_B, \p csr_row_ptr_B or \p csr_col_ind_B are invalid if \p alpha
*          is valid, \p descr_D, \p csr_val_D, \p csr_row_ptr_D or \p csr_col_ind_D is
*          invalid if \p beta is valid, \p csr_val_C, \p csr_row_ptr_C,
*          \p csr_col_ind_C, \p info_C or \p temp_buffer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example multiplies two CSR matrices with a scalar alpha and adds the result to
*  another CSR matrix.
*  \code{.c}
*  // Initialize scalar multipliers
*  float alpha = 2.0f;
*  float beta  = 1.0f;
*
*  // Create matrix descriptors
*  rocsparse_mat_descr descr_A;
*  rocsparse_mat_descr descr_B;
*  rocsparse_mat_descr descr_C;
*  rocsparse_mat_descr descr_D;
*
*  rocsparse_create_mat_descr(&descr_A);
*  rocsparse_create_mat_descr(&descr_B);
*  rocsparse_create_mat_descr(&descr_C);
*  rocsparse_create_mat_descr(&descr_D);
*
*  // Create matrix info structure
*  rocsparse_mat_info info_C;
*  rocsparse_create_mat_info(&info_C);
*
*  // Set pointer mode
*  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
*
*  // Query rocsparse for the required buffer size
*  size_t buffer_size;
*
*  rocsparse_scsrgemm_buffer_size(handle,
*                                 rocsparse_operation_none,
*                                 rocsparse_operation_none,
*                                 m,
*                                 n,
*                                 k,
*                                 &alpha,
*                                 descr_A,
*                                 nnz_A,
*                                 csr_row_ptr_A,
*                                 csr_col_ind_A,
*                                 descr_B,
*                                 nnz_B,
*                                 csr_row_ptr_B,
*                                 csr_col_ind_B,
*                                 &beta,
*                                 descr_D,
*                                 nnz_D,
*                                 csr_row_ptr_D,
*                                 csr_col_ind_D,
*                                 info_C,
*                                 &buffer_size);
*
*  // Allocate buffer
*  void* buffer;
*  hipMalloc(&buffer, buffer_size);
*
*  // Obtain number of total non-zero entries in C and row pointers of C
*  rocsparse_int nnz_C;
*  hipMalloc((void**)&csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
*
*  rocsparse_csrgemm_nnz(handle,
*                        rocsparse_operation_none,
*                        rocsparse_operation_none,
*                        m,
*                        n,
*                        k,
*                        descr_A,
*                        nnz_A,
*                        csr_row_ptr_A,
*                        csr_col_ind_A,
*                        descr_B,
*                        nnz_B,
*                        csr_row_ptr_B,
*                        csr_col_ind_B,
*                        descr_D,
*                        nnz_D,
*                        csr_row_ptr_D,
*                        csr_col_ind_D,
*                        descr_C,
*                        csr_row_ptr_C,
*                        &nnz_C,
*                        info_C,
*                        buffer);
*
*  // Compute column indices and values of C
*  hipMalloc((void**)&csr_col_ind_C, sizeof(rocsparse_int) * nnz_C);
*  hipMalloc((void**)&csr_val_C, sizeof(float) * nnz_C);
*
*  rocsparse_scsrgemm(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     m,
*                     n,
*                     k,
*                     &alpha,
*                     descr_A,
*                     nnz_A,
*                     csr_val_A,
*                     csr_row_ptr_A,
*                     csr_col_ind_A,
*                     descr_B,
*                     nnz_B,
*                     csr_val_B,
*                     csr_row_ptr_B,
*                     csr_col_ind_B,
*                     &beta,
*                     descr_D,
*                     nnz_D,
*                     csr_val_D,
*                     csr_row_ptr_D,
*                     csr_col_ind_D,
*                     descr_C,
*                     csr_val_C,
*                     csr_row_ptr_C,
*                     csr_col_ind_C,
*                     info_C,
*                     buffer);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgemm(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    rocsparse_int             k,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const float*              csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const float*              csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const float*              beta,
                                    const rocsparse_mat_descr descr_D,
                                    rocsparse_int             nnz_D,
                                    const float*              csr_val_D,
                                    const rocsparse_int*      csr_row_ptr_D,
                                    const rocsparse_int*      csr_col_ind_D,
                                    const rocsparse_mat_descr descr_C,
                                    float*                    csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C,
                                    const rocsparse_mat_info  info_C,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgemm(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    rocsparse_int             k,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const double*             csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const double*             csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const double*             beta,
                                    const rocsparse_mat_descr descr_D,
                                    rocsparse_int             nnz_D,
                                    const double*             csr_val_D,
                                    const rocsparse_int*      csr_row_ptr_D,
                                    const rocsparse_int*      csr_col_ind_D,
                                    const rocsparse_mat_descr descr_C,
                                    double*                   csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C,
                                    const rocsparse_mat_info  info_C,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgemm(rocsparse_handle               handle,
                                    rocsparse_operation            trans_A,
                                    rocsparse_operation            trans_B,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    rocsparse_int                  k,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr_A,
                                    rocsparse_int                  nnz_A,
                                    const rocsparse_float_complex* csr_val_A,
                                    const rocsparse_int*           csr_row_ptr_A,
                                    const rocsparse_int*           csr_col_ind_A,
                                    const rocsparse_mat_descr      descr_B,
                                    rocsparse_int                  nnz_B,
                                    const rocsparse_float_complex* csr_val_B,
                                    const rocsparse_int*           csr_row_ptr_B,
                                    const rocsparse_int*           csr_col_ind_B,
                                    const rocsparse_float_complex* beta,
                                    const rocsparse_mat_descr      descr_D,
                                    rocsparse_int                  nnz_D,
                                    const rocsparse_float_complex* csr_val_D,
                                    const rocsparse_int*           csr_row_ptr_D,
                                    const rocsparse_int*           csr_col_ind_D,
                                    const rocsparse_mat_descr      descr_C,
                                    rocsparse_float_complex*       csr_val_C,
                                    const rocsparse_int*           csr_row_ptr_C,
                                    rocsparse_int*                 csr_col_ind_C,
                                    const rocsparse_mat_info       info_C,
                                    void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgemm(rocsparse_handle                handle,
                                    rocsparse_operation             trans_A,
                                    rocsparse_operation             trans_B,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    rocsparse_int                   k,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr_A,
                                    rocsparse_int                   nnz_A,
                                    const rocsparse_double_complex* csr_val_A,
                                    const rocsparse_int*            csr_row_ptr_A,
                                    const rocsparse_int*            csr_col_ind_A,
                                    const rocsparse_mat_descr       descr_B,
                                    rocsparse_int                   nnz_B,
                                    const rocsparse_double_complex* csr_val_B,
                                    const rocsparse_int*            csr_row_ptr_B,
                                    const rocsparse_int*            csr_col_ind_B,
                                    const rocsparse_double_complex* beta,
                                    const rocsparse_mat_descr       descr_D,
                                    rocsparse_int                   nnz_D,
                                    const rocsparse_double_complex* csr_val_D,
                                    const rocsparse_int*            csr_row_ptr_D,
                                    const rocsparse_int*            csr_col_ind_D,
                                    const rocsparse_mat_descr       descr_C,
                                    rocsparse_double_complex*       csr_val_C,
                                    const rocsparse_int*            csr_row_ptr_C,
                                    rocsparse_int*                  csr_col_ind_C,
                                    const rocsparse_mat_info        info_C,
                                    void*                           temp_buffer);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix symbolic multiplication using CSR storage format
*
*  \details
*  \p rocsparse_csrgemm_symbolic multiplies two sparsity patterns and add an extra one: \f[ opA \cdot op(B) + D \f]
*  with \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format and the sparse \f$m \times n\f$ matrix \f$D\f$.
*  The *  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
*  storage format, such
*  that
*  \f[
*    C := op(A) \cdot op(B) + D,
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
*  It is assumed that \p csr_row_ptr_C has already been filled and that and
*  \p csr_col_ind_C is allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by rocsparse_csrgemm_nnz(). The
*  required buffer size for the computation can be obtained by
*  rocsparse_scsrgemm_buffer_size(), rocsparse_dcsrgemm_buffer_size(),
*  rocsparse_ccsrgemm_buffer_size() and rocsparse_zcsrgemm_buffer_size(), respectively.
*
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*  \note Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*  \note This function is blocking with respect to the host.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_C           number of non-zero entries of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csr_col_ind_C   array of \p nnz_C elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*  @param[in]
*  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned
*                  by rocsparse_scsrgemm_buffer_size(),
*                  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() or
*                  rocsparse_zcsrgemm_buffer_size().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
*          \p nnz_D is invalid.
*  \retval rocsparse_status_invalid_pointer
*          \p descr_A, \p csr_row_ptr_A, \p csr_col_ind_A, \p descr_B,
*          \p csr_row_ptr_B or \p csr_col_ind_B, \p descr_D, \p csr_row_ptr_D, \p csr_col_ind_D
*          \p csr_row_ptr_C,
*          \p csr_col_ind_C, \p info_C or \p temp_buffer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example multiplies symbolically two CSR matrices and adds the result to
*  another CSR matrix.
*  \code{.c}
*  // Initialize scalar multipliers
*  float alpha = 2.0f;
*  float beta  = 1.0f;
*
*  // Create matrix descriptors
*  rocsparse_mat_descr descr_A;
*  rocsparse_mat_descr descr_B;
*  rocsparse_mat_descr descr_C;
*  rocsparse_mat_descr descr_D;
*
*  rocsparse_create_mat_descr(&descr_A);
*  rocsparse_create_mat_descr(&descr_B);
*  rocsparse_create_mat_descr(&descr_C);
*  rocsparse_create_mat_descr(&descr_D);
*
*  // Create matrix info structure
*  rocsparse_mat_info info_C;
*  rocsparse_create_mat_info(&info_C);
*
*  // Set pointer mode
*  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
*
*  // Query rocsparse for the required buffer size
*  size_t buffer_size;
*
*  rocsparse_scsrgemm_buffer_size(handle,
*                                 rocsparse_operation_none,
*                                 rocsparse_operation_none,
*                                 m,
*                                 n,
*                                 k,
*                                 &alpha,
*                                 descr_A,
*                                 nnz_A,
*                                 csr_row_ptr_A,
*                                 csr_col_ind_A,
*                                 descr_B,
*                                 nnz_B,
*                                 csr_row_ptr_B,
*                                 csr_col_ind_B,
*                                 &beta,
*                                 descr_D,
*                                 nnz_D,
*                                 csr_row_ptr_D,
*                                 csr_col_ind_D,
*                                 info_C,
*                                 &buffer_size);
*
*  // Allocate buffer
*  void* buffer;
*  hipMalloc(&buffer, buffer_size);
*
*  // Obtain number of total non-zero entries in C and row pointers of C
*  rocsparse_int nnz_C;
*  hipMalloc((void**)&csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
*
*  rocsparse_csrgemm_nnz(handle,
*                        rocsparse_operation_none,
*                        rocsparse_operation_none,
*                        m,
*                        n,
*                        k,
*                        descr_A,
*                        nnz_A,
*                        csr_row_ptr_A,
*                        csr_col_ind_A,
*                        descr_B,
*                        nnz_B,
*                        csr_row_ptr_B,
*                        csr_col_ind_B,
*                        descr_D,
*                        nnz_D,
*                        csr_row_ptr_D,
*                        csr_col_ind_D,
*                        descr_C,
*                        csr_row_ptr_C,
*                        &nnz_C,
*                        info_C,
*                        buffer);
*
*  // Compute column indices of C
*  hipMalloc((void**)&csr_col_ind_C, sizeof(rocsparse_int) * nnz_C);
*
*  rocsparse_csrgemm_symbolic(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     m,
*                     n,
*                     k,
*                     descr_A,
*                     nnz_A,
*                     csr_row_ptr_A,
*                     csr_col_ind_A,
*                     descr_B,
*                     nnz_B,
*                     csr_row_ptr_B,
*                     csr_col_ind_B,
*                     descr_D,
*                     nnz_D,
*                     csr_row_ptr_D,
*                     csr_col_ind_D,
*                     descr_C,
*                     nnz_C,
*                     csr_row_ptr_C,
*                     csr_col_ind_C,
*                     info_C,
*                     buffer);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrgemm_symbolic(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            rocsparse_int             k,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnz_D,
                                            const rocsparse_int*      csr_row_ptr_D,
                                            const rocsparse_int*      csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            rocsparse_int             nnz_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C,
                                            const rocsparse_mat_info  info_C,
                                            void*                     temp_buffer);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix numeric multiplication using CSR storage format
*
*  \details
*  \p rocsparse_csrgemm_numeric multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
*  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, predefined in CSR
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
*  The symbolic part of the csr matrix C can be obtained by rocsparse_csrgemm_symbolic().
*  It is assumed that \p csr_row_ptr_C and \p csr_col_ind_C have already been filled and that \p csr_val_C is allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by rocsparse_csrgemm_nnz(). The
*  required buffer size for the computation can be obtained by
*  rocsparse_scsrgemm_buffer_size(), rocsparse_dcsrgemm_buffer_size(),
*  rocsparse_ccsrgemm_buffer_size() and rocsparse_zcsrgemm_buffer_size(), respectively.
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \cdot op(B)\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*  \note Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*  \note This function is blocking with respect to the host.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  trans_A         matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B         matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
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
*  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_val_B       array of \p nnz_B elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_val_D       array of \p nnz_D elements of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_C           number of non-zero entries of the sparse CSR matrix \f$C\f$.
*  @param[out]
*  csr_val_C       array of \p nnz_C elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[in]
*  csr_col_ind_C   array of \p nnz_C elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*  @param[in]
*  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned
*                  by rocsparse_scsrgemm_buffer_size(),
*                  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() or
*                  rocsparse_zcsrgemm_buffer_size().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
*          \p nnz_D is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p descr_A, \p csr_val_A, \p csr_row_ptr_A, \p csr_col_ind_A, \p descr_B,
*          \p csr_val_B, \p csr_row_ptr_B or \p csr_col_ind_B are invalid if \p alpha
*          is valid, \p descr_D, \p csr_val_D, \p csr_row_ptr_D or \p csr_col_ind_D is
*          invalid if \p beta is valid, \p csr_val_C, \p csr_row_ptr_C,
*          \p csr_col_ind_C, \p info_C or \p temp_buffer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none,
*          \p trans_B != \ref rocsparse_operation_none, or
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example multiplies two CSR matrices with a scalar alpha and adds the result to
*  another CSR matrix.
*  \code{.c}
*  // Initialize scalar multipliers
*  float alpha = 2.0f;
*  float beta  = 1.0f;
*
*  // Create matrix descriptors
*  rocsparse_mat_descr descr_A;
*  rocsparse_mat_descr descr_B;
*  rocsparse_mat_descr descr_C;
*  rocsparse_mat_descr descr_D;
*
*  rocsparse_create_mat_descr(&descr_A);
*  rocsparse_create_mat_descr(&descr_B);
*  rocsparse_create_mat_descr(&descr_C);
*  rocsparse_create_mat_descr(&descr_D);
*
*  // Create matrix info structure
*  rocsparse_mat_info info_C;
*  rocsparse_create_mat_info(&info_C);
*
*  // Set pointer mode
*  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
*
*  // Query rocsparse for the required buffer size
*  size_t buffer_size;
*
*  rocsparse_scsrgemm_buffer_size(handle,
*                                 rocsparse_operation_none,
*                                 rocsparse_operation_none,
*                                 m,
*                                 n,
*                                 k,
*                                 &alpha,
*                                 descr_A,
*                                 nnz_A,
*                                 csr_row_ptr_A,
*                                 csr_col_ind_A,
*                                 descr_B,
*                                 nnz_B,
*                                 csr_row_ptr_B,
*                                 csr_col_ind_B,
*                                 &beta,
*                                 descr_D,
*                                 nnz_D,
*                                 csr_row_ptr_D,
*                                 csr_col_ind_D,
*                                 info_C,
*                                 &buffer_size);
*
*  // Allocate buffer
*  void* buffer;
*  hipMalloc(&buffer, buffer_size);
*
*  // Obtain number of total non-zero entries in C and row pointers of C
*  rocsparse_int nnz_C;
*  hipMalloc((void**)&csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
*
*  rocsparse_csrgemm_nnz(handle,
*                        rocsparse_operation_none,
*                        rocsparse_operation_none,
*                        m,
*                        n,
*                        k,
*                        descr_A,
*                        nnz_A,
*                        csr_row_ptr_A,
*                        csr_col_ind_A,
*                        descr_B,
*                        nnz_B,
*                        csr_row_ptr_B,
*                        csr_col_ind_B,
*                        descr_D,
*                        nnz_D,
*                        csr_row_ptr_D,
*                        csr_col_ind_D,
*                        descr_C,
*                        csr_row_ptr_C,
*                        &nnz_C,
*                        info_C,
*                        buffer);
*
*  // Compute column indices and values of C
*  hipMalloc((void**)&csr_col_ind_C, sizeof(rocsparse_int) * nnz_C);
*  rocsparse_csrgemm_symbolic(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     m,
*                     n,
*                     k,
*                     descr_A,
*                     nnz_A,
*                     csr_row_ptr_A,
*                     csr_col_ind_A,
*                     descr_B,
*                     nnz_B,
*                     csr_row_ptr_B,
*                     csr_col_ind_B,
*                     descr_D,
*                     nnz_D,
*                     csr_row_ptr_D,
*                     csr_col_ind_D,
*                     descr_C,
*                     nnz_C,
*                     csr_row_ptr_C,
*                     csr_col_ind_C,
*                     info_C,
*                     buffer);
*  hipMalloc((void**)&csr_val_C, sizeof(float) * nnz_C);
*
*  rocsparse_scsrgemm_numeric(handle,
*                     rocsparse_operation_none,
*                     rocsparse_operation_none,
*                     m,
*                     n,
*                     k,
*                     &alpha,
*                     descr_A,
*                     nnz_A,
*                     csr_val_A,
*                     csr_row_ptr_A,
*                     csr_col_ind_A,
*                     descr_B,
*                     nnz_B,
*                     csr_val_B,
*                     csr_row_ptr_B,
*                     csr_col_ind_B,
*                     &beta,
*                     descr_D,
*                     nnz_D,
*                     csr_val_D,
*                     csr_row_ptr_D,
*                     csr_col_ind_D,
*                     descr_C,
*                     nnz_C,
*                     csr_val_C,
*                     csr_row_ptr_C,
*                     csr_col_ind_C,
*                     info_C,
*                     buffer);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgemm_numeric(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            rocsparse_int             k,
                                            const float*              alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const float*              csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const float*              csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const float*              beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnz_D,
                                            const float*              csr_val_D,
                                            const rocsparse_int*      csr_row_ptr_D,
                                            const rocsparse_int*      csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            rocsparse_int             nnz_C,
                                            float*                    csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            const rocsparse_int*      csr_col_ind_C,
                                            const rocsparse_mat_info  info_C,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgemm_numeric(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            rocsparse_int             k,
                                            const double*             alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const double*             csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const double*             csr_val_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const double*             beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnz_D,
                                            const double*             csr_val_D,
                                            const rocsparse_int*      csr_row_ptr_D,
                                            const rocsparse_int*      csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            rocsparse_int             nnz_C,
                                            double*                   csr_val_C,
                                            const rocsparse_int*      csr_row_ptr_C,
                                            const rocsparse_int*      csr_col_ind_C,
                                            const rocsparse_mat_info  info_C,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgemm_numeric(rocsparse_handle               handle,
                                            rocsparse_operation            trans_A,
                                            rocsparse_operation            trans_B,
                                            rocsparse_int                  m,
                                            rocsparse_int                  n,
                                            rocsparse_int                  k,
                                            const rocsparse_float_complex* alpha,
                                            const rocsparse_mat_descr      descr_A,
                                            rocsparse_int                  nnz_A,
                                            const rocsparse_float_complex* csr_val_A,
                                            const rocsparse_int*           csr_row_ptr_A,
                                            const rocsparse_int*           csr_col_ind_A,
                                            const rocsparse_mat_descr      descr_B,
                                            rocsparse_int                  nnz_B,
                                            const rocsparse_float_complex* csr_val_B,
                                            const rocsparse_int*           csr_row_ptr_B,
                                            const rocsparse_int*           csr_col_ind_B,
                                            const rocsparse_float_complex* beta,
                                            const rocsparse_mat_descr      descr_D,
                                            rocsparse_int                  nnz_D,
                                            const rocsparse_float_complex* csr_val_D,
                                            const rocsparse_int*           csr_row_ptr_D,
                                            const rocsparse_int*           csr_col_ind_D,
                                            const rocsparse_mat_descr      descr_C,
                                            rocsparse_int                  nnz_C,
                                            rocsparse_float_complex*       csr_val_C,
                                            const rocsparse_int*           csr_row_ptr_C,
                                            const rocsparse_int*           csr_col_ind_C,
                                            const rocsparse_mat_info       info_C,
                                            void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgemm_numeric(rocsparse_handle                handle,
                                            rocsparse_operation             trans_A,
                                            rocsparse_operation             trans_B,
                                            rocsparse_int                   m,
                                            rocsparse_int                   n,
                                            rocsparse_int                   k,
                                            const rocsparse_double_complex* alpha,
                                            const rocsparse_mat_descr       descr_A,
                                            rocsparse_int                   nnz_A,
                                            const rocsparse_double_complex* csr_val_A,
                                            const rocsparse_int*            csr_row_ptr_A,
                                            const rocsparse_int*            csr_col_ind_A,
                                            const rocsparse_mat_descr       descr_B,
                                            rocsparse_int                   nnz_B,
                                            const rocsparse_double_complex* csr_val_B,
                                            const rocsparse_int*            csr_row_ptr_B,
                                            const rocsparse_int*            csr_col_ind_B,
                                            const rocsparse_double_complex* beta,
                                            const rocsparse_mat_descr       descr_D,
                                            rocsparse_int                   nnz_D,
                                            const rocsparse_double_complex* csr_val_D,
                                            const rocsparse_int*            csr_row_ptr_D,
                                            const rocsparse_int*            csr_col_ind_D,
                                            const rocsparse_mat_descr       descr_C,
                                            rocsparse_int                   nnz_C,
                                            rocsparse_double_complex*       csr_val_C,
                                            const rocsparse_int*            csr_row_ptr_C,
                                            const rocsparse_int*            csr_col_ind_C,
                                            const rocsparse_mat_info        info_C,
                                            void*                           temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRGEMM_H */
