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

#ifndef ROCSPARSE_GEBSR2GEBSR_H
#define ROCSPARSE_GEBSR2GEBSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_gebsr2gebsr_buffer_size returns the size of the temporary storage buffer that is required by
*  \ref rocsparse_gebsr2gebsr_nnz() and \ref rocsparse_sgebsr2gebsr "rocsparse_Xgebsr2gebsr()". The temporary
*  storage buffer must be allocated by the user.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle           handle to the rocsparse library context queue.
*  @param[in]
*  dir              the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb               number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb               number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb             number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descr_A          the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is
*                   \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  bsr_val_A        array of \p nnzb*row_block_dim_A*col_block_dim_A containing the values of the sparse general BSR
*                   matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A    array of \p mb+1 elements that point to the start of every block row of the
*                   sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsr_col_ind_A    array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  row_block_dim_A  row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  col_block_dim_A  column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  row_block_dim_C  row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  col_block_dim_C  column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  buffer_size      number of bytes of the temporary storage buffer required by \ref rocsparse_gebsr2gebsr_nnz() and
*                   \ref rocsparse_sgebsr2gebsr "rocsparse_Xgebsr2gebsr()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p nnzb or \p row_block_dim_A or
*              \p col_block_dim_A or \p row_block_dim_C or \p col_block_dim_C is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr_A or \p bsr_col_ind_A
*              or \p descr_A or \p buffer_size pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr_A,
                                                    const float*              bsr_val_A,
                                                    const rocsparse_int*      bsr_row_ptr_A,
                                                    const rocsparse_int*      bsr_col_ind_A,
                                                    rocsparse_int             row_block_dim_A,
                                                    rocsparse_int             col_block_dim_A,
                                                    rocsparse_int             row_block_dim_C,
                                                    rocsparse_int             col_block_dim_C,
                                                    size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr_A,
                                                    const double*             bsr_val_A,
                                                    const rocsparse_int*      bsr_row_ptr_A,
                                                    const rocsparse_int*      bsr_col_ind_A,
                                                    rocsparse_int             row_block_dim_A,
                                                    rocsparse_int             col_block_dim_A,
                                                    rocsparse_int             row_block_dim_C,
                                                    rocsparse_int             col_block_dim_C,
                                                    size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsr_buffer_size(rocsparse_handle               handle,
                                                    rocsparse_direction            dir,
                                                    rocsparse_int                  mb,
                                                    rocsparse_int                  nb,
                                                    rocsparse_int                  nnzb,
                                                    const rocsparse_mat_descr      descr_A,
                                                    const rocsparse_float_complex* bsr_val_A,
                                                    const rocsparse_int*           bsr_row_ptr_A,
                                                    const rocsparse_int*           bsr_col_ind_A,
                                                    rocsparse_int                  row_block_dim_A,
                                                    rocsparse_int                  col_block_dim_A,
                                                    rocsparse_int                  row_block_dim_C,
                                                    rocsparse_int                  col_block_dim_C,
                                                    size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsr_buffer_size(rocsparse_handle                handle,
                                                    rocsparse_direction             dir,
                                                    rocsparse_int                   mb,
                                                    rocsparse_int                   nb,
                                                    rocsparse_int                   nnzb,
                                                    const rocsparse_mat_descr       descr_A,
                                                    const rocsparse_double_complex* bsr_val_A,
                                                    const rocsparse_int*            bsr_row_ptr_A,
                                                    const rocsparse_int*            bsr_col_ind_A,
                                                    rocsparse_int                   row_block_dim_A,
                                                    rocsparse_int                   col_block_dim_A,
                                                    rocsparse_int                   row_block_dim_C,
                                                    rocsparse_int                   col_block_dim_C,
                                                    size_t*                         buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \details
*  This function takes a sparse GEneral BSR matrix as input and computes the block row offset array, \p bsr_row_ptr_C,
*  and the total number of nonzero blocks, \p nnz_total_dev_host_ptr, that will result from converting the GEneral BSR
*  format input matrix to a GEneral BSR format output matrix. The input and output matrices can have different row and
*  column block dimensions. \p rocsparse_gebsr2gebsr_nnz is the second step in the conversion and is used in conjunction with
*  \ref rocsparse_sgebsr2gebsr_buffer_size "rocsparse_Xgebsr2gebsr_buffer_size()" and
*  \ref rocsparse_sgebsr2gebsr "rocsparse_Xgebsr2gebsr()".
*
*  \p rocsparse_gebsr2gebsr_nnz accepts both host and device pointers for \p nnz_total_dev_host_ptr which can be set by
*  calling \ref rocsparse_set_pointer_mode prior to calling \p rocsparse_gebsr2gebsr_nnz.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle                  handle to the rocsparse library context queue.
*  @param[in]
*  dir                     the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb                      number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb                      number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb                    number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descr_A                 the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is
*                          \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  bsr_row_ptr_A           array of \p mb+1 elements that point to the start of every block row of the
*                          sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsr_col_ind_A           array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  row_block_dim_A         row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  col_block_dim_A         column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  descr_C                 the descriptor of the general BSR sparse matrix \f$C\f$, the supported matrix type is
*                          \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  bsr_row_ptr_C           array of \p mb_C+1 elements that point to the start of every block row of the
*                          sparse general BSR matrix \f$C\f$ where \p mb_C=(m+row_block_dim_C-1)/row_block_dim_C.
*  @param[in]
*  row_block_dim_C         row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  col_block_dim_C         column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  nnz_total_dev_host_ptr  total number of nonzero blocks in general BSR sparse matrix \f$C\f$ stored using device or host memory.
*  @param[out]
*  temp_buffer             buffer allocated by the user whose size is determined by calling
*                          \ref rocsparse_sgebsr2gebsr_buffer_size "rocsparse_Xgebsr2gebsr_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p nnzb or \p row_block_dim_A or
*              \p col_block_dim_A or \p row_block_dim_C or \p col_block_dim_C is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr_A or \p bsr_col_ind_A
*              or \p bsr_row_ptr_C or \p descr_A or \p descr_C or \p temp_buffer pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_gebsr2gebsr_nnz(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_int             mb,
                                           rocsparse_int             nb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr_A,
                                           const rocsparse_int*      bsr_row_ptr_A,
                                           const rocsparse_int*      bsr_col_ind_A,
                                           rocsparse_int             row_block_dim_A,
                                           rocsparse_int             col_block_dim_A,
                                           const rocsparse_mat_descr descr_C,
                                           rocsparse_int*            bsr_row_ptr_C,
                                           rocsparse_int             row_block_dim_C,
                                           rocsparse_int             col_block_dim_C,
                                           rocsparse_int*            nnz_total_dev_host_ptr,
                                           void*                     temp_buffer);

/*! \ingroup conv_module
*  \brief
*  This function converts the general BSR sparse matrix \f$A\f$ to another general BSR sparse matrix \f$C\f$.
*
*  \details
*  \p rocsparse_gebsr2gebsr converts a GEneral BSR matrix \f$A\f$ into a GEneral BSR matrix \f$C\f$. The input
*  and output matrices can have different row and column block dimensions. The input matrix \f$A\f$ is assumed
*  to be allocated such that array \p bsr_row_ptr_A has length \p mb+1, \p bsr_col_ind_A has length \p nnzb, and
*  \p bsr_val_A has length \p nnzb*row_block_dim_A*col_block_dim_A. The output matrix \f$C\f$ is assumed to be
*  allocated such that array \p bsr_row_ptr_C has length \p mb_C+1, \p bsr_col_ind_C has length \p nnzb_C, and
*  \p bsr_val_C has length \p nnzb_C*row_block_dim_C*col_block_dim_C where:
*  \f[
*    m = mb * row\_block\_dim\_A \\
*    n = nb * col\_block\_dim\_A
*  \f]
*  and
*  \f[
*    mb\_C = (m + row\_block\_dim\_C - 1) / row\_block\_dim\_C \\
*    nb\_C = (n + col\_block\_dim\_C - 1) / col\_block\_dim\_C
*  \f]
*  The number of non-zero blocks in the output sparse \f$C\f$ matrix (i.e. \p nnzb_C) is computed using
*  \ref rocsparse_gebsr2gebsr_nnz() which also fills in \p bsr_row_ptr_C array.
*
*  Converting from a sparse GEneral BSR matrix to a sparse GEneral BSR matrix requires three steps. First,
*  the user calls \ref rocsparse_sgebsr2gebsr_buffer_size "rocsparse_Xgebsr2gebsr_buffer_size()" in
*  order to determine the size of the required temporary storage buffer. Once this has been determined,
*  the user allocates this buffer. The user also now allocates the \p bsr_row_ptr_C array to have length
*  \p mb_C+1 and passes this to the function \ref rocsparse_gebsr2gebsr_nnz. This will fill the \p bsr_row_ptr_C
*  array and also compute the total number of nonzero blocks in the GEneral BSR output \f$C\f$ matrix. Now that
*  the total number of nonzero blocks is known, the user can allocate the \p bsr_col_ind_C and \p bsr_val_C arrays.
*  Finally, the user calls \p rocsparse_gebsr2gebsr to complete the conversion. Once the conversion is complete,
*  the temporary storage buffer can be deallocated. See example below.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle           handle to the rocsparse library context queue.
*  @param[in]
*  dir              the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb               number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb               number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb             number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descr_A          the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is
*                   \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  bsr_val_A        array of \p nnzb*row_block_dim_A*col_block_dim_A containing the values of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsr_row_ptr_A    array of \p mb+1 elements that point to the start of every block row of the
*                   sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsr_col_ind_A    array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  row_block_dim_A  row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  col_block_dim_A  column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  descr_C          the descriptor of the general BSR sparse matrix \f$C\f$, the supported matrix type is
*                   \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  bsr_val_C        array of \p nnzb_C*row_block_dim_C*col_block_dim_C containing the values of the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  bsr_row_ptr_C    array of \p mb_C+1 elements that point to the start of every block row of the
*                   sparse general BSR matrix \f$C\f$.
*  @param[in]
*  bsr_col_ind_C    array of \p nnzb_C elements containing the block column indices of the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  row_block_dim_C  row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  col_block_dim_C  column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  temp_buffer      buffer allocated by the user whose size is determined by calling
*                   \ref rocsparse_sgebsr2gebsr_buffer_size "rocsparse_Xgebsr2gebsr_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p nnzb or \p row_block_dim_A or
*              \p col_block_dim_A or \p row_block_dim_C or \p col_block_dim_C is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr_A or \p bsr_col_ind_A or \p bsr_val_A
*              or \p bsr_row_ptr_C or \p bsr_col_ind_C or \p bsr_val_C or \p descr_A or \p descr_C
*              or \p temp_buffer pointer is invalid.
*
*  \par Example
*  This example converts a GEneral BSR matrix into an GEneral BSR matrix.
*  \snippet example_rocsparse_gebsr2gebsr.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsr(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const rocsparse_mat_descr descr_A,
                                        const float*              bsr_val_A,
                                        const rocsparse_int*      bsr_row_ptr_A,
                                        const rocsparse_int*      bsr_col_ind_A,
                                        rocsparse_int             row_block_dim_A,
                                        rocsparse_int             col_block_dim_A,
                                        const rocsparse_mat_descr descr_C,
                                        float*                    bsr_val_C,
                                        rocsparse_int*            bsr_row_ptr_C,
                                        rocsparse_int*            bsr_col_ind_C,
                                        rocsparse_int             row_block_dim_C,
                                        rocsparse_int             col_block_dim_C,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsr(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const rocsparse_mat_descr descr_A,
                                        const double*             bsr_val_A,
                                        const rocsparse_int*      bsr_row_ptr_A,
                                        const rocsparse_int*      bsr_col_ind_A,
                                        rocsparse_int             row_block_dim_A,
                                        rocsparse_int             col_block_dim_A,
                                        const rocsparse_mat_descr descr_C,
                                        double*                   bsr_val_C,
                                        rocsparse_int*            bsr_row_ptr_C,
                                        rocsparse_int*            bsr_col_ind_C,
                                        rocsparse_int             row_block_dim_C,
                                        rocsparse_int             col_block_dim_C,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsr(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_mat_descr      descr_A,
                                        const rocsparse_float_complex* bsr_val_A,
                                        const rocsparse_int*           bsr_row_ptr_A,
                                        const rocsparse_int*           bsr_col_ind_A,
                                        rocsparse_int                  row_block_dim_A,
                                        rocsparse_int                  col_block_dim_A,
                                        const rocsparse_mat_descr      descr_C,
                                        rocsparse_float_complex*       bsr_val_C,
                                        rocsparse_int*                 bsr_row_ptr_C,
                                        rocsparse_int*                 bsr_col_ind_C,
                                        rocsparse_int                  row_block_dim_C,
                                        rocsparse_int                  col_block_dim_C,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsr(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_mat_descr       descr_A,
                                        const rocsparse_double_complex* bsr_val_A,
                                        const rocsparse_int*            bsr_row_ptr_A,
                                        const rocsparse_int*            bsr_col_ind_A,
                                        rocsparse_int                   row_block_dim_A,
                                        rocsparse_int                   col_block_dim_A,
                                        const rocsparse_mat_descr       descr_C,
                                        rocsparse_double_complex*       bsr_val_C,
                                        rocsparse_int*                  bsr_row_ptr_C,
                                        rocsparse_int*                  bsr_col_ind_C,
                                        rocsparse_int                   row_block_dim_C,
                                        rocsparse_int                   col_block_dim_C,
                                        void*                           temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GEBSR2GEBSR_H */
