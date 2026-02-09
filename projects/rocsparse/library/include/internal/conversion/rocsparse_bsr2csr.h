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

#ifndef ROCSPARSE_BSR2CSR_H
#define ROCSPARSE_BSR2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \brief Convert a sparse BSR matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_bsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
*  for \p csr_row_ptr is \p m+1 where:
*  \f[
*    m = mb * block\_dim \\
*    n = nb * block\_dim
*  \f]
*  Allocation for \p csr_val and \p csr_col_ind is computed by the
*  the number of blocks in the BSR matrix multiplied by the block dimension squared:
*  \f[
*    nnz = nnzb * block\_dim * block\_dim
*  \f]
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
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb          number of block rows in the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse BSR matrix.
*  @param[in]
*  bsr_descr   descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  block_dim   size of the blocks in the sparse BSR matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val     array of \p nnzb*block_dim*block_dim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csr_row_ptr array of \p m+1 where \p m=mb*block_dim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csr_col_ind array of \p nnzb*block_dim*block_dim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a BSR matrix into an CSR matrix.
*  \snippet example_rocsparse_bsr2csr.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsr2csr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    const rocsparse_mat_descr bsr_descr,
                                    const float*              bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr csr_descr,
                                    float*                    csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsr2csr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    const rocsparse_mat_descr bsr_descr,
                                    const double*             bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr csr_descr,
                                    double*                   csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsr2csr(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    const rocsparse_mat_descr      bsr_descr,
                                    const rocsparse_float_complex* bsr_val,
                                    const rocsparse_int*           bsr_row_ptr,
                                    const rocsparse_int*           bsr_col_ind,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_mat_descr      csr_descr,
                                    rocsparse_float_complex*       csr_val,
                                    rocsparse_int*                 csr_row_ptr,
                                    rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsr2csr(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    const rocsparse_mat_descr       bsr_descr,
                                    const rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*            bsr_row_ptr,
                                    const rocsparse_int*            bsr_col_ind,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_mat_descr       csr_descr,
                                    rocsparse_double_complex*       csr_val,
                                    rocsparse_int*                  csr_row_ptr,
                                    rocsparse_int*                  csr_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSR2CSR_H */
