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

#ifndef ROCSPARSE_GEBSR2GEBSC_H
#define ROCSPARSE_GEBSR2GEBSC_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_gebsr2gebsc_buffer_size returns the size of the temporary storage buffer
*  required by \ref rocsparse_sgebsr2gebsc "rocsparse_Xgebsr2gebsc()".
*  The temporary storage buffer must be allocated by the user.
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
*  mb           number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb           number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb         number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb*row_block_dim*col_block_dim containing the values of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every row of the
*              sparse GEneral BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the column indices of the sparse
*              GEneral BSR matrix.
*  @param[in]
*  row_block_dim   row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  col_block_dim   col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  p_buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sgebsr2gebsc(), rocsparse_dgebsr2gebsc(), rocsparse_cgebsr2gebsc() and
*              rocsparse_zgebsr2gebsc().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb or \p nnzb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr, \p bsr_col_ind or
*              \p p_buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                    rocsparse_int        mb,
                                                    rocsparse_int        nb,
                                                    rocsparse_int        nnzb,
                                                    const float*         bsr_val,
                                                    const rocsparse_int* bsr_row_ptr,
                                                    const rocsparse_int* bsr_col_ind,
                                                    rocsparse_int        row_block_dim,
                                                    rocsparse_int        col_block_dim,
                                                    size_t*              p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                    rocsparse_int        mb,
                                                    rocsparse_int        nb,
                                                    rocsparse_int        nnzb,
                                                    const double*        bsr_val,
                                                    const rocsparse_int* bsr_row_ptr,
                                                    const rocsparse_int* bsr_col_ind,
                                                    rocsparse_int        row_block_dim,
                                                    rocsparse_int        col_block_dim,
                                                    size_t*              p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsc_buffer_size(rocsparse_handle               handle,
                                                    rocsparse_int                  mb,
                                                    rocsparse_int                  nb,
                                                    rocsparse_int                  nnzb,
                                                    const rocsparse_float_complex* bsr_val,
                                                    const rocsparse_int*           bsr_row_ptr,
                                                    const rocsparse_int*           bsr_col_ind,
                                                    rocsparse_int                  row_block_dim,
                                                    rocsparse_int                  col_block_dim,
                                                    size_t*                        p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsc_buffer_size(rocsparse_handle                handle,
                                                    rocsparse_int                   mb,
                                                    rocsparse_int                   nb,
                                                    rocsparse_int                   nnzb,
                                                    const rocsparse_double_complex* bsr_val,
                                                    const rocsparse_int*            bsr_row_ptr,
                                                    const rocsparse_int*            bsr_col_ind,
                                                    rocsparse_int                   row_block_dim,
                                                    rocsparse_int                   col_block_dim,
                                                    size_t*                         p_buffer_size);

/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p rocsparse_gebsr2gebsc converts a GEneral BSR matrix into a GEneral BSC matrix. The resulting
*  matrix can also be seen as the transpose of the input matrix. \p rocsparse_gebsr2gebsc can also
*  be used to convert a GEneral BSC matrix into a GEneral BSR matrix.
*
*  The conversion of a sparse matrix from GEneral BSR to GEneral BSC format involves two steps. First, the
*  user calls \ref rocsparse_sgebsr2gebsc_buffer_size "rocsparse_Xgebsr2gebsc_buffer_size()" in order to
*  determine the size of the required tempory storage buffer. The user then allocates this buffer. Secondly,
*  the user calls \p rocsparse_gebsr2gebsc to complete the conversion. Once the conversion is complete, the
*  user must free the temporary buffer.
*
*  \p rocsparse_gebsr2gebsc takes a \ref rocsparse_action parameter as input. This \p copy_values parameter
*  decides whether \p bsc_row_ind and \p bsc_val are filled during conversion (\ref rocsparse_action_numeric)
*  or whether only \p bsc_row_ind is filled (\ref rocsparse_action_symbolic). Using
*  \ref rocsparse_action_symbolic is useful for example if only the sparsity pattern is required.
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle         handle to the rocsparse library context queue.
*  @param[in]
*  mb             number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb             number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb           number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_val        array of \p nnzb * \p row_block_dim * \p col_block_dim  elements of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_row_ptr    array of \p mb+1 elements that point to the start of every row of the
*                 sparse GEneral BSR matrix.
*  @param[in]
*  bsr_col_ind    array of \p nnz elements containing the column indices of the sparse
*                 GEneral BSR matrix.
*  @param[in]
*  row_block_dim  row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  col_block_dim  col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  bsc_val        array of \p nnz elements of the sparse BSC matrix.
*  @param[out]
*  bsc_row_ind    array of \p nnz elements containing the row indices of the sparse BSC
*                 matrix.
*  @param[out]
*  bsc_col_ptr    array of \p nb+1 elements that point to the start of every column of the
*                 sparse BSC matrix.
*  @param[in]
*  copy_values    \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
*  @param[in]
*  idx_base       \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  temp_buffer    temporary storage buffer allocated by the user, size is returned by
*                 \ref rocsparse_sgebsr2gebsc_buffer_size "rocsparse_Xgebsr2gebsc_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb or \p nnzb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind, \p bsc_val, \p bsc_row_ind, \p bsc_col_ptr or
*              \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  This example computes the transpose of a GEneral BSR matrix.
*  \snippet example_rocsparse_gebsr2gebsc.cpp doc example
*/
/**@{*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsc(rocsparse_handle     handle,
                                        rocsparse_int        mb,
                                        rocsparse_int        nb,
                                        rocsparse_int        nnzb,
                                        const float*         bsr_val,
                                        const rocsparse_int* bsr_row_ptr,
                                        const rocsparse_int* bsr_col_ind,
                                        rocsparse_int        row_block_dim,
                                        rocsparse_int        col_block_dim,
                                        float*               bsc_val,
                                        rocsparse_int*       bsc_row_ind,
                                        rocsparse_int*       bsc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsc(rocsparse_handle     handle,
                                        rocsparse_int        mb,
                                        rocsparse_int        nb,
                                        rocsparse_int        nnzb,
                                        const double*        bsr_val,
                                        const rocsparse_int* bsr_row_ptr,
                                        const rocsparse_int* bsr_col_ind,
                                        rocsparse_int        row_block_dim,
                                        rocsparse_int        col_block_dim,
                                        double*              bsc_val,
                                        rocsparse_int*       bsc_row_ind,
                                        rocsparse_int*       bsc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsc(rocsparse_handle               handle,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  row_block_dim,
                                        rocsparse_int                  col_block_dim,
                                        rocsparse_float_complex*       bsc_val,
                                        rocsparse_int*                 bsc_row_ind,
                                        rocsparse_int*                 bsc_col_ptr,
                                        rocsparse_action               copy_values,
                                        rocsparse_index_base           idx_base,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsc(rocsparse_handle                handle,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   row_block_dim,
                                        rocsparse_int                   col_block_dim,
                                        rocsparse_double_complex*       bsc_val,
                                        rocsparse_int*                  bsc_row_ind,
                                        rocsparse_int*                  bsc_col_ptr,
                                        rocsparse_action                copy_values,
                                        rocsparse_index_base            idx_base,
                                        void*                           temp_buffer);

/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GEBSR2GEBSC_H */
