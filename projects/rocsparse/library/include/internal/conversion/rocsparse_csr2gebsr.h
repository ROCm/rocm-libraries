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

#ifndef ROCSPARSE_CSR2GEBSR_H
#define ROCSPARSE_CSR2GEBSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \details
*  \p rocsparse_csr2gebsr_buffer_size returns the size of the temporary buffer that is required
*  by \ref rocsparse_csr2gebsr_nnz and \ref rocsparse_scsr2gebsr "rocsparse_Xcsr2gebsr()". The
*  temporary storage buffer must be allocated by the user.
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
*
*  @param[in]
*  dir         direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*              \ref rocsparse_direction_column.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*
*  @param[in]
*  csr_val      array of \p nnz elements containing the values of the sparse CSR matrix.
*
*  @param[in]
*  csr_row_ptr  integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind  integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  row_block_dim   the row block dimension of the GEneral BSR matrix. Between 1 and \p m
*
*  @param[in]
*  col_block_dim   the col block dimension of the GEneral BSR matrix. Between 1 and \p n
*
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer required by \ref rocsparse_csr2gebsr_nnz
*               and \ref rocsparse_scsr2gebsr "rocsparse_Xcsr2gebsr()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p row_block_dim  \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_val or \p csr_row_ptr or \p csr_col_ind or \p buffer_size
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const float*              csr_val,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_int             row_block_dim,
                                                  rocsparse_int             col_block_dim,
                                                  size_t*                   buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const double*             csr_val,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_int             row_block_dim,
                                                  rocsparse_int             col_block_dim,
                                                  size_t*                   buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2gebsr_buffer_size(rocsparse_handle               handle,
                                                  rocsparse_direction            dir,
                                                  rocsparse_int                  m,
                                                  rocsparse_int                  n,
                                                  const rocsparse_mat_descr      csr_descr,
                                                  const rocsparse_float_complex* csr_val,
                                                  const rocsparse_int*           csr_row_ptr,
                                                  const rocsparse_int*           csr_col_ind,
                                                  rocsparse_int                  row_block_dim,
                                                  rocsparse_int                  col_block_dim,
                                                  size_t*                        buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2gebsr_buffer_size(rocsparse_handle                handle,
                                                  rocsparse_direction             dir,
                                                  rocsparse_int                   m,
                                                  rocsparse_int                   n,
                                                  const rocsparse_mat_descr       csr_descr,
                                                  const rocsparse_double_complex* csr_val,
                                                  const rocsparse_int*            csr_row_ptr,
                                                  const rocsparse_int*            csr_col_ind,
                                                  rocsparse_int                   row_block_dim,
                                                  rocsparse_int                   col_block_dim,
                                                  size_t*                         buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \details
*  This function takes a sparse CSR matrix as input and computes the block row offset array, \p bsr_row_ptr,
*  and the total number of nonzero blocks, \p bsr_nnz_devhost, that will result from converting the CSR format
*  input matrix to a GEneral BSR format output matrix. This function is the second step in the conversion and
*  is used in conjunction with \ref rocsparse_scsr2gebsr_buffer_size "rocsparse_Xcsr2gebsr_buffer_size()" and
*  \ref rocsparse_scsr2gebsr "rocsparse_Xcsr2gebsr()".
*
*  \p rocsparse_csr2gebsr_nnz accepts both host and device pointers for \p bsr_nnz_devhost which can be set by
*  calling \ref rocsparse_set_pointer_mode prior to calling \p rocsparse_csr2gebsr_nnz.
*
*  \note
*  This function is blocking with respect to the host.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle           handle to the rocsparse library context queue.
*  @param[in]
*  dir              direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*                   \ref rocsparse_direction_column.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  n                number of columns of the sparse CSR matrix.
*  @param[in]
*  csr_descr        descriptor of the sparse CSR matrix. Currently, only
*                   \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_row_ptr      integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind      integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  bsr_descr        descriptor of the sparse GEneral BSR matrix. Currently, only
*                   \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_row_ptr      integer array containing \p mb+1 elements that point to the start of each block row of the
*                   General BSR matrix
*  @param[in]
*  row_block_dim    the row block dimension of the GEneral BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*  @param[in]
*  col_block_dim    the col block dimension of the GEneral BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*  @param[out]
*  bsr_nnz_devhost  total number of nonzero elements in device or host memory.
*  @param[in]
*  temp_buffer      buffer allocated by the user whose size is determined by calling
*                   \ref rocsparse_scsr2gebsr_buffer_size "rocsparse_Xcsr2gebsr_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p row_block_dim \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr or \p csr_col_ind or \p bsr_row_ptr or \p bsr_nnz_devhost
*              pointer is invalid.
*/
/**@{*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2gebsr_nnz(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const rocsparse_mat_descr csr_descr,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_int*      csr_col_ind,
                                         const rocsparse_mat_descr bsr_descr,
                                         rocsparse_int*            bsr_row_ptr,
                                         rocsparse_int             row_block_dim,
                                         rocsparse_int             col_block_dim,
                                         rocsparse_int*            bsr_nnz_devhost,
                                         void*                     temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEneral BSR matrix
*
*  \details
*  \p rocsparse_csr2gebsr converts a CSR matrix into a GEneral BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows
*  and \p nb is the number of block columns in the GEneral BSR matrix:
*  \f[
*    mb = (m + row\_block\_dim - 1) / row\_block\_dim \\
*    nb = (n + col\_block\_dim - 1) / col\_block\_dim
*  \f]
*  Allocation size for \p bsr_val and \p bsr_col_ind is computed using \ref rocsparse_csr2bsr_nnz()
*  which also fills in \p bsr_row_ptr.
*
*  Converting from a sparse CSR matrix to a sparse GEneral BSR matrix requires three steps. First,
*  the user calls \ref rocsparse_scsr2gebsr_buffer_size "rocsparse_Xcsr2gebsr_buffer_size()" in
*  order to determine the size of the required temporary storage buffer. Once this has been determined,
*  the user allocates this buffer. The user also now allocates the \p bsr_row_ptr array to have length
*  \p mb+1 and passes this to the function \ref rocsparse_csr2gebsr_nnz. This will fill the \p bsr_row_ptr
*  array and also compute the total number of nonzero blocks in the GEneral BSR matrix. Now that the total
*  number of nonzero blocks is known, the user can allocate the \p bsr_col_ind and \p bsr_val arrays.
*  Finally, the user calls \p rocsparse_csr2gebsr to complete the conversion. See example below.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle         handle to the rocsparse library context queue.
*  @param[in]
*  dir            the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  m              number of rows in the sparse CSR matrix.
*  @param[in]
*  n              number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_descr      descriptor of the sparse CSR matrix. Currently, only
*                 \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val        array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr    array of \p m+1 elements that point to the start of every row of the
*                 sparse CSR matrix.
*  @param[in]
*  csr_col_ind    array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  bsr_descr      descriptor of the sparse BSR matrix. Currently, only
*                 \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_val        array of \p nnzb* \p row_block_dim* \p col_block_dim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsr_row_ptr    array of \p mb+1 elements that point to the start of every block row of the
*                 sparse BSR matrix.
*  @param[out]
*  bsr_col_ind    array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  row_block_dim  row size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  col_block_dim  col size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  temp_buffer    buffer allocated by the user whose size is determined by calling
*                 \ref rocsparse_scsr2gebsr_buffer_size "rocsparse_Xcsr2gebsr_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p row_block_dim or \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a CSR matrix into an BSR matrix.
*  \snippet example_rocsparse_csr2gebsr.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2gebsr(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr csr_descr,
                                      const float*              csr_val,
                                      const rocsparse_int*      csr_row_ptr,
                                      const rocsparse_int*      csr_col_ind,
                                      const rocsparse_mat_descr bsr_descr,
                                      float*                    bsr_val,
                                      rocsparse_int*            bsr_row_ptr,
                                      rocsparse_int*            bsr_col_ind,
                                      rocsparse_int             row_block_dim,
                                      rocsparse_int             col_block_dim,
                                      void*                     temp_buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2gebsr(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr csr_descr,
                                      const double*             csr_val,
                                      const rocsparse_int*      csr_row_ptr,
                                      const rocsparse_int*      csr_col_ind,
                                      const rocsparse_mat_descr bsr_descr,
                                      double*                   bsr_val,
                                      rocsparse_int*            bsr_row_ptr,
                                      rocsparse_int*            bsr_col_ind,
                                      rocsparse_int             row_block_dim,
                                      rocsparse_int             col_block_dim,
                                      void*                     temp_buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2gebsr(rocsparse_handle               handle,
                                      rocsparse_direction            dir,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      csr_descr,
                                      const rocsparse_float_complex* csr_val,
                                      const rocsparse_int*           csr_row_ptr,
                                      const rocsparse_int*           csr_col_ind,
                                      const rocsparse_mat_descr      bsr_descr,
                                      rocsparse_float_complex*       bsr_val,
                                      rocsparse_int*                 bsr_row_ptr,
                                      rocsparse_int*                 bsr_col_ind,
                                      rocsparse_int                  row_block_dim,
                                      rocsparse_int                  col_block_dim,
                                      void*                          temp_buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2gebsr(rocsparse_handle                handle,
                                      rocsparse_direction             dir,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       csr_descr,
                                      const rocsparse_double_complex* csr_val,
                                      const rocsparse_int*            csr_row_ptr,
                                      const rocsparse_int*            csr_col_ind,
                                      const rocsparse_mat_descr       bsr_descr,
                                      rocsparse_double_complex*       bsr_val,
                                      rocsparse_int*                  bsr_row_ptr,
                                      rocsparse_int*                  bsr_col_ind,
                                      rocsparse_int                   row_block_dim,
                                      rocsparse_int                   col_block_dim,
                                      void*                           temp_buffer);

/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSR2GEBSR_H */
