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

#ifndef ROCSPARSE_CHECK_MATRIX_GEBSR_H
#define ROCSPARSE_CHECK_MATRIX_GEBSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup utility_module
*  \brief Check matrix to see if it is valid.
*
*  \details
*  \p rocsparse_check_matrix_gebsr_buffer_size computes the required buffer size needed when
*  calling \ref rocsparse_scheck_matrix_gebsr "rocsparse_Xcheck_matrix_gebsr()".
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir          matrix storage of GEBSR blocks.
*  @param[in]
*  mb           number of block rows of the sparse GEBSR matrix.
*  @param[in]
*  nb           number of block columns of the sparse GEBSR matrix.
*  @param[in]
*  nnzb         number of non-zero blocks of the sparse GEBSR matrix.
*  @param[in]
*  row_block_dim row block dimension of the sparse GEBSR matrix.
*  @param[in]
*  col_block_dim column block dimension of the sparse GEBSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb elements of the sparse GEBSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every row of the
*              sparse GEBSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the column indices of the sparse
*              GEBSR matrix.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  matrix_type \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric,
*              \ref rocsparse_matrix_type_hermitian or \ref rocsparse_matrix_type_triangular.
*  @param[in]
*  uplo        \ref rocsparse_fill_mode_lower or \ref rocsparse_fill_mode_upper.
*  @param[in]
*  storage     \ref rocsparse_storage_mode_sorted or \ref rocsparse_storage_mode_sorted.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_scheck_matrix_gebsr(), rocsparse_dcheck_matrix_gebsr(),
*              rocsparse_ccheck_matrix_gebsr() and rocsparse_zcheck_matrix_gebsr().
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value \p dir or \p idx_base or \p matrix_type or \p uplo or \p storage is invalid.
*  \retval rocsparse_status_invalid_size \p mb \p nb \p nnzb \p row_block_dim or \p col_block_dim is invalid.
*  \retval rocsparse_status_invalid_pointer \p bsr_val, \p bsr_row_ptr, \p bsr_col_ind or \p buffer_size pointer
*          is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scheck_matrix_gebsr_buffer_size(rocsparse_handle       handle,
                                                           rocsparse_direction    dir,
                                                           rocsparse_int          mb,
                                                           rocsparse_int          nb,
                                                           rocsparse_int          nnzb,
                                                           rocsparse_int          row_block_dim,
                                                           rocsparse_int          col_block_dim,
                                                           const float*           bsr_val,
                                                           const rocsparse_int*   bsr_row_ptr,
                                                           const rocsparse_int*   bsr_col_ind,
                                                           rocsparse_index_base   idx_base,
                                                           rocsparse_matrix_type  matrix_type,
                                                           rocsparse_fill_mode    uplo,
                                                           rocsparse_storage_mode storage,
                                                           size_t*                buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcheck_matrix_gebsr_buffer_size(rocsparse_handle       handle,
                                                           rocsparse_direction    dir,
                                                           rocsparse_int          mb,
                                                           rocsparse_int          nb,
                                                           rocsparse_int          nnzb,
                                                           rocsparse_int          row_block_dim,
                                                           rocsparse_int          col_block_dim,
                                                           const double*          bsr_val,
                                                           const rocsparse_int*   bsr_row_ptr,
                                                           const rocsparse_int*   bsr_col_ind,
                                                           rocsparse_index_base   idx_base,
                                                           rocsparse_matrix_type  matrix_type,
                                                           rocsparse_fill_mode    uplo,
                                                           rocsparse_storage_mode storage,
                                                           size_t*                buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccheck_matrix_gebsr_buffer_size(rocsparse_handle    handle,
                                                           rocsparse_direction dir,
                                                           rocsparse_int       mb,
                                                           rocsparse_int       nb,
                                                           rocsparse_int       nnzb,
                                                           rocsparse_int       row_block_dim,
                                                           rocsparse_int       col_block_dim,
                                                           const rocsparse_float_complex* bsr_val,
                                                           const rocsparse_int*   bsr_row_ptr,
                                                           const rocsparse_int*   bsr_col_ind,
                                                           rocsparse_index_base   idx_base,
                                                           rocsparse_matrix_type  matrix_type,
                                                           rocsparse_fill_mode    uplo,
                                                           rocsparse_storage_mode storage,
                                                           size_t*                buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcheck_matrix_gebsr_buffer_size(rocsparse_handle    handle,
                                                           rocsparse_direction dir,
                                                           rocsparse_int       mb,
                                                           rocsparse_int       nb,
                                                           rocsparse_int       nnzb,
                                                           rocsparse_int       row_block_dim,
                                                           rocsparse_int       col_block_dim,
                                                           const rocsparse_double_complex* bsr_val,
                                                           const rocsparse_int*   bsr_row_ptr,
                                                           const rocsparse_int*   bsr_col_ind,
                                                           rocsparse_index_base   idx_base,
                                                           rocsparse_matrix_type  matrix_type,
                                                           rocsparse_fill_mode    uplo,
                                                           rocsparse_storage_mode storage,
                                                           size_t*                buffer_size);
/**@}*/

/*! \ingroup utility_module
*  \brief Check matrix to see if it is valid.
*
*  \details
*  \p rocsparse_check_matrix_gebsr checks if the input GEBSR matrix is valid. It performs basic sanity checks on the input
*  matrix and tries to detect issues in the data. This includes looking for 'nan' or 'inf' values in the data arrays,
*  invalid column indices and row offsets, whether the matrix is triangular or not, whether there are duplicate
*  indices or whether the column indices are not sorted when they should be. If an issue is found, it is written to the
*  \p data_status parameter.
*
*  Performing the above checks involves two steps. First the user calls \p rocsparse_Xcheck_matrix_gebsr_buffer_size in order
*  to determine the required buffer size. The user then allocates this buffer and passes it to \p rocsparse_Xcheck_matrix_gebsr.
*  Any issues detected will be written to the \p data_status parameter which is always a host variable regardless of pointer mode.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir          matrix storage of GEBSR blocks.
*  @param[in]
*  mb           number of block rows of the sparse GEBSR matrix.
*  @param[in]
*  nb           number of block columns of the sparse GEBSR matrix.
*  @param[in]
*  nnzb         number of non-zero blocks of the sparse GEBSR matrix.
*  @param[in]
*  row_block_dim row block dimension of the sparse GEBSR matrix.
*  @param[in]
*  col_block_dim column block dimension of the sparse GEBSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb elements of the sparse GEBSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every row of the
*              sparse GEBSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the column indices of the sparse
*              GEBSR matrix.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  matrix_type \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric,
*              \ref rocsparse_matrix_type_hermitian or \ref rocsparse_matrix_type_triangular.
*  @param[in]
*  uplo        \ref rocsparse_fill_mode_lower or \ref rocsparse_fill_mode_upper.
*  @param[in]
*  storage     \ref rocsparse_storage_mode_sorted or \ref rocsparse_storage_mode_sorted.
*  @param[out]
*  data_status modified to indicate the status of the data
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value \p dir or \p idx_base or \p matrix_type or \p uplo or \p storage is invalid.
*  \retval rocsparse_status_invalid_size \p mb \p nb \p nnzb \p row_block_dim or \p col_block_dim is invalid.
*  \retval rocsparse_status_invalid_pointer \p bsr_val, \p bsr_row_ptr, \p bsr_col_ind, \p temp_buffer or \p data_status pointer
*          is invalid.
*
*  \par Example
*  In this example we want to check whether a GEBSR matrix has valid values. The matrix passed
*  is invalid because it contains a nan entry in the values array.
*
*  \code{.c}
*   // 1 2 | 0 0
*   // 0 3 | 0 0
*   // ---------
*   // 4 5 | 7 8
*   // 0 6 | 0 9
*   std::vector<int> hbsr_row_ptr = {0, 1, 3};
*   std::vector<int> hbsr_col_ind = {0, 0, 1};
*   std::vector<float> hbsr_val = {1, 2, 0, 3, 4, 5, 0, 6, 7, 8, std::numeric_limits<double>::quiet_NaN(), 9}; //<---contains nan
*
*   int mb = 2;
*   int nb = 2;
*   int nnzb = 3;
*   int block_dim = 2;
*
*   int* dbsr_row_ptr = nullptr;
*   int* dbsr_col_ind = nullptr;
*   float* dbsr_val = nullptr;
*   hipMalloc((void**)&dbsr_row_ptr, sizeof(int) * (mb + 1));
*   hipMalloc((void**)&dbsr_col_ind, sizeof(int) * nnzb);
*   hipMalloc((void**)&dbsr_val, sizeof(float) * nnzb * block_dim * block_dim);
*
*   hipMemcpy(dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*   hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice);
*   hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(float) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice);
*
*   rocsparse_handle handle;
*   rocsparse_create_handle(&handle);
*
*   const rocsparse_direction direction = rocsparse_direction_row;
*   const rocsparse_index_base idx_base = rocsparse_index_base_zero;
*   const rocsparse_fill_mode fill_mode = rocsparse_fill_mode_upper;
*   const rocsparse_matrix_type matrix_type = rocsparse_matrix_type_triangular;
*   const rocsparse_storage_mode storage_mode = rocsparse_storage_mode_sorted;
*
*   rocsparse_data_status data_status;
*
*   size_t buffer_size;
*   rocsparse_scheck_matrix_gebsr_buffer_size(handle, direction, mb, nb, nnzb, block_dim, block_dim,
*       dbsr_val, dbsr_row_ptr, dbsr_col_ind, idx_base, matrix_type, fill_mode, storage_mode, &buffer_size);
*
*   void* dbuffer = nullptr;
*   hipMalloc((void**)&dbuffer, buffer_size);
*
*   rocsparse_scheck_matrix_gebsr(handle, direction, mb, nb, nnzb, block_dim, block_dim, dbsr_val, dbsr_row_ptr,
*       dbsr_col_ind, idx_base, matrix_type, fill_mode, storage_mode, &data_status, dbuffer);
*
*   std::cout << "data_status: " << data_status << std::endl;
*
*   hipFree(dbuffer);
*
*   rocsparse_destroy_handle(handle);
*
*   hipFree(dbsr_row_ptr);
*   hipFree(dbsr_col_ind);
*   hipFree(dbsr_val);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scheck_matrix_gebsr(rocsparse_handle       handle,
                                               rocsparse_direction    dir,
                                               rocsparse_int          mb,
                                               rocsparse_int          nb,
                                               rocsparse_int          nnzb,
                                               rocsparse_int          row_block_dim,
                                               rocsparse_int          col_block_dim,
                                               const float*           bsr_val,
                                               const rocsparse_int*   bsr_row_ptr,
                                               const rocsparse_int*   bsr_col_ind,
                                               rocsparse_index_base   idx_base,
                                               rocsparse_matrix_type  matrix_type,
                                               rocsparse_fill_mode    uplo,
                                               rocsparse_storage_mode storage,
                                               rocsparse_data_status* data_status,
                                               void*                  temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcheck_matrix_gebsr(rocsparse_handle       handle,
                                               rocsparse_direction    dir,
                                               rocsparse_int          mb,
                                               rocsparse_int          nb,
                                               rocsparse_int          nnzb,
                                               rocsparse_int          row_block_dim,
                                               rocsparse_int          col_block_dim,
                                               const double*          bsr_val,
                                               const rocsparse_int*   bsr_row_ptr,
                                               const rocsparse_int*   bsr_col_ind,
                                               rocsparse_index_base   idx_base,
                                               rocsparse_matrix_type  matrix_type,
                                               rocsparse_fill_mode    uplo,
                                               rocsparse_storage_mode storage,
                                               rocsparse_data_status* data_status,
                                               void*                  temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccheck_matrix_gebsr(rocsparse_handle               handle,
                                               rocsparse_direction            dir,
                                               rocsparse_int                  mb,
                                               rocsparse_int                  nb,
                                               rocsparse_int                  nnzb,
                                               rocsparse_int                  row_block_dim,
                                               rocsparse_int                  col_block_dim,
                                               const rocsparse_float_complex* bsr_val,
                                               const rocsparse_int*           bsr_row_ptr,
                                               const rocsparse_int*           bsr_col_ind,
                                               rocsparse_index_base           idx_base,
                                               rocsparse_matrix_type          matrix_type,
                                               rocsparse_fill_mode            uplo,
                                               rocsparse_storage_mode         storage,
                                               rocsparse_data_status*         data_status,
                                               void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcheck_matrix_gebsr(rocsparse_handle                handle,
                                               rocsparse_direction             dir,
                                               rocsparse_int                   mb,
                                               rocsparse_int                   nb,
                                               rocsparse_int                   nnzb,
                                               rocsparse_int                   row_block_dim,
                                               rocsparse_int                   col_block_dim,
                                               const rocsparse_double_complex* bsr_val,
                                               const rocsparse_int*            bsr_row_ptr,
                                               const rocsparse_int*            bsr_col_ind,
                                               rocsparse_index_base            idx_base,
                                               rocsparse_matrix_type           matrix_type,
                                               rocsparse_fill_mode             uplo,
                                               rocsparse_storage_mode          storage,
                                               rocsparse_data_status*          data_status,
                                               void*                           temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CHECK_MATRIX_GEBSR_H */
