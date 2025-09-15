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

#ifndef ROCSPARSE_CHECK_MATRIX_CSC_H
#define ROCSPARSE_CHECK_MATRIX_CSC_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup utility_module
*  \details
*  \p rocsparse_check_matrix_csc_buffer_size computes the required buffer size needed when
*  calling \ref rocsparse_scheck_matrix_csc "rocsparse_Xcheck_matrix_csc()".
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSC matrix.
*  @param[in]
*  n           number of columns of the sparse CSC matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSC matrix.
*  @param[in]
*  csc_val     array of \p nnz elements of the sparse CSC matrix.
*  @param[in]
*  csc_col_ptr array of \p m+1 elements that point to the start of every column of the
*              sparse CSC matrix.
*  @param[in]
*  csc_row_ind array of \p nnz elements containing the row indices of the sparse
*              CSC matrix.
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
*              \ref rocsparse_scheck_matrix_csc "rocsparse_Xcheck_matrix_csc()".
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value \p idx_base or \p matrix_type or \p uplo or \p storage is invalid.
*  \retval rocsparse_status_invalid_size \p m \p n or \p nnz is invalid.
*  \retval rocsparse_status_invalid_pointer \p csc_val, \p csc_col_ptr, \p csc_row_ind or \p buffer_size pointer
*          is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scheck_matrix_csc_buffer_size(rocsparse_handle       handle,
                                                         rocsparse_int          m,
                                                         rocsparse_int          n,
                                                         rocsparse_int          nnz,
                                                         const float*           csc_val,
                                                         const rocsparse_int*   csc_col_ptr,
                                                         const rocsparse_int*   csc_row_ind,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_matrix_type  matrix_type,
                                                         rocsparse_fill_mode    uplo,
                                                         rocsparse_storage_mode storage,
                                                         size_t*                buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcheck_matrix_csc_buffer_size(rocsparse_handle       handle,
                                                         rocsparse_int          m,
                                                         rocsparse_int          n,
                                                         rocsparse_int          nnz,
                                                         const double*          csc_val,
                                                         const rocsparse_int*   csc_col_ptr,
                                                         const rocsparse_int*   csc_row_ind,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_matrix_type  matrix_type,
                                                         rocsparse_fill_mode    uplo,
                                                         rocsparse_storage_mode storage,
                                                         size_t*                buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccheck_matrix_csc_buffer_size(rocsparse_handle               handle,
                                                         rocsparse_int                  m,
                                                         rocsparse_int                  n,
                                                         rocsparse_int                  nnz,
                                                         const rocsparse_float_complex* csc_val,
                                                         const rocsparse_int*           csc_col_ptr,
                                                         const rocsparse_int*           csc_row_ind,
                                                         rocsparse_index_base           idx_base,
                                                         rocsparse_matrix_type          matrix_type,
                                                         rocsparse_fill_mode            uplo,
                                                         rocsparse_storage_mode         storage,
                                                         size_t* buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcheck_matrix_csc_buffer_size(rocsparse_handle                handle,
                                                         rocsparse_int                   m,
                                                         rocsparse_int                   n,
                                                         rocsparse_int                   nnz,
                                                         const rocsparse_double_complex* csc_val,
                                                         const rocsparse_int*   csc_col_ptr,
                                                         const rocsparse_int*   csc_row_ind,
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
*  \p rocsparse_check_matrix_csc checks if the input CSC matrix is valid. It performs basic sanity checks on the input
*  matrix and tries to detect issues in the data. This includes looking for 'nan' or 'inf' values in the data arrays,
*  invalid row indices or invalid column offsets, whether the matrix is triangular or not, whether there are duplicate row
*  indices or whether the row indices are not sorted when they should be. If an issue is found, it is written to the
*  \p data_status parameter.
*
*  Performing the above checks involves two steps. First the user calls \p rocsparse_Xcheck_matrix_csc_buffer_size in order
*  to determine the required buffer size. The user then allocates this buffer and passes it to \p rocsparse_Xcheck_matrix_csc.
*  Any issues detected will be written to the \p data_status parameter which is always a host variable regardless of pointer mode.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSC matrix.
*  @param[in]
*  n           number of columns of the sparse CSC matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSC matrix.
*  @param[in]
*  csc_val     array of \p nnz elements of the sparse CSC matrix.
*  @param[in]
*  csc_col_ptr array of \p m+1 elements that point to the start of every column of the
*              sparse CSC matrix.
*  @param[in]
*  csc_row_ind array of \p nnz elements containing the row indices of the sparse
*              CSC matrix.
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
*  \retval rocsparse_status_invalid_value \p idx_base or \p matrix_type or \p uplo or \p storage is invalid.
*  \retval rocsparse_status_invalid_size \p m \p n or \p nnz is invalid.
*  \retval rocsparse_status_invalid_pointer \p csc_val, \p csc_col_ptr, \p csc_row_ind, \p temp_buffer or \p data_status pointer
*          is invalid.
*
*  \par Example
*  In this example we want to check whether a CSC matrix has correct row indices. The matrix passed
*  is invalid because it contains a duplicate entry in the row indices array.
*
*  \code{.c}
*   // 1 2 0 0
*   // 0 3 4 0
*   // 2 0 1 1
*   // 0 3 0 2
*   std::vector<int> hcsc_row_ind = {0, 2, 0, 1, 1, 1, 2, 2, 3}; //<---duplicate row index in second column
*   std::vector<int> hcsc_col_ptr = {0, 2, 5, 7, 9};
*   std::vector<float> hcsc_val = {1, 2, 2, 3, 3, 4, 1, 1, 2};
*
*   int m = 4;
*   int n = 4;
*   int nnz = 9;
*
*   int* dcsc_row_ind = nullptr;
*   int* dcsc_col_ptr = nullptr;
*   float* dcsc_val = nullptr;
*   hipMalloc((void**)&dcsc_row_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dcsc_col_ptr, sizeof(int) * (n + 1));
*   hipMalloc((void**)&dcsc_val, sizeof(float) * nnz);
*
*   hipMemcpy(dcsc_row_ind, hcsc_row_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dcsc_col_ptr, hcsc_col_ptr.data(), sizeof(int) * (n + 1), hipMemcpyHostToDevice);
*   hipMemcpy(dcsc_val, hcsc_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*   rocsparse_handle handle;
*   rocsparse_create_handle(&handle);
*
*   const rocsparse_index_base idx_base = rocsparse_index_base_zero;
*   const rocsparse_fill_mode fill_mode = rocsparse_fill_mode_upper;
*   const rocsparse_matrix_type matrix_type = rocsparse_matrix_type_general;
*   const rocsparse_storage_mode storage_mode = rocsparse_storage_mode_sorted;
*
*   rocsparse_data_status data_status;
*
*   size_t buffer_size;
*   rocsparse_scheck_matrix_csc_buffer_size(handle, m, n, nnz, dcsc_val, dcsc_col_ptr, dcsc_row_ind,
*       idx_base, matrix_type, fill_mode, storage_mode, &buffer_size);
*
*   void* dbuffer = nullptr;
*   hipMalloc((void**)&dbuffer, buffer_size);
*
*   rocsparse_scheck_matrix_csc(handle, m, n, nnz, dcsc_val, dcsc_col_ptr, dcsc_row_ind, idx_base,
*       matrix_type, fill_mode, storage_mode, &data_status, dbuffer);
*
*   std::cout << "data_status: " << data_status << std::endl;
*
*   hipFree(dbuffer);
*
*   rocsparse_destroy_handle(handle);
*
*   hipFree(dcsc_row_ind);
*   hipFree(dcsc_col_ptr);
*   hipFree(dcsc_val);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scheck_matrix_csc(rocsparse_handle       handle,
                                             rocsparse_int          m,
                                             rocsparse_int          n,
                                             rocsparse_int          nnz,
                                             const float*           csc_val,
                                             const rocsparse_int*   csc_col_ptr,
                                             const rocsparse_int*   csc_row_ind,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_matrix_type  matrix_type,
                                             rocsparse_fill_mode    uplo,
                                             rocsparse_storage_mode storage,
                                             rocsparse_data_status* data_status,
                                             void*                  temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcheck_matrix_csc(rocsparse_handle       handle,
                                             rocsparse_int          m,
                                             rocsparse_int          n,
                                             rocsparse_int          nnz,
                                             const double*          csc_val,
                                             const rocsparse_int*   csc_col_ptr,
                                             const rocsparse_int*   csc_row_ind,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_matrix_type  matrix_type,
                                             rocsparse_fill_mode    uplo,
                                             rocsparse_storage_mode storage,
                                             rocsparse_data_status* data_status,
                                             void*                  temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccheck_matrix_csc(rocsparse_handle               handle,
                                             rocsparse_int                  m,
                                             rocsparse_int                  n,
                                             rocsparse_int                  nnz,
                                             const rocsparse_float_complex* csc_val,
                                             const rocsparse_int*           csc_col_ptr,
                                             const rocsparse_int*           csc_row_ind,
                                             rocsparse_index_base           idx_base,
                                             rocsparse_matrix_type          matrix_type,
                                             rocsparse_fill_mode            uplo,
                                             rocsparse_storage_mode         storage,
                                             rocsparse_data_status*         data_status,
                                             void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcheck_matrix_csc(rocsparse_handle                handle,
                                             rocsparse_int                   m,
                                             rocsparse_int                   n,
                                             rocsparse_int                   nnz,
                                             const rocsparse_double_complex* csc_val,
                                             const rocsparse_int*            csc_col_ptr,
                                             const rocsparse_int*            csc_row_ind,
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

#endif /* ROCSPARSE_CHECK_MATRIX_CSC_H */
