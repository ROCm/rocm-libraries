/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocsparse_spmat_descr_backward.h provides auxilary functions in rocsparse only supported for backward compatibility.
 */

#ifndef ROCSPARSE_SPMAT_DESCR_BACKWARD_H
#define ROCSPARSE_SPMAT_DESCR_BACKWARD_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a sparse COO matrix descriptor
 *  \details
 *  \p rocsparse_create_coo_descr creates a sparse COO matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO matrix.
 *  @param[in]
 *  cols        number of columns in the COO matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO matrix.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  coo_row_ind,
                                            void*                  coo_col_ind,
                                            void*                  coo_val,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_coo_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  coo_row_ind,
                                                  const void*                  coo_col_ind,
                                                  const void*                  coo_val,
                                                  rocsparse_indextype          idx_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse COO AoS matrix descriptor
 *  \details
 *  \p rocsparse_create_coo_aos_descr creates a sparse COO AoS matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse COO AoS matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO AoS matrix.
 *  @param[in]
 *  cols        number of columns in the COO AoS matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO AoS matrix.
 *  @param[in]
 *  coo_ind     <row, column> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                nnz,
                                                void*                  coo_ind,
                                                void*                  coo_val,
                                                rocsparse_indextype    idx_type,
                                                rocsparse_index_base   idx_base,
                                                rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse BSR matrix descriptor
 *  \details
 *  \p rocsparse_create_bsr_descr creates a sparse BSR matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[in]
 *  mb           number of rows in the BSR matrix.
 *  @param[in]
 *  nb           number of columns in the BSR matrix
 *  @param[in]
 *  nnzb         number of non-zeros in the BSR matrix.
 *  @param[in]
 *  block_dir    direction of the internal block storage.
 *  @param[in]
 *  block_dim    dimension of the blocks.
 *  @param[in]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p mb+1 ).
 *  @param[in]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p nnzb ).
 *  @param[in]
 *  bsr_val      values of the BSR matrix (must be array of length \p nnzb * \p block_dim * \p block_dim ).
 *  @param[in]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p bsr_row_ptr or \p bsr_col_ind or \p bsr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p mb or \p nb or \p nnzb \p block_dim is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type or \p block_dir is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_bsr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                mb,
                                            int64_t                nb,
                                            int64_t                nnzb,
                                            rocsparse_direction    block_dir,
                                            int64_t                block_dim,
                                            void*                  bsr_row_ptr,
                                            void*                  bsr_col_ind,
                                            void*                  bsr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse CSR matrix descriptor
 *  \details
 *  \p rocsparse_create_csr_descr creates a sparse CSR matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSR matrix.
 *  @param[in]
 *  cols         number of columns in the CSR matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csr_row_ptr,
                                            void*                  csr_col_ind,
                                            void*                  csr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_csr_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csr_row_ptr,
                                                  const void*                  csr_col_ind,
                                                  const void*                  csr_val,
                                                  rocsparse_indextype          row_ptr_type,
                                                  rocsparse_indextype          col_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse CSC matrix descriptor
 *  \details
 *  \p rocsparse_create_csc_descr creates a sparse CSC matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSC matrix.
 *  @param[in]
 *  cols         number of columns in the CSC matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[in]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  row_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csc_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csc_col_ptr,
                                            void*                  csc_row_ind,
                                            void*                  csc_val,
                                            rocsparse_indextype    col_ptr_type,
                                            rocsparse_indextype    row_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_csc_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csc_col_ptr,
                                                  const void*                  csc_row_ind,
                                                  const void*                  csc_val,
                                                  rocsparse_indextype          col_ptr_type,
                                                  rocsparse_indextype          row_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_ell_descr creates a sparse ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the ELL matrix.
 *  @param[in]
 *  cols        number of columns in the ELL matrix
 *  @param[in]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_width   width of the ELL matrix.
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_width is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_ell_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            void*                  ell_col_ind,
                                            void*                  ell_val,
                                            int64_t                ell_width,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse blocked ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_bell_descr creates a sparse blocked ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  Currently the only routine that supports the Blocked ELL format is \ref rocsparse_spmm.
 *
 *  @param[out]
 *  descr         the pointer to the sparse blocked ELL matrix descriptor.
 *  @param[in]
 *  rows          number of rows in the blocked ELL matrix.
 *  @param[in]
 *  cols          number of columns in the blocked ELL matrix
 *  @param[in]
 *  ell_block_dir \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[in]
 *  ell_block_dim block dimension of the sparse blocked ELL matrix.
 *  @param[in]
 *  ell_cols      column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_col_ind   column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val       values of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  idx_type      \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base      \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type     \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_cols or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_bell_descr(rocsparse_spmat_descr* descr,
                                             int64_t                rows,
                                             int64_t                cols,
                                             rocsparse_direction    ell_block_dir,
                                             int64_t                ell_block_dim,
                                             int64_t                ell_cols,
                                             void*                  ell_col_ind,
                                             void*                  ell_val,
                                             rocsparse_indextype    idx_type,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_bell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   rocsparse_direction          ell_block_dir,
                                                   int64_t                      ell_block_dim,
                                                   int64_t                      ell_cols,
                                                   const void*                  ell_col_ind,
                                                   const void*                  ell_val,
                                                   rocsparse_indextype          idx_type,
                                                   rocsparse_index_base         idx_base,
                                                   rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a sparse matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_spmat_descr destroys a sparse matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  Currently the only routine that supports the Blocked ELL format is \ref rocsparse_spmm.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_const_spmat_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse COO matrix descriptor
 *  \details
 *  \p rocsparse_coo_get gets the fields of the sparse COO matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse COO matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse COO matrix.
 *  @param[out]
 *  cols        number of columns in the sparse COO matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse COO matrix.
 *  @param[out]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      coo_row_ind,
                                   void**                      coo_col_ind,
                                   void**                      coo_val,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_coo_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                coo_row_ind,
                                         const void**                coo_col_ind,
                                         const void**                coo_val,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse COO AoS matrix descriptor
 *  \details
 *  \p rocsparse_coo_aos_get gets the fields of the sparse COO AoS matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse COO AoS matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse COO AoS matrix.
 *  @param[out]
 *  cols        number of columns in the sparse COO AoS matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse COO AoS matrix.
 *  @param[out]
 *  coo_ind     <row, columns> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr,
                                       int64_t*                    rows,
                                       int64_t*                    cols,
                                       int64_t*                    nnz,
                                       void**                      coo_ind,
                                       void**                      coo_val,
                                       rocsparse_indextype*        idx_type,
                                       rocsparse_index_base*       idx_base,
                                       rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_coo_aos_get(rocsparse_const_spmat_descr descr,
                                             int64_t*                    rows,
                                             int64_t*                    cols,
                                             int64_t*                    nnz,
                                             const void**                coo_ind,
                                             const void**                coo_val,
                                             rocsparse_indextype*        idx_type,
                                             rocsparse_index_base*       idx_base,
                                             rocsparse_datatype*         data_type);

/**@}*/
/*! \ingroup aux_module
 *  \brief Get the fields of the sparse CSR matrix descriptor
 *  \details
 *  \p rocsparse_csr_get gets the fields of the sparse CSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSR matrix.
 *  @param[out]
 *  cols         number of columns in the CSR matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[out]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[out]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csr_row_ptr,
                                   void**                      csr_col_ind,
                                   void**                      csr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_csr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csr_row_ptr,
                                         const void**                csr_col_ind,
                                         const void**                csr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse CSC matrix descriptor
 *  \details
 *  \p rocsparse_csc_get gets the fields of the sparse CSC matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse CSC matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSC matrix.
 *  @param[out]
 *  cols         number of columns in the CSC matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[out]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[out]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  col_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  row_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csc_col_ptr,
                                   void**                      csc_row_ind,
                                   void**                      csc_val,
                                   rocsparse_indextype*        col_ptr_type,
                                   rocsparse_indextype*        row_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_csc_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csc_col_ptr,
                                         const void**                csc_row_ind,
                                         const void**                csc_val,
                                         rocsparse_indextype*        col_ptr_type,
                                         rocsparse_indextype*        row_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse ELL matrix descriptor
 *  \details
 *  \p rocsparse_ell_get gets the fields of the sparse ELL matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the ELL matrix.
 *  @param[out]
 *  cols        number of columns in the ELL matrix
 *  @param[out]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_width   width of the ELL matrix.
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_width is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ell_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   void**                      ell_col_ind,
                                   void**                      ell_val,
                                   int64_t*                    ell_width,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_ell_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         const void**                ell_col_ind,
                                         const void**                ell_val,
                                         int64_t*                    ell_width,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse blocked ELL matrix descriptor
 *  \details
 *  \p rocsparse_bell_get gets the fields of the sparse blocked ELL matrix descriptor
 *
 *  @param[in]
 *  descr         the pointer to the sparse blocked ELL matrix descriptor.
 *  @param[out]
 *  rows          number of rows in the blocked ELL matrix.
 *  @param[out]
 *  cols          number of columns in the blocked ELL matrix
 *  @param[out]
 *  ell_block_dir \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[out]
 *  ell_block_dim block dimension of the sparse blocked ELL matrix.
 *  @param[out]
 *  ell_cols      column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_col_ind   column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_val       values of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  idx_type      \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base      \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type     \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_cols or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_block_dim is invalid.
 *  \retval rocsparse_status_invalid_value if \p ell_block_dir or \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    rocsparse_direction*        ell_block_dir,
                                    int64_t*                    ell_block_dim,
                                    int64_t*                    ell_cols,
                                    void**                      ell_col_ind,
                                    void**                      ell_val,
                                    rocsparse_indextype*        idx_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_bell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          rocsparse_direction*        ell_block_dir,
                                          int64_t*                    ell_block_dim,
                                          int64_t*                    ell_cols,
                                          const void**                ell_col_ind,
                                          const void**                ell_val,
                                          rocsparse_indextype*        idx_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse BSR matrix descriptor
 *  \details
 *  \p rocsparse_bsr_get gets the fields of the sparse BSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[out]
 *  brows         number of rows in the BSR matrix.
 *  @param[out]
 *  bcols         number of columns in the BSR matrix
 *  @param[out]
 *  bnnz          number of non-zeros in the BSR matrix.
 *  @param[out]
 *  bdir          storage layout of the dense block matrices.
 *  @param[out]
 *  bdim          block dimension.
 *  @param[out]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p brows+1 ).
 *  @param[out]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p bnnz ).
 *  @param[out]
 *  bsr_val      values of the BSR matrix (must be array of length \p bnnz * \p bdim * \p bdim ).
 *  @param[out]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    brows,
                                   int64_t*                    bcols,
                                   int64_t*                    bnnz,
                                   rocsparse_direction*        bdir,
                                   int64_t*                    bdim,
                                   void**                      bsr_row_ptr,
                                   void**                      bsr_col_ind,
                                   void**                      bsr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_bsr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    brows,
                                         int64_t*                    bcols,
                                         int64_t*                    bnnz,
                                         rocsparse_direction*        bdir,
                                         int64_t*                    bdim,
                                         const void**                bsr_row_ptr,
                                         const void**                bsr_col_ind,
                                         const void**                bsr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the row indices, column indices and values array in the sparse COO matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val);

/*! \ingroup aux_module
 *  \brief Set the <row, column> indices and values array in the sparse COO AoS matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  coo_ind <row, column> indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the sparse CSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val);

/*! \ingroup aux_module
 *  \brief Set the column offsets, row indices and values array in the sparse CSC matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse vector descriptor.
 *  @param[in]
 *  csc_col_ptr column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val     values of the CSC matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val);

/*! \ingroup aux_module
 *  \brief Set the column indices and values array in the sparse ELL matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse vector descriptor.
 *  @param[in]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the sparse BSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p nnzb ).
 *  @param[in]
 *  bsr_val      values of the BSR matrix (must be array of length \p nnzb*block_dim*block_dim ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p bsr_row_ptr or \p bsr_col_ind or \p bsr_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 bsr_row_ptr,
                                            void*                 bsr_col_ind,
                                            void*                 bsr_val);

/*! \ingroup aux_module
 *  \brief Get the number of rows, columns and non-zeros from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse matrix.
 *  @param[out]
 *  cols        number of columns in the sparse matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_size(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz);

/*! \ingroup aux_module
 *  \brief Get the sparse matrix format from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  format      \ref rocsparse_format_coo or \ref rocsparse_format_coo_aos or
 *              \ref rocsparse_format_csr or \ref rocsparse_format_csc or
 *              \ref rocsparse_format_ell
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p format is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_format(rocsparse_const_spmat_descr descr,
                                            rocsparse_format*           format);

/*! \ingroup aux_module
 *  \brief Get the sparse matrix index base from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_index_base(rocsparse_const_spmat_descr descr,
                                                rocsparse_index_base*       idx_base);

/*! \ingroup aux_module
 *  \brief Get the values array from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr     the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  values    values array of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_spmat_get_values(rocsparse_const_spmat_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in the sparse matrix descriptor
 *
 *  @param[inout]
 *  descr     the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  values    values array of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the number of non-zeros from the sparse matrix descriptor
 *
 *  \note The returned number of non-zeros is the number of elements of the array of values of the sparse matrix.
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  nnz the number of non-zeros of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_nnz(rocsparse_const_spmat_descr descr, int64_t* nnz);

/*! \ingroup aux_module
 *  \brief Set the number of non-zeros in the sparse matrix descriptor
 *
 *  \note In the case of a sparse matrix with the format \ref rocsparse_format_bsr, \p nnz is the number of blocks.
 *  \note In the case of a sparse matrix with the format \ref rocsparse_format_ell, the operation will return an error.
 *  \note In the case of a sparse matrix with the format \ref rocsparse_format_bell, the operation will return an error.
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  nnz         number of non-zeros of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_nnz(rocsparse_spmat_descr descr, int64_t nnz);

/*! \ingroup aux_module
 *  \brief Get the strided batch count from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  batch_count batch_count of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_const_spmat_descr descr,
                                                   rocsparse_int*              batch_count);

/*! \ingroup aux_module
 *  \brief Set the strided batch count in the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  batch_count batch_count of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_strided_batch(rocsparse_spmat_descr descr,
                                                   rocsparse_int         batch_count);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the sparse COO matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  batch_count  batch_count of the sparse COO matrix.
 *  @param[in]
 *  batch_stride batch stride of the sparse COO matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, row offset batch stride and the column indices batch stride in the sparse CSR matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the sparse CSR matrix.
 *  @param[in]
 *  offsets_batch_stride        row offset batch stride of the sparse CSR matrix.
 *  @param[in]
 *  columns_values_batch_stride column indices batch stride of the sparse CSR matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p columns_values_batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               columns_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, column offset batch stride and the row indices batch stride in the sparse CSC matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the sparse CSC matrix.
 *  @param[in]
 *  offsets_batch_stride        column offset batch stride of the sparse CSC matrix.
 *  @param[in]
 *  rows_values_batch_stride    row indices batch stride of the sparse CSC matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p rows_values_batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               rows_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Get the requested attribute data from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  attribute \ref rocsparse_spmat_fill_mode or \ref rocsparse_spmat_diag_type or
 *            \ref rocsparse_spmat_matrix_type or \ref rocsparse_spmat_storage_mode
 *  @param[out]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p attribute is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_attribute(rocsparse_const_spmat_descr descr,
                                               rocsparse_spmat_attribute   attribute,
                                               void*                       data,
                                               size_t                      data_size);

/*! \ingroup aux_module
 *  \brief Set the requested attribute data in the sparse matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  attribute \ref rocsparse_spmat_fill_mode or \ref rocsparse_spmat_diag_type or
 *            \ref rocsparse_spmat_matrix_type or \ref rocsparse_spmat_storage_mode
 *  @param[in]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p attribute is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               const void*               data,
                                               size_t                    data_size);

#ifdef __cplusplus
}
#endif

#endif



#if 0


<<<<<<< HEAD
=======
/*! \ingroup aux_module
 *  \brief Get the fields of the sparse COO matrix descriptor
 *  \details
 *  \p rocsparse_coo_get gets the fields of the sparse COO matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse COO matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse COO matrix.
 *  @param[out]
 *  cols        number of columns in the sparse COO matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse COO matrix.
 *  @param[out]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      coo_row_ind,
                                   void**                      coo_col_ind,
                                   void**                      coo_val,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_coo_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                coo_row_ind,
                                         const void**                coo_col_ind,
                                         const void**                coo_val,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse COO AoS matrix descriptor
 *  \details
 *  \p rocsparse_coo_aos_get gets the fields of the sparse COO AoS matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse COO AoS matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse COO AoS matrix.
 *  @param[out]
 *  cols        number of columns in the sparse COO AoS matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse COO AoS matrix.
 *  @param[out]
 *  coo_ind     <row, columns> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr,
                                       int64_t*                    rows,
                                       int64_t*                    cols,
                                       int64_t*                    nnz,
                                       void**                      coo_ind,
                                       void**                      coo_val,
                                       rocsparse_indextype*        idx_type,
                                       rocsparse_index_base*       idx_base,
                                       rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_coo_aos_get(rocsparse_const_spmat_descr descr,
                                             int64_t*                    rows,
                                             int64_t*                    cols,
                                             int64_t*                    nnz,
                                             const void**                coo_ind,
                                             const void**                coo_val,
                                             rocsparse_indextype*        idx_type,
                                             rocsparse_index_base*       idx_base,
                                             rocsparse_datatype*         data_type);

/**@}*/
/*! \ingroup aux_module
 *  \brief Get the fields of the sparse CSR matrix descriptor
 *  \details
 *  \p rocsparse_csr_get gets the fields of the sparse CSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSR matrix.
 *  @param[out]
 *  cols         number of columns in the CSR matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[out]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[out]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csr_row_ptr,
                                   void**                      csr_col_ind,
                                   void**                      csr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_csr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csr_row_ptr,
                                         const void**                csr_col_ind,
                                         const void**                csr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse CSC matrix descriptor
 *  \details
 *  \p rocsparse_csc_get gets the fields of the sparse CSC matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse CSC matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSC matrix.
 *  @param[out]
 *  cols         number of columns in the CSC matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[out]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[out]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  col_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  row_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csc_col_ptr,
                                   void**                      csc_row_ind,
                                   void**                      csc_val,
                                   rocsparse_indextype*        col_ptr_type,
                                   rocsparse_indextype*        row_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_csc_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csc_col_ptr,
                                         const void**                csc_row_ind,
                                         const void**                csc_val,
                                         rocsparse_indextype*        col_ptr_type,
                                         rocsparse_indextype*        row_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse ELL matrix descriptor
 *  \details
 *  \p rocsparse_ell_get gets the fields of the sparse ELL matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the ELL matrix.
 *  @param[out]
 *  cols        number of columns in the ELL matrix
 *  @param[out]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_width   width of the ELL matrix.
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_width is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ell_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   void**                      ell_col_ind,
                                   void**                      ell_val,
                                   int64_t*                    ell_width,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_ell_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         const void**                ell_col_ind,
                                         const void**                ell_val,
                                         int64_t*                    ell_width,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse blocked ELL matrix descriptor
 *  \details
 *  \p rocsparse_bell_get gets the fields of the sparse blocked ELL matrix descriptor
 *
 *  @param[in]
 *  descr         the pointer to the sparse blocked ELL matrix descriptor.
 *  @param[out]
 *  rows          number of rows in the blocked ELL matrix.
 *  @param[out]
 *  cols          number of columns in the blocked ELL matrix
 *  @param[out]
 *  ell_block_dir \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[out]
 *  ell_block_dim block dimension of the sparse blocked ELL matrix.
 *  @param[out]
 *  ell_cols      column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_col_ind   column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_val       values of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  idx_type      \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base      \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type     \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_cols or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_block_dim is invalid.
 *  \retval rocsparse_status_invalid_value if \p ell_block_dir or \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    rocsparse_direction*        ell_block_dir,
                                    int64_t*                    ell_block_dim,
                                    int64_t*                    ell_cols,
                                    void**                      ell_col_ind,
                                    void**                      ell_val,
                                    rocsparse_indextype*        idx_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_bell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          rocsparse_direction*        ell_block_dir,
                                          int64_t*                    ell_block_dim,
                                          int64_t*                    ell_cols,
                                          const void**                ell_col_ind,
                                          const void**                ell_val,
                                          rocsparse_indextype*        idx_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse sliced ELL matrix descriptor
 *  \details
 *  \p rocsparse_sell_get gets the fields of the sparse sliced ELL matrix descriptor
 *
 *  @param[in]
 *  descr                  the pointer to the sparse sliced ELL matrix descriptor.
 *  @param[out]
 *  rows                   number of rows in the sliced ELL matrix.
 *  @param[out]
 *  cols                   number of columns in the sliced ELL matrix
 *  @param[out]
 *  nnz                    number of non-zeros in the sliced ELL matix.
 *  @param[out]
 *  sell_slice_size        slice size in the sliced ELL matrix.
 *  @param[out]
 *  sell_colval_size       actual number of elements stored in the sliced ELL matrix.
 *  @param[out]
 *  sell_slice_offsets     slice offsets array in the sliced ELL matrix (must be array of length \p nslices + 1
 *                         where \p nslices=(rows-1)/sell_slice_size+1 ).
 *  @param[out]
 *  sell_col_ind            column indices of the sliced ELL matrix (must be array of length \p sell_colval_size ).
 *  @param[out]
 *  sell_val                values of the sliced ELL matrix (must be array of length \p sell_colval_size ).
 *  @param[out]
 *  sell_slice_offsets_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  sell_col_ind_type       \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base                \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type               \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                          \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p sell_slice_offsets or \p sell_col_ind or \p sell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz or \p sell_colval_size or \p sell_slice_size is invalid.
 *  \retval rocsparse_status_invalid_value if \p sell_slice_offsets_type or \p sell_col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    nnz,
                                    int64_t*                    sell_slice_size,
                                    int64_t*                    sell_colval_size,
                                    void**                      sell_slice_offsets,
                                    void**                      sell_col_ind,
                                    void**                      sell_val,
                                    rocsparse_indextype*        sell_slice_offsets_type,
                                    rocsparse_indextype*        sell_col_ind_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_sell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz,
                                          int64_t*                    sell_slice_size,
                                          int64_t*                    sell_colval_size,
                                          const void**                sell_slice_offsets,
                                          const void**                sell_col_ind,
                                          const void**                sell_val,
                                          rocsparse_indextype*        sell_slice_offsets_type,
                                          rocsparse_indextype*        sell_col_ind_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse BSR matrix descriptor
 *  \details
 *  \p rocsparse_bsr_get gets the fields of the sparse BSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[out]
 *  brows         number of rows in the BSR matrix.
 *  @param[out]
 *  bcols         number of columns in the BSR matrix
 *  @param[out]
 *  bnnz          number of non-zeros in the BSR matrix.
 *  @param[out]
 *  bdir          storage layout of the dense block matrices.
 *  @param[out]
 *  bdim          block dimension.
 *  @param[out]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p brows+1 ).
 *  @param[out]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p bnnz ).
 *  @param[out]
 *  bsr_val      values of the BSR matrix (must be array of length \p bnnz * \p bdim * \p bdim ).
 *  @param[out]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    brows,
                                   int64_t*                    bcols,
                                   int64_t*                    bnnz,
                                   rocsparse_direction*        bdir,
                                   int64_t*                    bdim,
                                   void**                      bsr_row_ptr,
                                   void**                      bsr_col_ind,
                                   void**                      bsr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_bsr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    brows,
                                         int64_t*                    bcols,
                                         int64_t*                    bnnz,
                                         rocsparse_direction*        bdir,
                                         int64_t*                    bdim,
                                         const void**                bsr_row_ptr,
                                         const void**                bsr_col_ind,
                                         const void**                bsr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the row indices, column indices and values array in the sparse COO matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val);

/*! \ingroup aux_module
 *  \brief Set the <row, column> indices and values array in the sparse COO AoS matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  coo_ind <row, column> indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the sparse CSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val);

/*! \ingroup aux_module
 *  \brief Set the column offsets, row indices and values array in the sparse CSC matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse vector descriptor.
 *  @param[in]
 *  csc_col_ptr column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val     values of the CSC matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val);

/*! \ingroup aux_module
 *  \brief Set the column indices and values array in the sparse ELL matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse vector descriptor.
 *  @param[in]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the sparse BSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p nnzb ).
 *  @param[in]
 *  bsr_val      values of the BSR matrix (must be array of length \p nnzb*block_dim*block_dim ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p bsr_row_ptr or \p bsr_col_ind or \p bsr_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 bsr_row_ptr,
                                            void*                 bsr_col_ind,
                                            void*                 bsr_val);

/*! \ingroup aux_module
 *  \brief Get the number of rows, columns and non-zeros from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse matrix.
 *  @param[out]
 *  cols        number of columns in the sparse matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_size(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz);

/*! \ingroup aux_module
 *  \brief Get the sparse matrix format from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  format      \ref rocsparse_format_coo or \ref rocsparse_format_coo_aos or
 *              \ref rocsparse_format_csr or \ref rocsparse_format_csc or
 *              \ref rocsparse_format_ell
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p format is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_format(rocsparse_const_spmat_descr descr,
                                            rocsparse_format*           format);

/*! \ingroup aux_module
 *  \brief Get the sparse matrix index base from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_index_base(rocsparse_const_spmat_descr descr,
                                                rocsparse_index_base*       idx_base);

/*! \ingroup aux_module
 *  \brief Get the values array from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr     the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  values    values array of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_spmat_get_values(rocsparse_const_spmat_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in the sparse matrix descriptor
 *
 *  @param[inout]
 *  descr     the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  values    values array of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the number of non-zeros from the sparse matrix descriptor
 *
 *  \note The returned number of non-zeros is the number of elements of the array of values of the sparse matrix.
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  nnz the number of non-zeros of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_nnz(rocsparse_const_spmat_descr descr, int64_t* nnz);

/*! \ingroup aux_module
 *  \brief Set the number of non-zeros in the sparse matrix descriptor
 *
 *  \note In the case of a sparse matrix with the format \ref rocsparse_format_bsr, \p nnz is the number of blocks.
 *  \note In the case of a sparse matrix with the format \ref rocsparse_format_ell, the operation will return an error.
 *  \note In the case of a sparse matrix with the format \ref rocsparse_format_bell, the operation will return an error.
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  nnz         number of non-zeros of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_nnz(rocsparse_spmat_descr descr, int64_t nnz);

/*! \ingroup aux_module
 *  \brief Get the strided batch count from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  batch_count batch_count of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_const_spmat_descr descr,
                                                   rocsparse_int*              batch_count);

/*! \ingroup aux_module
 *  \brief Set the strided batch count in the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  batch_count batch_count of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_strided_batch(rocsparse_spmat_descr descr,
                                                   rocsparse_int         batch_count);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the sparse COO matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  batch_count  batch_count of the sparse COO matrix.
 *  @param[in]
 *  batch_stride batch stride of the sparse COO matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, row offset batch stride and the column indices batch stride in the sparse CSR matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the sparse CSR matrix.
 *  @param[in]
 *  offsets_batch_stride        row offset batch stride of the sparse CSR matrix.
 *  @param[in]
 *  columns_values_batch_stride column indices batch stride of the sparse CSR matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p columns_values_batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               columns_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, column offset batch stride and the row indices batch stride in the sparse CSC matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the sparse CSC matrix.
 *  @param[in]
 *  offsets_batch_stride        column offset batch stride of the sparse CSC matrix.
 *  @param[in]
 *  rows_values_batch_stride    row indices batch stride of the sparse CSC matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p rows_values_batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               rows_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Get the requested attribute data from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  attribute \ref rocsparse_spmat_fill_mode or \ref rocsparse_spmat_diag_type or
 *            \ref rocsparse_spmat_matrix_type or \ref rocsparse_spmat_storage_mode
 *  @param[out]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p attribute is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_attribute(rocsparse_const_spmat_descr descr,
                                               rocsparse_spmat_attribute   attribute,
                                               void*                       data,
                                               size_t                      data_size);

/*! \ingroup aux_module
 *  \brief Set the requested attribute data in the sparse matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  attribute \ref rocsparse_spmat_fill_mode or \ref rocsparse_spmat_diag_type or
 *            \ref rocsparse_spmat_matrix_type or \ref rocsparse_spmat_storage_mode
 *  @param[in]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p attribute is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               const void*               data,
                                               size_t                    data_size);

/*! \ingroup aux_module
 *  \brief Create a dense vector descriptor
 *  \details
 *  \p rocsparse_create_dnvec_descr creates a dense vector descriptor. It should be
 *  destroyed at the end using rocsparse_destroy_dnvec_descr().
 *
 *  @param[out]
 *  descr   the pointer to the dense vector descriptor.
 *  @param[in]
 *  size   size of the dense vector.
 *  @param[in]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr,
                                              int64_t                size,
                                              void*                  values,
                                              rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_dnvec_descr(rocsparse_const_dnvec_descr* descr,
                                                    int64_t                      size,
                                                    const void*                  values,
                                                    rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a dense vector descriptor
 *
 *  \details
 *  \p rocsparse_destroy_dnvec_descr destroys a dense vector descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_const_dnvec_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the dense vector descriptor
 *  \details
 *  \p rocsparse_dnvec_get gets the fields of the dense vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense vector descriptor.
 *  @param[out]
 *  size   size of the dense vector.
 *  @param[out]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get(const rocsparse_dnvec_descr descr,
                                     int64_t*                    size,
                                     void**                      values,
                                     rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnvec_get(rocsparse_const_dnvec_descr descr,
                                           int64_t*                    size,
                                           const void**                values,
                                           rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the values array from a dense vector descriptor
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *  @param[out]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get_values(const rocsparse_dnvec_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnvec_get_values(rocsparse_const_dnvec_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in a dense vector descriptor
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_set_values(rocsparse_dnvec_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Create a dense matrix descriptor
 *  \details
 *  \p rocsparse_create_dnmat_descr creates a dense matrix descriptor. It should be
 *  destroyed at the end using rocsparse_destroy_dnmat_descr().
 *
 *  @param[out]
 *  descr     the pointer to the dense matrix descriptor.
 *  @param[in]
 *  rows      number of rows in the dense matrix.
 *  @param[in]
 *  cols      number of columns in the dense matrix.
 *  @param[in]
 *  ld        leading dimension of the dense matrix.
 *  @param[in]
 *  values    non-zero values in the dense vector (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *  @param[in]
 *  data_type \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *            \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *  @param[in]
 *  order     \ref rocsparse_order_row or \ref rocsparse_order_column.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ld is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type or \p order is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr,
                                              int64_t                rows,
                                              int64_t                cols,
                                              int64_t                ld,
                                              void*                  values,
                                              rocsparse_datatype     data_type,
                                              rocsparse_order        order);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_dnmat_descr(rocsparse_const_dnmat_descr* descr,
                                                    int64_t                      rows,
                                                    int64_t                      cols,
                                                    int64_t                      ld,
                                                    const void*                  values,
                                                    rocsparse_datatype           data_type,
                                                    rocsparse_order              order);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a dense matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_dnmat_descr destroys a dense matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_const_dnmat_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the dense matrix descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense matrix descriptor.
 *  @param[out]
 *  rows   number of rows in the dense matrix.
 *  @param[out]
 *  cols   number of columns in the dense matrix.
 *  @param[out]
 *  ld        leading dimension of the dense matrix.
 *  @param[out]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *  @param[out]
 *  order     \ref rocsparse_order_row or \ref rocsparse_order_column.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ld is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type or \p order is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    ld,
                                     void**                      values,
                                     rocsparse_datatype*         data_type,
                                     rocsparse_order*            order);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnmat_get(rocsparse_const_dnmat_descr descr,
                                           int64_t*                    rows,
                                           int64_t*                    cols,
                                           int64_t*                    ld,
                                           const void**                values,
                                           rocsparse_datatype*         data_type,
                                           rocsparse_order*            order);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the values array from the dense matrix descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense matrix descriptor.
 *  @param[out]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get_values(const rocsparse_dnmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnmat_get_values(rocsparse_const_dnmat_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in a dense matrix descriptor
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_set_values(rocsparse_dnmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the batch count and batch stride from the dense matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the dense matrix descriptor.
 *  @param[out]
 *  batch_count  the batch count in the dense matrix.
 *  @param[out]
 *  batch_stride the batch stride in the dense matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get_strided_batch(rocsparse_const_dnmat_descr descr,
                                                   rocsparse_int*              batch_count,
                                                   int64_t*                    batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the dense matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the dense matrix descriptor.
 *  @param[in]
 *  batch_count  the batch count in the dense matrix.
 *  @param[in]
 *  batch_stride the batch stride in the dense matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_set_strided_batch(rocsparse_dnmat_descr descr,
                                                   rocsparse_int         batch_count,
                                                   int64_t               batch_stride);

>>>>>>> develop


/*! \ingroup aux_module
 *  \brief Create a sparse COO matrix descriptor
 *  \details
 *  \p rocsparse_create_coo_descr creates a sparse COO matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO matrix.
 *  @param[in]
 *  cols        number of columns in the COO matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO matrix.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  coo_row_ind,
                                            void*                  coo_col_ind,
                                            void*                  coo_val,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_coo_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  coo_row_ind,
                                                  const void*                  coo_col_ind,
                                                  const void*                  coo_val,
                                                  rocsparse_indextype          idx_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse COO AoS matrix descriptor
 *  \details
 *  \p rocsparse_create_coo_aos_descr creates a sparse COO AoS matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse COO AoS matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO AoS matrix.
 *  @param[in]
 *  cols        number of columns in the COO AoS matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO AoS matrix.
 *  @param[in]
 *  coo_ind     <row, column> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                nnz,
                                                void*                  coo_ind,
                                                void*                  coo_val,
                                                rocsparse_indextype    idx_type,
                                                rocsparse_index_base   idx_base,
                                                rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse BSR matrix descriptor
 *  \details
 *  \p rocsparse_create_bsr_descr creates a sparse BSR matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[in]
 *  mb           number of rows in the BSR matrix.
 *  @param[in]
 *  nb           number of columns in the BSR matrix
 *  @param[in]
 *  nnzb         number of non-zeros in the BSR matrix.
 *  @param[in]
 *  block_dir    direction of the internal block storage.
 *  @param[in]
 *  block_dim    dimension of the blocks.
 *  @param[in]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p mb+1 ).
 *  @param[in]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p nnzb ).
 *  @param[in]
 *  bsr_val      values of the BSR matrix (must be array of length \p nnzb * \p block_dim * \p block_dim ).
 *  @param[in]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p bsr_row_ptr or \p bsr_col_ind or \p bsr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p mb or \p nb or \p nnzb \p block_dim is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type or \p block_dir is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_bsr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                mb,
                                            int64_t                nb,
                                            int64_t                nnzb,
                                            rocsparse_direction    block_dir,
                                            int64_t                block_dim,
                                            void*                  bsr_row_ptr,
                                            void*                  bsr_col_ind,
                                            void*                  bsr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse CSR matrix descriptor
 *  \details
 *  \p rocsparse_create_csr_descr creates a sparse CSR matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSR matrix.
 *  @param[in]
 *  cols         number of columns in the CSR matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csr_row_ptr,
                                            void*                  csr_col_ind,
                                            void*                  csr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_csr_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csr_row_ptr,
                                                  const void*                  csr_col_ind,
                                                  const void*                  csr_val,
                                                  rocsparse_indextype          row_ptr_type,
                                                  rocsparse_indextype          col_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse CSC matrix descriptor
 *  \details
 *  \p rocsparse_create_csc_descr creates a sparse CSC matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSC matrix.
 *  @param[in]
 *  cols         number of columns in the CSC matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[in]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  row_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csc_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csc_col_ptr,
                                            void*                  csc_row_ind,
                                            void*                  csc_val,
                                            rocsparse_indextype    col_ptr_type,
                                            rocsparse_indextype    row_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_csc_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csc_col_ptr,
                                                  const void*                  csc_row_ind,
                                                  const void*                  csc_val,
                                                  rocsparse_indextype          col_ptr_type,
                                                  rocsparse_indextype          row_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_ell_descr creates a sparse ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the ELL matrix.
 *  @param[in]
 *  cols        number of columns in the ELL matrix
 *  @param[in]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_width   width of the ELL matrix.
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_width is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_ell_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            void*                  ell_col_ind,
                                            void*                  ell_val,
                                            int64_t                ell_width,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse blocked ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_bell_descr creates a sparse blocked ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  Currently the only routine that supports the Blocked ELL format is \ref rocsparse_spmm.
 *
 *  @param[out]
 *  descr         the pointer to the sparse blocked ELL matrix descriptor.
 *  @param[in]
 *  rows          number of rows in the blocked ELL matrix.
 *  @param[in]
 *  cols          number of columns in the blocked ELL matrix
 *  @param[in]
 *  ell_block_dir \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[in]
 *  ell_block_dim block dimension of the sparse blocked ELL matrix.
 *  @param[in]
 *  ell_cols      column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_col_ind   column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val       values of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  idx_type      \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base      \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type     \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_cols or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_bell_descr(rocsparse_spmat_descr* descr,
                                             int64_t                rows,
                                             int64_t                cols,
                                             rocsparse_direction    ell_block_dir,
                                             int64_t                ell_block_dim,
                                             int64_t                ell_cols,
                                             void*                  ell_col_ind,
                                             void*                  ell_val,
                                             rocsparse_indextype    idx_type,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_bell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   rocsparse_direction          ell_block_dir,
                                                   int64_t                      ell_block_dim,
                                                   int64_t                      ell_cols,
                                                   const void*                  ell_col_ind,
                                                   const void*                  ell_val,
                                                   rocsparse_indextype          idx_type,
                                                   rocsparse_index_base         idx_base,
                                                   rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse sliced ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_sell_descr creates a sparse slice ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  Currently the only routine that supports the sliced ELL format is \ref rocsparse_spmv.
 *
 *  @param[out]
 *  descr                   the pointer to the sparse sliced ELL matrix descriptor.
 *  @param[in]
 *  rows                    number of rows in the sliced ELL matrix.
 *  @param[in]
 *  cols                    number of columns in the sliced ELL matrix
 *  @param[in]
 *  nnz                     number of non-zeros in the sliced ELL matrix.
 *  @param[in]
 *  sell_slice_size         slice size in the sliced ELL matrix.
 *  @param[in]
 *  sell_colval_size        size of the column and value arrays in the sliced ELL matrix.
 *  @param[in]
 *  sell_slice_offsets      slice offsets into column and value arrays (must be array of length \p nslices+1 where \p nslice=m/sell_slice_size ).
 *  @param[in]
 *  sell_col_ind            column indices of the sliced ELL matrix (must be array of length \p sell_colval_size ).
 *  @param[in]
 *  sell_val                values of the sliced ELL matrix (must be array of length \p sell_colval_size ).
 *  @param[in]
 *  sell_slice_offsets_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  sell_col_ind_type       \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base                \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type               \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                          \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p sell_slice_offsets or \p sell_col_ind or \p sell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz pr \p sell_slice_size or \p sell_colval_size is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_sell_descr(rocsparse_spmat_descr* descr,
                                             int64_t                rows,
                                             int64_t                cols,
                                             int64_t                nnz,
                                             int64_t                sell_slice_size,
                                             int64_t                sell_colval_size,
                                             void*                  sell_slice_offsets,
                                             void*                  sell_col_ind,
                                             void*                  sell_val,
                                             rocsparse_indextype    sell_slice_offsets_type,
                                             rocsparse_indextype    sell_col_ind_type,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_sell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   int64_t                      nnz,
                                                   int64_t                      sell_slice_size,
                                                   int64_t                      sell_colval_size,
                                                   const void*                  sell_slice_offsets,
                                                   const void*                  sell_col_ind,
                                                   const void*                  sell_val,
                                                   rocsparse_indextype  sell_slice_offsets_type,
                                                   rocsparse_indextype  sell_col_ind_type,
                                                   rocsparse_index_base idx_base,
                                                   rocsparse_datatype   data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a sparse matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_spmat_descr destroys a sparse matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  Currently the only routine that supports the Blocked ELL format is \ref rocsparse_spmm.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_const_spmat_descr descr);
#endif
