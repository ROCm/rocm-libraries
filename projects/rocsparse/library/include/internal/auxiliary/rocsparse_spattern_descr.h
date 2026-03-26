/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 *  \brief rocsparse_spattern_descr.h provides auxiliary functions in rocsparse
 */

#ifndef ROCSPARSE_SPATTERN_DESCR_H
#define ROCSPARSE_SPATTERN_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Destroy a sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_destroy destroys a sparsity pattern descriptor.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[in]
 *  descr       the pointer to the sparsity pattern descriptor.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_destroy(rocsparse_handle         handle,
                                                  rocsparse_spattern_descr descr,
                                                  rocsparse_error*         p_error);

/*! \ingroup aux_module
 *  \brief Create a CSR sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_csr creates a sparsity pattern descriptor using the CSR format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSR matrix.
 *  @param[in]
 *  cols         number of columns in the CSR matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[in]
 *  row_data     row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  col_data  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_csr(rocsparse_handle          handle,
                                                     rocsparse_spattern_descr* p_descr,
                                                     int64_t                   rows,
                                                     int64_t                   cols,
                                                     int64_t                   nnz,
                                                     rocsparse_idvec_descr     row_data,
                                                     rocsparse_idvec_descr     col_data,
                                                     rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a sparse sliced ELL matrix descriptor
 *  \details
 *  \p rocsparse_descr_create_sell creates a sparse slice ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_spmat_descr_destroy.
 *
 *  Currently the only routine that supports the sliced ELL format is \ref rocsparse_spmv.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr                   the pointer to the sparse sliced ELL matrix descriptor.
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
 *  row_data     slice offsets into column and value arrays (must be array of length \p nslices+1 where \p nslice=m/sell_slice_size ).
 *  @param[in]
 *  col_data            column indices of the sliced ELL matrix (must be array of length \p sell_colval_size ).
 *  @param[out]
 *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p p_descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz pr \p sell_slice_size or \p sell_colval_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_sell(rocsparse_handle          handle,
                                                      rocsparse_spattern_descr* p_descr,
                                                      int64_t                   rows,
                                                      int64_t                   cols,
                                                      int64_t                   nnz,
                                                      int64_t                   sell_slice_size,
                                                      int64_t                   sell_colval_size,
                                                      rocsparse_idvec_descr     row_data,
                                                      rocsparse_idvec_descr     col_data,
                                                      rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a BSR sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_bsr creates a sparsity pattern descriptor using the BSR format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[in]
 *  rowsb         number of rows in the BSR matrix.
 *  @param[in]
 *  colsb         number of columns in the BSR matrix
 *  @param[in]
 *  nnzb          number of non-zeros in the BSR matrix.
 *  @param[in]
 *  block_direction block direction, or storage.
 *  @param[in]
 *  block_dim     dimension of the block
 *  @param[in]
 *  row_data  row offsets of the BSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  col_data  column indices of the BSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p p_descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_bsr(rocsparse_handle          handle,
                                                     rocsparse_spattern_descr* p_descr,
                                                     int64_t                   rowsb,
                                                     int64_t                   colsb,
                                                     int64_t                   nnzb,
                                                     rocsparse_direction       block_direction,
                                                     int64_t                   block_dim,
                                                     rocsparse_idvec_descr     row_data,
                                                     rocsparse_idvec_descr     col_data,
                                                     rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a CSC sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_csr creates a sparsity pattern descriptor using the CSC format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr        the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSC matrix.
 *  @param[in]
 *  cols         number of columns in the CSC matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[in]
 *  row_data     row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_data  column offsets of the CSC matrix (must be array of length \p cols+1 ).
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_csc(rocsparse_handle          handle,
                                                     rocsparse_spattern_descr* p_descr,
                                                     int64_t                   rows,
                                                     int64_t                   cols,
                                                     int64_t                   nnz,
                                                     rocsparse_idvec_descr     row_data,
                                                     rocsparse_idvec_descr     col_data,
                                                     rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a COO sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_csr creates a sparsity pattern descriptor using the COO format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr       the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO matrix.
 *  @param[in]
 *  cols        number of columns in the COO matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO matrix.
 *  @param[in]
 *  row_data row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_data column indices of the COO matrix (must be array of length \p nnz ).
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_coo(rocsparse_handle          handle,
                                                     rocsparse_spattern_descr* p_descr,
                                                     int64_t                   rows,
                                                     int64_t                   cols,
                                                     int64_t                   nnz,
                                                     rocsparse_idvec_descr     row_data,
                                                     rocsparse_idvec_descr     col_data,
                                                     rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a COOAOS sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_csr creates a sparsity pattern descriptor using the COOAOS format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr       the pointer to the sparse COOAOS matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COOAOS matrix.
 *  @param[in]
 *  cols        number of columns in the COOAOS matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COOAOS matrix.
 *  @param[in]
 *  row_data row indices of the COOAOS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_data column indices of the COOAOS matrix (must be array of length \p nnz ).
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_coo_aos(rocsparse_handle          handle,
                                                         rocsparse_spattern_descr* p_descr,
                                                         int64_t                   rows,
                                                         int64_t                   cols,
                                                         int64_t                   nnz,
                                                         rocsparse_idvec_descr     row_data,
                                                         rocsparse_idvec_descr     col_data,
                                                         rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a ELL sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_csr creates a sparsity pattern descriptor using the ELL format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the ELL matrix.
 *  @param[in]
 *  cols        number of columns in the ELL matrix
 *  @param[in]
 *  width         width of the ELLPACK Format.
 *  @param[in]
 *  col_data column indices of the ELL matrix (must be array of length \p nnz ).
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_ell(rocsparse_handle          handle,
                                                     rocsparse_spattern_descr* p_descr,
                                                     int64_t                   rows,
                                                     int64_t                   cols,
                                                     int64_t                   width,
                                                     rocsparse_idvec_descr     col_data,
                                                     rocsparse_error*          p_error);

/*! \ingroup aux_module
 *  \brief Create a BELL sparsity pattern descriptor
 *  \details
 *  \p rocsparse_spattern_descr_create_csr creates a sparsity pattern descriptor using the BELL format. It should be
 *  destroyed at the end using \p rocsparse_spattern_descr_destroy.
 *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
 *  @param[out]
 *  p_descr       the pointer to the sparse BELL matrix descriptor.
 *  @param[in]
 *  rowsb        number of rows in the BELL matrix.
 *  @param[in]
 *  colsb        number of columns in the BELL matrix
 *  @param[in]
 *  width         width of the BELLPACK Format.
 *  @param[in]
 *  block_direction block direction, or storage.
 *  @param[in]
 *  block_dim     dimension of the block
 *  @param[in]
 *  col_data column indices of the BELL matrix (must be array of length \p nnz ).
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_descr_create_bell(rocsparse_handle          handle,
                                                      rocsparse_spattern_descr* p_descr,
                                                      int64_t                   rowsb,
                                                      int64_t                   colsb,
                                                      int64_t                   width,
                                                      rocsparse_direction       block_direction,
                                                      int64_t                   block_dim,
                                                      rocsparse_idvec_descr     col_data,
                                                      rocsparse_error*          p_error);

/*! \ingroup aux_module
   *  \brief Get sparsity pattern property.
   *
   *  \details
   *  \p rocsparse_spattern_get_prop gets the sparsity pattern property.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  prop   select from \ref rocsparse_spattern_prop.
   *  @param[out]
   *  value   pointer to the value.
   *  @param[in]
   *  value_size_in_bytes size in bytes of the memory \p value points to, this must match the required size given from the underlying type given by the documentation of \ref rocsparse_idvec_prop.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p value is invalid.
   *  \retval rocsparse_status_invalid_value if \p prop is invalid or if \p value_size_in_bytes does not match the required size.
   */

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_get_prop(rocsparse_handle               handle,
                                             rocsparse_const_spattern_descr descr,
                                             rocsparse_spattern_prop        prop,
                                             void*                          value,
                                             size_t                         value_size_in_bytes,
                                             rocsparse_error*               p_error);

/*! \ingroup aux_module
   *  \brief Set sparsity pattern property.
   *
   *  \details
   *  \p rocsparse_spattern_set_prop sets the sparsity pattern property.
   *
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  prop   select from \ref rocsparse_spattern_prop.
   *  @param[out]
   *  value   pointer to the value.
   *  @param[in]
   *  value_size_in_bytes size in bytes of the memory \p value points to, this must match the required size given from the underlying type given by the documentation of \ref rocsparse_spattern_prop.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p value is invalid.
   *  \retval rocsparse_status_invalid_value if \p prop is invalid or if \p value_size_in_bytes does not match the required size.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_set_prop(rocsparse_handle         handle,
                                             rocsparse_spattern_descr descr,
                                             rocsparse_spattern_prop  prop,
                                             const void*              value,
                                             size_t                   value_size_in_bytes,
                                             rocsparse_error*         p_error);

/*! \ingroup aux_module
   *  \brief Get sparsity pattern data.
   *
   *  \details
   *  \p rocsparse_spattern_get_data gets pointer to the sparsity pattern data.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  spattern_data data selection.
   *  @param[out]
   *  p_data   get pointer to \ref rocsparse_idvec_descr.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_get_data(rocsparse_handle               handle,
                                             rocsparse_const_spattern_descr descr,
                                             rocsparse_spattern_data        spattern_data,
                                             rocsparse_idvec_descr*         p_data,
                                             rocsparse_error*               p_error);

/*! \ingroup aux_module
   *  \brief Set sparsity pattern data.
   *
   *  \details
   *  \p rocsparse_spattern_set_data sets pointer to the sparsity pattern data.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  spattern_data data selection.
   *  @param[out]
   *  data   set pointer to \ref rocsparse_idvec_descr.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_set_data(rocsparse_handle         handle,
                                             rocsparse_spattern_descr descr,
                                             rocsparse_spattern_data  spattern_data,
                                             rocsparse_idvec_descr    data,
                                             rocsparse_error*         p_error);

#ifdef __cplusplus
}
#endif

#endif
