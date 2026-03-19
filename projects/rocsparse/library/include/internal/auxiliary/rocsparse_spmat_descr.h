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
 *  \brief rocsparse_spmat_descr.h provides auxilary functions in rocsparse
 */

#ifndef ROCSPARSE_SPMAT_DESCR_H
#define ROCSPARSE_SPMAT_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Destroy a sparse matrix descriptor
 *  \details
 *  \p rocsparse_spmat_descr_destroy destroys a sparse matrix descriptor.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_descr_destroy(rocsparse_handle      handle,
                                               rocsparse_spmat_descr descr,
                                               rocsparse_error*      p_error);

/*! \ingroup aux_module
 *  \brief Create a sparse matrix descriptor
 *  \details
 *  \p rocsparse_spmat_descr_create creates a sparse matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_spmat_descr_destroy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  p_descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  spattern      sparsity pattern of the sparse matrix descriptor.
 *  @param[in]
 *  values        values of the sparse matrix.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p row_data or \p col_data is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_descr_create(rocsparse_handle         handle,
                                              rocsparse_spmat_descr*   p_descr,
                                              rocsparse_spattern_descr spattern,
                                              rocsparse_dnvec_descr    values,
                                              rocsparse_error*         p_error);

/*! \ingroup aux_module
   *  \brief Get sparse matrix property.
   *
   *  \details
   *  \p rocsparse_spmat_get_prop gets the sparse matrix property.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  prop   select from \ref rocsparse_spmat_prop.
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
rocsparse_status rocsparse_spmat_get_prop(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr descr,
                                          rocsparse_spmat_prop        prop,
                                          void*                       value,
                                          size_t                      value_size_in_bytes,
                                          rocsparse_error*            p_error);

/*! \ingroup aux_module
   *  \brief Get the sparsity pattern from the sparse matrix.
   *
   *  \details
   *  \p rocsparse_spmat_get_spattern gets pointer to the sparsity pattern.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[out]
   *  p_value   get pointer to \ref rocsparse_spattern_descr.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_value is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_spattern(rocsparse_handle          handle,
                                              rocsparse_spmat_descr     descr,
                                              rocsparse_spattern_descr* p_value,
                                              rocsparse_error*          p_error);

/*! \ingroup aux_module
   *  \brief Set the sparsity pattern to the sparse matrix.
   *
   *  \details
   *  \p rocsparse_spmat_set_spattern sets pointer to the sparsity pattern.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[in]
   *  value   pointer to \ref rocsparse_spattern_descr.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p value is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_spattern(rocsparse_handle         handle,
                                              rocsparse_spmat_descr    descr,
                                              rocsparse_spattern_descr value,
                                              rocsparse_error*         p_error);

/*! \ingroup aux_module
   *  \brief Get sparse mstrix data.
   *
   *  \details
   *  \p rocsparse_spmat_get_data gets pointer to the sparse mstrix data.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[out]
   *  p_value   get pointer to \ref rocsparse_dnvec_descr.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p p_data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_data(rocsparse_handle       handle,
                                          rocsparse_spmat_descr  descr,
                                          rocsparse_dnvec_descr* p_value,
                                          rocsparse_error*       p_error);

/*! \ingroup aux_module
   *  \brief Set sparse mstrix data.
   *
   *  \details
   *  \p rocsparse_spattern_set_data sets pointer to the sparse mstrix data.
   *
   *  @param[in]
   *  handle  handle to the rocsparse library context queue.
   *  @param[in]
   *  descr   the matrix descriptor.
   *  @param[out]
   *  value   set pointer to \ref rocsparse_dnvec_descr.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle if \p handle is invalid.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_data(rocsparse_handle      handle,
                                          rocsparse_spmat_descr descr,
                                          rocsparse_dnvec_descr value,
                                          rocsparse_error*      p_error);

#ifdef __cplusplus
}
#endif

#endif
