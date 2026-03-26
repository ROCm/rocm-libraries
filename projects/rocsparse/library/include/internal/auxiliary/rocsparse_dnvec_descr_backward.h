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
 *  \brief rocsparse_dnvec_backward_descr.h provides auxiliary functions in rocsparse
 */

#ifndef ROCSPARSE_DNVEC_BACKWARD_DESCR_H
#define ROCSPARSE_DNVEC_BACKWARD_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 *  \brief Set the batch count and batch stride in the dense vector descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the dense vector descriptor.
 *  @param[in]
 *  batch_count  the batch count in the dense vector.
 *  @param[in]
 *  batch_stride the batch stride in the dense vector.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_set_strided_batch(rocsparse_dnvec_descr descr,
                                                   rocsparse_int         batch_count,
                                                   int64_t               batch_stride);

/*! \ingroup aux_module
 *  \brief Get the batch count and batch stride from the dense vector descriptor
 *
 *  @param[in]
 *  descr        the pointer to the dense vector descriptor.
 *  @param[out]
 *  batch_count  the batch count in the dense vector.
 *  @param[out]
 *  batch_stride the batch stride in the dense vector.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get_strided_batch(rocsparse_const_dnvec_descr descr,
                                                   rocsparse_int*              batch_count,
                                                   int64_t*                    batch_stride);

#ifdef __cplusplus
}
#endif

#endif
