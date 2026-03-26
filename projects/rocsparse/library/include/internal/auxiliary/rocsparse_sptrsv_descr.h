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
 *  \brief rocsparse_sptrsv_descr.h provides auxiliary functions in rocsparse
 *  but without using a rocsparse_handle.
 * Causing a disruption in the stream to use, as the default is the only available.
 */

#ifndef ROCSPARSE_SPTRSV_DESCR_H
#define ROCSPARSE_SPTRSV_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
*  \brief Sparse matrix sptrsv.
*
*  \details
*  \p rocsparse_create_sptrsv_descr creates the descriptor of the \ref rocsparse_sptrsv_buffer_size and
*  \ref rocsparse_sptrsv routines.

 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
*  @param[out]
*  p_sptrsv_descr        pointer to the descriptor of the Sptrsv routine.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
*  \retval      rocsparse_status_invalid_handle \p handle pointer is invalid.
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_pointer \p descr pointer is invalid.
*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv_descr_create(rocsparse_handle        handle,
                                               rocsparse_sptrsv_descr* p_sptrsv_descr,
                                               rocsparse_error*        p_error);

/*! \ingroup aux_module
*  \brief Sparse matrix sptrsv.
*
*  \details
*  \p rocsparse_destroy_sptrsv_descr destroys the descriptor of the \ref rocsparse_sptrsv_buffer_size and
*  \ref rocsparse_sptrsv routines.
*
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
*  @param[in]
*  sptrsv_descr        descriptor of the sptrsv routine.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
*  \retval      rocsparse_status_invalid_handle \p handle pointer is invalid.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv_descr_destroy(rocsparse_handle       handle,
                                                rocsparse_sptrsv_descr sptrsv_descr,
                                                rocsparse_error*       p_error);

/*! \ingroup aux_module
 *  \brief Set the requested \ref rocsparse_sptrsv_input data in the SpTRSV descriptor
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  descr       the pointer to the SpTRSV descriptor.
 *  @param[in]
 *  input       value of \ref rocsparse_sptrsv_input.
 *  @param[in]
 *  data        input data
 *  @param[in]
 *  data_size_in_bytes   input data size in bytes.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p input is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size_in_bytes is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv_set_input(rocsparse_handle       handle,
                                            rocsparse_sptrsv_descr descr,
                                            rocsparse_sptrsv_input input,
                                            const void*            data,
                                            size_t                 data_size_in_bytes,
                                            rocsparse_error*       p_error);

/*! \ingroup aux_module
 *  \brief Get the requested \ref rocsparse_sptrsv_output data from the SpTRSV descriptor
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  descr       the pointer to the SpTRSV descriptor.
 *  @param[in]
 *  output      value of \ref rocsparse_sptrsv_output.
 *  @param[out]
 *  data        output data
 *  @param[in]
 *  data_size_in_bytes   output data size in bytes.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p output is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size_in_bytes is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv_get_output(rocsparse_handle        handle,
                                             rocsparse_sptrsv_descr  descr,
                                             rocsparse_sptrsv_output output,
                                             void*                   data,
                                             size_t                  data_size_in_bytes,
                                             rocsparse_error*        p_error);

#ifdef __cplusplus
}
#endif

#endif
