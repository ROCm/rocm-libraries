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
 *  \brief rocsparse_spmv_descr.h provides auxilary functions in rocsparse
 *  but without using a rocsparse_handle.
 * Causing a disruption in the stream to use, as the default is the only available.
 */

#ifndef ROCSPARSE_SPMV_DESCR_H
#define ROCSPARSE_SPMV_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
   *  \brief Sparse matrix spmv.
   *
   *  \details
   *  \p rocsparse_create_spmv_descr creates the descriptor of the \ref rocsparse_v2_spmv_buffer_size and
   *  \ref rocsparse_v2_spmv routines.

   *  @param[out]
   *  descr        pointer to the descriptor of the SpMV routine.
   *
   *  \retval      rocsparse_status_success the operation completed successfully.
   *  \retval      rocsparse_status_invalid_pointer \p descr pointer is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_spmv_descr(rocsparse_spmv_descr* descr);

/*! \ingroup aux_module
   *  \brief Sparse matrix spmv.
   *
   *  \details
   *  \p rocsparse_destroy_spmv_descr destroys the descriptor of the \ref rocsparse_v2_spmv_buffer_size and
   *  \ref rocsparse_v2_spmv routines.
   *
   *  @param[in]
   *  descr        descriptor of the v2_spmv routine.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spmv_descr(rocsparse_spmv_descr descr);

/*! \ingroup aux_module
   *  \brief Set the requested \ref rocsparse_spmv_input data in the SpMV descriptor
   *
   *  @param[in]
   *  handle      the pointer to the handle to the rocSPARSE library context.
   *  @param[inout]
   *  descr       the pointer to the SpMV descriptor.
   *  @param[in]
   *  input       one possible value of \ref rocsparse_spmv_input
   *  @param[in]
   *  in          input value
   *  @param[in]
   *  size_in_bytes input value size in bytes.
   *  @param[out]
   *  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_pointer if \p descr or \p in is invalid.
   *  \retval rocsparse_status_invalid_value if \p input is invalid.
   *  \retval rocsparse_status_invalid_size if \p size_in_bytes is zero.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmv_set_input(rocsparse_handle     handle,
                                          rocsparse_spmv_descr descr,
                                          rocsparse_spmv_input input,
                                          const void*          in,
                                          size_t               size_in_bytes,
                                          rocsparse_error*     error);

#ifdef __cplusplus
}
#endif

#endif
