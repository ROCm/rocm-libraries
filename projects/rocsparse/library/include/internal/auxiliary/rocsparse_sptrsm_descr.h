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
 *  \brief rocsparse_sptrsm_descr.h provides auxilary functions in rocsparse
 *  but without using a rocsparse_handle.
 * Causing a disruption in the stream to use, as the default is the only available.
 */

#ifndef ROCSPARSE_SPTRSM_DESCR_H
#define ROCSPARSE_SPTRSM_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 *  \brief Set the requested \ref rocsparse_sptrsm_input data in the SpTRSM descriptor
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  descr       the pointer to the SpTRSM descriptor.
 *  @param[in]
 *  input      value of \ref rocsparse_sptrsm_input.
 *  @param[in]
 *  data        input data
 *  @param[in]
 *  data_size   input data size.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p input is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsm_set_input(rocsparse_handle       handle,
                                            rocsparse_sptrsm_descr descr,
                                            rocsparse_sptrsm_input input,
                                            const void*            data,
                                            size_t                 data_size,
                                            rocsparse_error*       p_error);

/*! \ingroup aux_module
 *  \brief Get the requested \ref rocsparse_sptrsm_output data from the SpTRSM descriptor
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  descr       the pointer to the SpTRSM descriptor.
 *  @param[in]
 *  output      value of \ref rocsparse_sptrsm_output.
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
rocsparse_status rocsparse_sptrsm_get_output(rocsparse_handle        handle,
                                             rocsparse_sptrsm_descr  descr,
                                             rocsparse_sptrsm_output output,
                                             void*                   data,
                                             size_t                  data_size_in_bytes,
                                             rocsparse_error*        p_error);

#ifdef __cplusplus
}
#endif

#endif
