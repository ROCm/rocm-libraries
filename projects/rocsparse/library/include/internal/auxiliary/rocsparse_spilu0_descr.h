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
 *  \brief rocsparse_spilu0_descr.h provides auxilary functions in rocsparse
 *  but without using a rocsparse_handle.
 * Causing a disruption in the stream to use, as the default is the only available.
 */

#ifndef ROCSPARSE_SPILU0_DESCR_H
#define ROCSPARSE_SPILU0_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
*  \brief Create SpILU0 descriptor.
*
*  \details
*  \p rocsparse_spilu0_descr_create creates the descriptor of the configuration of the sparse Incomplete LU of level 0.

 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
*  @param[out]
*  p_spilu0_descr        pointer to the descriptor of the Spilu0 routine.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
*
*  \retval      rocsparse_status_invalid_handle \p handle pointer is invalid.
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_pointer \p descr pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spilu0_descr_create(rocsparse_handle        handle,
                                               rocsparse_spilu0_descr* p_spilu0_descr,
                                               rocsparse_error*        p_error);

/*! \ingroup aux_module
*  \brief Destroy SpILU0 descriptor.
*
*  \details
*  \p rocsparse_spilu0_descr_destroy destroys the descriptor of the configuration of the sparse Incomplete LU of level 0.
*
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
*  @param[in]
*  spilu0_descr        descriptor of the spilu0 routine.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
*  \retval      rocsparse_status_invalid_handle \p handle pointer is invalid.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spilu0_descr_destroy(rocsparse_handle       handle,
                                                rocsparse_spilu0_descr spilu0_descr,
                                                rocsparse_error*       p_error);

/*! \ingroup aux_module
 *  \brief Set the requested \ref rocsparse_spilu0_input data in the SpILU0 descriptor
 *
 *  \note
 *  -     \ref rocsparse_spilu0_input_alg is \ref rocsparse_spilu0_alg, and can only be set before applying any phase.
 *  -     \ref rocsparse_spilu0_input_compute_datatype is \ref rocsparse_datatype, and can only be set before applying any phase. For now, it must be of value type of A.
 *  -     \ref rocsparse_spilu0_input_analysis_policy is \ref rocsparse_analysis_policy, can only be set before applying any phase.
 *  -     \ref rocsparse_spilu0_input_singularity_tolerance is a device/host double pointer. Its device mode is determined from the \ref rocsparse_handle. No batched tolerances can be specified.
 *  -     \ref rocsparse_spilu0_input_boost_enable is a host int32_t, 1 to enable, 0 to disable
 *  -     \ref rocsparse_spilu0_input_boost_value is a pointer to a scalar of value type of A. Its device mode is determined from the \ref rocsparse_handle. No batched boost values can be specified.
 *  -     \ref rocsparse_spilu0_input_boost_tolerance is a double pointer. Its device mode is determined from the \ref rocsparse_handle. No batched boost tolerances can be specified.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  spilu0_descr       the pointer to the SpILU0 descriptor.
 *  @param[in]
 *  spilu0_input       value of \ref rocsparse_spilu0_input.
 *  @param[in]
 *  input        input data
 *  @param[in]
 *  input_size_in_bytes   input data size in bytes.
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
rocsparse_status rocsparse_spilu0_set_input(rocsparse_handle       handle,
                                            rocsparse_spilu0_descr spilu0_descr,
                                            rocsparse_spilu0_input spilu0_input,
                                            const void*            input,
                                            size_t                 input_size_in_bytes,
                                            rocsparse_error*       p_error);

/*! \ingroup aux_module
 *  \brief Get the requested \ref rocsparse_spilu0_output data from the SpILU0 descriptor
 *  \note
 *  -     \ref rocsparse_spilu0_output_singularity is \ref rocsparse_singularity, it will be considered as an array of size batch_count.
 *  -     \ref rocsparse_spilu0_output_singularity_position is int64_t, it will be considered as an array of size batch_count.
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  spilu0_descr       the pointer to the SpILU0 descriptor.
 *  @param[in]
 *  spilu0_output      value of \ref rocsparse_spilu0_output.
 *  @param[out]
 *  output        output data
 *  @param[in]
 *  output_size_in_bytes   output data size in bytes.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p output is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size_in_bytes is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spilu0_get_output(rocsparse_handle        handle,
                                             rocsparse_spilu0_descr  spilu0_descr,
                                             rocsparse_spilu0_output spilu0_output,
                                             void*                   output,
                                             size_t                  output_size_in_bytes,
                                             rocsparse_error*        p_error);

#ifdef __cplusplus
}
#endif

#endif
