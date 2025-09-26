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

#ifndef ROCSPARSE_CHECK_SPMAT_H
#define ROCSPARSE_CHECK_SPMAT_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Check matrix to see if it is valid.
*
*  \details
*  \p rocsparse_check_spmat checks if the input matrix is valid.
*
*  \p rocsparse_check_spmat requires two steps to complete. First the user calls \p rocsparse_check_spmat
*  with the stage parameter set to \ref rocsparse_check_spmat_stage_buffer_size which determines the
*  size of the temporary buffer needed in the second step. The user allocates this buffer and calls
*  \p rocsparse_check_spmat with the stage parameter set to \ref rocsparse_check_spmat_stage_compute
*  which checks the input matrix for errors. Any detected errors in the input matrix are reported in the
*  \p data_status (passed to the function as a host pointer).
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="check_spmat_uniform">Uniform Precisions</caption>
*  <tr><th>A
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the checking operation, when stage is equal to
*  \ref rocsparse_check_spmat_stage_buffer_size.
*
*  \note
*  The sparse matrix formats currently supported are: rocsparse_format_coo, rocsparse_format_csr,
*  rocsparse_format_csc and rocsparse_format_ell.
*
*  \note check_spmat requires two stages to complete. The first stage
*  \ref rocsparse_check_spmat_stage_buffer_size will return the size of the temporary storage buffer
*  that is required for subsequent calls to \ref rocsparse_check_spmat.
*  In the final stage \ref rocsparse_check_spmat_stage_compute, the actual computation is performed.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  mat         matrix descriptor.
*  @param[out]
*  data_status modified to indicate the status of the data
*  @param[in]
*  stage       check_matrix stage for the matrix computation.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer. buffer_size is set when
*              \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user. When a nullptr is passed,
*              the required allocation size (in bytes) is written to \p buffer_size and
*              function returns without performing the checking operation.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p mat, \p buffer_size, \p temp_buffer or \p data_status pointer
*              is invalid.
*  \retval     rocsparse_status_invalid_value the value of stage is incorrect.
*
*  \par Example
*  In this example we want to check whether a matrix is upper triangular. The matrix passed to
*  \ref rocsparse_check_spmat is invalid because it contains an entry in the lower triangular
*  part of the matrix.
*  \snippet example_rocsparse_check_spmat.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_check_spmat(rocsparse_handle            handle,
                                       rocsparse_const_spmat_descr mat,
                                       rocsparse_data_status*      data_status,
                                       rocsparse_check_spmat_stage stage,
                                       size_t*                     buffer_size,
                                       void*                       temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CHECK_SPMAT_H */
