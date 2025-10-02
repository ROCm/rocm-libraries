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

#ifndef ROCSPARSE_HYB2CSR_H
#define ROCSPARSE_HYB2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_hyb2csr_buffer_size returns the size of the temporary storage buffer
*  required by \ref rocsparse_shyb2csr "rocsparse_Xhyb2csr()". The temporary storage
*  buffer must be allocated by the user.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  descr           descriptor of the sparse HYB matrix. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  hyb             sparse matrix in HYB format.
*  @param[in]
*  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[out]
*  buffer_size     number of bytes of the temporary storage buffer required by
*                  \ref rocsparse_shyb2csr "rocsparse_Xhyb2csr()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p hyb, \p csr_row_ptr or
*              \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_hyb2csr_buffer_size(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               const rocsparse_int*      csr_row_ptr,
                                               size_t*                   buffer_size);

/*! \ingroup conv_module
*  \brief Convert a sparse HYB matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_hyb2csr converts a HYB matrix into a CSR matrix. This requires a HYB input structure,
*  \ref rocsparse_hyb_mat, which is created using \ref rocsparse_create_hyb_mat and is filled with data
*  using the conversion routine \ref rocsparse_scsr2hyb "rocsparse_Xcsr2hyb()".
*
*  Converting back to a sparse CSR matrix from a sparse HYB matrix requires two steps. First, the user calls
*  \ref rocsparse_hyb2csr_buffer_size in order to determine the size of the required temporary
*  storage buffer. Once this is determined, the user allocates this buffer. Finally, the user calls
*  \ref rocsparse_shyb2csr "rocsparse_Xhyb2csr()" to complete the conversion.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  descr           descriptor of the sparse HYB matrix. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  hyb             sparse matrix in HYB format.
*  @param[out]
*  csr_val         array containing the values of the sparse CSR matrix.
*  @param[out]
*  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[out]
*  csr_col_ind     array containing the column indices of the sparse CSR matrix.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned by
*                  \ref rocsparse_hyb2csr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p hyb, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example converts a HYB matrix into a CSR matrix.
*  \snippet example_rocsparse_hyb2csr.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_shyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    float*                    csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dhyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    double*                   csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_chyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    rocsparse_float_complex*  csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zhyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    rocsparse_double_complex* csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_HYB2CSR_H */
