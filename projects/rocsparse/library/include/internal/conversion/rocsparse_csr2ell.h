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

#ifndef ROCSPARSE_CSR2ELL_H
#define ROCSPARSE_CSR2ELL_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_csr2ell_width computes the maximum of the per row non-zero elements
*  over all rows, the \p ell_width, for a given CSR matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  ell_descr   descriptor of the sparse ELL matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  ell_width   pointer to the number of non-zero elements per row in ELL storage
*              format.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_descr, \p csr_row_ptr, or
*              \p ell_width pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2ell_width(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         const rocsparse_mat_descr csr_descr,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_mat_descr ell_descr,
                                         rocsparse_int*            ell_width);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse ELL matrix
*
*  \details
*  \p rocsparse_csr2ell converts a CSR matrix into an ELL matrix. It is assumed,
*  that \p ell_val and \p ell_col_ind are allocated. Allocation size is computed by the
*  number of rows times the number of ELL non-zero elements per row, such that
*  \f$\text{nnz}_{\text{ELL}} = m \cdot \text{ell_width}\f$. The number of ELL
*  non-zero elements per row is obtained by rocsparse_csr2ell_width().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val     array containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array containing the column indices of the sparse CSR matrix.
*  @param[in]
*  ell_descr   descriptor of the sparse ELL matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  ell_width   number of non-zero elements per row in ELL storage format.
*  @param[out]
*  ell_val     array of \p m times \p ell_width elements of the sparse ELL matrix.
*  @param[out]
*  ell_col_ind array of \p m times \p ell_width elements containing the column indices
*              of the sparse ELL matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p ell_width is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_descr, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p ell_descr, \p ell_val or
*              \p ell_col_ind pointer is invalid.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example converts a CSR matrix into an ELL matrix.
*  \snippet example_rocsparse_csr2ell.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2ell(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    const rocsparse_mat_descr csr_descr,
                                    const float*              csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    float*                    ell_val,
                                    rocsparse_int*            ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2ell(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    const rocsparse_mat_descr csr_descr,
                                    const double*             csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    double*                   ell_val,
                                    rocsparse_int*            ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2ell(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    const rocsparse_mat_descr      csr_descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    const rocsparse_mat_descr      ell_descr,
                                    rocsparse_int                  ell_width,
                                    rocsparse_float_complex*       ell_val,
                                    rocsparse_int*                 ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2ell(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    const rocsparse_mat_descr       csr_descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    const rocsparse_mat_descr       ell_descr,
                                    rocsparse_int                   ell_width,
                                    rocsparse_double_complex*       ell_val,
                                    rocsparse_int*                  ell_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSR2ELL_H */
