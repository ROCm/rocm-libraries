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

#ifndef ROCSPARSE_BSRPAD_VALUE_H
#define ROCSPARSE_BSRPAD_VALUE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Pads a value to the diagonal of the last block (if the last block is a diagonal block) in the sparse BSR matrix
*  when the matrix expands outside m x m
*
*  \details When converting from a CSR matrix to a BSR matrix the resulting BSR matrix will be larger when m < mb * block_dim.
*  In these situations, the CSR to BSR conversion will expand the BSR matrix to have zeros when outside m x m. This routine
*  converts the resulting BSR matrix to one that has a value on the last diagonal blocks diagonal if this last block is a diagonal
*  block in the BSR matrix.
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
*  m           number of rows of the sparse BSR matrix.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix.
*  @param[in]
*  value       scalar value that is set on the diagonal of the last block when the matrix expands outside of \p m x \p m
*  @param[in]
*  bsr_descr   descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[inout]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p mb, \p nnzb or \p block_dim is
*              invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_descr, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrpad_value(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             mb,
                                         rocsparse_int             nnzb,
                                         rocsparse_int             block_dim,
                                         float                     value,
                                         const rocsparse_mat_descr bsr_descr,
                                         float*                    bsr_val,
                                         const rocsparse_int*      bsr_row_ptr,
                                         const rocsparse_int*      bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrpad_value(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             mb,
                                         rocsparse_int             nnzb,
                                         rocsparse_int             block_dim,
                                         double                    value,
                                         const rocsparse_mat_descr bsr_descr,
                                         double*                   bsr_val,
                                         const rocsparse_int*      bsr_row_ptr,
                                         const rocsparse_int*      bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrpad_value(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             mb,
                                         rocsparse_int             nnzb,
                                         rocsparse_int             block_dim,
                                         rocsparse_float_complex   value,
                                         const rocsparse_mat_descr bsr_descr,
                                         rocsparse_float_complex*  bsr_val,
                                         const rocsparse_int*      bsr_row_ptr,
                                         const rocsparse_int*      bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrpad_value(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             mb,
                                         rocsparse_int             nnzb,
                                         rocsparse_int             block_dim,
                                         rocsparse_double_complex  value,
                                         const rocsparse_mat_descr bsr_descr,
                                         rocsparse_double_complex* bsr_val,
                                         const rocsparse_int*      bsr_row_ptr,
                                         const rocsparse_int*      bsr_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRPAD_VALUE_H */
