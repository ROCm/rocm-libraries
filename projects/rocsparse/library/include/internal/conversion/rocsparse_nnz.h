/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_NNZ_H
#define ROCSPARSE_NNZ_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  This function computes the number of nonzero elements per row or column and the total number of nonzero elements
*  in a dense matrix.
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle                  handle to the rocsparse library context queue.
*  @param[in]
*  dir                     direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*                          \ref rocsparse_direction_column.
*  @param[in]
*  m                       number of rows of the dense matrix \p A.
*  @param[in]
*  n                       number of columns of the dense matrix \p A.
*  @param[in]
*  descr                   the descriptor of the dense matrix \p A.
*  @param[in]
*  A                       array of dimensions (\p ld, \p n)
*  @param[in]
*  ld                      leading dimension of dense array \p A.
*  @param[out]
*  nnz_per_row_columns     array of size \p m or \p n containing the number of nonzero elements per row or column, respectively.
*  @param[out]
*  nnz_total_dev_host_ptr  total number of nonzero elements in device or host memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_row_columns or \p nnz_total_dev_host_ptr
*              pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_nnz.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_snnz(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                const rocsparse_mat_descr descr,
                                const float*              A,
                                rocsparse_int             ld,
                                rocsparse_int*            nnz_per_row_columns,
                                rocsparse_int*            nnz_total_dev_host_ptr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnnz(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                const rocsparse_mat_descr descr,
                                const double*             A,
                                rocsparse_int             ld,
                                rocsparse_int*            nnz_per_row_columns,
                                rocsparse_int*            nnz_total_dev_host_ptr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cnnz(rocsparse_handle               handle,
                                rocsparse_direction            dir,
                                rocsparse_int                  m,
                                rocsparse_int                  n,
                                const rocsparse_mat_descr      descr,
                                const rocsparse_float_complex* A,
                                rocsparse_int                  ld,
                                rocsparse_int*                 nnz_per_row_columns,
                                rocsparse_int*                 nnz_total_dev_host_ptr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_znnz(rocsparse_handle                handle,
                                rocsparse_direction             dir,
                                rocsparse_int                   m,
                                rocsparse_int                   n,
                                const rocsparse_mat_descr       descr,
                                const rocsparse_double_complex* A,
                                rocsparse_int                   ld,
                                rocsparse_int*                  nnz_per_row_columns,
                                rocsparse_int*                  nnz_total_dev_host_ptr);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_NNZ_H */
