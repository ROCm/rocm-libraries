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

#ifndef ROCSPARSE_DENSE2COO_H
#define ROCSPARSE_DENSE2COO_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix \f$A\f$ in column-oriented dense format into a sparse matrix in COO format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are
*  filled in based on nnz_per_rows, which can be pre-computed with \ref rocsparse_snnz "rocsparse_Xnnz()".
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  m            number of rows of the column-oriented dense matrix \p A.
*  @param[in]
*  n            number of columns of the column-oriented dense matrix \p A.
*  @param[in]
*  descr        the descriptor of the column-oriented dense matrix \p A, the supported matrix type is
*               \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  A            column-oriented dense matrix of dimensions (\p ld, \p n)
*  @param[in]
*  ld           leading dimension of column-oriented dense matrix \p A.
*  @param[in]
*  nnz_per_rows array of size \p n containing the number of non-zero elements per row.
*  @param[out]
*  coo_val
*               array of nnz nonzero elements of matrix \p A.
*  @param[out]
*  coo_row_ind  integer array of nnz row indices of the non-zero elements of matrix \p A.
*  @param[out]
*  coo_col_ind  integer array of nnz column indices of the non-zero elements of matrix \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_rows or \p coo_val \p coo_col_ind or \p coo_row_ind
*              pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_dense2coo.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdense2coo(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      float*                    coo_val,
                                      rocsparse_int*            coo_row_ind,
                                      rocsparse_int*            coo_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddense2coo(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      double*                   coo_val,
                                      rocsparse_int*            coo_row_ind,
                                      rocsparse_int*            coo_col_ind);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdense2coo(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* A,
                                      rocsparse_int                  ld,
                                      const rocsparse_int*           nnz_per_rows,
                                      rocsparse_float_complex*       coo_val,
                                      rocsparse_int*                 coo_row_ind,
                                      rocsparse_int*                 coo_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdense2coo(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* A,
                                      rocsparse_int                   ld,
                                      const rocsparse_int*            nnz_per_rows,
                                      rocsparse_double_complex*       coo_val,
                                      rocsparse_int*                  coo_row_ind,
                                      rocsparse_int*                  coo_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_DENSE2COO_H */
