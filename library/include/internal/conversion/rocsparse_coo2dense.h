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

#ifndef ROCSPARSE_COO2DENSE_H
#define ROCSPARSE_COO2DENSE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in COO format into a column-oriented dense matrix.
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
*  m           number of rows of the column-oriented dense matrix \p A.
*  @param[in]
*  n           number of columns of the column-oriented dense matrix \p A.
*  @param[in]
*  nnz         number of non-zero entries of the sparse COO matrix.
*  @param[in]
*  descr       the descriptor of the column-oriented dense matrix \p A, the supported matrix type is 
*              \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  coo_val     array of nnz nonzero elements of matrix \p A.
*  @param[in]
*  coo_row_ind integer array of nnz row indices of the non-zero elements of matrix \p A.
*  @param[in]
*  coo_col_ind integer array of nnz column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[out]
*  ld          leading dimension of column-oriented dense matrix \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p coo_val \p coo_col_ind or \p coo_row_ind
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // 1 2 3 0
*    // 0 0 4 5
*    // 0 6 0 0
*    // 7 0 0 8
*    rocsparse_int m = 4;
*    rocsparse_int n = 4;
*    rocsparse_int nnz = 8;
*    rocsparse_int ld = m;
*
*    std::vector<rocsparse_int> hcoo_row_ind = {0, 0, 0, 1, 1, 2, 3, 3};
*    std::vector<rocsparse_int> hcoo_col_ind = {0, 1, 2, 2, 3, 1, 0, 3};
*    std::vector<float> hcoo_val = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
*
*    rocsparse_int* dcoo_row_ind = nullptr;
*    rocsparse_int* dcoo_col_ind = nullptr;
*    float* dcoo_val = nullptr;
*    hipMalloc((void**)&dcoo_row_ind, sizeof(rocsparse_int) * nnz);
*    hipMalloc((void**)&dcoo_col_ind, sizeof(rocsparse_int) * nnz);
*    hipMalloc((void**)&dcoo_val, sizeof(float) * nnz);
*
*    hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcoo_col_ind, hcoo_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    float* ddense = nullptr;
*    hipMalloc((void**)&ddense, sizeof(float) * ld * n);
*
*    rocsparse_handle handle;
*    rocsparse_create_handle(&handle);
*
*    rocsparse_mat_descr descr;
*    rocsparse_create_mat_descr(&descr);
*
*    rocsparse_scoo2dense(handle,
*                        m,
*                        n,
*                        nnz,
*                        descr,
*                        dcoo_val,
*                        dcoo_row_ind,
*                        dcoo_col_ind,
*                        ddense,
*                        ld);
*
*    rocsparse_destroy_handle(handle);
*    rocsparse_destroy_mat_descr(descr);
*
*    hipFree(dcoo_row_ind);
*    hipFree(dcoo_col_ind);
*    hipFree(dcoo_val);
*
*    hipFree(ddense);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scoo2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      rocsparse_int             nnz,
                                      const rocsparse_mat_descr descr,
                                      const float*              coo_val,
                                      const rocsparse_int*      coo_row_ind,
                                      const rocsparse_int*      coo_col_ind,
                                      float*                    A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcoo2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      rocsparse_int             nnz,
                                      const rocsparse_mat_descr descr,
                                      const double*             coo_val,
                                      const rocsparse_int*      coo_row_ind,
                                      const rocsparse_int*      coo_col_ind,
                                      double*                   A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccoo2dense(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      rocsparse_int                  nnz,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* coo_val,
                                      const rocsparse_int*           coo_row_ind,
                                      const rocsparse_int*           coo_col_ind,
                                      rocsparse_float_complex*       A,
                                      rocsparse_int                  ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcoo2dense(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      rocsparse_int                   nnz,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* coo_val,
                                      const rocsparse_int*            coo_row_ind,
                                      const rocsparse_int*            coo_col_ind,
                                      rocsparse_double_complex*       A,
                                      rocsparse_int                   ld);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_COO2DENSE_H */
