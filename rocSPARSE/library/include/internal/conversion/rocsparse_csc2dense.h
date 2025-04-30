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

#ifndef ROCSPARSE_CSC2DENSE_H
#define ROCSPARSE_CSC2DENSE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in CSC format into a column-oriented dense matrix.
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
*  descr       the descriptor of the column-oriented dense matrix \p A, the supported matrix type is 
*              \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  csc_val     array of nnz ( = \p csc_col_ptr[n] - \p csc_col_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csc_col_ptr integer array of \p n+1 elements that contains the start of every column and the end of the last 
*              column plus one.
*  @param[in]
*  csc_row_ind integer array of nnz ( = \p csc_col_ptr[n] - \p csc_col_ptr[0] ) column indices of the non-zero 
*              elements of matrix \p A.
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*  @param[out]
*  ld          leading dimension of column-oriented dense matrix \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p csc_val \p csc_col_ptr or \p csc_row_ind
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
*    std::vector<rocsparse_int> hcsc_col_ptr = {0, 2, 4, 6, 8};
*    std::vector<rocsparse_int> hcsc_row_ind = {0, 3, 0, 2, 0, 1, 1, 3};
*    std::vector<float> hcsc_val = {1.0f, 7.0f, 2.0f, 6.0f, 3.0f, 4.0f, 5.0f, 8.0f};
*
*    rocsparse_int* dcsc_col_ptr = nullptr;
*    rocsparse_int* dcsc_row_ind = nullptr;
*    float* dcsc_val = nullptr;
*    hipMalloc((void**)&dcsc_col_ptr, sizeof(rocsparse_int) * (n + 1));
*    hipMalloc((void**)&dcsc_row_ind, sizeof(rocsparse_int) * nnz);
*    hipMalloc((void**)&dcsc_val, sizeof(float) * nnz);
*
*    hipMemcpy(dcsc_col_ptr, hcsc_col_ptr.data(), sizeof(rocsparse_int) * (n + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsc_row_ind, hcsc_row_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsc_val, hcsc_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
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
*    rocsparse_scsc2dense(handle,
*                         m,
*                         n,
*                         descr,
*                         dcsc_val,
*                         dcsc_col_ptr,
*                         dcsc_row_ind,
*                         ddense,
*                         ld);
*
*    rocsparse_destroy_handle(handle);
*    rocsparse_destroy_mat_descr(descr);
*
*    hipFree(dcsc_col_ptr);
*    hipFree(dcsc_row_ind);
*    hipFree(dcsc_val);
*
*    hipFree(ddense);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsc2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              csc_val,
                                      const rocsparse_int*      csc_col_ptr,
                                      const rocsparse_int*      csc_row_ind,
                                      float*                    A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsc2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             csc_val,
                                      const rocsparse_int*      csc_col_ptr,
                                      const rocsparse_int*      csc_row_ind,
                                      double*                   A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsc2dense(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* csc_val,
                                      const rocsparse_int*           csc_col_ptr,
                                      const rocsparse_int*           csc_row_ind,
                                      rocsparse_float_complex*       A,
                                      rocsparse_int                  ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsc2dense(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* csc_val,
                                      const rocsparse_int*            csc_col_ptr,
                                      const rocsparse_int*            csc_row_ind,
                                      rocsparse_double_complex*       A,
                                      rocsparse_int                   ld);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSC2DENSE_H */
