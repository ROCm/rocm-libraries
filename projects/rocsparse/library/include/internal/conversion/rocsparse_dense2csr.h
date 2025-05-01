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

#ifndef ROCSPARSE_DENSE2CSR_H
#define ROCSPARSE_DENSE2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix \f$A\f$ in column-oriented dense format into a sparse matrix in CSR format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based 
*  on nnz_per_row, which can be pre-computed with \ref rocsparse_snnz "rocsparse_Xnnz()".
*
*  \note
*  This function is blocking with respect to the host.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle        handle to the rocsparse library context queue.
*  @param[in]
*  m             number of rows of the column-oriented dense matrix \p A.
*  @param[in]
*  n             number of columns of the column-oriented dense dense matrix \p A.
*  @param[in]
*  descr         the descriptor of the column-oriented dense matrix \p A, the supported matrix type is 
*                \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  A             column-oriented dense matrix of dimensions (\p ld, \p n)
*  @param[in]
*  ld            leading dimension of column-oriented dense matrix \p A.
*  @param[in]
*  nnz_per_rows  array of size \p n containing the number of non-zero elements per row.
*  @param[out]
*  csr_val       array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  csr_row_ptr   integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csr_col_ind   integer array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) column indices of the non-zero elements of 
*                matrix \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_rows or \p csr_val \p csr_row_ptr or \p csr_col_ind
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // Matrix descriptor
*      rocsparse_mat_descr descr;
*      rocsparse_create_mat_descr(&descr);
*
*      // Column-Oriented Dense matrix in column order
*      //     1 2 0 3 0
*      // A = 0 4 5 0 0
*      //     6 0 0 7 8
*      float hdense_A[15] = {1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};
*
*      rocsparse_int m         = 3;
*      rocsparse_int n         = 5;
*      rocsparse_direction dir = rocsparse_direction_row;
*
*      float* ddense_A = nullptr;
*      hipMalloc((void**)&ddense_A, sizeof(float) * m * n);
*      hipMemcpy(ddense_A, hdense_A, sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*      // Allocate memory for the nnz_per_row_columns array
*      rocsparse_int* dnnz_per_row;
*      hipMalloc((void**)&dnnz_per_row, sizeof(rocsparse_int) * m);
*
*      rocsparse_int nnz_A;
*      rocsparse_snnz(handle, dir, m, n, descr, ddense_A, m, dnnz_per_row, &nnz_A);
*
*      // Allocate sparse CSR matrix
*      rocsparse_int* dcsr_row_ptr = nullptr;
*      rocsparse_int* dcsr_col_ind = nullptr;
*      float* dcsr_val = nullptr;
*      hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz_A);
*      hipMalloc((void**)&dcsr_val, sizeof(float) * nnz_A);
*
*      rocsparse_sdense2csr(handle, m, n, descr, ddense_A, m, dnnz_per_row, dcsr_val, dcsr_row_ptr, dcsr_col_ind);
*
*      hipFree(dcsr_row_ptr);
*      hipFree(dcsr_col_ind);
*      hipFree(dcsr_val);
*      hipFree(dnnz_per_row);
*
*      rocsparse_destroy_mat_descr(descr);
*      rocsparse_destroy_handle(handle);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdense2csr(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      float*                    csr_val,
                                      rocsparse_int*            csr_row_ptr,
                                      rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddense2csr(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      double*                   csr_val,
                                      rocsparse_int*            csr_row_ptr,
                                      rocsparse_int*            csr_col_ind);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdense2csr(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* A,
                                      rocsparse_int                  ld,
                                      const rocsparse_int*           nnz_per_rows,
                                      rocsparse_float_complex*       csr_val,
                                      rocsparse_int*                 csr_row_ptr,
                                      rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdense2csr(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* A,
                                      rocsparse_int                   ld,
                                      const rocsparse_int*            nnz_per_rows,
                                      rocsparse_double_complex*       csr_val,
                                      rocsparse_int*                  csr_row_ptr,
                                      rocsparse_int*                  csr_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_DENSE2CSR_H */
