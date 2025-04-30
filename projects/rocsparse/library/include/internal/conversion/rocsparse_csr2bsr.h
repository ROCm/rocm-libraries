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

#ifndef ROCSPARSE_CSR2BSR_H
#define ROCSPARSE_CSR2BSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \details
*  This function takes a sparse CSR matrix as input and computes the block row offset array, \p bsr_row_ptr, 
*  and the total number of nonzero blocks, \p bsr_nnz, that will result from converting the CSR format input 
*  matrix to a BSR format output matrix. This function is the first step in the conversion and is used in 
*  conjunction with \ref rocsparse_scsr2bsr "rocsparse_Xcsr2bsr()".
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  dir         direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*              \ref rocsparse_direction_column.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_row_ptr integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  block_dim   the block dimension of the BSR matrix. Between 1 and min(m, n)
*
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_row_ptr integer array containing \p mb+1 elements that point to the start of each block row of the BSR matrix
*
*  @param[out]
*  bsr_nnz     total number of nonzero elements in device or host memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr or \p csr_col_ind or \p bsr_row_ptr or \p bsr_nnz
*              pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2bsr_nnz(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr csr_descr,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_int             block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       rocsparse_int*            bsr_row_ptr,
                                       rocsparse_int*            bsr_nnz);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse BSR matrix
*
*  \details
*  \p rocsparse_csr2bsr converts a CSR matrix into a BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows 
*  and \p nb is the number of block columns in the BSR matrix:
*  \f[
*    mb = (m + block\_dim - 1) / block\_dim \\
*    nb = (n + block\_dim - 1) / block\_dim
*  \f]
*  Allocation size for \p bsr_val and \p bsr_col_ind is computed using \ref rocsparse_csr2bsr_nnz() 
*  which also fills in \p bsr_row_ptr.
*
*  Converting from a sparse CSR matrix to a sparse BSR matrix requires two steps. First, the user 
*  allocates the \p bsr_row_ptr array to have length \p mb+1 and passes this to the function 
*  \ref rocsparse_csr2bsr_nnz. This will fill the \p bsr_row_ptr array and also compute the total
*  number of nonzero blocks in the BSR matrix. Now that the total number of nonzero blocks is known,
*  the user can allocate the \p bsr_col_ind and \p bsr_val arrays. Finally, the user calls 
*  \p rocsparse_csr2bsr to complete the conversion. See example below.
*
*  \p rocsparse_csr2bsr requires extra temporary storage that is allocated internally if \p block_dim>16
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
*  dir          the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val      array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr  array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csr_col_ind  array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  block_dim    size of the blocks in the sparse BSR matrix.
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_val      array of \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsr_row_ptr  array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsr_col_ind  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a CSR matrix into an BSR matrix.
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      rocsparse_int m   = 4;
*      rocsparse_int n   = 6;
*      rocsparse_int block_dim = 2;
*      rocsparse_int nnz = 9;
*      rocsparse_int mb = (m + block_dim - 1) / block_dim;
*      rocsparse_int nb = (n + block_dim - 1) / block_dim;
*
*      csr_row_ptr[m+1]  = {0, 2, 4, 7, 9};             // device memory
*      csr_col_ind[nnz]  = {0, 1, 1, 2, 0, 3, 4, 2, 4}; // device memory
*      csr_val[nnz]      = {1, 4, 2, 3, 5, 7, 8, 9, 6}; // device memory
*
*      hipMalloc(&bsr_row_ptr, sizeof(rocsparse_int) *(mb + 1));
*      rocsparse_int nnzb;
*      rocsparse_int* nnzTotalHostPtr = &nnzb;
*      csr2bsr_nnz(handle,
*                  rocsparse_direction_row,
*                  m,
*                  n,
*                  csr_descr,
*                  csr_row_ptr,
*                  csr_col_ind,
*                  block_dim,
*                  bsr_descr,
*                  bsr_row_ptr,
*                  nnzTotalHostPtr);
*      nnzb = *nnzTotalDevHostPtr;
*      hipMalloc(&bsr_col_ind, sizeof(int)*nnzb);
*      hipMalloc(&bsr_val, sizeof(float)*(block_dim * block_dim) * nnzb);
*      scsr2bsr(handle,
*               rocsparse_direction_row,
*               m,
*               n,
*               csr_descr,
*               csr_val,
*               csr_row_ptr,
*               csr_col_ind,
*               block_dim,
*               bsr_descr,
*               bsr_val,
*               bsr_row_ptr,
*               bsr_col_ind);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2bsr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr csr_descr,
                                    const float*              csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr bsr_descr,
                                    float*                    bsr_val,
                                    rocsparse_int*            bsr_row_ptr,
                                    rocsparse_int*            bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2bsr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr csr_descr,
                                    const double*             csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr bsr_descr,
                                    double*                   bsr_val,
                                    rocsparse_int*            bsr_row_ptr,
                                    rocsparse_int*            bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2bsr(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_mat_descr      csr_descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_mat_descr      bsr_descr,
                                    rocsparse_float_complex*       bsr_val,
                                    rocsparse_int*                 bsr_row_ptr,
                                    rocsparse_int*                 bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2bsr(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_mat_descr       csr_descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_mat_descr       bsr_descr,
                                    rocsparse_double_complex*       bsr_val,
                                    rocsparse_int*                  bsr_row_ptr,
                                    rocsparse_int*                  bsr_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSR2BSR_H */
