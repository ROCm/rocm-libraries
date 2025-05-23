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

#ifndef ROCSPARSE_BSR2CSR_H
#define ROCSPARSE_BSR2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \brief Convert a sparse BSR matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_bsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
*  for \p csr_row_ptr is \p m+1 where:
*  \f[
*    m = mb * block\_dim \\
*    n = nb * block\_dim
*  \f]
*  Allocation for \p csr_val and \p csr_col_ind is computed by the
*  the number of blocks in the BSR matrix multiplied by the block dimension squared:
*  \f[
*    nnz = nnzb * block\_dim * block\_dim
*  \f]
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
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb          number of block rows in the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse BSR matrix.
*  @param[in]
*  bsr_descr   descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  block_dim   size of the blocks in the sparse BSR matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val     array of \p nnzb*block_dim*block_dim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csr_row_ptr array of \p m+1 where \p m=mb*block_dim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csr_col_ind array of \p nnzb*block_dim*block_dim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a BSR matrix into an CSR matrix.
*  \code{.c}
*    //     1 4 2 1 0 0
*    // A = 0 2 3 5 0 0
*    //     5 2 2 7 8 6
*    //     9 3 9 1 6 1
*    rocsparse_int mb = 2;
*    rocsparse_int nb = 3;
*    rocsparse_int block_dim = 2;
*    rocsparse_int m = mb * block_dim;
*    rocsparse_int n = nb * block_dim;
*    rocsparse_int nnzb = 5;
*    rocsparse_int nnz = nnzb * block_dim * block_dim;
*
*    std::vector<rocsparse_int> hbsr_row_ptr = {0, 2, 5};
*    std::vector<rocsparse_int> hbsr_col_ind = {0, 1, 0, 1, 2};
*    std::vector<float> hbsr_val = {1.0f, 0.0f, 4.0f, 2.0f, 
*                                   2.0f, 3.0f, 1.0f, 5.0f, 
*                                   5.0f, 9.0f, 2.0f, 3.0f,
*                                   2.0f, 9.0f, 7.0f, 1.0f,
*                                   8.0f, 6.0f, 6.0f, 1.0f};
*
*    rocsparse_int* dbsr_row_ptr = nullptr;
*    rocsparse_int* dbsr_col_ind = nullptr;
*    float* dbsr_val = nullptr;
*    hipMalloc((void**)&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1));
*    hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb);
*    hipMalloc((void**)&dbsr_val, sizeof(float) * nnzb * block_dim * block_dim);
*
*    hipMemcpy(dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice);
*    hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(float) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice);
*
*    // Create CSR arrays on device
*    rocsparse_int* dcsr_row_ptr = nullptr;
*    rocsparse_int* dcsr_col_ind = nullptr;
*    float* dcsr_val = nullptr;
*    hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*    hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz);
*    hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*
*    // Create rocsparse handle
*    rocsparse_handle handle;
*    rocsparse_create_handle(&handle);
*
*    rocsparse_mat_descr bsr_descr = nullptr;
*    rocsparse_create_mat_descr(&bsr_descr);
*
*    rocsparse_mat_descr csr_descr = nullptr;
*    rocsparse_create_mat_descr(&csr_descr);
*
*    rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero);
*    rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero);
*
*    // Format conversion
*    rocsparse_sbsr2csr(handle,
*                        rocsparse_direction_column,
*                        mb,
*                        nb,
*                        bsr_descr,
*                        dbsr_val,
*                        dbsr_row_ptr,
*                        dbsr_col_ind,
*                        block_dim,
*                        csr_descr,
*                        dcsr_val,
*                        dcsr_row_ptr,
*                        dcsr_col_ind);
*
*    rocsparse_destroy_handle(handle);
*    rocsparse_destroy_mat_descr(csr_descr);
*    rocsparse_destroy_mat_descr(bsr_descr);
*
*    hipFree(dbsr_row_ptr);
*    hipFree(dbsr_col_ind);
*    hipFree(dbsr_val);
*
*    hipFree(dcsr_row_ptr);
*    hipFree(dcsr_col_ind);
*    hipFree(dcsr_val);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsr2csr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    const rocsparse_mat_descr bsr_descr,
                                    const float*              bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr csr_descr,
                                    float*                    csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsr2csr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    const rocsparse_mat_descr bsr_descr,
                                    const double*             bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr csr_descr,
                                    double*                   csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsr2csr(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    const rocsparse_mat_descr      bsr_descr,
                                    const rocsparse_float_complex* bsr_val,
                                    const rocsparse_int*           bsr_row_ptr,
                                    const rocsparse_int*           bsr_col_ind,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_mat_descr      csr_descr,
                                    rocsparse_float_complex*       csr_val,
                                    rocsparse_int*                 csr_row_ptr,
                                    rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsr2csr(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    const rocsparse_mat_descr       bsr_descr,
                                    const rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*            bsr_row_ptr,
                                    const rocsparse_int*            bsr_col_ind,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_mat_descr       csr_descr,
                                    rocsparse_double_complex*       csr_val,
                                    rocsparse_int*                  csr_row_ptr,
                                    rocsparse_int*                  csr_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSR2CSR_H */
