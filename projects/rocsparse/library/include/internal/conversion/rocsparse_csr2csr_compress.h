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

#ifndef ROCSPARSE_CSR2CSR_COMPRESS_H
#define ROCSPARSE_CSR2CSR_COMPRESS_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_csr2csr_compress converts a CSR matrix into a compressed CSR matrix by
 *  removing entries in the input CSR matrix that are below a non-negative threshold \p tol
 *
 *  Compressing a CSR matrix involves two steps. First we use
 *  \ref rocsparse_snnz_compress "rocsparse_Xnnz_compress()" to determine how many entries will
 *  be in the final compressed CSR matrix. Then we call \p rocsparse_csr2csr_compress to finish
 *  the compression and fill in the column indices and values arrays of the compressed CSR matrix.
 *
 *  \note
 *  In the case of complex matrices only the magnitude of the real part of \p tol is used.
 *
 *  \note
 *  This function is blocking with respect to the host.
 *
 *  \note
 *  This routine does not support execution in a hipGraph context.
 *
 *  @param[in]
 *  handle        handle to the rocsparse library context queue.
 *  @param[in]
 *  m             number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns of the sparse CSR matrix.
 *  @param[in]
 *  descr_A       matrix descriptor for the CSR matrix
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                uncompressed sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the uncompressed
 *                sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of elements in the column indices and values arrays of the uncompressed
 *                sparse CSR matrix.
 *  @param[in]
 *  nnz_per_row   array of length \p m containing the number of entries that will be kept per row in
 *                the final compressed CSR matrix.
 *  @param[out]
 *  csr_val_C     array of \p nnz_C elements of the compressed sparse CSC matrix.
 *  @param[out]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every column of the compressed
 *                sparse CSR matrix.
 *  @param[out]
 *  csr_col_ind_C array of \p nnz_C elements containing the row indices of the compressed
 *                sparse CSR matrix.
 *  @param[in]
 *  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
 *                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
 *                are removed in output compressed CSR matrix.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz_A is invalid.
 *  \retval     rocsparse_status_invalid_value \p tol is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_val_A, \p csr_row_ptr_A,
 *              \p csr_col_ind_A, \p csr_val_C, \p csr_row_ptr_C, \p csr_col_ind_C or
 *              \p nnz_per_row pointer is invalid.
 *
 *  \par Example
 *  This example demonstrates how to compress a CSR matrix.
 *  \snippet example_rocsparse_csr2csr_compress.cpp doc example
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2csr_compress(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr_A,
                                             const float*              csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_int*      nnz_per_row,
                                             float*                    csr_val_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            csr_col_ind_C,
                                             float                     tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2csr_compress(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr_A,
                                             const double*             csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_int*      nnz_per_row,
                                             double*                   csr_val_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            csr_col_ind_C,
                                             double                    tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2csr_compress(rocsparse_handle               handle,
                                             rocsparse_int                  m,
                                             rocsparse_int                  n,
                                             const rocsparse_mat_descr      descr_A,
                                             const rocsparse_float_complex* csr_val_A,
                                             const rocsparse_int*           csr_row_ptr_A,
                                             const rocsparse_int*           csr_col_ind_A,
                                             rocsparse_int                  nnz_A,
                                             const rocsparse_int*           nnz_per_row,
                                             rocsparse_float_complex*       csr_val_C,
                                             rocsparse_int*                 csr_row_ptr_C,
                                             rocsparse_int*                 csr_col_ind_C,
                                             rocsparse_float_complex        tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2csr_compress(rocsparse_handle                handle,
                                             rocsparse_int                   m,
                                             rocsparse_int                   n,
                                             const rocsparse_mat_descr       descr_A,
                                             const rocsparse_double_complex* csr_val_A,
                                             const rocsparse_int*            csr_row_ptr_A,
                                             const rocsparse_int*            csr_col_ind_A,
                                             rocsparse_int                   nnz_A,
                                             const rocsparse_int*            nnz_per_row,
                                             rocsparse_double_complex*       csr_val_C,
                                             rocsparse_int*                  csr_row_ptr_C,
                                             rocsparse_int*                  csr_col_ind_C,
                                             rocsparse_double_complex        tol);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSR2CSR_COMPRESS_H */
