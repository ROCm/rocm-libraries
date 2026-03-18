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

#ifndef ROCSPARSE_NNZ_COMPRESS_H
#define ROCSPARSE_NNZ_COMPRESS_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
*  in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
*  of remaining elements in the matrix.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle        handle to the rocsparse library context queue.
*
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*
*  @param[in]
*  descr_A       the descriptor of the sparse CSR matrix.
*
*  @param[in]
*  csr_val_A     array of \p nnz_A elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
*                uncompressed sparse CSR matrix.
*  @param[out]
*  nnz_per_row   array of length \p m containing the number of entries that will be kept per row in
*                the final compressed CSR matrix.
*  @param[out]
*  nnz_C         number of elements in the column indices and values arrays of the compressed
*                sparse CSR matrix. Can be either host or device pointer.
*  @param[in]
*  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
*                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
*                are removed in output compressed CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n is invalid.
*  \retval     rocsparse_status_invalid_value \p tol is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_val_A or \p csr_row_ptr_A or \p nnz_per_row or \p nnz_C
*              pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_nnz_compress.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_snnz_compress(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         const rocsparse_mat_descr descr_A,
                                         const float*              csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         rocsparse_int*            nnz_per_row,
                                         rocsparse_int*            nnz_C,
                                         float                     tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnnz_compress(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         const rocsparse_mat_descr descr_A,
                                         const double*             csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         rocsparse_int*            nnz_per_row,
                                         rocsparse_int*            nnz_C,
                                         double                    tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cnnz_compress(rocsparse_handle               handle,
                                         rocsparse_int                  m,
                                         const rocsparse_mat_descr      descr_A,
                                         const rocsparse_float_complex* csr_val_A,
                                         const rocsparse_int*           csr_row_ptr_A,
                                         rocsparse_int*                 nnz_per_row,
                                         rocsparse_int*                 nnz_C,
                                         rocsparse_float_complex        tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_znnz_compress(rocsparse_handle                handle,
                                         rocsparse_int                   m,
                                         const rocsparse_mat_descr       descr_A,
                                         const rocsparse_double_complex* csr_val_A,
                                         const rocsparse_int*            csr_row_ptr_A,
                                         rocsparse_int*                  nnz_per_row,
                                         rocsparse_int*                  nnz_C,
                                         rocsparse_double_complex        tol);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_NNZ_COMPRESS_H */
