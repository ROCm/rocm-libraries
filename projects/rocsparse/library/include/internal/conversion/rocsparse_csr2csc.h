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

#ifndef ROCSPARSE_CSR2CSC_H
#define ROCSPARSE_CSR2CSC_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_csr2csc_buffer_size returns the size of the temporary storage buffer
*  required by \ref rocsparse_scsr2csc "rocsparse_Xcsr2csc()".
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
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_scsr2csc "rocsparse_Xcsr2csc()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr, \p csr_col_ind or
*              \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_action     copy_values,
                                               size_t*              buffer_size);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p rocsparse_csr2csc converts a CSR matrix into a CSC matrix. The resulting matrix can also
*  be seen as the transpose of the input matrix. \p rocsparse_csr2csc can also be used to convert
*  a CSC matrix into a CSR matrix.
*
*  The conversion of a sparse matrix from CSR to CSC format involves two steps. First, the
*  user calls \ref rocsparse_csr2csc_buffer_size in order to determine the size of the required
*  tempory storage buffer. The user then allocates this buffer. Secondly, the user calls
*  \p rocsparse_csr2csc to complete the conversion. Once the conversion is complete, the
*  user must free the temporary buffer.
*
*  Both \ref rocsparse_csr2csc_buffer_size and \p rocsparse_csr2csc take a \ref rocsparse_action
*  parameter as input. This \p copy_values parameter decides whether \p csc_row_ind and \p csc_val
*  are filled during conversion (\ref rocsparse_action_numeric) or whether only \p csc_row_ind is filled
*  (\ref rocsparse_action_symbolic). Using \ref rocsparse_action_symbolic is useful for example if only
*  the sparsity pattern is required.
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
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[out]
*  csc_val     array of \p nnz elements of the sparse CSC matrix.
*  @param[out]
*  csc_row_ind array of \p nnz elements containing the row indices of the sparse CSC
*              matrix.
*  @param[out]
*  csc_col_ptr array of \p n+1 elements that point to the start of every column of the
*              sparse CSC matrix.
*  @param[in]
*  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              rocsparse_csr2csc_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p csc_val, \p csc_row_ind, \p csc_col_ptr or
*              \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  This example computes the transpose of a CSR matrix.
*  \code{.c}
*      //     1 2 0 3 0
*      // A = 0 4 5 0 0
*      //     6 0 0 7 8
*
*      rocsparse_int m_A   = 3;
*      rocsparse_int n_A   = 5;
*      rocsparse_int nnz_A = 8;
*
*      csr_row_ptr_A[m_A + 1] = {0, 3, 5, 8};             // device memory
*      csr_col_ind_A[nnz_A] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
*      csr_val_A[nnz_A]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
*
*      // Allocate memory for transposed CSR matrix
*      rocsparse_int m_T   = n_A;
*      rocsparse_int n_T   = m_A;
*      rocsparse_int nnz_T = nnz_A;
*
*      rocsparse_int* csr_row_ptr_T;
*      rocsparse_int* csr_col_ind_T;
*      float* csr_val_T;
*
*      hipMalloc((void**)&csr_row_ptr_T, sizeof(rocsparse_int) * (m_T + 1));
*      hipMalloc((void**)&csr_col_ind_T, sizeof(rocsparse_int) * nnz_T);
*      hipMalloc((void**)&csr_val_T, sizeof(float) * nnz_T);
*
*      // Obtain the temporary buffer size
*      size_t buffer_size;
*      rocsparse_csr2csc_buffer_size(handle,
*                                    m_A,
*                                    n_A,
*                                    nnz_A,
*                                    csr_row_ptr_A,
*                                    csr_col_ind_A,
*                                    rocsparse_action_numeric,
*                                    &buffer_size);
*
*      // Allocate temporary buffer
*      void* temp_buffer;
*      hipMalloc(&temp_buffer, buffer_size);
*
*      rocsparse_scsr2csc(handle,
*                         m_A,
*                         n_A,
*                         nnz_A,
*                         csr_val_A,
*                         csr_row_ptr_A,
*                         csr_col_ind_A,
*                         csr_val_T,
*                         csr_col_ind_T,
*                         csr_row_ptr_T,
*                         rocsparse_action_numeric,
*                         rocsparse_index_base_zero,
*                         temp_buffer);
*  \endcode
*
*  \par Example
*  This example computes the symbolic transpose of A
*  \snippet example_rocsparse_csr2csc.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2csc(rocsparse_handle     handle,
                                    rocsparse_int        m,
                                    rocsparse_int        n,
                                    rocsparse_int        nnz,
                                    const float*         csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    float*               csc_val,
                                    rocsparse_int*       csc_row_ind,
                                    rocsparse_int*       csc_col_ptr,
                                    rocsparse_action     copy_values,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2csc(rocsparse_handle     handle,
                                    rocsparse_int        m,
                                    rocsparse_int        n,
                                    rocsparse_int        nnz,
                                    const double*        csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    double*              csc_val,
                                    rocsparse_int*       csc_row_ind,
                                    rocsparse_int*       csc_col_ptr,
                                    rocsparse_action     copy_values,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2csc(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    rocsparse_int                  nnz,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    rocsparse_float_complex*       csc_val,
                                    rocsparse_int*                 csc_row_ind,
                                    rocsparse_int*                 csc_col_ptr,
                                    rocsparse_action               copy_values,
                                    rocsparse_index_base           idx_base,
                                    void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2csc(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    rocsparse_int                   nnz,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    rocsparse_double_complex*       csc_val,
                                    rocsparse_int*                  csc_row_ind,
                                    rocsparse_int*                  csc_col_ptr,
                                    rocsparse_action                copy_values,
                                    rocsparse_index_base            idx_base,
                                    void*                           temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSR2CSC_H */
