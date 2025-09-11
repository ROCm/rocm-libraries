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

#ifndef ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H
#define ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_prune_dense2csr_by_percentage_buffer_size returns the size of the temporary buffer that
*  is required by \ref rocsparse_sprune_dense2csr_nnz_by_percentage "rocsparse_Xprune_dense2csr_nnz_by_percentage()"
*  and \ref rocsparse_sprune_dense2csr_by_percentage "rocsparse_Xprune_dense2csr_by_percentage()". The temporary
*  storage buffer must be allocated by the user.
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
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*  @param[in]
*  lda         leading dimension of dense array \p A.
*  @param[in]
*  percentage  \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref rocsparse_matrix_type_general
*              and also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  csr_val     array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csr_row_ptr integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info prune  information structure
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_sprune_dense2csr_nnz_by_percentage "rocsparse_Xprune_dense2csr_nnz_by_percentage()" and
*              \ref rocsparse_sprune_dense2csr_by_percentage "rocsparse_Xprune_dense2csr_by_percentage()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const float*              A,
                                                         rocsparse_int             lda,
                                                         float                     percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const float*              csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const double*             A,
                                                         rocsparse_int             lda,
                                                         double                    percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const double*             csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \details
*  \p rocsparse_sprune_dense2csr_nnz_by_percentage computes the number of nonzero elements per row and the total
*  number of nonzero elements in a sparse CSR matrix once a \p percentage of the smallest magnitude elements
*  have been pruned from the dense input matrix. See
*  \ref rocsparse_sprune_dense2csr_by_percentage "rocsparse_sprune_dense2csr_by_percentage()" for a more detailed
*  description of how this pruning based on \p percentage works.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle                 handle to the rocsparse library context queue.
*  @param[in]
*  m                      number of rows of the dense matrix \p A.
*  @param[in]
*  n                      number of columns of the dense matrix \p A.
*  @param[in]
*  A                      array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                    leading dimension of dense array \p A.
*  @param[in]
*  percentage             \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr                  the descriptor of the dense matrix \p A.
*  @param[out]
*  csr_row_ptr            integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnz_total_dev_host_ptr total number of nonzero elements in device or host memory.
*  @param[in]
*  info prune             information structure
*  @param[out]
*  temp_buffer            buffer allocated by the user whose size is determined by calling
*                         \ref rocsparse_sprune_dense2csr_by_percentage_buffer_size "rocsparse_Xprune_dense2csr_by_percentage_buffer_size()"..
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda or \p percentage is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p descr or \p info or \p csr_row_ptr
*              or \p nnz_total_dev_host_ptr or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                              rocsparse_int             m,
                                                              rocsparse_int             n,
                                                              const float*              A,
                                                              rocsparse_int             lda,
                                                              float                     percentage,
                                                              const rocsparse_mat_descr descr,
                                                              rocsparse_int*            csr_row_ptr,
                                                              rocsparse_int* nnz_total_dev_host_ptr,
                                                              rocsparse_mat_info info,
                                                              void*              temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                              rocsparse_int             m,
                                                              rocsparse_int             n,
                                                              const double*             A,
                                                              rocsparse_int             lda,
                                                              double                    percentage,
                                                              const rocsparse_mat_descr descr,
                                                              rocsparse_int*            csr_row_ptr,
                                                              rocsparse_int* nnz_total_dev_host_ptr,
                                                              rocsparse_mat_info info,
                                                              void*              temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix \f$A\f$ in dense format into a sparse matrix in CSR format while pruning values
*  based on percentage.
*
*  \details
*  This function converts the dense column oriented matrix \f$A\f$ into a sparse CSR matrix \f$C\f$ by pruning values in \f$A\f$
*  that are less than a threshold. This threshold is determined by using a \p percentage and the following steps:
*
*  <b>Step 1</b>: First the \p A array is sorted in ascending order using the absolute value of each entry:
*  \f[
*    A\_sorted = sort(abs(A))
*  \f]
*
*  <b>Step 2</b>: Next we use the \p percentage parameter to determine the threshold:
*  \f[
*    pos = ceil(m \times n \times (percentage/100)) - 1 \\
*    pos = \min(pos, m \times n - 1) \\
*    pos = \max(pos, 0) \\
*    threshold = A\_sorted[pos]
*  \f]
*
*  <b>Step 3</b>: Finally, we use this threshold with the routine
*  \ref rocsparse_sprune_dense2csr "rocsparse_Xprune_dense2csr()" to complete the conversion.
*
*  The conversion involves three steps. The user first calls
*  \ref rocsparse_sprune_dense2csr_by_percentage_buffer_size "rocsparse_Xprune_dense2csr_by_percentage_buffer_size()"
*  to determine the size of the temporary storage buffer. The user allocates this buffer as well as the array
*  \p csr_row_ptr to have \p m+1 elements. The user then calls
*  \ref rocsparse_sprune_dense2csr_nnz_by_percentage "rocsparse_Xprune_dense2csr_nnz_by_percentage()" which fills
*  in the \p csr_row_ptr array and stores the number of elements that are larger than the pruning threshold
*  in \p nnz_total_dev_host_ptr. Now that the number of nonzeros larger than the pruning threshold is known, the
*  user uses this information to allocate the \p csr_col_ind and \p csr_val arrays and then calls
*  \p rocsparse_prune_dense2csr_by_percentage to complete the conversion. Once the conversion is complete, the
*  temporary storage buffer can be freed.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*  @param[in]
*  lda         leading dimension of dense array \p A.
*  @param[in]
*  percentage  \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref rocsparse_matrix_type_general and
*              also any valid value of the \ref rocsparse_index_base.
*  @param[out]
*  csr_val     array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csr_row_ptr integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info prune  information structure
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              \ref rocsparse_sprune_dense2csr_by_percentage_buffer_size "rocsparse_Xprune_dense2csr_by_percentage_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda or \p percentage is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p descr or \p info or \p csr_val
*              or \p csr_row_ptr or \p csr_col_ind or \p temp_buffer pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_prune_dense2csr_by_percentage.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                                          rocsparse_int             m,
                                                          rocsparse_int             n,
                                                          const float*              A,
                                                          rocsparse_int             lda,
                                                          float                     percentage,
                                                          const rocsparse_mat_descr descr,
                                                          float*                    csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          rocsparse_int*            csr_col_ind,
                                                          rocsparse_mat_info        info,
                                                          void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                                          rocsparse_int             m,
                                                          rocsparse_int             n,
                                                          const double*             A,
                                                          rocsparse_int             lda,
                                                          double                    percentage,
                                                          const rocsparse_mat_descr descr,
                                                          double*                   csr_val,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          rocsparse_int*            csr_col_ind,
                                                          rocsparse_mat_info        info,
                                                          void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H */
