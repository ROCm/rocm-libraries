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

#ifndef ROCSPARSE_PRUNE_DENSE2CSR_H
#define ROCSPARSE_PRUNE_DENSE2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_prune_dense2csr_buffer_size returns the size of the temporary buffer that
*  is required by \ref rocsparse_sprune_dense2csr_nnz "rocsparse_Xprune_dense2csr_nnz()" and
*  \ref rocsparse_sprune_dense2csr "rocsparse_Xprune_dense2csr()". The temporary storage
*  buffer must be allocated by the user.
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
*  threshold   pointer to the pruning non-negative threshold which can exist in either host or device memory.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref rocsparse_matrix_type_general and 
*              also any valid value of the \ref rocsparse_index_base.
*  @param[in]
*  csr_val     array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csr_row_ptr integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_sprune_dense2csr_nnz "rocsparse_Xprune_dense2csr_nnz()" and
*              \ref rocsparse_sprune_dense2csr "rocsparse_Xprune_dense2csr()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const float*              A,
                                                        rocsparse_int             lda,
                                                        const float*              threshold,
                                                        const rocsparse_mat_descr descr,
                                                        const float*              csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const double*             A,
                                                        rocsparse_int             lda,
                                                        const double*             threshold,
                                                        const rocsparse_mat_descr descr,
                                                        const double*             csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \details
*  \p rocsparse_prune_dense2csr_nnz computes the number of nonzero elements per row and the total
*  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
*  pruned from the matrix.
* 
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
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
*  threshold              pointer to the pruning non-negative threshold which can exist in either host or device memory.
*  @param[in]
*  descr                  the descriptor of the dense matrix \p A.
*  @param[out]
*  csr_row_ptr            integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnz_total_dev_host_ptr total number of nonzero elements in device or host memory.
*  @param[out]
*  temp_buffer            buffer allocated by the user whose size is determined by calling 
*                         \ref rocsparse_sprune_dense2csr_buffer_size "rocsparse_Xprune_dense2csr_buffer_size()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p threshold or \p descr or \p csr_row_ptr
*              or \p nnz_total_dev_host_ptr or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_nnz(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const float*              A,
                                                rocsparse_int             lda,
                                                const float*              threshold,
                                                const rocsparse_mat_descr descr,
                                                rocsparse_int*            csr_row_ptr,
                                                rocsparse_int*            nnz_total_dev_host_ptr,
                                                void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_nnz(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const double*             A,
                                                rocsparse_int             lda,
                                                const double*             threshold,
                                                const rocsparse_mat_descr descr,
                                                rocsparse_int*            csr_row_ptr,
                                                rocsparse_int*            nnz_total_dev_host_ptr,
                                                void*                     temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert and prune dense matrix \f$A\f$ into a sparse CSR matrix \f$C\f$
*
*  \details
*  This function converts the dense matrix \f$A\f$ into a sparse CSR matrix \f$C\f$ by pruning values in \f$A\f$
*  that are less than a threshold.
*
*  The conversion involves three steps. The user first calls 
*  \ref rocsparse_sprune_dense2csr_buffer_size "rocsparse_Xprune_dense2csr_buffer_size()" 
*  to determine the size of the temporary storage buffer. The user allocates this buffer as well as the array 
*  \p csr_row_ptr to have \p m+1 elements. The user then calls 
*  \ref rocsparse_sprune_dense2csr_nnz "rocsparse_Xprune_dense2csr_nnz()" which fills
*  in the \p csr_row_ptr array and stores the number of elements that are larger than the pruning \p threshold
*  in \p nnz_total_dev_host_ptr. Now that the number of nonzeros larger than the pruning \p threshold is known, the 
*  user uses this information to allocate the \p csr_col_ind and \p csr_val arrays and then calls
*  \p rocsparse_prune_dense2csr to complete the conversion. Once the conversion is complete, the temporary storage 
*  buffer can be freed.
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
*  threshold   pointer to the non-negative pruning threshold which can exist in either host or device memory.
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
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              \ref rocsparse_sprune_dense2csr_buffer_size "rocsparse_Xprune_dense2csr_buffer_size()" .
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p descr or \p threshold or \p csr_val
*              or \p csr_row_ptr or \p csr_col_ind or \p temp_buffer pointer is invalid.
*
*  \par Example
*  \code{.c}
*    //     1 2 0 7
*    // A = 3 0 0 4
*    //     5 6 0 4
*    //     0 4 2 5
*    rocsparse_int m   = 4;
*    rocsparse_int n   = 4;
*    rocsparse_int lda = m;
*    float threshold = 3.0f;
*
*    std::vector<float> hdense = {1.0f, 3.0f, 5.0f, 0.0f, 2.0f, 0.0f, 6.0f, 4.0f, 0.0f, 0.0f, 0.0f, 2.0f, 7.0f, 4.0f, 4.0f, 5.0f};
*
*    rocsparse_handle handle;
*    rocsparse_create_handle(&handle);
*
*    rocsparse_mat_descr descr;
*    rocsparse_create_mat_descr(&descr);
*
*    rocsparse_mat_info info;
*    rocsparse_create_mat_info(&info);
*
*    float* ddense = nullptr;
*    hipMalloc((void**)&ddense, sizeof(float) * lda * n);
*    hipMemcpy(ddense, hdense.data(), sizeof(float) * lda * n, hipMemcpyHostToDevice);
*
*    rocsparse_int* dcsr_row_ptr = nullptr;
*    hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*
*    // Obtain the temporary buffer size
*    size_t buffer_size;
*    rocsparse_sprune_dense2csr_buffer_size(handle,
*                                           m,
*                                           n,
*                                           ddense,
*                                           lda,
*                                           &threshold,
*                                           descr,
*                                           nullptr,
*                                           nullptr,
*                                           nullptr,
*                                           &buffer_size);
*
*    // Allocate temporary buffer
*    void* temp_buffer;
*    hipMalloc(&temp_buffer, buffer_size);
*
*    rocsparse_int nnz;
*    rocsparse_sprune_dense2csr_nnz(handle,
*                                   m,
*                                   n,
*                                   ddense,
*                                   lda,
*                                   &threshold,
*                                   descr,
*                                   dcsr_row_ptr,
*                                   &nnz,
*                                   temp_buffer);
*
*    rocsparse_int* dcsr_col_ind = nullptr;
*    float* dcsr_val = nullptr;
*    hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * nnz);
*    hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*
*    rocsparse_sprune_dense2csr(handle,
*                               m,
*                               n,
*                               ddense,
*                               lda,
*                               &threshold,
*                               descr,
*                               dcsr_val,
*                               dcsr_row_ptr,
*                               dcsr_col_ind,
*                               temp_buffer);
*    
*    rocsparse_destroy_handle(handle);
*    rocsparse_destroy_mat_descr(descr);
*    rocsparse_destroy_mat_info(info);
*
*    hipFree(temp_buffer);
*
*    hipFree(ddense);
*
*    hipFree(dcsr_row_ptr);
*    hipFree(dcsr_col_ind);
*    hipFree(dcsr_val);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const float*              A,
                                            rocsparse_int             lda,
                                            const float*              threshold,
                                            const rocsparse_mat_descr descr,
                                            float*                    csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            rocsparse_int*            csr_col_ind,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const double*             A,
                                            rocsparse_int             lda,
                                            const double*             threshold,
                                            const rocsparse_mat_descr descr,
                                            double*                   csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            rocsparse_int*            csr_col_ind,
                                            void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_PRUNE_DENSE2CSR_H */
