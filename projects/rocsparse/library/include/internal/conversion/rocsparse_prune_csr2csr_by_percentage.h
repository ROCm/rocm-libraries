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

#ifndef ROCSPARSE_PRUNE_CSR2CSR_BY_PERCENTAGE_H
#define ROCSPARSE_PRUNE_CSR2CSR_BY_PERCENTAGE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
 *  \details
 *  \p rocsparse_prune_csr2csr_by_percentage_buffer_size returns the size of the temporary buffer that
 *  is required by \ref rocsparse_sprune_csr2csr_nnz_by_percentage "rocsparse_Xprune_csr2csr_nnz_by_percentage()"
 *  and \ref rocsparse_sprune_csr2csr_by_percentage "rocsparse_Xprune_csr2csr_by_percentage()". The temporary 
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
 *  handle        handle to the rocsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of non-zeros in the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_descr_A   descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements containing the values of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  percentage    \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  csr_descr_C   descriptor of the sparse CSR matrix \f$C\f$. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_C     array of \p nnz_C elements containing the values of the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  csr_col_ind_C array of \p nnz_C elements containing the column indices of the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  info          prune info structure.
 *  @param[out]
 *  buffer_size   number of bytes of the temporary storage buffer required by 
 *                \ref rocsparse_sprune_csr2csr_nnz_by_percentage "rocsparse_Xprune_csr2csr_nnz_by_percentage()" and
 *                \ref rocsparse_sprune_csr2csr_by_percentage "rocsparse_Xprune_csr2csr_by_percentage()"
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_sprune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const float*              csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       float                     percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       const float*              csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       const rocsparse_int*      csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_dprune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const double*             csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       double                    percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       const double*             csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       const rocsparse_int*      csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
 *  \details
 *  \p rocsparse_prune_csr2csr_nnz_by_percentage computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once a \p percentage of the smallest magnitude elements 
 *  have been pruned from the sparse CSR input matrix. See 
 *  \ref rocsparse_sprune_csr2csr_by_percentage "rocsparse_sprune_csr2csr_by_percentage()" for a more detailed 
 *  description of how this pruning based on \p percentage works.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
 *
 *  \note
 *  This routine does not support execution in a hipGraph context.
 *
 *  @param[in]
 *  handle                 handle to the rocsparse library context queue.
 *  @param[in]
 *  m                      number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                      number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnz_A                  number of non-zeros in the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_descr_A            descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *                         \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_A              array of \p nnz_A elements containing the values of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr_A          array of \p m+1 elements that point to the start of every row of the
 *                         sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_col_ind_A          array of \p nnz_A elements containing the column indices of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  percentage             \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  csr_descr_C            descriptor of the sparse CSR matrix \f$C\f$. Currently, only
 *                         \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_row_ptr_C          array of \p m+1 elements that point to the start of every row of the
 *                         sparse CSR matrix \f$C\f$.
 *  @param[out]
 *  nnz_total_dev_host_ptr total number of nonzero elements in device or host memory.
 *  @param[in]
 *  info                   prune info structure.
 *  @param[out]
 *  temp_buffer            buffer allocated by the user whose size is determined by calling
 *                         \ref rocsparse_sprune_csr2csr_by_percentage_buffer_size "rocsparse_Xprune_csr2csr_by_percentage_buffer_size()".
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz_A or \p percentage is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr_A or \p csr_descr_C or \p info or \p csr_val_A
 *              or \p csr_row_ptr_A or \p csr_col_ind_A or \p csr_row_ptr_C or \p nnz_total_dev_host_ptr
 *              or \p temp_buffer pointer is invalid.
 *
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                            rocsparse_int             m,
                                                            rocsparse_int             n,
                                                            rocsparse_int             nnz_A,
                                                            const rocsparse_mat_descr csr_descr_A,
                                                            const float*              csr_val_A,
                                                            const rocsparse_int*      csr_row_ptr_A,
                                                            const rocsparse_int*      csr_col_ind_A,
                                                            float                     percentage,
                                                            const rocsparse_mat_descr csr_descr_C,
                                                            rocsparse_int*            csr_row_ptr_C,
                                                            rocsparse_int* nnz_total_dev_host_ptr,
                                                            rocsparse_mat_info info,
                                                            void*              temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                            rocsparse_int             m,
                                                            rocsparse_int             n,
                                                            rocsparse_int             nnz_A,
                                                            const rocsparse_mat_descr csr_descr_A,
                                                            const double*             csr_val_A,
                                                            const rocsparse_int*      csr_row_ptr_A,
                                                            const rocsparse_int*      csr_col_ind_A,
                                                            double                    percentage,
                                                            const rocsparse_mat_descr csr_descr_C,
                                                            rocsparse_int*            csr_row_ptr_C,
                                                            rocsparse_int* nnz_total_dev_host_ptr,
                                                            rocsparse_mat_info info,
                                                            void*              temp_buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix \f$A\f$ into a sparse CSR matrix \f$C\f$
 *
 *  \details
 *  This function converts the sparse CSR matrix \f$A\f$ into a sparse CSR matrix \f$C\f$ by pruning values in \f$A\f$
 *  that are less than a threshold. This threshold is determined by using a \p percentage and the following steps:
 *
 *  <b>Step 1</b>: First the \p csr_val_A array is sorted in ascending order using the absolute value of each entry:
 *  \f[
 *    csr\_val\_A\_sorted = sort(abs(csr\_val\_A))
 *  \f]
 *  
 *  <b>Step 2</b>: Next we use the \p percentage parameter to determine the threshold:
 *  \f[
 *    pos = ceil(nnz\_A \times (percentage/100)) - 1 \\
 *    pos = \min(pos, nnz\_A - 1) \\
 *    pos = \max(pos, 0) \\
 *    threshold = csr\_val\_A\_sorted[pos]
 *  \f]
 *  
 *  <b>Step 3</b>: Finally, we use this threshold with the routine 
 *  \ref rocsparse_sprune_csr2csr "rocsparse_Xprune_csr2csr()" to complete the conversion.
 *
 *  The conversion involves three steps. The user first calls 
 *  \ref rocsparse_sprune_csr2csr_by_percentage_buffer_size "rocsparse_Xprune_csr2csr_by_percentage_buffer_size()" 
 *  to determine the size of the temporary storage buffer. The user allocates this buffer as well as the array 
 *  \p csr_row_ptr_C to have \p m+1 elements. The user then calls 
 *  \ref rocsparse_sprune_csr2csr_nnz_by_percentage "rocsparse_Xprune_csr2csr_nnz_by_percentage()" which fills
 *  in the \p csr_row_ptr_C array and stores the number of elements that are larger than the pruning threshold
 *  in \p nnz_total_dev_host_ptr. Now that the number of nonzeros larger than the pruning threshold is known, the 
 *  user uses this information to allocate the \p csr_col_ind_C and \p csr_val_C arrays and then calls
 *  \p rocsparse_prune_csr2csr_by_percentage to complete the conversion. Once the conversion is complete, the 
 *  temporary storage buffer can be freed.
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
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of non-zeros in the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_descr_A   descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements containing the values of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  percentage    \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  csr_descr_C   descriptor of the sparse CSR matrix \f$C\f$. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_val_C     array of \p nnz_C elements containing the values of the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix \f$C\f$.
 *  @param[out]
 *  csr_col_ind_C array of \p nnz_C elements containing the column indices of the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  info          prune info structure.
 *  @param[in]
 *  temp_buffer   buffer allocated by the user whose size is determined by calling 
 *                \ref rocsparse_sprune_csr2csr_buffer_size "rocsparse_Xprune_csr2csr_buffer_size()".
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz_A or \p percentage is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr_A or \p csr_descr_C or \p info or \p csr_val_A
 *              or \p csr_row_ptr_A or \p csr_col_ind_A or \p csr_val_C or \p csr_row_ptr_C or \p csr_col_ind_C
 *              or \p temp_buffer pointer is invalid.
 *
 *  \par Example
 *  \code{.c}
 *    //     1 2 0 0
 *    // A = 3 0 0 4
 *    //     5 6 0 4
 *    //     7 4 2 5
 *    rocsparse_int m   = 4;
 *    rocsparse_int n   = 4;
 *    rocsparse_int nnz_A = 11;
 *    float percentage = 50.0f;
 *
 *    std::vector<rocsparse_int> hcsr_row_ptr_A = {0, 2, 4, 7, 11}; 
 *    std::vector<rocsparse_int> hcsr_col_ind_A = {0, 1, 0, 3, 0, 1, 3, 0, 1, 2, 3}; 
 *    std::vector<float> hcsr_val_A = {1, 2, 3, 4, 5, 6, 4, 7, 4, 2, 5};
 *
 *    rocsparse_int* dcsr_row_ptr_A = nullptr;
 *    rocsparse_int* dcsr_col_ind_A = nullptr;
 *    float* dcsr_val_A = nullptr;
 *    hipMalloc((void**)&dcsr_row_ptr_A, sizeof(rocsparse_int) * (m + 1));
 *    hipMalloc((void**)&dcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A);
 *    hipMalloc((void**)&dcsr_val_A, sizeof(float) * nnz_A);
 *
 *    hipMemcpy(dcsr_row_ptr_A, hcsr_row_ptr_A.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
 *    hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A.data(), sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice);
 *    hipMemcpy(dcsr_val_A, hcsr_val_A.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice);
 *
 *    rocsparse_handle handle;
 *    rocsparse_create_handle(&handle);
 *
 *    rocsparse_mat_descr descr_A;
 *    rocsparse_create_mat_descr(&descr_A);
 *
 *    rocsparse_mat_descr descr_C;
 *    rocsparse_create_mat_descr(&descr_C);
 *
 *    rocsparse_mat_info info;
 *    rocsparse_create_mat_info(&info);
 *
 *    rocsparse_int* dcsr_row_ptr_C = nullptr;
 *    hipMalloc((void**)&dcsr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
 *
 *    // Obtain the temporary buffer size
 *    size_t buffer_size;
 *    rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle,
 *                                                       m,
 *                                                       n,
 *                                                       nnz_A,
 *                                                       descr_A,
 *                                                       dcsr_val_A,
 *                                                       dcsr_row_ptr_A,
 *                                                       dcsr_col_ind_A,
 *                                                       percentage,
 *                                                       descr_C,
 *                                                       nullptr,
 *                                                       nullptr,
 *                                                       nullptr,
 *                                                       info,
 *                                                       &buffer_size);
 *
 *    // Allocate temporary buffer
 *    void* temp_buffer;
 *    hipMalloc(&temp_buffer, buffer_size);
 *
 *    rocsparse_int nnz_C;
 *    rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
 *                                               m,
 *                                               n,
 *                                               nnz_A,
 *                                               descr_A,
 *                                               dcsr_val_A,
 *                                               dcsr_row_ptr_A,
 *                                               dcsr_col_ind_A,
 *                                               percentage,
 *                                               descr_C,
 *                                               dcsr_row_ptr_C,
 *                                               &nnz_C,
 *                                               info,
 *                                               temp_buffer);
 *
 *    rocsparse_int* dcsr_col_ind_C = nullptr;
 *    float* dcsr_val_C = nullptr;
 *    hipMalloc((void**)&dcsr_col_ind_C, sizeof(rocsparse_int) * nnz_C);
 *    hipMalloc((void**)&dcsr_val_C, sizeof(float) * nnz_C);
 *
 *    rocsparse_sprune_csr2csr_by_percentage(handle,
 *                                           m,
 *                                           n,
 *                                           nnz_A,
 *                                           descr_A,
 *                                           dcsr_val_A,
 *                                           dcsr_row_ptr_A,
 *                                           dcsr_col_ind_A,
 *                                           percentage,
 *                                           descr_C,
 *                                           dcsr_val_C,
 *                                           dcsr_row_ptr_C,
 *                                           dcsr_col_ind_C,
 *                                           info,
 *                                           temp_buffer);
 *   
 *    rocsparse_destroy_handle(handle);
 *    rocsparse_destroy_mat_descr(descr_A);
 *    rocsparse_destroy_mat_descr(descr_C);
 *    rocsparse_destroy_mat_info(info);
 *
 *    hipFree(temp_buffer);
 *
 *    hipFree(dcsr_row_ptr_A);
 *    hipFree(dcsr_col_ind_A);
 *    hipFree(dcsr_val_A);
 *
 *    hipFree(dcsr_row_ptr_C);
 *    hipFree(dcsr_col_ind_C);
 *    hipFree(dcsr_val_C);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_mat_descr csr_descr_A,
                                                        const float*              csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        float                     percentage,
                                                        const rocsparse_mat_descr csr_descr_C,
                                                        float*                    csr_val_C,
                                                        const rocsparse_int*      csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        rocsparse_mat_info        info,
                                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_mat_descr csr_descr_A,
                                                        const double*             csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        double                    percentage,
                                                        const rocsparse_mat_descr csr_descr_C,
                                                        double*                   csr_val_C,
                                                        const rocsparse_int*      csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        rocsparse_mat_info        info,
                                                        void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_PRUNE_CSR2CSR_BY_PERCENTAGE_H */
