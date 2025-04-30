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

#ifndef ROCSPARSE_ELL2CSR_H
#define ROCSPARSE_ELL2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \details
*  This function takes a sparse ELL matrix as input and computes the row offset array, \p csr_row_ptr, 
*  and the total number of nonzeros, \p csr_nnz, that will result from converting the ELL format input 
*  matrix to a CSR format output matrix. This function is the first step in the conversion and is used in 
*  conjunction with \ref rocsparse_sell2csr "rocsparse_Xell2csr()". It is assumed that \p csr_row_ptr has 
*  been allocated with size \p m+1.
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
*  m           number of rows of the sparse ELL matrix.
*  @param[in]
*  n           number of columns of the sparse ELL matrix.
*  @param[in]
*  ell_descr   descriptor of the sparse ELL matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  ell_width   number of non-zero elements per row in ELL storage format.
*  @param[in]
*  ell_col_ind array of \p m times \p ell_width elements containing the column indices
*              of the sparse ELL matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csr_nnz     pointer to the total number of non-zero elements in CSR storage
*              format.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
*  \retval     rocsparse_status_invalid_pointer \p ell_descr, \p ell_col_ind,
*              \p csr_descr, \p csr_row_ptr or \p csr_nnz pointer is invalid.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int             ell_width,
                                       const rocsparse_int*      ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       rocsparse_int*            csr_row_ptr,
                                       rocsparse_int*            csr_nnz);

/*! \ingroup conv_module
*  \brief Convert a sparse ELL matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_ell2csr converts a ELL matrix into a CSR matrix. It is assumed
*  that \p csr_row_ptr has already been filled and that \p csr_val and \p csr_col_ind
*  are allocated by the user. Allocation size for \p csr_row_ptr is computed as 
*  \p m+1. Allocation size for \p csr_val and \p csr_col_ind is computed using 
*  \ref rocsparse_ell2csr_nnz() which also fills in \p csr_row_ptr.
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
*  m           number of rows of the sparse ELL matrix.
*  @param[in]
*  n           number of columns of the sparse ELL matrix.
*  @param[in]
*  ell_descr   descriptor of the sparse ELL matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  ell_width   number of non-zero elements per row in ELL storage format.
*  @param[in]
*  ell_val     array of \p m times \p ell_width elements of the sparse ELL matrix.
*  @param[in]
*  ell_col_ind array of \p m times \p ell_width elements containing the column indices
*              of the sparse ELL matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val     array containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csr_col_ind array containing the column indices of the sparse CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_descr, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p ell_descr, \p ell_val or
*              \p ell_col_ind pointer is invalid.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example converts an ELL matrix into a CSR matrix.
*  \code{.c}
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    rocsparse_int m         = 3;
*    rocsparse_int n         = 5;
*    rocsparse_int nnz       = 8;
*    rocsparse_int ell_width = 3;
*
*    std::vector<rocsparse_int> hell_col_ind = {0, 1, 0, 1, 2, 3, 3, -1, 4};
*    std::vector<float> hell_val = {1, 4, 6, 2, 5, 7, 3, 0, 8};
*
*    rocsparse_int* dell_col_ind = nullptr;
*    float* dell_val = nullptr;
*    hipMalloc((void**)&dell_col_ind, sizeof(rocsparse_int) * m * ell_width);
*    hipMalloc((void**)&dell_val, sizeof(float) * m * ell_width);
*
*    hipMemcpy(dell_col_ind, hell_col_ind.data(), sizeof(rocsparse_int) * m * ell_width, hipMemcpyHostToDevice);
*    hipMemcpy(dell_val, hell_val.data(), sizeof(float) * m * ell_width, hipMemcpyHostToDevice);
*
*    rocsparse_handle handle;
*    rocsparse_create_handle(&handle);
*
*    // Create ELL matrix descriptor
*    rocsparse_mat_descr ell_descr;
*    rocsparse_create_mat_descr(&ell_descr);
*
*    // Create CSR matrix descriptor
*    rocsparse_mat_descr csr_descr;
*    rocsparse_create_mat_descr(&csr_descr);
*
*    // Allocate csr_row_ptr array for row offsets
*    rocsparse_int* dcsr_row_ptr;
*    hipMalloc((void**)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*
*    // Obtain the number of CSR non-zero entries
*    // and fill csr_row_ptr array with row offsets
*    rocsparse_int csr_nnz;
*    rocsparse_ell2csr_nnz(handle,
*                          m,
*                          n,
*                          ell_descr,
*                          ell_width,
*                          dell_col_ind,
*                          csr_descr,
*                          dcsr_row_ptr,
*                          &csr_nnz);
*
*    // Allocate CSR column and value arrays
*    rocsparse_int* dcsr_col_ind = nullptr;
*    float* dcsr_val = nullptr;
*    hipMalloc((void**)&dcsr_col_ind, sizeof(rocsparse_int) * csr_nnz);
*    hipMalloc((void**)&dcsr_val, sizeof(float) * csr_nnz);
*
*    // Format conversion
*    rocsparse_sell2csr(handle,
*                       m,
*                       n,
*                       ell_descr,
*                       ell_width,
*                       dell_val,
*                       dell_col_ind,
*                       csr_descr,
*                       dcsr_val,
*                       dcsr_row_ptr,
*                       dcsr_col_ind);
*
*    hipFree(dell_col_ind);
*    hipFree(dell_val);
*
*    hipFree(dcsr_row_ptr);
*    hipFree(dcsr_col_ind);
*    hipFree(dcsr_val);
*
*    rocsparse_destroy_mat_descr(ell_descr);
*    rocsparse_destroy_mat_descr(csr_descr);
*
*    rocsparse_destroy_handle(handle);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sell2csr(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    const float*              ell_val,
                                    const rocsparse_int*      ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    float*                    csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dell2csr(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    const double*             ell_val,
                                    const rocsparse_int*      ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    double*                   csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cell2csr(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_mat_descr      ell_descr,
                                    rocsparse_int                  ell_width,
                                    const rocsparse_float_complex* ell_val,
                                    const rocsparse_int*           ell_col_ind,
                                    const rocsparse_mat_descr      csr_descr,
                                    rocsparse_float_complex*       csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zell2csr(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_mat_descr       ell_descr,
                                    rocsparse_int                   ell_width,
                                    const rocsparse_double_complex* ell_val,
                                    const rocsparse_int*            ell_col_ind,
                                    const rocsparse_mat_descr       csr_descr,
                                    rocsparse_double_complex*       csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*                  csr_col_ind);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_ELL2CSR_H */
