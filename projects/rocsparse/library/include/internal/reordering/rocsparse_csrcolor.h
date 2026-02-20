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

#ifndef ROCSPARSE_CSRCOLOR_H
#define ROCSPARSE_CSRCOLOR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup reordering_module
*  \brief Coloring of the adjacency graph of the matrix \f$A\f$ stored in the CSR format.
*
*  \details
*  \p rocsparse_csrcolor performs the coloring of the undirected graph represented by the (symmetric) sparsity pattern of the
*  matrix \f$A\f$ stored in CSR format. Graph coloring is a way of coloring the nodes of a graph such that no two adjacent nodes
*  are of the same color. The \p fraction_to_color is a parameter to only color a given percentage of the graph nodes, the
*  remaining uncolored nodes receive distinct new colors. The optional \p reordering array is a permutation array such that
*  unknowns of the same color are grouped. The matrix \f$A\f$ must be stored as a general matrix with a symmetric sparsity pattern,
*  and if the matrix \f$A\f$ is non-symmetric then the user is responsible to provide the symmetric part \f$\frac{A+A^T}{2}\f$.
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
*  m           number of rows of sparse matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of sparse matrix \f$A\f$.
*  @param[in]
*  descr      sparse matrix descriptor.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  fraction_to_color  fraction of nodes to be colored, which should be in the interval [0.0,1.0], for example 0.8 implies that 80 percent of nodes will be colored.
*  @param[out]
*  ncolors      resulting number of distinct colors.
*  @param[out]
*  coloring     resulting mapping of colors.
*  @param[out]
*  reordering   optional resulting reordering permutation if \p reordering is a non-null pointer.
*  @param[inout]
*  info    structure that holds the information collected during the coloring algorithm.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr, \p csr_col_ind, \p fraction_to_color, \p ncolors, \p coloring or \p info pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrcolor(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     const float*              fraction_to_color,
                                     rocsparse_int*            ncolors,
                                     rocsparse_int*            coloring,
                                     rocsparse_int*            reordering,
                                     rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrcolor(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     const double*             fraction_to_color,
                                     rocsparse_int*            ncolors,
                                     rocsparse_int*            coloring,
                                     rocsparse_int*            reordering,
                                     rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrcolor(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  nnz,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* csr_val,
                                     const rocsparse_int*           csr_row_ptr,
                                     const rocsparse_int*           csr_col_ind,
                                     const float*                   fraction_to_color,
                                     rocsparse_int*                 ncolors,
                                     rocsparse_int*                 coloring,
                                     rocsparse_int*                 reordering,
                                     rocsparse_mat_info             info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrcolor(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   nnz,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* csr_val,
                                     const rocsparse_int*            csr_row_ptr,
                                     const rocsparse_int*            csr_col_ind,
                                     const double*                   fraction_to_color,
                                     rocsparse_int*                  ncolors,
                                     rocsparse_int*                  coloring,
                                     rocsparse_int*                  reordering,
                                     rocsparse_mat_info              info);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRCOLOR_H */
