/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HOST_HOST_SPARSE_HPP_
#define ROCALUTION_HOST_HOST_SPARSE_HPP_

#include <cstddef>

#include "../../utils/type_traits.hpp"

typedef enum host_sparse_operation_
{
    host_sparse_operation_none                = 111, /**< Operate with matrix. */
    host_sparse_operation_transpose           = 112, /**< Operate with transpose. */
    host_sparse_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} host_sparse_operation;

typedef enum host_sparse_matrix_type_
{
    host_sparse_matrix_type_general    = 0, /**< general matrix type. */
    host_sparse_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    host_sparse_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    host_sparse_matrix_type_triangular = 3 /**< triangular matrix type. */
} host_sparse_matrix_type;

typedef enum host_sparse_diag_type_
{
    host_sparse_diag_type_non_unit = 0, /**< diagonal entries are non-unity. */
    host_sparse_diag_type_unit     = 1 /**< diagonal entries are unity */
} host_sparse_diag_type;

typedef enum host_sparse_fill_mode_
{
    host_sparse_fill_mode_lower = 0, /**< lower triangular part is stored. */
    host_sparse_fill_mode_upper = 1 /**< upper triangular part is stored. */
} host_sparse_fill_mode;

namespace rocalution
{
    template <typename I, typename J, typename T>
    bool host_csritsv_buffer_size(host_sparse_operation   trans,
                                  J                       m,
                                  I                       nnz,
                                  host_sparse_fill_mode   fill_mode,
                                  host_sparse_diag_type   diag_type,
                                  host_sparse_matrix_type mat_type,
                                  const T*                csr_val,
                                  const I*                csr_row_ptr,
                                  const J*                csr_col_ind,
                                  size_t*                 buffer_size);

    template <typename I, typename J, typename T>
    bool host_csritsv_solve(int*                       host_nmaxiter,
                            const numeric_traits_t<T>* host_tol,
                            numeric_traits_t<T>*       host_history,
                            host_sparse_operation      trans,
                            J                          m,
                            I                          nnz,
                            const T*                   alpha,
                            host_sparse_fill_mode      fill_mode,
                            host_sparse_diag_type      diag_type,
                            host_sparse_matrix_type    mat_type,
                            const T*                   csr_val,
                            const I*                   csr_row_ptr,
                            const J*                   csr_col_ind,
                            const T*                   x,
                            T*                         y,
                            void*                      temp_buffer,
                            J*                         zero_pivot);
} // namespace rocalution

#endif // ROCALUTION_HOST_HOST_SPARSE_HPP_
