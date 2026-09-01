/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_diagonal_solve.hpp"

#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)

#include "rocsparse_diagonal_solve_device.h"
#include "rocsparse_indextype_utils.hpp"

namespace rocsparse
{
    static constexpr uint32_t DIAGONAL_SOLVE_BLOCKSIZE = 256;

    static constexpr uint32_t DIAGONAL_SOLVE_GRID_CAP = 65535;

    template <typename I, typename J, typename T>
    static rocsparse_status diagonal_solve_launch(rocsparse_handle            handle,
                                                  int64_t                     batch_count,
                                                  int64_t                     m,
                                                  int64_t                     nrhs,
                                                  const void*                 alpha_,
                                                  const void*                 diag_ind,
                                                  const void*                 transposed_perm,
                                                  const void*                 val,
                                                  int64_t                     val_batch_stride,
                                                  const void*                 x,
                                                  int64_t                     x_row_stride,
                                                  int64_t                     x_col_stride,
                                                  int64_t                     x_batch_stride,
                                                  void*                       y,
                                                  int64_t                     y_row_stride,
                                                  int64_t                     y_col_stride,
                                                  int64_t                     y_batch_stride,
                                                  void*                       zero_pivot,
                                                  int64_t                     zero_pivot_stride,
                                                  rocsparse_index_base        base,
                                                  rocsparse_diagonal_modifier modifier,
                                                  bool                        conj,
                                                  bool                        conj_x,
                                                  bool                        is_host_mode)
    {
        auto           alpha = reinterpret_cast<const T*>(alpha_);
        const uint32_t gy    = static_cast<uint32_t>(
            (nrhs < static_cast<int64_t>(DIAGONAL_SOLVE_GRID_CAP)) ? nrhs
                                                                      : DIAGONAL_SOLVE_GRID_CAP);
        dim3 blocks((m - 1) / DIAGONAL_SOLVE_BLOCKSIZE + 1, gy, batch_count);
        dim3 threads(DIAGONAL_SOLVE_BLOCKSIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::diagonal_solve_kernel<DIAGONAL_SOLVE_BLOCKSIZE, I, J, T>),
            blocks,
            threads,
            0,
            handle->stream,
            static_cast<J>(m),
            static_cast<J>(nrhs),
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
            reinterpret_cast<const I*>(diag_ind),
            reinterpret_cast<const I*>(transposed_perm),
            reinterpret_cast<const T*>(val),
            val_batch_stride,
            reinterpret_cast<const T*>(x),
            x_row_stride,
            x_col_stride,
            x_batch_stride,
            reinterpret_cast<T*>(y),
            y_row_stride,
            y_col_stride,
            y_batch_stride,
            reinterpret_cast<J*>(zero_pivot),
            zero_pivot_stride,
            base,
            modifier,
            conj,
            conj_x,
            is_host_mode);
        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status diagonal_solve_dispatch(rocsparse_handle            handle,
                                                    rocsparse_indextype         diag_ind_type,
                                                    rocsparse_indextype         col_type,
                                                    int64_t                     batch_count,
                                                    int64_t                     m,
                                                    int64_t                     nrhs,
                                                    const void*                 alpha,
                                                    const void*                 diag_ind,
                                                    const void*                 transposed_perm,
                                                    const void*                 val,
                                                    int64_t                     val_batch_stride,
                                                    const void*                 x,
                                                    int64_t                     x_row_stride,
                                                    int64_t                     x_col_stride,
                                                    int64_t                     x_batch_stride,
                                                    void*                       y,
                                                    int64_t                     y_row_stride,
                                                    int64_t                     y_col_stride,
                                                    int64_t                     y_batch_stride,
                                                    void*                       zero_pivot,
                                                    int64_t                     zero_pivot_stride,
                                                    rocsparse_index_base        base,
                                                    rocsparse_diagonal_modifier modifier,
                                                    bool                        conj,
                                                    bool                        conj_x,
                                                    bool                        is_host_mode)
    {
#define DIAGONAL_SOLVE(I_, J_)                                              \
    diagonal_solve_launch<typename rocsparse::indextype_traits<I_>::type_t, \
                          typename rocsparse::indextype_traits<J_>::type_t, \
                          T>(handle,                                        \
                             batch_count,                                   \
                             m,                                             \
                             nrhs,                                          \
                             alpha,                                         \
                             diag_ind,                                      \
                             transposed_perm,                               \
                             val,                                           \
                             val_batch_stride,                              \
                             x,                                             \
                             x_row_stride,                                  \
                             x_col_stride,                                  \
                             x_batch_stride,                                \
                             y,                                             \
                             y_row_stride,                                  \
                             y_col_stride,                                  \
                             y_batch_stride,                                \
                             zero_pivot,                                    \
                             zero_pivot_stride,                             \
                             base,                                          \
                             modifier,                                      \
                             conj,                                          \
                             conj_x,                                        \
                             is_host_mode)

        if(diag_ind_type == rocsparse_indextype_i32 && col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (DIAGONAL_SOLVE(rocsparse_indextype_i32, rocsparse_indextype_i32)));
        }
        else if(diag_ind_type == rocsparse_indextype_i64 && col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (DIAGONAL_SOLVE(rocsparse_indextype_i64, rocsparse_indextype_i32)));
        }
        else if(diag_ind_type == rocsparse_indextype_i64 && col_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (DIAGONAL_SOLVE(rocsparse_indextype_i64, rocsparse_indextype_i64)));
        }
        else if(diag_ind_type == rocsparse_indextype_i32 && col_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (DIAGONAL_SOLVE(rocsparse_indextype_i32, rocsparse_indextype_i64)));
        }
        else
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                "unsupported index type combination in diagonal_solve");
        }
#undef DIAGONAL_SOLVE
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::diagonal_solve(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_diagonal_modifier modifier,
                                           const void*                 alpha,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_indextype         diag_ind_type,
                                           const void*                 diag_ind,
                                           const void*                 transposed_perm,
                                           int64_t                     nrhs,
                                           const void*                 x,
                                           int64_t                     x_row_stride,
                                           int64_t                     x_col_stride,
                                           int64_t                     x_batch_stride,
                                           void*                       y,
                                           int64_t                     y_row_stride,
                                           int64_t                     y_col_stride,
                                           int64_t                     y_batch_stride,
                                           int64_t                     batch_count,
                                           bool                        conj_x,
                                           void*                       zero_pivot,
                                           int64_t                     zero_pivot_stride,
                                           bool                        is_host_mode)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool conj = (trans == rocsparse_operation_conjugate_transpose);

#define DIAGONAL_SOLVE_DISPATCH(T_)                           \
    rocsparse::diagonal_solve_dispatch<T_>(handle,            \
                                           diag_ind_type,     \
                                           A->col_type,       \
                                           batch_count,       \
                                           A->rows,           \
                                           nrhs,              \
                                           alpha,             \
                                           diag_ind,          \
                                           transposed_perm,   \
                                           A->const_val_data, \
                                           A->batch_stride,   \
                                           x,                 \
                                           x_row_stride,      \
                                           x_col_stride,      \
                                           x_batch_stride,    \
                                           y,                 \
                                           y_row_stride,      \
                                           y_col_stride,      \
                                           y_batch_stride,    \
                                           zero_pivot,        \
                                           zero_pivot_stride, \
                                           A->descr->base,    \
                                           modifier,          \
                                           conj,              \
                                           conj_x,            \
                                           is_host_mode)

    switch(A->data_type)
    {
    case rocsparse_datatype_f32_r:
    {
        RETURN_IF_ROCSPARSE_ERROR((DIAGONAL_SOLVE_DISPATCH(float)));
        return rocsparse_status_success;
    }
    case rocsparse_datatype_f64_r:
    {
        RETURN_IF_ROCSPARSE_ERROR((DIAGONAL_SOLVE_DISPATCH(double)));
        return rocsparse_status_success;
    }
    case rocsparse_datatype_f32_c:
    {
        RETURN_IF_ROCSPARSE_ERROR((DIAGONAL_SOLVE_DISPATCH(rocsparse_float_complex)));
        return rocsparse_status_success;
    }
    case rocsparse_datatype_f64_c:
    {
        RETURN_IF_ROCSPARSE_ERROR((DIAGONAL_SOLVE_DISPATCH(rocsparse_double_complex)));
        return rocsparse_status_success;
    }
    default:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                               "unsupported data type in diagonal_solve");
    }
    }
#undef DIAGONAL_SOLVE_DISPATCH
    return rocsparse_status_success;
}

#endif
