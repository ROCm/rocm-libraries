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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_csc_to_csr_descr.hpp"
#include "rocsparse_cscsv.hpp"
#include "rocsparse_csrsv_info.hpp"
#include "rocsparse_trm_info.hpp"

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

rocsparse_status rocsparse::diagonal_solve(rocsparse_handle              handle,
                                           rocsparse_operation           trans,
                                           rocsparse_diagonal_modifier   modifier,
                                           const void*                   alpha,
                                           rocsparse_const_spmat_descr   A,
                                           const rocsparse::spdiag_view& diag,
                                           int64_t                       nrhs,
                                           const void*                   x,
                                           int64_t                       x_row_stride,
                                           int64_t                       x_col_stride,
                                           int64_t                       x_batch_stride,
                                           void*                         y,
                                           int64_t                       y_row_stride,
                                           int64_t                       y_col_stride,
                                           int64_t                       y_batch_stride,
                                           int64_t                       batch_count,
                                           bool                          conj_x,
                                           void*                         zero_pivot,
                                           int64_t                       zero_pivot_stride,
                                           bool                          is_host_mode)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool conj = (trans == rocsparse_operation_conjugate_transpose);

    // The kernel reads the diagonal from the value array with a per-batch stride.
    // The CSC strided-batch setter stores that stride in columns_values_batch_stride
    // and leaves batch_stride unset (unlike CSR, which fills batch_stride). Pick the
    // right field per format, otherwise batched CSC solves reuse batch 0's diagonal.
    const int64_t val_batch_stride
        = (A->format == rocsparse_format_csc) ? A->columns_values_batch_stride : A->batch_stride;

    // The zero-pivot buffer is typed like the analysis pivot, i.e. the CSR col
    // index type. build_csr_from_csc swaps row/col, so for CSC that is row_type;
    // using col_type would mismatch the analysis pivot on mixed-index matrices.
    const rocsparse_indextype zero_pivot_indextype
        = (A->format == rocsparse_format_csc) ? A->row_type : A->col_type;

#define DIAGONAL_SOLVE_DISPATCH(T_)                              \
    rocsparse::diagonal_solve_dispatch<T_>(handle,               \
                                           diag.offset_type,     \
                                           zero_pivot_indextype, \
                                           batch_count,          \
                                           A->rows,              \
                                           nrhs,                 \
                                           alpha,                \
                                           diag.diag_ind,        \
                                           diag.transposed_perm, \
                                           A->const_val_data,    \
                                           val_batch_stride,     \
                                           x,                    \
                                           x_row_stride,         \
                                           x_col_stride,         \
                                           x_batch_stride,       \
                                           y,                    \
                                           y_row_stride,         \
                                           y_col_stride,         \
                                           y_batch_stride,       \
                                           zero_pivot,           \
                                           zero_pivot_stride,    \
                                           A->descr->base,       \
                                           modifier,             \
                                           conj,                 \
                                           conj_x,               \
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

rocsparse_status rocsparse::build_spdiag_view(rocsparse_const_spmat_descr A,
                                              rocsparse_operation         trans,
                                              rocsparse_csrsv_info        info,
                                              rocsparse::spdiag_view*     view)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
        (info == nullptr) ? rocsparse_status_invalid_pointer : rocsparse_status_success,
        "the analysis stage must be executed before a diagonal solve");

    const rocsparse::trm_info_t* trm = nullptr;
    switch(A->format)
    {
    case rocsparse_format_csr:
    {
        trm = info->get(trans, A->descr->fill_mode);
        break;
    }
    case rocsparse_format_csc:
    {
#if defined(ROCSPARSE_WITH_CSC_TRSV) || defined(ROCSPARSE_WITH_CSC_TRSM)
        _rocsparse_mat_descr   descr_csr;
        _rocsparse_spmat_descr mat_csr;
        rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);
        trm = info->get(rocsparse::cscsv_operation_to_csr(trans), descr_csr.fill_mode);
        break;
#else
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
#endif
    }
    default:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_not_implemented,
            "diagonal solve is only implemented for CSR and CSC matrices");
    }
    }

    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
        (trm == nullptr || trm->get_diag_ind() == nullptr) ? rocsparse_status_invalid_pointer
                                                           : rocsparse_status_success,
        "the analysis stage did not provide the diagonal offsets required by the diagonal solve");

    view->offset_type     = trm->get_offset_indextype();
    view->diag_ind        = trm->get_diag_ind();
    view->transposed_perm = trm->get_transposed_perm();
    return rocsparse_status_success;
}

namespace rocsparse
{
    // Shared implementation for the format-specific diagonal solves. Everything is
    // identical between CSR and CSC except the index type used for the analysis
    // pivot, which the callers pass in.
    static rocsparse_status diagonal_solve_from_info(rocsparse_handle            handle,
                                                     rocsparse_operation         trans,
                                                     rocsparse_diagonal_modifier modifier,
                                                     const void*                 alpha,
                                                     rocsparse_const_spmat_descr A,
                                                     rocsparse_csrsv_info        info,
                                                     rocsparse_indextype         pivot_indextype,
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
                                                     bool                        conj_x)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::spdiag_view diag{};
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::build_spdiag_view(A, trans, info, &diag));

        hipStream_t stream = handle->stream;

        info->create_singularity_numeric_exact(batch_count, pivot_indextype, stream);
        auto numeric_exact = info->get_singularity_numeric_exact();
        if(pivot_indextype == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::assign_device_async<int32_t>(batch_count,
                                                        (int32_t*)numeric_exact->get_position(),
                                                        (const int32_t*)info->get_position(),
                                                        stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::assign_device_async<int64_t>(batch_count,
                                                        (int64_t*)numeric_exact->get_position(),
                                                        (const int64_t*)info->get_position(),
                                                        stream));
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::diagonal_solve(handle,
                                      trans,
                                      modifier,
                                      alpha,
                                      A,
                                      diag,
                                      nrhs,
                                      x,
                                      x_row_stride,
                                      x_col_stride,
                                      x_batch_stride,
                                      y,
                                      y_row_stride,
                                      y_col_stride,
                                      y_batch_stride,
                                      batch_count,
                                      conj_x,
                                      numeric_exact->get_position(),
                                      1,
                                      handle->pointer_mode == rocsparse_pointer_mode_host));

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::diagonal_solve_csr(rocsparse_handle            handle,
                                               rocsparse_operation         trans,
                                               rocsparse_diagonal_modifier modifier,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_csrsv_info        info,
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
                                               bool                        conj_x)
{
    ROCSPARSE_ROUTINE_TRACE;

    // The analysis pivot is typed like the CSR column index.
    return rocsparse::diagonal_solve_from_info(handle,
                                               trans,
                                               modifier,
                                               alpha,
                                               A,
                                               info,
                                               A->col_type,
                                               nrhs,
                                               x,
                                               x_row_stride,
                                               x_col_stride,
                                               x_batch_stride,
                                               y,
                                               y_row_stride,
                                               y_col_stride,
                                               y_batch_stride,
                                               batch_count,
                                               conj_x);
}

rocsparse_status rocsparse::diagonal_solve_csc(rocsparse_handle            handle,
                                               rocsparse_operation         trans,
                                               rocsparse_diagonal_modifier modifier,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_csrsv_info        info,
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
                                               bool                        conj_x)
{
    ROCSPARSE_ROUTINE_TRACE;

    // build_csr_from_csc swaps row/col for CSC, so the analysis pivot is typed
    // like the CSC row index rather than the column index.
    return rocsparse::diagonal_solve_from_info(handle,
                                               trans,
                                               modifier,
                                               alpha,
                                               A,
                                               info,
                                               A->row_type,
                                               nrhs,
                                               x,
                                               x_row_stride,
                                               x_col_stride,
                                               x_batch_stride,
                                               y,
                                               y_row_stride,
                                               y_col_stride,
                                               y_batch_stride,
                                               batch_count,
                                               conj_x);
}

#endif
