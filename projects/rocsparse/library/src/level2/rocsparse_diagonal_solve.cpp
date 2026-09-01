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
#include "rocsparse_dnmat_descr.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_trm_info.hpp"

namespace rocsparse
{
    // Layout of a dense operand of the diagonal solve: element (row, col) of batch
    // entry b lives at values[b * batch_stride + row * row_stride + col * col_stride].
    // A dense vector has a single column and folds its increment into row_stride,
    // while a dense matrix folds its leading dimension, its storage order and, for
    // the right-hand side, an optional transposition into the two remaining strides.
    struct dense_layout
    {
        int64_t row_stride{1};
        int64_t col_stride{0};
        int64_t batch_stride{0};
    };

    // Grid mapping: x walks the rows, y walks the right-hand sides and z walks the
    // batch. The y and z dimensions of a grid are capped, so both are consumed with
    // a grid-stride loop: over the batch here, and over the right-hand sides inside
    // the device function.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void diagonal_solve_kernel(J       m,
                               J       nrhs,
                               int64_t batch_count,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                               const I* __restrict__ diag_ind,
                               const I* __restrict__ transposed_perm,
                               const T* __restrict__ val,
                               int64_t val_batch_stride,
                               const T* __restrict__ x,
                               int64_t x_row_stride,
                               int64_t x_col_stride,
                               int64_t x_batch_stride,
                               T* __restrict__ y,
                               int64_t y_row_stride,
                               int64_t y_col_stride,
                               int64_t y_batch_stride,
                               J* __restrict__ zero_pivot,
                               int64_t                     zero_pivot_stride,
                               rocsparse_index_base        base,
                               rocsparse_diagonal_modifier modifier,
                               bool                        conj,
                               bool                        conj_x,
                               bool                        is_host_mode)
    {
        const auto row = static_cast<J>(hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x);
        if(row >= m)
        {
            return;
        }

        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        for(int64_t batch = hipBlockIdx_z; batch < batch_count; batch += hipGridDim_z)
        {
            rocsparse::diagonal_solve_device(row,
                                             nrhs,
                                             static_cast<J>(hipBlockIdx_y),
                                             static_cast<J>(hipGridDim_y),
                                             alpha,
                                             diag_ind,
                                             transposed_perm,
                                             val + batch * val_batch_stride,
                                             x + batch * x_batch_stride,
                                             x_row_stride,
                                             x_col_stride,
                                             y + batch * y_batch_stride,
                                             y_row_stride,
                                             y_col_stride,
                                             zero_pivot + batch * zero_pivot_stride,
                                             base,
                                             modifier,
                                             conj,
                                             conj_x);
        }
    }

    template <typename I, typename J, typename T>
    static rocsparse_status diagonal_solve_launch(rocsparse_handle               handle,
                                                  int64_t                        batch_count,
                                                  int64_t                        m,
                                                  int64_t                        nrhs,
                                                  const void*                    alpha_,
                                                  const void*                    diag_ind,
                                                  const void*                    transposed_perm,
                                                  const void*                    val,
                                                  int64_t                        val_batch_stride,
                                                  const void*                    x,
                                                  const rocsparse::dense_layout& x_layout,
                                                  void*                          y,
                                                  const rocsparse::dense_layout& y_layout,
                                                  void*                          zero_pivot,
                                                  int64_t                        zero_pivot_stride,
                                                  rocsparse_index_base           base,
                                                  rocsparse_diagonal_modifier    modifier,
                                                  bool                           conj,
                                                  bool                           conj_x,
                                                  bool                           is_host_mode)
    {
        static constexpr uint32_t BLOCKSIZE = 1024;

        auto alpha = reinterpret_cast<const T*>(alpha_);

        // The y and z dimensions of a grid are capped, so both counts are clamped
        // here and the kernel loops over whatever does not fit.
        dim3 blocks((m - 1) / BLOCKSIZE + 1,
                    rocsparse::get_batch_grid_size(nrhs),
                    rocsparse::get_batch_grid_size(batch_count));
        dim3 threads(BLOCKSIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::diagonal_solve_kernel<BLOCKSIZE, I, J, T>),
                                           blocks,
                                           threads,
                                           0,
                                           handle->stream,
                                           static_cast<J>(m),
                                           static_cast<J>(nrhs),
                                           batch_count,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                                           reinterpret_cast<const I*>(diag_ind),
                                           reinterpret_cast<const I*>(transposed_perm),
                                           reinterpret_cast<const T*>(val),
                                           val_batch_stride,
                                           reinterpret_cast<const T*>(x),
                                           x_layout.row_stride,
                                           x_layout.col_stride,
                                           x_layout.batch_stride,
                                           reinterpret_cast<T*>(y),
                                           y_layout.row_stride,
                                           y_layout.col_stride,
                                           y_layout.batch_stride,
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
    static rocsparse_status diagonal_solve_dispatch(rocsparse_handle               handle,
                                                    rocsparse_indextype            diag_ind_type,
                                                    rocsparse_indextype            col_type,
                                                    int64_t                        batch_count,
                                                    int64_t                        m,
                                                    int64_t                        nrhs,
                                                    const void*                    alpha,
                                                    const void*                    diag_ind,
                                                    const void*                    transposed_perm,
                                                    const void*                    val,
                                                    int64_t                        val_batch_stride,
                                                    const void*                    x,
                                                    const rocsparse::dense_layout& x_layout,
                                                    void*                          y,
                                                    const rocsparse::dense_layout& y_layout,
                                                    void*                          zero_pivot,
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
                             x_layout,                                      \
                             y,                                             \
                             y_layout,                                      \
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

    static rocsparse_status diagonal_solve(rocsparse_handle               handle,
                                           rocsparse_operation            trans,
                                           rocsparse_diagonal_modifier    modifier,
                                           const void*                    alpha,
                                           rocsparse_const_spmat_descr    A,
                                           const rocsparse::spdiag_view&  diag,
                                           int64_t                        nrhs,
                                           const void*                    x,
                                           const rocsparse::dense_layout& x_layout,
                                           void*                          y,
                                           const rocsparse::dense_layout& y_layout,
                                           int64_t                        batch_count,
                                           bool                           conj_x,
                                           void*                          zero_pivot,
                                           int64_t                        zero_pivot_stride,
                                           bool                           is_host_mode)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool conj = (trans == rocsparse_operation_conjugate_transpose);

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
                                           A->batch_stride,      \
                                           x,                    \
                                           x_layout,             \
                                           y,                    \
                                           y_layout,             \
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
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_bsr:
    case rocsparse_format_ell:
    case rocsparse_format_bell:
    case rocsparse_format_sell:
    {
        // LCOV_EXCL_START
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_not_implemented,
            "diagonal solve is only implemented for CSR and CSC matrices");
        // LCOV_EXCL_STOP
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
    // Consumes what the analysis stage produced - the diagonal offsets and the
    // structural pivot - seeds the numeric pivot buffer from it, then solves.
    // Everything is identical between CSR and CSC except the index type used for
    // the analysis pivot, which the callers pass in.
    static rocsparse_status diagonal_solve_using_analysis(rocsparse_handle            handle,
                                                          rocsparse_operation         trans,
                                                          rocsparse_diagonal_modifier modifier,
                                                          const void*                 alpha,
                                                          rocsparse_const_spmat_descr A,
                                                          rocsparse_csrsv_info        info,
                                                          rocsparse_indextype pivot_indextype,
                                                          int64_t             nrhs,
                                                          const void*         x,
                                                          const rocsparse::dense_layout& x_layout,
                                                          void*                          y,
                                                          const rocsparse::dense_layout& y_layout,
                                                          int64_t batch_count,
                                                          bool    conj_x)
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
                                      x_layout,
                                      y,
                                      y_layout,
                                      batch_count,
                                      conj_x,
                                      numeric_exact->get_position(),
                                      1,
                                      handle->pointer_mode == rocsparse_pointer_mode_host));

        return rocsparse_status_success;
    }

    // A dense vector carries a single right-hand side, so only its increment and its
    // batch stride are needed to address it.
    static rocsparse_status diagonal_solve_dnvec(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 rocsparse_diagonal_modifier modifier,
                                                 const void*                 alpha,
                                                 rocsparse_const_spmat_descr A,
                                                 rocsparse_csrsv_info        info,
                                                 rocsparse_indextype         pivot_indextype,
                                                 rocsparse_const_dnvec_descr x,
                                                 rocsparse_dnvec_descr       y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse::dense_layout x_layout{x->inc, 0, x->batch_stride};
        const rocsparse::dense_layout y_layout{y->inc, 0, y->batch_stride};

        return rocsparse::diagonal_solve_using_analysis(handle,
                                                        trans,
                                                        modifier,
                                                        alpha,
                                                        A,
                                                        info,
                                                        pivot_indextype,
                                                        1,
                                                        x->const_values,
                                                        x_layout,
                                                        y->values,
                                                        y_layout,
                                                        y->batch_count,
                                                        false);
    }

    // A dense matrix addresses (row, col) with its leading dimension on one side and
    // a unit stride on the other, depending on its storage order. Transposing the
    // right-hand side swaps the two strides, and conjugating it is deferred to the
    // kernel so that no temporary copy of X is needed.
    static rocsparse_status diagonal_solve_dnmat(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 rocsparse_diagonal_modifier modifier,
                                                 const void*                 alpha,
                                                 rocsparse_const_spmat_descr A,
                                                 rocsparse_csrsv_info        info,
                                                 rocsparse_indextype         pivot_indextype,
                                                 rocsparse_operation         x_operation,
                                                 rocsparse_const_dnmat_descr X,
                                                 rocsparse_dnmat_descr       Y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool x_transposed = (x_operation != rocsparse_operation_none);
        const bool x_col_major  = (X->order == rocsparse_order_column);
        const bool y_col_major  = (Y->order == rocsparse_order_column);

        const rocsparse::dense_layout x_layout{
            x_transposed ? (x_col_major ? X->ld : 1) : (x_col_major ? 1 : X->ld),
            x_transposed ? (x_col_major ? 1 : X->ld) : (x_col_major ? X->ld : 1),
            0};
        const rocsparse::dense_layout y_layout{y_col_major ? 1 : Y->ld, y_col_major ? Y->ld : 1, 0};

        // rocsparse_sptrsm rejects batched operands, hence the single batch entry.
        return rocsparse::diagonal_solve_using_analysis(
            handle,
            trans,
            modifier,
            alpha,
            A,
            info,
            pivot_indextype,
            Y->cols,
            X->const_values,
            x_layout,
            Y->values,
            y_layout,
            1,
            x_operation == rocsparse_operation_conjugate_transpose);
    }
}

// The analysis pivot is typed like the CSR column index, whereas build_csr_from_csc
// swaps row and col, so for CSC it is typed like the CSC row index instead.

rocsparse_status rocsparse::diagonal_solve_csr(rocsparse_handle            handle,
                                               rocsparse_operation         trans,
                                               rocsparse_diagonal_modifier modifier,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_csrsv_info        info,
                                               rocsparse_const_dnvec_descr x,
                                               rocsparse_dnvec_descr       y)
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse::diagonal_solve_dnvec(
        handle, trans, modifier, alpha, A, info, A->col_type, x, y);
}

rocsparse_status rocsparse::diagonal_solve_csc(rocsparse_handle            handle,
                                               rocsparse_operation         trans,
                                               rocsparse_diagonal_modifier modifier,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_csrsv_info        info,
                                               rocsparse_const_dnvec_descr x,
                                               rocsparse_dnvec_descr       y)
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse::diagonal_solve_dnvec(
        handle, trans, modifier, alpha, A, info, A->row_type, x, y);
}

rocsparse_status rocsparse::diagonal_solve_csr(rocsparse_handle            handle,
                                               rocsparse_operation         trans,
                                               rocsparse_diagonal_modifier modifier,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_csrsv_info        info,
                                               rocsparse_operation         x_operation,
                                               rocsparse_const_dnmat_descr X,
                                               rocsparse_dnmat_descr       Y)
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse::diagonal_solve_dnmat(
        handle, trans, modifier, alpha, A, info, A->col_type, x_operation, X, Y);
}

rocsparse_status rocsparse::diagonal_solve_csc(rocsparse_handle            handle,
                                               rocsparse_operation         trans,
                                               rocsparse_diagonal_modifier modifier,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_csrsv_info        info,
                                               rocsparse_operation         x_operation,
                                               rocsparse_const_dnmat_descr X,
                                               rocsparse_dnmat_descr       Y)
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse::diagonal_solve_dnmat(
        handle, trans, modifier, alpha, A, info, A->row_type, x_operation, X, Y);
}

#endif
