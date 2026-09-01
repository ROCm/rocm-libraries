/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

#include "../../level1/rocsparse_gthr.hpp"
#include "../../level2/rocsparse_csrsv.hpp"
#include "../csrsm_device.h"

#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "../../level2/rocsparse_csrsv.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    template <uint32_t BLOCKSIZE,
              bool     SLEEP,
              typename I,
              typename J,
              typename T,
              bool A_OP_CONJUGATE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsm(rocsparse_operation transB,
               J                   m,
               J                   nrhs,
               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
               const I* __restrict__ csr_row_ptr,
               const J* __restrict__ csr_col_ind,
               const T* __restrict__ csr_val,
               int64_t csr_val_stride,
               T*      B,
               int64_t ldb,
               int64_t B_stride,
               int* __restrict__ done_array,
               int64_t done_array_stride,
               const J* __restrict__ map,
               J*                   zero_pivot,
               int64_t              zero_pivot_stride,
               rocsparse_index_base idx_base,
               rocsparse_fill_mode  fill_mode,
               rocsparse_diag_type  diag_type,
               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        //
        // Batch index.
        //
        const auto batch_index = hipBlockIdx_y;

        //
        // Call batch sample.
        //
        rocsparse::csrsm_device<BLOCKSIZE, SLEEP, I, J, T, A_OP_CONJUGATE>(
            transB,
            m,
            nrhs,
            alpha,
            csr_row_ptr,
            csr_col_ind,
            csr_val + csr_val_stride * batch_index,
            B + B_stride * batch_index,
            ldb,
            done_array + done_array_stride * batch_index,
            map,
            zero_pivot + zero_pivot_stride * batch_index,
            idx_base,
            fill_mode,
            diag_type);
    }

    template <uint32_t BLOCKSIZE,
              bool     SLEEP,
              typename I,
              typename J,
              typename T,
              bool A_OP_CONJUGATE>
    static rocsparse_status csrsm_launch(rocsparse_handle            handle,
                                         int64_t                     nrhs,
                                         rocsparse_operation         op_B,
                                         rocsparse_const_spmat_descr A,
                                         rocsparse_const_dnvec_descr alpha,
                                         rocsparse_dnmat_descr       B,
                                         int32_t*                    done_array,
                                         int64_t                     done_array_stride,
                                         const void*                 map_,
                                         void*                       zero_pivot,
                                         int64_t                     zero_pivot_stride,
                                         rocsparse_fill_mode         fill_mode,
                                         rocsparse_diag_type         diag_type,
                                         bool                        is_host_mode)
    {

        const int64_t m = A->rows;

        int32_t blockdim = 512;
        while(nrhs <= blockdim && blockdim > 32)
        {
            blockdim >>= 1;
        }
        blockdim <<= 1;

        const auto alpha_scalar = reinterpret_cast<const T*>(alpha->const_values);

        const dim3 csrsm_blocks(((nrhs - 1) / blockdim + 1) * m, B->batch_count);

        const dim3 csrsm_threads(blockdim);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrsm<BLOCKSIZE, SLEEP, I, J, T, A_OP_CONJUGATE>),
                                           csrsm_blocks,
                                           csrsm_threads,
                                           0,
                                           handle->stream,
                                           op_B,
                                           m,
                                           nrhs,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_scalar),
                                           reinterpret_cast<const I*>(A->const_row_data),
                                           reinterpret_cast<const J*>(A->const_col_data),
                                           reinterpret_cast<const T*>(A->const_val_data),
                                           A->batch_stride,
                                           reinterpret_cast<T*>(B->values),
                                           B->ld,
                                           B->batch_stride,
                                           reinterpret_cast<int32_t*>(done_array),
                                           done_array_stride,
                                           reinterpret_cast<const J*>(map_),
                                           reinterpret_cast<J*>(zero_pivot),
                                           zero_pivot_stride,
                                           A->idx_base,
                                           fill_mode,
                                           diag_type,
                                           is_host_mode);
        return rocsparse_status_success;
    }

}

typedef rocsparse_status (*csrsm_launch_t)(rocsparse_handle            handle,
                                           int64_t                     nrhs,
                                           rocsparse_operation         op_B,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_const_dnvec_descr alpha,
                                           rocsparse_dnmat_descr       B,
                                           int32_t*                    done_array,
                                           int64_t                     done_array_stride,
                                           const void*                 map_,
                                           void*                       zero_pivot,
                                           int64_t                     zero_pivot_stride,
                                           rocsparse_fill_mode         fill_mode,
                                           rocsparse_diag_type         diag_type,
                                           bool                        is_host_mode);

template <uint32_t BLOCKSIZE, bool SLEEP, typename I, typename J>
static csrsm_launch_t find_csrsm_launch_T(rocsparse_datatype val_type, bool A_load_conjugate)
{
    switch(val_type)
    {
    case rocsparse_datatype_f32_r:
    {
        return rocsparse::csrsm_launch<BLOCKSIZE, SLEEP, I, J, float, false>;
    }
    case rocsparse_datatype_f64_r:
    {
        return rocsparse::csrsm_launch<BLOCKSIZE, SLEEP, I, J, double, false>;
    }

    case rocsparse_datatype_f32_c:
    {

        if(A_load_conjugate)
            return rocsparse::csrsm_launch<BLOCKSIZE, SLEEP, I, J, rocsparse_float_complex, true>;
        else
            return rocsparse::csrsm_launch<BLOCKSIZE, SLEEP, I, J, rocsparse_float_complex, false>;
    }
    case rocsparse_datatype_f64_c:
    {
        if(A_load_conjugate)
            return rocsparse::csrsm_launch<BLOCKSIZE, SLEEP, I, J, rocsparse_double_complex, true>;
        else
            return rocsparse::csrsm_launch<BLOCKSIZE, SLEEP, I, J, rocsparse_double_complex, false>;
    }

    case rocsparse_datatype_bf16_r:
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
    }
}

template <uint32_t BLOCKSIZE, bool SLEEP, typename I>
static csrsm_launch_t find_csrsm_launch_J(rocsparse_indextype col_type,
                                          rocsparse_datatype  val_type,
                                          bool                A_load_conjugate)
{
    switch(col_type)
    {
    case rocsparse_indextype_i32:
    {
        return find_csrsm_launch_T<BLOCKSIZE, SLEEP, I, int32_t>(val_type, A_load_conjugate);
    }
    case rocsparse_indextype_i64:
    {
        return find_csrsm_launch_T<BLOCKSIZE, SLEEP, I, int64_t>(val_type, A_load_conjugate);
    }
    case deprecated_rocsparse_indextype_u16:
    {
        return nullptr;
    }
    }
    return nullptr;
}

template <uint32_t BLOCKSIZE, bool SLEEP>
static csrsm_launch_t find_csrsm_launch_I(rocsparse_indextype row_type,
                                          rocsparse_indextype col_type,
                                          rocsparse_datatype  val_type,
                                          bool                A_load_conjugate)
{
    switch(row_type)
    {
    case rocsparse_indextype_i32:
    {
        return find_csrsm_launch_J<BLOCKSIZE, SLEEP, int32_t>(col_type, val_type, A_load_conjugate);
    }
    case rocsparse_indextype_i64:
    {
        return find_csrsm_launch_J<BLOCKSIZE, SLEEP, int64_t>(col_type, val_type, A_load_conjugate);
    }
    case deprecated_rocsparse_indextype_u16:
    {
        return nullptr;
    }
    }
    return nullptr;
}

static csrsm_launch_t find_csrsm_launch(rocsparse_handle    handle,
                                        int32_t             blockdim,
                                        rocsparse_indextype row_type,
                                        rocsparse_indextype col_type,
                                        rocsparse_datatype  val_type,
                                        bool                A_load_conjugate)
{
    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
    const int         asicRev       = handle->asic_rev;
    const bool SLEEP = (gcn_arch_name == rocsparse::rocsparse_arch_names::gfx908 && asicRev < 2);

    if(blockdim == 64)
    {
        if(SLEEP)
        {
            return find_csrsm_launch_I<64, true>(row_type, col_type, val_type, A_load_conjugate);
        }
        else
        {
            return find_csrsm_launch_I<64, false>(row_type, col_type, val_type, A_load_conjugate);
        }
    }
    else if(blockdim == 128)
    {
        if(SLEEP)
        {
            return find_csrsm_launch_I<128, true>(row_type, col_type, val_type, A_load_conjugate);
        }
        else
        {
            return find_csrsm_launch_I<128, false>(row_type, col_type, val_type, A_load_conjugate);
        }
    }
    else if(blockdim == 256)
    {
        if(SLEEP)
        {
            return find_csrsm_launch_I<256, true>(row_type, col_type, val_type, A_load_conjugate);
        }
        else
        {
            return find_csrsm_launch_I<256, false>(row_type, col_type, val_type, A_load_conjugate);
        }
    }
    else if(blockdim == 512)
    {
        if(SLEEP)
        {
            return find_csrsm_launch_I<512, true>(row_type, col_type, val_type, A_load_conjugate);
        }
        else
        {
            return find_csrsm_launch_I<512, false>(row_type, col_type, val_type, A_load_conjugate);
        }
    }
    else if(blockdim == 1024)
    {
        if(SLEEP)
        {
            return find_csrsm_launch_I<1024, true>(row_type, col_type, val_type, A_load_conjugate);
        }
        else
        {
            return find_csrsm_launch_I<1024, false>(row_type, col_type, val_type, A_load_conjugate);
        }
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

rocsparse_status rocsparse::spmat_transpose_update_values(rocsparse_handle            handle,
                                                          rocsparse_spmat_descr       target,
                                                          rocsparse_const_spmat_descr source,
                                                          rocsparse::trm_info_t*      trm_info)
{

    // Gather values
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::gthr_strided_batched(handle,
                                                               target->batch_count,
                                                               source->nnz,
                                                               source->data_type,
                                                               source->const_val_data,
                                                               source->batch_stride,
                                                               source->data_type,
                                                               target->val_data,
                                                               target->batch_stride,
                                                               trm_info->get_offset_indextype(),
                                                               trm_info->get_transposed_perm(),
                                                               rocsparse_index_base_zero)));

#if 0
  if(conjugate)
    {

      RETURN_IF_ROCSPARSE_ERROR(rocsparse::conjugate_strided_batched(handle,
								     target->batch_count,
								     target->nnz,
								     target->data_type,
								     target->val_data,
								     target->batch_stride));
    }
#endif
    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_compute(rocsparse_handle            handle,
                                          const int64_t               nrhs,
                                          rocsparse_operation         op_A,
                                          rocsparse_operation         op_B,
                                          rocsparse_const_dnvec_descr alpha,
                                          rocsparse_const_spmat_descr A,
                                          rocsparse_dnmat_descr       B,
                                          rocsparse_csrsm_info        csrsm_info,
                                          size_t                      buffer_size_in_bytes,
                                          void*                       buffer,
                                          rocsparse_error*            p_error)
{
    static constexpr bool A_load_conjugate = false;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_compute(handle,
                                                       nrhs,
                                                       op_A,
                                                       A_load_conjugate,
                                                       op_B,
                                                       alpha,
                                                       A,
                                                       B,
                                                       csrsm_info,
                                                       buffer_size_in_bytes,
                                                       buffer,
                                                       p_error));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_compute(rocsparse_handle            handle,
                                          const int64_t               nrhs,
                                          rocsparse_operation         op_A,
                                          bool                        A_load_conjugate,
                                          rocsparse_operation         op_B,
                                          rocsparse_const_dnvec_descr alpha,
                                          rocsparse_const_spmat_descr A,
                                          rocsparse_dnmat_descr       B,
                                          rocsparse_csrsm_info        csrsm_info,
                                          size_t                      buffer_size_in_bytes,
                                          void*                       buffer,
                                          rocsparse_error*            p_error)
{

    rocsparse_host_assert(A_load_conjugate && op_A != rocsparse_operation_none,
                          "That's not the supposed configuration");
    //
    // It is assumed that B is transposed and has dimension nrhs x M
    //
    ROCSPARSE_ROUTINE_TRACE;
    const int64_t M = A->rows;
    if(M == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    const auto case_transpose_A = (op_A == rocsparse_operation_transpose)
                                  || (op_A == rocsparse_operation_conjugate_transpose);

    const auto case_transform_B
        = (op_B == rocsparse_operation_none && B->order == rocsparse_order_column);
    if(nrhs == 1)
    {
        const int64_t          b_batch_count  = B->batch_count;
        const int64_t          b_batch_stride = B->batch_stride;
        auto                   buf            = reinterpret_cast<char*>(buffer);
        _rocsparse_dnvec_descr b_st(
            b_batch_count,
            M,
            B->data_type,
            B->const_values,
            B->values,
            (op_B == rocsparse_operation_none && B->order == rocsparse_order_column) ? 1 : B->ld,
            b_batch_stride);
        rocsparse_dnvec_descr b = &b_st;

        const int64_t          y_size         = M;
        const int64_t          y_batch_count  = B->batch_count;
        const int64_t          y_batch_stride = (B->batch_count == 1) ? 0 : M;
        const int64_t          y_inc          = 1;
        const auto             y_datatype     = B->data_type;
        _rocsparse_dnvec_descr y_st(
            y_batch_count, y_size, y_datatype, buf, buf, y_inc, y_batch_stride);

        rocsparse_dnvec_descr y = &y_st;

        // y->size because y_inc = 1;
        const size_t nbytes = rocsparse::align_size(rocsparse::datatype_sizeof(B->data_type)
                                                    * y->size * y->batch_count);
        buf += nbytes;
        buffer_size_in_bytes -= nbytes;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                         op_A,
                                                         A_load_conjugate,
                                                         alpha->data_type,
                                                         alpha->const_values,
                                                         alpha->batch_stride,
                                                         A,
                                                         b,
                                                         y,
                                                         rocsparse_solve_policy_auto,
                                                         csrsm_info,
                                                         buffer_size_in_bytes,
                                                         buf));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnvec_copy_data(handle, nullptr, y, b, p_error));

        return rocsparse_status_success;
    }

    const auto A_batch_count  = (A->batch_stride > 0) ? A->batch_count : 1;
    const auto A_batch_stride = A->batch_stride;

    // Stream
    int32_t blockdim = 512;
    while(nrhs <= blockdim && blockdim > 32)
    {
        blockdim >>= 1;
    }
    blockdim <<= 1;

    const int narrays = (nrhs - 1) / blockdim + 1;
    //
    // Buffer
    // header: 256
    // done_array: sizeof(int32_t) * M * narrays * B->batch_count
    //
    size_t nbytes = 256 + rocsparse::align_size(sizeof(int32_t) * M * narrays * B->batch_count);

    if(case_transpose_A)
    {
        nbytes += rocsparse::align_size(rocsparse::datatype_sizeof(A->data_type) * A->nnz
                                        * A_batch_count);
    }

    if(case_transform_B)
    {
        nbytes += rocsparse::align_size(rocsparse::datatype_sizeof(B->data_type) * (A->rows * nrhs)
                                        * B->batch_count);
    }
    RETURN_IF_ROCSPARSE_ERROR((nbytes <= buffer_size_in_bytes) ? rocsparse_status_success
                                                               : rocsparse_status_invalid_size);

    // rocsparse::align_size(rocsparse::datatype_sizeof(B->data_type) * y->size * y->batch_count);

    auto buf = reinterpret_cast<char*>(buffer);
    buf += 256;

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.

    const size_t  done_array_nbytes = sizeof(int32_t) * M * narrays * B->batch_count;
    int32_t*      done_array        = reinterpret_cast<int32_t*>(buf);
    const int64_t done_array_stride = M * narrays;
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(done_array, 0, done_array_nbytes, handle->stream));
    buf += rocsparse::align_size(done_array_nbytes);

    rocsparse_dnmat_descr Bt{};
    rocsparse_spmat_descr At{};

    _rocsparse_dnmat_descr Bt_st{true,
                                 A->rows,
                                 nrhs,
                                 nrhs,
                                 buf,
                                 buf,
                                 B->data_type,
                                 rocsparse_order_row,
                                 B->batch_count,
                                 (B->batch_stride == 0) ? 0 : (A->rows * nrhs)};

    if(case_transform_B)
    {
        Bt = &Bt_st;
        const size_t Bt_nbytes
            = rocsparse::datatype_sizeof(Bt->data_type) * Bt->rows * Bt->cols * Bt->batch_count;
        buf += rocsparse::align_size(Bt_nbytes);
    }

    rocsparse::trm_info_t* trm_info = csrsm_info->get(op_A, A->descr->fill_mode);

    _rocsparse_spmat_descr At_st(rocsparse_format_csr,
                                 A_batch_count,
                                 M,
                                 M,
                                 A->nnz,
                                 A->data_type,
                                 buf,
                                 buf,
                                 A_batch_stride,
                                 trm_info->get_offset_indextype(),
                                 trm_info->get_transposed_row_ptr(),
                                 trm_info->get_transposed_row_ptr(),
                                 0,
                                 trm_info->get_index_indextype(),
                                 trm_info->get_transposed_col_ind(),
                                 trm_info->get_transposed_col_ind(),
                                 0,
                                 A->descr->base,
                                 nullptr, //A->descr,
                                 nullptr); //A->info);

    if(case_transpose_A)
    {
        At = &At_st;
    }

    rocsparse_const_spmat_descr matrix = (case_transpose_A) ? At : A;
    rocsparse_dnmat_descr       rhs    = (case_transform_B) ? Bt : B;
    // If diag type is unit, re-initialize zero pivot to remove structural zeros
    const auto diag_type = A->descr->diag_type;

    csrsm_info->create_singularity_numeric_exact(B->batch_count, A->col_type, handle->stream);
    auto numeric_exact_position = csrsm_info->get_singularity_numeric_exact();

    switch(diag_type)
    {
    case rocsparse_diag_type_unit:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
            1, A->col_type, csrsm_info->get_position(), handle->stream));
        if(A->col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                B->batch_count,
                (int32_t*)csrsm_info->get_singularity_numeric_exact()->get_position(),
                (const int32_t*)csrsm_info->get_position(),
                handle->stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                B->batch_count,
                (int64_t*)csrsm_info->get_singularity_numeric_exact()->get_position(),
                (const int64_t*)csrsm_info->get_position(),
                handle->stream));
        }

        break;
    }
    case rocsparse_diag_type_non_unit:
    {
        if(A->col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                B->batch_count,
                (int32_t*)csrsm_info->get_singularity_numeric_exact()->get_position(),
                (const int32_t*)csrsm_info->get_position(),
                handle->stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                B->batch_count,
                (int64_t*)csrsm_info->get_singularity_numeric_exact()->get_position(),
                (const int64_t*)csrsm_info->get_position(),
                handle->stream));
        }
        break;
    }
    }

    // Transpose B if B is not transposed yet to improve performance
    if(case_transform_B)
    {
        // Leading dimension for transposed B
        rocsparse_const_dnvec_descr no_scale = nullptr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnmat_copy_data(handle, no_scale, B, rhs, p_error));
    }

    const rocsparse_fill_mode fill_mode
        = (case_transpose_A)
              ? ((A->descr->fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                                    : rocsparse_fill_mode_lower)
              : A->descr->fill_mode;

    // When computing transposed triangular solve, we first need to update the
    // transposed matrix values

    //
    // Update the values of A^T if needed.
    //
    if(case_transpose_A)
    {

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::spmat_transpose_update_values(handle, At, A, trm_info));

        if(op_A == rocsparse_operation_conjugate_transpose)
        {
            A_load_conjugate = true;
        }
    }

    csrsm_launch_t launch = find_csrsm_launch(
        handle, blockdim, matrix->row_type, matrix->col_type, matrix->data_type, A_load_conjugate);

    RETURN_IF_ROCSPARSE_ERROR(launch(handle,
                                     nrhs,
                                     op_B,
                                     matrix,
                                     alpha,
                                     rhs,
                                     done_array,
                                     done_array_stride,
                                     trm_info->get_row_map(),
                                     numeric_exact_position->get_position(),
                                     1,
                                     fill_mode,
                                     diag_type,
                                     alpha->pointer_mode == rocsparse_pointer_mode_host));

    if(case_transform_B)
    {
        rocsparse_const_dnvec_descr no_scale = nullptr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnmat_copy_data(handle, no_scale, rhs, B, p_error));
    }
    return rocsparse_status_success;
}
