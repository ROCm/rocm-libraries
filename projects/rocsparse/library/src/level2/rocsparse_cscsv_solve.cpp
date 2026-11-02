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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_cscsv.hpp"
#include "rocsparse_cscsv_common.hpp"
#include "rocsparse_diagonal_solve.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_singularity.hpp"

#include "cscsv_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t L, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_solve_kernel(J m,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const I* __restrict__ csc_col_ptr,
                            const J* __restrict__ csc_row_ind,
                            const T* __restrict__ csc_val,
                            int64_t csc_val_stride,
                            const I* __restrict__ csc_diag_ind,
                            const T* __restrict__ x,
                            int64_t x_inc,
                            int64_t x_stride,
                            T*      y,
                            int64_t y_inc,
                            int64_t y_stride,
                            int64_t batch_count,
                            J* __restrict__ rem,
                            T* __restrict__ sum,
                            int32_t* __restrict__ ticket,
                            J*                   zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_fill_mode  fill_mode,
                            rocsparse_diag_type  diag_type,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        for(int64_t batch_index = hipBlockIdx_y; batch_index < batch_count;
            batch_index += hipGridDim_y)
        {
            rocsparse::cscsv_solve_device<BLOCKSIZE, L>(m,
                                                        alpha,
                                                        csc_col_ptr,
                                                        csc_row_ind,
                                                        csc_val + batch_index * csc_val_stride,
                                                        csc_diag_ind,
                                                        x + batch_index * x_stride,
                                                        x_inc,
                                                        y + batch_index * y_stride,
                                                        y_inc,
                                                        rem + batch_index * static_cast<int64_t>(m),
                                                        sum + batch_index * static_cast<int64_t>(m),
                                                        ticket + batch_index,
                                                        zero_pivot + batch_index,
                                                        idx_base,
                                                        fill_mode,
                                                        diag_type);
            // The next batch instance reuses the shared ticket of the block.
            __syncthreads();
        }
    }

    template <uint32_t L, typename I, typename J, typename T>
    static rocsparse_status cscsv_solve_launch(rocsparse_handle             handle,
                                               const void*                  alpha,
                                               rocsparse_const_spmat_descr  A,
                                               rocsparse_const_dnvec_descr  x,
                                               rocsparse_dnvec_descr        y,
                                               const rocsparse::trm_info_t* trm_info,
                                               void*                        rem,
                                               void*                        sum,
                                               int32_t*                     ticket,
                                               void*                        zero_pivot)
    {
        static constexpr uint32_t BLOCKSIZE = rocsparse::s_cscsv_blocksize;
        const int64_t             m         = A->cols;
        const int64_t val_stride = (A->batch_count > 1) ? A->columns_values_batch_stride : 0;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_solve_kernel<BLOCKSIZE, L, I, J, T>),
            dim3((m * L - 1) / BLOCKSIZE + 1, rocsparse::get_grid_size_y(handle, y->batch_count)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, reinterpret_cast<const T*>(alpha)),
            reinterpret_cast<const I*>(A->const_col_data),
            reinterpret_cast<const J*>(A->const_row_data),
            reinterpret_cast<const T*>(A->const_val_data),
            val_stride,
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            reinterpret_cast<const T*>(x->const_values),
            x->inc,
            x->batch_stride,
            reinterpret_cast<T*>(y->values),
            y->inc,
            y->batch_stride,
            y->batch_count,
            reinterpret_cast<J*>(rem),
            reinterpret_cast<T*>(sum),
            ticket,
            reinterpret_cast<J*>(zero_pivot),
            A->descr->base,
            A->descr->fill_mode,
            A->descr->diag_type,
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T, typename... P>
    static rocsparse_status cscsv_solve_dispatch_group(uint32_t group_size, P&&... p)
    {
        switch(group_size)
        {
        case 2:
            return rocsparse::cscsv_solve_launch<2, I, J, T>(p...);
        case 4:
            return rocsparse::cscsv_solve_launch<4, I, J, T>(p...);
        case 8:
            return rocsparse::cscsv_solve_launch<8, I, J, T>(p...);
        case 16:
            return rocsparse::cscsv_solve_launch<16, I, J, T>(p...);
        case 32:
            return rocsparse::cscsv_solve_launch<32, I, J, T>(p...);
        case 64:
            return rocsparse::cscsv_solve_launch<64, I, J, T>(p...);
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        // LCOV_EXCL_STOP
    }

    template <typename I, typename J, typename... P>
    static rocsparse_status cscsv_solve_dispatch_datatype(rocsparse_datatype datatype, P&&... p)
    {
        switch(datatype)
        {
        case rocsparse_datatype_f32_r:
            return rocsparse::cscsv_solve_dispatch_group<I, J, float>(p...);
        case rocsparse_datatype_f64_r:
            return rocsparse::cscsv_solve_dispatch_group<I, J, double>(p...);
        case rocsparse_datatype_f32_c:
            return rocsparse::cscsv_solve_dispatch_group<I, J, rocsparse_float_complex>(p...);
        case rocsparse_datatype_f64_c:
            return rocsparse::cscsv_solve_dispatch_group<I, J, rocsparse_double_complex>(p...);
        default:
            break;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

namespace rocsparse
{
    // Each batch instance starts from the structural zero pivot of the analysis.
    static rocsparse_status cscsv_seed_zero_pivot(rocsparse_handle            handle,
                                                  rocsparse_const_spmat_descr A,
                                                  int64_t                     batch_count,
                                                  rocsparse_cscsv_info        info)
    {
        hipStream_t stream = handle->stream;
        info->create_singularity_numeric_exact(batch_count, A->row_type, stream);
        void* zero_pivot = info->get_singularity_numeric_exact()->get_position();
        if(A->row_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int32_t>(
                batch_count, (int32_t*)zero_pivot, (const int32_t*)info->get_position(), stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<int64_t>(
                batch_count, (int64_t*)zero_pivot, (const int64_t*)info->get_position(), stream));
        }
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::cscsv_column_solve(rocsparse_handle            handle,
                                               rocsparse_datatype          alpha_datatype,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_const_dnvec_descr x,
                                               rocsparse_dnvec_descr       y,
                                               rocsparse_cscsv_info        info,
                                               void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    hipStream_t               stream      = handle->stream;
    const rocsparse_mat_descr descr       = A->descr;
    const int64_t             batch_count = y->batch_count;
    const size_t              m           = A->cols;
    const rocsparse_indextype J_type      = A->row_type;

    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((alpha_datatype != A->data_type)
                                               ? rocsparse_status_not_implemented
                                               : rocsparse_status_success,
                                           "the CSC column solve requires the compute "
                                           "datatype to be the datatype of the matrix");

    const rocsparse::trm_info_t* trm_info = info->get(rocsparse_operation_none, descr->fill_mode);
    if(trm_info == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_seed_zero_pivot(handle, A, batch_count, info));
    void* zero_pivot = info->get_singularity_numeric_exact()->get_position();

    if(m == 0)
    {
        return rocsparse_status_success;
    }

    char*        ptr               = reinterpret_cast<char*>(temp_buffer);
    int32_t*     ticket            = reinterpret_cast<int32_t*>(ptr);
    const size_t ticket_size_bytes = rocsparse::cscsv_aligned(sizeof(int32_t) * batch_count);
    ptr += ticket_size_bytes;

    void* rem = ptr;
    ptr += rocsparse::cscsv_aligned(rocsparse::indextype_sizeof(J_type) * m * batch_count);

    void* sum = ptr;

    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(ticket, 0, ticket_size_bytes, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
        sum, 0, rocsparse::datatype_sizeof(A->data_type) * m * batch_count, stream));

    // Smallest power of two covering the longest column, capped at the wavefront size.
    uint32_t group_size = 2;
    while(static_cast<int64_t>(group_size) < trm_info->get_max_nnz()
          && group_size < static_cast<uint32_t>(handle->wavefront_size))
    {
        group_size *= 2;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::cscsv_dispatch_indextype(A->col_type, J_type, [&](auto i, auto j) {
            using I                             = decltype(i);
            using J                             = decltype(j);
            static constexpr uint32_t BLOCKSIZE = rocsparse::s_cscsv_blocksize;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::cscsv_init_kernel<BLOCKSIZE, J>),
                dim3(rocsparse::get_grid_size_x(
                    handle, rocsparse::ceil_div(m * batch_count, BLOCKSIZE), BLOCKSIZE)),
                dim3(BLOCKSIZE),
                0,
                stream,
                static_cast<J>(m),
                batch_count,
                reinterpret_cast<const J*>(info->get_dependency_count()),
                reinterpret_cast<J*>(rem));

            return rocsparse::cscsv_solve_dispatch_datatype<I, J>(A->data_type,
                                                                  group_size,
                                                                  handle,
                                                                  alpha,
                                                                  A,
                                                                  x,
                                                                  y,
                                                                  trm_info,
                                                                  rem,
                                                                  sum,
                                                                  ticket,
                                                                  zero_pivot);
        }));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_column_diagonal_solve(rocsparse_handle            handle,
                                                        rocsparse_diagonal_modifier modifier,
                                                        const void*                 alpha,
                                                        rocsparse_const_spmat_descr A,
                                                        rocsparse_const_dnvec_descr x,
                                                        rocsparse_dnvec_descr       y,
                                                        rocsparse_cscsv_info        info)
{
    ROCSPARSE_ROUTINE_TRACE;
    const int64_t batch_count = y->batch_count;

    const rocsparse::trm_info_t* trm_info
        = info->get(rocsparse_operation_none, A->descr->fill_mode);
    if(trm_info == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    // The column analysis indexes the diagonal directly in the CSC values.
    rocsparse::spdiag_view diag{};
    diag.offset_type     = trm_info->get_offset_indextype();
    diag.diag_ind        = trm_info->get_diag_ind();
    diag.transposed_perm = nullptr;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_seed_zero_pivot(handle, A, batch_count, info));

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::diagonal_solve(handle,
                                  rocsparse_operation_none,
                                  modifier,
                                  alpha,
                                  A,
                                  diag,
                                  static_cast<int64_t>(1),
                                  x->const_values,
                                  x->inc,
                                  static_cast<int64_t>(0),
                                  x->batch_stride,
                                  y->values,
                                  y->inc,
                                  static_cast<int64_t>(0),
                                  y->batch_stride,
                                  batch_count,
                                  false,
                                  info->get_singularity_numeric_exact()->get_position(),
                                  1,
                                  handle->pointer_mode == rocsparse_pointer_mode_host));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_column_zero_pivot(rocsparse_handle     handle,
                                                    rocsparse_cscsv_info info,
                                                    rocsparse_indextype  indextype,
                                                    void*                position)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::singularity_get_position_async(handle,
                                                  1,
                                                  info,
                                                  info->get_singularity_numeric_exact(),
                                                  nullptr,
                                                  handle->pointer_mode,
                                                  indextype,
                                                  position));

    // position lives where the handle's pointer mode says it does.
    const hipMemcpyKind kind = (handle->pointer_mode == rocsparse_pointer_mode_host)
                                   ? hipMemcpyHostToHost
                                   : hipMemcpyDeviceToHost;

    int64_t p = -1;
    if(indextype == rocsparse_indextype_i32)
    {
        int32_t p32;
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(&p32, position, sizeof(int32_t), kind, handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
        p = p32;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(&p, position, sizeof(int64_t), kind, handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
    }
    return (p != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success;
}
