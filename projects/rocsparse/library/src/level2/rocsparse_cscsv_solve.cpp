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
#include "rocsparse_cscsv_kernel.hpp"
#include "rocsparse_diagonal_solve.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_scalar.hpp"
#include "rocsparse_singularity.hpp"
#include "rocsparse_utility.hpp"

#include "cscsv_device.h"

#include <map>
#include <sstream>
#include <tuple>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_solve_kernel(J m,
                            const I* __restrict__ csc_col_ptr,
                            const J* __restrict__ csc_row_ind,
                            const T* __restrict__ csc_val,
                            int64_t csc_val_stride,
                            const I* __restrict__ csc_diag_ind,
                            int64_t num_tasks,
                            const J* __restrict__ task_col,
                            const I* __restrict__ task_begin,
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
                            rocsparse_diag_type  diag_type)
    {
        for(int64_t batch_index = hipBlockIdx_y; batch_index < batch_count;
            batch_index += hipGridDim_y)
        {

            rocsparse::cscsv_solve_device<BLOCKSIZE, NUM_LANES, rocsparse::s_cscsv_items>(
                num_tasks,
                csc_col_ptr,
                csc_row_ind,
                csc_val + batch_index * csc_val_stride,
                csc_diag_ind,
                task_col,
                task_begin,
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

    template <uint32_t BLOCKSIZE, typename J, typename T>
    static rocsparse_status cscsv_init_launch(rocsparse_handle handle,
                                              int64_t          m,
                                              int64_t          batch_count,
                                              const void*      alpha,
                                              int64_t          alpha_stride,
                                              const void*      dependency_count,
                                              const void*      x,
                                              int64_t          x_inc,
                                              int64_t          x_stride,
                                              void*            rem,
                                              void*            sum)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_init_kernel<BLOCKSIZE, J, T>),
            dim3(rocsparse::get_grid_size_x(
                handle, rocsparse::ceil_div(m * batch_count, BLOCKSIZE), BLOCKSIZE)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            batch_count,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, reinterpret_cast<const T*>(alpha)),
            alpha_stride,
            reinterpret_cast<const J*>(dependency_count),
            reinterpret_cast<const T*>(x),
            x_inc,
            x_stride,
            reinterpret_cast<J*>(rem),
            reinterpret_cast<T*>(sum),
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J, typename T>
    static rocsparse_status cscsv_solve_launch(rocsparse_handle     handle,
                                               int64_t              batch_count,
                                               int64_t              m,
                                               const void*          csc_col_ptr,
                                               const void*          csc_row_ind,
                                               const void*          csc_val,
                                               int64_t              csc_val_stride,
                                               const void*          csc_diag_ind,
                                               int64_t              num_tasks,
                                               const void*          task_col,
                                               const void*          task_begin,
                                               void*                y,
                                               int64_t              y_inc,
                                               int64_t              y_stride,
                                               void*                rem,
                                               void*                sum,
                                               int32_t*             ticket,
                                               void*                zero_pivot,
                                               rocsparse_index_base idx_base,
                                               rocsparse_fill_mode  fill_mode,
                                               rocsparse_diag_type  diag_type)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_solve_kernel<BLOCKSIZE, NUM_LANES, I, J, T>),
            dim3(rocsparse::ceil_div(num_tasks * NUM_LANES, BLOCKSIZE),
                 rocsparse::get_grid_size_y(handle, batch_count)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            reinterpret_cast<const I*>(csc_col_ptr),
            reinterpret_cast<const J*>(csc_row_ind),
            reinterpret_cast<const T*>(csc_val),
            csc_val_stride,
            reinterpret_cast<const I*>(csc_diag_ind),
            num_tasks,
            reinterpret_cast<const J*>(task_col),
            reinterpret_cast<const I*>(task_begin),
            reinterpret_cast<T*>(y),
            y_inc,
            y_stride,
            batch_count,
            reinterpret_cast<J*>(rem),
            reinterpret_cast<T*>(sum),
            ticket,
            reinterpret_cast<J*>(zero_pivot),
            idx_base,
            fill_mode,
            diag_type);
        return rocsparse_status_success;
    }

    using cscsv_init_tpl_t = std::tuple<rocsparse_indextype, rocsparse_datatype>;

#define CONFIG(J_, T_)                                                          \
    {                                                                           \
        cscsv_init_tpl_t(J_, T_),                                               \
            cscsv_init_launch<rocsparse::s_cscsv_blocksize,                     \
                              typename rocsparse::indextype_traits<J_>::type_t, \
                              typename rocsparse::datatype_traits<T_>::type_t>  \
    }
#define CONFIG_TYPES(J_)                                                        \
    CONFIG(J_, rocsparse_datatype_f32_r), CONFIG(J_, rocsparse_datatype_f64_r), \
        CONFIG(J_, rocsparse_datatype_f32_c), CONFIG(J_, rocsparse_datatype_f64_c)

    static const std::map<cscsv_init_tpl_t, rocsparse::cscsv_init_launch_t> s_cscsv_init_dispatch{
        {CONFIG_TYPES(rocsparse_indextype_i32), CONFIG_TYPES(rocsparse_indextype_i64)}};

#undef CONFIG_TYPES
#undef CONFIG

    using cscsv_solve_tpl_t
        = std::tuple<uint32_t, rocsparse_indextype, rocsparse_indextype, rocsparse_datatype>;

    // clang-format off
#define CONFIG(N_, I_, J_, T_)                                                   \
    {cscsv_solve_tpl_t(N_, I_, J_, T_),                                          \
     cscsv_solve_launch<rocsparse::s_cscsv_blocksize,                            \
                        N_,                                                      \
                        typename rocsparse::indextype_traits<I_>::type_t,        \
                        typename rocsparse::indextype_traits<J_>::type_t,        \
                        typename rocsparse::datatype_traits<T_>::type_t>}
#define CONFIG_LANES(I_, J_, T_)                                                 \
    CONFIG(2, I_, J_, T_), CONFIG(4, I_, J_, T_), CONFIG(8, I_, J_, T_),         \
    CONFIG(16, I_, J_, T_), CONFIG(32, I_, J_, T_), CONFIG(64, I_, J_, T_)
#define CONFIG_TYPES(I_, J_)                                                     \
    CONFIG_LANES(I_, J_, rocsparse_datatype_f32_r),                              \
    CONFIG_LANES(I_, J_, rocsparse_datatype_f64_r),                              \
    CONFIG_LANES(I_, J_, rocsparse_datatype_f32_c),                              \
    CONFIG_LANES(I_, J_, rocsparse_datatype_f64_c)
    // clang-format on

    static const std::map<cscsv_solve_tpl_t, rocsparse::cscsv_solve_launch_t>
        s_cscsv_solve_dispatch{{CONFIG_TYPES(rocsparse_indextype_i32, rocsparse_indextype_i32),
                                CONFIG_TYPES(rocsparse_indextype_i32, rocsparse_indextype_i64),
                                CONFIG_TYPES(rocsparse_indextype_i64, rocsparse_indextype_i32),
                                CONFIG_TYPES(rocsparse_indextype_i64, rocsparse_indextype_i64)}};

#undef CONFIG_TYPES
#undef CONFIG_LANES
#undef CONFIG
}

rocsparse_status rocsparse::cscsv_init_launch_find(rocsparse::cscsv_init_launch_t* launch,
                                                   rocsparse_indextype             j_type,
                                                   rocsparse_datatype              t_type)
{
    const auto& it
        = rocsparse::s_cscsv_init_dispatch.find(rocsparse::cscsv_init_tpl_t(j_type, t_type));
    if(it == rocsparse::s_cscsv_init_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid type configuration: j_type: " << rocsparse::enum_utils::to_string(j_type)
             << ", t_type: " << rocsparse::enum_utils::to_string(t_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_solve_launch_find(rocsparse::cscsv_solve_launch_t* launch,
                                                    uint32_t                         num_lanes,
                                                    rocsparse_indextype              i_type,
                                                    rocsparse_indextype              j_type,
                                                    rocsparse_datatype               t_type)
{
    const auto& it = rocsparse::s_cscsv_solve_dispatch.find(
        rocsparse::cscsv_solve_tpl_t(num_lanes, i_type, j_type, t_type));
    if(it == rocsparse::s_cscsv_solve_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid configuration: num_lanes: " << num_lanes
             << ", i_type: " << rocsparse::enum_utils::to_string(i_type)
             << ", j_type: " << rocsparse::enum_utils::to_string(j_type)
             << ", t_type: " << rocsparse::enum_utils::to_string(t_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}

namespace rocsparse
{
    // Each batch instance starts from the structural zero pivot of the analysis, or from
    // no pivot if the diagonal is unit.
    static rocsparse_status cscsv_seed_zero_pivot(rocsparse_handle     handle,
                                                  rocsparse_indextype  indextype,
                                                  rocsparse_diag_type  diag_type,
                                                  int64_t              batch_count,
                                                  rocsparse_cscsv_info info)
    {
        hipStream_t stream = handle->stream;
        info->create_singularity_numeric_exact(batch_count, indextype, stream);
        void* zero_pivot = info->get_singularity_numeric_exact()->get_position();
        if(diag_type == rocsparse_diag_type_unit)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
                info->get_batch_count(), indextype, info->get_position(), stream));
        }
        if(indextype == rocsparse_indextype_i32)
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

rocsparse_status rocsparse::cscsv_solve(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_datatype          alpha_datatype,
                                        const void*                 alpha,
                                        int64_t                     alpha_stride,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_const_dnvec_descr x,
                                        rocsparse_dnvec_descr       y,
                                        rocsparse_solve_policy      solve_policy,
                                        rocsparse_cscsv_info        info,
                                        size_t                      buffer_size_in_bytes,
                                        void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(trans == rocsparse_operation_none)
    {
        hipStream_t               stream      = handle->stream;
        const rocsparse_mat_descr descr       = A->descr;
        const int64_t             batch_count = y->batch_count;
        const int64_t             m           = A->cols;
        const rocsparse_indextype J_type      = A->row_type;

        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((alpha_datatype != A->data_type)
                                                   ? rocsparse_status_not_implemented
                                                   : rocsparse_status_success,
                                               "the CSC solve requires the compute "
                                               "datatype to be the datatype of the matrix");

        size_t required_buffer_size{};
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::cscsv_solve_buffer_size(handle, trans, A, x, y, &required_buffer_size));
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((buffer_size_in_bytes < required_buffer_size)
                                                   ? rocsparse_status_invalid_size
                                                   : rocsparse_status_success,
                                               "the buffer of the CSC solve is too small");

        const rocsparse::trm_info_t* trm_info
            = info->get(rocsparse_operation_none, descr->fill_mode);
        if(trm_info == nullptr)
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            // LCOV_EXCL_STOP
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::cscsv_seed_zero_pivot(handle, J_type, descr->diag_type, batch_count, info));
        void* zero_pivot = info->get_singularity_numeric_exact()->get_position();

        if(m == 0)
        {
            return rocsparse_status_success;
        }

        char*        ptr               = reinterpret_cast<char*>(buffer);
        int32_t*     ticket            = reinterpret_cast<int32_t*>(ptr);
        const size_t ticket_size_bytes = rocsparse::align_size(sizeof(int32_t) * batch_count);
        ptr += ticket_size_bytes;

        void* rem = ptr;
        ptr += rocsparse::align_size(rocsparse::indextype_sizeof(J_type) * m * batch_count);

        void* sum = ptr;

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(ticket, 0, ticket_size_bytes, stream));

        rocsparse::cscsv_init_launch_t init_launch;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::cscsv_init_launch_find(&init_launch, J_type, A->data_type));
        RETURN_IF_ROCSPARSE_ERROR(init_launch(handle,
                                              m,
                                              batch_count,
                                              alpha,
                                              alpha_stride,
                                              info->get_dependency_count(descr->fill_mode),
                                              x->const_values,
                                              x->inc,
                                              x->batch_stride,
                                              rem,
                                              sum));

        const uint32_t num_lanes = info->get_num_lanes(descr->fill_mode);

        rocsparse::cscsv_solve_launch_t solve_launch;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_solve_launch_find(
            &solve_launch, num_lanes, A->col_type, J_type, A->data_type));
        RETURN_IF_ROCSPARSE_ERROR(
            solve_launch(handle,
                         batch_count,
                         m,
                         A->const_col_data,
                         A->const_row_data,
                         A->const_val_data,
                         (A->batch_count > 1) ? A->columns_values_batch_stride : 0,
                         trm_info->get_diag_ind(),
                         info->get_num_tasks(descr->fill_mode),
                         info->get_task_col(descr->fill_mode),
                         info->get_task_begin(descr->fill_mode),
                         y->values,
                         y->inc,
                         y->batch_stride,
                         rem,
                         sum,
                         ticket,
                         zero_pivot,
                         descr->base,
                         descr->fill_mode,
                         descr->diag_type));
        return rocsparse_status_success;
    }
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

    // A conjugate-transpose CSC solve becomes a non-transposed CSR solve with
    // on-the-fly conjugation of the matrix values.
    const bool force_conj = (trans == rocsparse_operation_conjugate_transpose);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                     rocsparse::cscsv_operation_to_csr(trans),
                                                     alpha_datatype,
                                                     alpha,
                                                     alpha_stride,
                                                     &mat_csr,
                                                     x,
                                                     y,
                                                     solve_policy,
                                                     info->get_csrsv_info(),
                                                     buffer,
                                                     force_conj));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_diagonal_solve(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 rocsparse_diagonal_modifier modifier,
                                                 const void*                 alpha,
                                                 rocsparse_const_spmat_descr A,
                                                 rocsparse_const_dnvec_descr x,
                                                 rocsparse_dnvec_descr       y,
                                                 rocsparse_cscsv_info        info)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(trans == rocsparse_operation_none)
    {
        const int64_t batch_count = y->batch_count;

        const rocsparse::trm_info_t* trm_info
            = info->get(rocsparse_operation_none, A->descr->fill_mode);
        if(trm_info == nullptr)
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            // LCOV_EXCL_STOP
        }

        // The CSC analysis indexes the diagonal directly in the CSC values.
        rocsparse::spdiag_view diag{};
        diag.offset_type     = trm_info->get_offset_indextype();
        diag.diag_ind        = trm_info->get_diag_ind();
        diag.transposed_perm = nullptr;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_seed_zero_pivot(
            handle, A->row_type, A->descr->diag_type, batch_count, info));

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
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::diagonal_solve_csc(handle,
                                                            trans,
                                                            modifier,
                                                            alpha,
                                                            A,
                                                            info->get_csrsv_info(),
                                                            static_cast<int64_t>(1),
                                                            x->const_values,
                                                            x->inc,
                                                            static_cast<int64_t>(0),
                                                            x->batch_stride,
                                                            y->values,
                                                            y->inc,
                                                            static_cast<int64_t>(0),
                                                            y->batch_stride,
                                                            y->batch_count,
                                                            false));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_zero_pivot(rocsparse_handle     handle,
                                             rocsparse_cscsv_info info,
                                             rocsparse_indextype  indextype,
                                             void*                position)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(info == nullptr)
    {
        return rocsparse::csrsv_zero_pivot(handle, nullptr, indextype, position);
    }
    if(info->get_csrsv_info() == nullptr)
    {
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
    return rocsparse::csrsv_zero_pivot(handle, info->get_csrsv_info(), indextype, position);
}
