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
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_dichotomic_search.hpp"
#include "rocsparse_primitives.hpp"

#include <map>
#include <sstream>
#include <tuple>

namespace rocsparse
{
    // One thread per nonzero, so that a long column does not serialize the analysis. The
    // diagonal offsets are 0-based offsets into csc_row_ind and csc_val.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_analysis_entries_kernel(int64_t m,
                                       const I* __restrict__ csc_col_ptr,
                                       const J* __restrict__ csc_row_ind,
                                       I* __restrict__ csc_diag_ind,
                                       J* __restrict__ dependency_count,
                                       rocsparse_index_base idx_base,
                                       rocsparse_fill_mode  fill_mode)
    {
        const I    last  = csc_col_ptr[m];
        const bool lower = (fill_mode == rocsparse_fill_mode_lower);

        // k increases along the loop, so the column of the previous entry bounds the search.
        int64_t col = 0;
        for(int64_t k = static_cast<int64_t>(csc_col_ptr[0] - idx_base)
                        + static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            k < static_cast<int64_t>(last - idx_base);
            k += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            col = rocsparse::dichotomic_search<I, int64_t>(
                col, m - 1, static_cast<I>(k + idx_base), last, csc_col_ptr);
            const int64_t row = csc_row_ind[k] - idx_base;
            if(row == col)
            {
                csc_diag_ind[col] = static_cast<I>(k);
            }
            else if(lower ? (row > col) : (row < col))
            {
                rocsparse::atomic_add(&dependency_count[row], static_cast<J>(1));
            }
        }
    }

    // Structural zero pivot: the first column without a diagonal entry.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_analysis_pivot_kernel(int64_t m,
                                     const I* __restrict__ csc_diag_ind,
                                     J* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base)
    {
        for(int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; col < m;
            col += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            if(csc_diag_ind[col] < 0)
            {
                rocsparse::atomic_min(zero_pivot, static_cast<J>(col + idx_base));
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    static rocsparse_status cscsv_analysis_launch(rocsparse_handle     handle,
                                                  int64_t              m,
                                                  int64_t              nnz,
                                                  const void*          csc_col_ptr,
                                                  const void*          csc_row_ind,
                                                  void*                csc_diag_ind,
                                                  void*                dependency_count,
                                                  void*                zero_pivot,
                                                  rocsparse_index_base idx_base,
                                                  rocsparse_diag_type  diag_type,
                                                  rocsparse_fill_mode  fill_mode)
    {
        if(nnz > 0)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::cscsv_analysis_entries_kernel<BLOCKSIZE, I, J>),
                dim3(rocsparse::get_grid_size_x(
                    handle, rocsparse::ceil_div(nnz, static_cast<int64_t>(BLOCKSIZE)), BLOCKSIZE)),
                dim3(BLOCKSIZE),
                0,
                handle->stream,
                m,
                reinterpret_cast<const I*>(csc_col_ptr),
                reinterpret_cast<const J*>(csc_row_ind),
                reinterpret_cast<I*>(csc_diag_ind),
                reinterpret_cast<J*>(dependency_count),
                idx_base,
                fill_mode);
        }
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::cscsv_analysis_pivot_kernel<BLOCKSIZE, I, J>),
                dim3(rocsparse::get_grid_size_x(
                    handle, rocsparse::ceil_div(m, static_cast<int64_t>(BLOCKSIZE)), BLOCKSIZE)),
                dim3(BLOCKSIZE),
                0,
                handle->stream,
                m,
                reinterpret_cast<const I*>(csc_diag_ind),
                reinterpret_cast<J*>(zero_pivot),
                idx_base);
        }
        return rocsparse_status_success;
    }

    // Tasks per column: one per chunk of NUM_LANES * NUM_ITEMS entries, at least one.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_count_tasks_kernel(int64_t m,
                                  const I* __restrict__ csc_col_ptr,
                                  int64_t chunk,
                                  I* __restrict__ task_count)
    {
        for(int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; col < m;
            col += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t len = csc_col_ptr[col + 1] - csc_col_ptr[col];
            task_count[col]   = static_cast<I>(
                rocsparse::max(static_cast<int64_t>(1), (len + chunk - 1) / chunk));
        }
    }

    template <uint32_t BLOCKSIZE, typename I>
    static rocsparse_status cscsv_count_tasks_launch(
        rocsparse_handle handle, int64_t m, const I* csc_col_ptr, int64_t chunk, I* task_count)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_count_tasks_kernel<BLOCKSIZE, I>),
            dim3(rocsparse::get_grid_size_x(
                handle, rocsparse::ceil_div(m, static_cast<int64_t>(BLOCKSIZE)), BLOCKSIZE)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            csc_col_ptr,
            chunk,
            task_count);
        return rocsparse_status_success;
    }

    // Tasks in elimination order: column order, reversed for an upper triangle.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsv_expand_tasks_kernel(int64_t m,
                                   const I* __restrict__ csc_col_ptr,
                                   const I* __restrict__ task_ptr,
                                   int64_t chunk,
                                   int64_t num_tasks,
                                   bool    reverse,
                                   J* __restrict__ task_col,
                                   I* __restrict__ task_begin,
                                   rocsparse_index_base idx_base)
    {
        for(int64_t col = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; col < m;
            col += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const I begin = csc_col_ptr[col] - idx_base;
            for(I t = task_ptr[col]; t < task_ptr[col + 1]; ++t)
            {
                const int64_t pos = reverse ? (num_tasks - 1 - t) : t;
                task_col[pos]     = static_cast<J>(col);
                task_begin[pos]   = begin + static_cast<I>((t - task_ptr[col]) * chunk);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    static rocsparse_status cscsv_expand_tasks_launch(rocsparse_handle     handle,
                                                      int64_t              m,
                                                      const I*             csc_col_ptr,
                                                      const I*             task_ptr,
                                                      int64_t              chunk,
                                                      int64_t              num_tasks,
                                                      bool                 reverse,
                                                      J*                   task_col,
                                                      I*                   task_begin,
                                                      rocsparse_index_base idx_base)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_expand_tasks_kernel<BLOCKSIZE, I, J>),
            dim3(rocsparse::get_grid_size_x(
                handle, rocsparse::ceil_div(m, static_cast<int64_t>(BLOCKSIZE)), BLOCKSIZE)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            csc_col_ptr,
            task_ptr,
            chunk,
            num_tasks,
            reverse,
            task_col,
            task_begin,
            idx_base);
        return rocsparse_status_success;
    }

    // Temporaries of cscsv_build_tasks in the analysis buffer: task counts, their exclusive scan
    // and the scan buffer.
    template <typename I>
    static rocsparse_status
        cscsv_build_tasks_buffer_size(rocsparse_handle handle, int64_t m, size_t* buffer_size)
    {
        size_t scan_size{};
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
            handle, static_cast<I>(0), m + 1, &scan_size)));
        buffer_size[0] = 2 * rocsparse::align_size(sizeof(I) * (m + 1)) + scan_size;
        return rocsparse_status_success;
    }

    // Splits the columns into tasks of at most num_lanes * NUM_ITEMS entries and stores them in
    // info, in elimination order.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    static rocsparse_status cscsv_build_tasks(rocsparse_handle     handle,
                                              int64_t              m,
                                              const void*          csc_col_ptr_,
                                              rocsparse_index_base idx_base,
                                              rocsparse_fill_mode  fill_mode,
                                              uint32_t             num_lanes,
                                              rocsparse_cscsv_info info,
                                              void*                buffer)
    {
        hipStream_t   stream      = handle->stream;
        const I*      csc_col_ptr = reinterpret_cast<const I*>(csc_col_ptr_);
        const int64_t chunk       = static_cast<int64_t>(num_lanes) * s_cscsv_num_items;

        char* ptr        = reinterpret_cast<char*>(buffer);
        I*    task_count = reinterpret_cast<I*>(ptr);
        ptr += rocsparse::align_size(sizeof(I) * (m + 1));
        I* task_ptr = reinterpret_cast<I*>(ptr);
        ptr += rocsparse::align_size(sizeof(I) * (m + 1));
        void* scan_buffer = ptr;

        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(task_count + m, 0, sizeof(I), stream));
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::cscsv_count_tasks_launch<BLOCKSIZE, I>(
            handle, m, csc_col_ptr, chunk, task_count)));

        size_t scan_size{};
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
            handle, static_cast<I>(0), m + 1, &scan_size)));
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan<I, I>(
            handle, task_count, task_ptr, static_cast<I>(0), m + 1, scan_size, scan_buffer)));

        I num_tasks{};
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            &num_tasks, task_ptr + m, sizeof(I), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        RETURN_IF_ROCSPARSE_ERROR(
            info->allocate_tasks(fill_mode, num_tasks, num_lanes, sizeof(J), sizeof(I), stream));
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::cscsv_expand_tasks_launch<BLOCKSIZE, I, J>(
            handle,
            m,
            csc_col_ptr,
            task_ptr,
            chunk,
            static_cast<int64_t>(num_tasks),
            fill_mode == rocsparse_fill_mode_upper,
            reinterpret_cast<J*>(info->get_task_col(fill_mode)),
            reinterpret_cast<I*>(info->get_task_begin(fill_mode)),
            idx_base)));
        return rocsparse_status_success;
    }

    using cscsv_analysis_tpl_t = std::tuple<rocsparse_indextype, rocsparse_indextype>;

#define CONFIG(I_, J_)                                                              \
    {                                                                               \
        cscsv_analysis_tpl_t(I_, J_),                                               \
            cscsv_analysis_launch<rocsparse::s_cscsv_blocksize,                     \
                                  typename rocsparse::indextype_traits<I_>::type_t, \
                                  typename rocsparse::indextype_traits<J_>::type_t> \
    }

    static const std::map<cscsv_analysis_tpl_t, rocsparse::cscsv_analysis_launch_t>
        s_cscsv_analysis_dispatch{{CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
                                   CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i64),
                                   CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
                                   CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64)}};
#undef CONFIG

#define CONFIG(I_, J_)                                                          \
    {                                                                           \
        cscsv_analysis_tpl_t(I_, J_),                                           \
            cscsv_build_tasks<rocsparse::s_cscsv_blocksize,                     \
                              typename rocsparse::indextype_traits<I_>::type_t, \
                              typename rocsparse::indextype_traits<J_>::type_t> \
    }

    static const std::map<cscsv_analysis_tpl_t, rocsparse::cscsv_build_tasks_t>
        s_cscsv_build_tasks_dispatch{{CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
                                      CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i64),
                                      CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
                                      CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64)}};
#undef CONFIG

    static const std::map<rocsparse_indextype, rocsparse::cscsv_build_tasks_buffer_size_t>
        s_cscsv_build_tasks_buffer_size_dispatch{
            {{rocsparse_indextype_i32, cscsv_build_tasks_buffer_size<int32_t>},
             {rocsparse_indextype_i64, cscsv_build_tasks_buffer_size<int64_t>}}};
}

rocsparse_status rocsparse::cscsv_build_tasks_buffer_size_find(
    rocsparse::cscsv_build_tasks_buffer_size_t* launch, rocsparse_indextype i_type)
{
    const auto& it = rocsparse::s_cscsv_build_tasks_buffer_size_dispatch.find(i_type);
    if(it == rocsparse::s_cscsv_build_tasks_buffer_size_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid index type configuration: i_type: "
             << rocsparse::enum_utils::to_string(i_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_analysis_launch_find(rocsparse::cscsv_analysis_launch_t* launch,
                                                       rocsparse_indextype                 i_type,
                                                       rocsparse_indextype                 j_type)
{
    const auto& it = rocsparse::s_cscsv_analysis_dispatch.find(
        rocsparse::cscsv_analysis_tpl_t(i_type, j_type));
    if(it == rocsparse::s_cscsv_analysis_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid index type configuration: i_type: "
             << rocsparse::enum_utils::to_string(i_type)
             << ", j_type: " << rocsparse::enum_utils::to_string(j_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_build_tasks_find(rocsparse::cscsv_build_tasks_t* build,
                                                   rocsparse_indextype             i_type,
                                                   rocsparse_indextype             j_type)
{
    const auto& it = rocsparse::s_cscsv_build_tasks_dispatch.find(
        rocsparse::cscsv_analysis_tpl_t(i_type, j_type));
    if(it == rocsparse::s_cscsv_build_tasks_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid index type configuration: i_type: "
             << rocsparse::enum_utils::to_string(i_type)
             << ", j_type: " << rocsparse::enum_utils::to_string(j_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    build[0] = it->second;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_analysis(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_analysis_policy   analysis_policy,
                                           rocsparse_solve_policy      solve_policy,
                                           rocsparse_cscsv_info        info,
                                           size_t                      buffer_size_in_bytes,
                                           void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_check_matrix(A));
    info->set_csc_analysis(trans == rocsparse_operation_none);
    if(trans == rocsparse_operation_none)
    {
        size_t required_buffer_size{};
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::cscsv_analysis_buffer_size(handle, trans, A, &required_buffer_size));
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((buffer_size_in_bytes < required_buffer_size)
                                                   ? rocsparse_status_invalid_size
                                                   : rocsparse_status_success,
                                               "the buffer of the CSC analysis is too small");

        // The CSC analysis is not shared through the matrix info.
        hipStream_t               stream = handle->stream;
        const rocsparse_mat_descr descr  = A->descr;
        const int64_t             m      = A->cols;

        // Column pointers and row indices
        const rocsparse_indextype I_type   = A->col_type;
        const rocsparse_indextype J_type   = A->row_type;
        const size_t              sizeof_I = rocsparse::indextype_sizeof(I_type);
        const size_t              sizeof_J = rocsparse::indextype_sizeof(J_type);

        // Owned here until info->set takes it, so that an early return does not leak it.
        std::unique_ptr<rocsparse::trm_info_t> trm_info(new rocsparse::trm_info_t());
        trm_info->set_m(m);
        trm_info->set_nnz(A->nnz);
        trm_info->set_descr(descr);
        trm_info->set_row_ptr(A->const_col_data);
        trm_info->set_col_ind(A->const_row_data);
        trm_info->set_offset_indextype(I_type);
        trm_info->set_index_indextype(J_type);

        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(trm_info->get_ref_diag_ind(), sizeof_I * m, stream));
        RETURN_IF_ROCSPARSE_ERROR(
            info->allocate_dependency_count(descr->fill_mode, sizeof_J * m, stream));
        void* dependency_count = info->get_dependency_count(descr->fill_mode);
        RETURN_IF_ROCSPARSE_ERROR(info->create_zero_pivot_async(J_type, stream));

        void* zero_pivot = info->get_position();
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::assign_max_async(info->get_batch_count(), J_type, zero_pivot, stream));

        // -1 marks a missing diagonal entry
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(trm_info->get_diag_ind(), 0xFF, sizeof_I * m, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(dependency_count, 0, sizeof_J * m, stream));

        if(m > 0)
        {
            rocsparse::cscsv_analysis_launch_t launch;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::cscsv_analysis_launch_find(&launch, I_type, J_type));
            RETURN_IF_ROCSPARSE_ERROR(launch(handle,
                                             m,
                                             A->nnz,
                                             A->const_col_data,
                                             A->const_row_data,
                                             trm_info->get_diag_ind(),
                                             dependency_count,
                                             zero_pivot,
                                             descr->base,
                                             descr->diag_type,
                                             descr->fill_mode));
        }

        // Lanes per task from the average column length: longer columns are split into tasks.
        const uint32_t num_lanes = rocsparse::cscsv_num_lanes(
            (m > 0) ? rocsparse::ceil_div(A->nnz, m * static_cast<int64_t>(s_cscsv_num_items)) : 1,
            static_cast<uint32_t>(handle->wavefront_size));
        if(m > 0)
        {
            rocsparse::cscsv_build_tasks_t build;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_build_tasks_find(&build, I_type, J_type));
            RETURN_IF_ROCSPARSE_ERROR(build(handle,
                                            m,
                                            A->const_col_data,
                                            descr->base,
                                            descr->fill_mode,
                                            num_lanes,
                                            info,
                                            buffer));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                info->allocate_tasks(descr->fill_mode, 0, num_lanes, sizeof_J, sizeof_I, stream));
        }

        // Replaces the previous analysis of this fill mode, if any.
        delete info->get(rocsparse_operation_none, descr->fill_mode);
        info->set(rocsparse_operation_none, descr->fill_mode, trm_info.release());
        return rocsparse_status_success;
    }

    rocsparse_csrsv_info csrsv_info{};
    switch(analysis_policy)
    {
    case rocsparse_analysis_policy_reuse:
    {
        info->set_shared_csrsv_info(A->info->get_shared_csrsv_info());
        csrsv_info = info->get_csrsv_info();
        break;
    }
    case rocsparse_analysis_policy_force:
    {
        // Recomputed in place when the transposed analysis is kept in the matrix info.
        csrsv_info = info->get_csrsv_info();
        break;
    }
    }

    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis(handle,
                                                        rocsparse::cscsv_operation_to_csr(trans),
                                                        &mat_csr,
                                                        analysis_policy,
                                                        solve_policy,
                                                        &csrsv_info,
                                                        buffer));

    switch(analysis_policy)
    {
    case rocsparse_analysis_policy_reuse:
    {
        break;
    }
    case rocsparse_analysis_policy_force:
    {
        if(info->get_csrsv_info() == nullptr)
        {
            info->set_csrsv_info(csrsv_info);
        }
        break;
    }
    }
    return rocsparse_status_success;
}

bool rocsparse::cscsv_is_analyzed(rocsparse_cscsv_info        info,
                                  rocsparse_operation         trans,
                                  rocsparse_const_spmat_descr A)
{
    if(trans == rocsparse_operation_none)
    {
        return info->get(rocsparse_operation_none, A->descr->fill_mode) != nullptr;
    }
    if(info->get_csrsv_info() == nullptr)
    {
        return false;
    }

    // The transposed analysis is stored under the CSR operation and the flipped fill mode.
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);
    return info->get_csrsv_info()->get(rocsparse::cscsv_operation_to_csr(trans),
                                       descr_csr.fill_mode)
           != nullptr;
}
