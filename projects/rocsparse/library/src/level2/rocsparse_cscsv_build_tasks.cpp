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

#include "rocsparse_cscsv.hpp"
#include "rocsparse_cscsv_build_tasks_kernel.hpp"
#include "rocsparse_cscsv_solve_kernel.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_primitives.hpp"

#include <map>
#include <sstream>
#include <tuple>

#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
namespace rocsparse
{
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
    static rocsparse_status cscsv_build_tasks_buffer_size(rocsparse_handle handle,
                                                          int64_t          m,
                                                          size_t*          buffer_size_in_bytes)
    {
        size_t scan_size{};
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
            handle, static_cast<I>(0), m + 1, &scan_size)));
        buffer_size_in_bytes[0] = 2 * rocsparse::align_size(sizeof(I) * (m + 1)) + scan_size;
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
        const int64_t chunk
            = static_cast<int64_t>(num_lanes) * rocsparse::cscsv_kernel_constants_t::s_num_items;

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
    using cscsv_build_tasks_tpl_t = std::tuple<rocsparse_indextype, rocsparse_indextype>;

#define CONFIG(I_, J_)                                                          \
    {                                                                           \
        cscsv_build_tasks_tpl_t(I_, J_),                                        \
            cscsv_build_tasks<rocsparse::cscsv_kernel_constants_t::s_blocksize, \
                              typename rocsparse::indextype_traits<I_>::type_t, \
                              typename rocsparse::indextype_traits<J_>::type_t> \
    }

    static const std::map<cscsv_build_tasks_tpl_t, rocsparse::cscsv_build_tasks_t>
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
rocsparse_status rocsparse::cscsv_build_tasks_find(rocsparse::cscsv_build_tasks_t* build,
                                                   rocsparse_indextype             i_type,
                                                   rocsparse_indextype             j_type)
{
    const auto& it = rocsparse::s_cscsv_build_tasks_dispatch.find(
        rocsparse::cscsv_build_tasks_tpl_t(i_type, j_type));
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
#endif
