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

#include "rocsparse_control.hpp"
#include "rocsparse_cscsm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

#include "../../level2/cscsv_device.h"
#include "../../level2/rocsparse_cscsv.hpp"
#include "../../level2/rocsparse_cscsv_analysis_kernel.hpp"
#include "../../level2/rocsparse_cscsv_build_tasks_kernel.hpp"
#include "../../level2/rocsparse_cscsv_solve_kernel.hpp"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_primitives.hpp"

#include <cstdlib>
#include <memory>
#include <type_traits>

#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
namespace rocsparse
{
    // Level of each unknown, one plus the highest level of
    // the unknowns it depends on, by a push pass on integers over the tasks in elimination order.
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsm_level_kernel(int64_t num_tasks,
                            const I* __restrict__ csc_col_ptr,
                            const J* __restrict__ csc_row_ind,
                            const J* __restrict__ task_col,
                            const I* __restrict__ task_begin,
                            J* __restrict__ rem,
                            int32_t* __restrict__ level,
                            int32_t* __restrict__ ticket,
                            rocsparse_index_base idx_base,
                            rocsparse_fill_mode  fill_mode)
    {
        constexpr uint32_t ITEMS = rocsparse::cscsv_kernel_constants_t::s_num_items;
        __shared__ int32_t block_id;
        if(hipThreadIdx_x == 0)
        {
            block_id = rocsparse::atomic_add(ticket, 1);
        }
        __syncthreads();

        const uint32_t lid = hipThreadIdx_x & (NUM_LANES - 1);
        const int64_t  t
            = static_cast<int64_t>(block_id) * (BLOCKSIZE / NUM_LANES) + hipThreadIdx_x / NUM_LANES;
        if(t >= num_tasks)
        {
            return;
        }

        const bool lower   = (fill_mode == rocsparse_fill_mode_lower);
        const J    col     = task_col[t];
        const I    kbeg    = task_begin[t];
        const I    col_end = csc_col_ptr[col + 1] - idx_base;

        J    rows[ITEMS];
        bool deps[ITEMS];
        for(uint32_t i = 0; i < ITEMS; ++i)
        {
            const int64_t k = static_cast<int64_t>(kbeg) + lid + i * NUM_LANES;
            rows[i]         = (k < col_end) ? csc_row_ind[k] - idx_base : col;
            deps[i]         = lower ? (rows[i] > col) : (rows[i] < col);
        }

        bool done = false;
        while(!done)
        {
            J r = (lid == 0)
                      ? rocsparse::atomic_load(&rem[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      : static_cast<J>(0);
            r   = rocsparse::cscsv_group_broadcast<NUM_LANES>(r);
            if(r == 0)
            {
                rocsparse::acquire_done_rows();
                const int32_t lvl
                    = rocsparse::atomic_load(&level[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                      + 1;
                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_max(&level[rows[i]], lvl);
                    }
                }
                rocsparse::release_done_row_data();
                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    if(deps[i])
                    {
                        rocsparse::atomic_fetch_sub(&rem[rows[i]],
                                                    static_cast<J>(1),
                                                    __ATOMIC_RELAXED,
                                                    __MEMORY_SCOPE_DEVICE);
                    }
                }
                done = true;
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsm_level_keys_kernel(int64_t num_tasks,
                                 const J* __restrict__ task_col,
                                 const int32_t* __restrict__ level,
                                 int32_t* __restrict__ keys,
                                 I* __restrict__ perm)
    {
        for(int64_t t = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            t < num_tasks;
            t += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            keys[t] = level[task_col[t]];
            perm[t] = static_cast<I>(t);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsm_level_gather_kernel(int64_t num_tasks,
                                   const I* __restrict__ perm,
                                   const J* __restrict__ task_col,
                                   const I* __restrict__ task_begin,
                                   J* __restrict__ sorted_task_col,
                                   I* __restrict__ sorted_task_begin)
    {
        for(int64_t t = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
            t < num_tasks;
            t += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            sorted_task_col[t]   = task_col[perm[t]];
            sorted_task_begin[t] = task_begin[perm[t]];
        }
    }

    // Sorts the tasks of trm by level (stable, so the elimination order is kept inside a level):
    // every task still comes after the tasks it depends on.
    template <typename I, typename J>
    static rocsparse_status cscsm_level_order(rocsparse_handle            handle,
                                              rocsparse_const_spmat_descr A,
                                              uint32_t                    num_lanes,
                                              rocsparse::trm_info_t*      trm)
    {
        constexpr uint32_t BS        = rocsparse::cscsv_kernel_constants_t::s_blocksize;
        hipStream_t        stream    = handle->stream;
        const int64_t      m         = A->cols;
        const int64_t      num_tasks = trm->get_max_nnz();
        const J*           task_col  = reinterpret_cast<const J*>(trm->get_row_map());
        const I*           task_beg  = reinterpret_cast<const I*>(trm->get_transposed_perm());
        if(num_tasks == 0)
        {
            return rocsparse_status_success;
        }

        J*       rem{};
        int32_t* level{};
        int32_t* ticket{};
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&rem, sizeof(J) * m, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&level, sizeof(int32_t) * m, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&ticket, sizeof(int32_t), stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            rem, trm->get_transposed_col_ind(), sizeof(J) * m, hipMemcpyDeviceToDevice, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(level, 0, sizeof(int32_t) * m, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(ticket, 0, sizeof(int32_t), stream));

        const auto launch = [&](auto lanes) -> rocsparse_status {
            constexpr uint32_t N = decltype(lanes)::value;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::cscsm_level_kernel<BS, N, I, J>),
                dim3(rocsparse::ceil_div(num_tasks * N, static_cast<int64_t>(BS))),
                dim3(BS),
                0,
                stream,
                num_tasks,
                reinterpret_cast<const I*>(A->const_col_data),
                reinterpret_cast<const J*>(A->const_row_data),
                task_col,
                task_beg,
                rem,
                level,
                ticket,
                A->descr->base,
                A->descr->fill_mode);
            return rocsparse_status_success;
        };
        switch(num_lanes)
        {
        case 2:
            RETURN_IF_ROCSPARSE_ERROR(launch(std::integral_constant<uint32_t, 2>{}));
            break;
        case 4:
            RETURN_IF_ROCSPARSE_ERROR(launch(std::integral_constant<uint32_t, 4>{}));
            break;
        case 8:
            RETURN_IF_ROCSPARSE_ERROR(launch(std::integral_constant<uint32_t, 8>{}));
            break;
        case 16:
            RETURN_IF_ROCSPARSE_ERROR(launch(std::integral_constant<uint32_t, 16>{}));
            break;
        case 32:
            RETURN_IF_ROCSPARSE_ERROR(launch(std::integral_constant<uint32_t, 32>{}));
            break;
        case 64:
            RETURN_IF_ROCSPARSE_ERROR(launch(std::integral_constant<uint32_t, 64>{}));
            break;
        default:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(rem, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(ticket, stream));

        int32_t* keys0{};
        int32_t* keys1{};
        I*       perm0{};
        I*       perm1{};
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&keys0, sizeof(int32_t) * num_tasks, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&keys1, sizeof(int32_t) * num_tasks, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&perm0, sizeof(I) * num_tasks, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&perm1, sizeof(I) * num_tasks, stream));
        const dim3 grid(rocsparse::get_grid_size_x(
            handle, rocsparse::ceil_div(num_tasks, static_cast<int64_t>(BS)), BS));
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::cscsm_level_keys_kernel<BS, I, J>),
                                           grid,
                                           dim3(BS),
                                           0,
                                           stream,
                                           num_tasks,
                                           task_col,
                                           level,
                                           keys0,
                                           perm0);
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(level, stream));

        const uint32_t endbit = rocsparse::clz(m + 1);
        size_t         sort_size{};
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, I>(
            handle, num_tasks, 0, endbit, &sort_size)));
        void* sort_buffer{};
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&sort_buffer, sort_size, stream));
        rocsparse::primitives::double_buffer<int32_t> keys(keys0, keys1);
        rocsparse::primitives::double_buffer<I>       perm(perm0, perm1);
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, perm, num_tasks, 0, endbit, sort_size, sort_buffer));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(sort_buffer, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(keys0, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(keys1, stream));

        J* sorted_col{};
        I* sorted_begin{};
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&sorted_col, sizeof(J) * num_tasks, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&sorted_begin, sizeof(I) * num_tasks, stream));
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::cscsm_level_gather_kernel<BS, I, J>),
                                           grid,
                                           dim3(BS),
                                           0,
                                           stream,
                                           num_tasks,
                                           perm.current(),
                                           task_col,
                                           task_beg,
                                           sorted_col,
                                           sorted_begin);
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(perm0, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(perm1, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(trm->get_row_map(), stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(trm->get_transposed_perm(), stream));
        *trm->get_ref_row_map()         = sorted_col;
        *trm->get_ref_transposed_perm() = sorted_begin;
        return rocsparse_status_success;
    }

    // PROTOTYPE: the analysis of the CSC sptrsv (diagonal, dependency counts, structural pivot,
    // tasks in elimination order), stored in csrsm_info (see rocsparse_cscsm.hpp).
    static rocsparse_status cscsm_push_analysis(rocsparse_handle            handle,
                                                int64_t                     nrhs,
                                                rocsparse_const_spmat_descr A,
                                                rocsparse_csrsm_info        csrsm_info,
                                                void*                       buffer)
    {
        hipStream_t               stream   = handle->stream;
        const rocsparse_mat_descr descr    = A->descr;
        const int64_t             m        = A->cols;
        const rocsparse_indextype I_type   = A->col_type;
        const rocsparse_indextype J_type   = A->row_type;
        const size_t              sizeof_I = rocsparse::indextype_sizeof(I_type);
        const size_t              sizeof_J = rocsparse::indextype_sizeof(J_type);

        std::unique_ptr<rocsparse::trm_info_t> trm(new rocsparse::trm_info_t());
        trm->set_m(m);
        trm->set_nnz(A->nnz);
        trm->set_descr(descr);
        trm->set_row_ptr(A->const_col_data);
        trm->set_col_ind(A->const_row_data);
        trm->set_offset_indextype(I_type);
        trm->set_index_indextype(J_type);
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(trm->get_ref_diag_ind(), sizeof_I * m, stream));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(trm->get_ref_transposed_col_ind(), sizeof_J * m, stream));

        RETURN_IF_ROCSPARSE_ERROR(csrsm_info->create_zero_pivot_async(J_type, stream));
        void* zero_pivot = csrsm_info->get_position();
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::assign_max_async(csrsm_info->get_batch_count(), J_type, zero_pivot, stream));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(trm->get_diag_ind(), 0xFF, sizeof_I * m, stream));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(trm->get_transposed_col_ind(), 0, sizeof_J * m, stream));

        rocsparse::cscsv_analysis_launch_t launch;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_analysis_launch_find(&launch, I_type, J_type));
        RETURN_IF_ROCSPARSE_ERROR(launch(handle,
                                         m,
                                         A->nnz,
                                         A->const_col_data,
                                         A->const_row_data,
                                         trm->get_diag_ind(),
                                         trm->get_transposed_col_ind(),
                                         zero_pivot,
                                         descr->base,
                                         descr->diag_type,
                                         descr->fill_mode));

        const uint32_t num_lanes = rocsparse::cscsv_num_lanes(
            rocsparse::ceil_div(
                A->nnz, m * static_cast<int64_t>(rocsparse::cscsv_kernel_constants_t::s_num_items)),
            static_cast<uint32_t>(handle->wavefront_size));
        _rocsparse_cscsv_info          tasks;
        rocsparse::cscsv_build_tasks_t build;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_build_tasks_find(&build, I_type, J_type));
        RETURN_IF_ROCSPARSE_ERROR(build(handle,
                                        m,
                                        A->const_col_data,
                                        descr->base,
                                        descr->fill_mode,
                                        num_lanes,
                                        &tasks,
                                        buffer));
        trm->set_max_nnz(tasks.get_num_tasks(descr->fill_mode));
        tasks.release_tasks(
            descr->fill_mode, trm->get_ref_row_map(), trm->get_ref_transposed_perm());

        // Level order with several right-hand sides: in natural order, the tasks in flight depend
        // on each other on matrices with narrow levels. One right-hand side keeps the natural
        // order of the CSC sptrsv. PROTOTYPE: ROCSPARSE_CSCSM_LEVEL=0/1 forces the order.
        const char* level_env = std::getenv("ROCSPARSE_CSCSM_LEVEL");
        const bool  level     = (level_env != nullptr) ? (level_env[0] != '0') : (nrhs > 1);
        if(level)
        {
            const bool i32 = (I_type == rocsparse_indextype_i32);
            const bool j32 = (J_type == rocsparse_indextype_i32);
            if(i32 && j32)
                RETURN_IF_ROCSPARSE_ERROR(
                    (cscsm_level_order<int32_t, int32_t>(handle, A, num_lanes, trm.get())));
            else if(i32)
                RETURN_IF_ROCSPARSE_ERROR(
                    (cscsm_level_order<int32_t, int64_t>(handle, A, num_lanes, trm.get())));
            else if(j32)
                RETURN_IF_ROCSPARSE_ERROR(
                    (cscsm_level_order<int64_t, int32_t>(handle, A, num_lanes, trm.get())));
            else
                RETURN_IF_ROCSPARSE_ERROR(
                    (cscsm_level_order<int64_t, int64_t>(handle, A, num_lanes, trm.get())));
        }

        // Replaces the previous push analysis of this fill mode, if any.
        delete csrsm_info->get(rocsparse_operation_none, descr->fill_mode);
        csrsm_info->set(rocsparse_operation_none, descr->fill_mode, trm.release());
        return rocsparse_status_success;
    }
}
#endif

rocsparse_status rocsparse::cscsm_analysis(rocsparse_handle            handle,
                                           int64_t                     nrhs,
                                           rocsparse_operation         op_A,
                                           rocsparse_operation         op_B,
                                           rocsparse_const_dnvec_descr alpha,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_const_dnmat_descr B,
                                           rocsparse_analysis_policy   analysis,
                                           rocsparse_csrsm_info*       p_csrsm_info,
                                           size_t                      buffer_size_in_bytes,
                                           void*                       buffer,
                                           rocsparse_error*            p_error,
                                           bool                        limited_memory)
{

    ROCSPARSE_ROUTINE_TRACE;

#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
    if(rocsparse::cscsm_use_limited_memory(op_A, op_B, alpha, A, B, limited_memory))
    {
        if(A->cols == 0 || nrhs == 0)
        {
            return rocsparse_status_success;
        }
        if(p_csrsm_info[0] == nullptr)
        {
            p_csrsm_info[0] = new _rocsparse_csrsm_info();
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::cscsm_push_analysis(handle, nrhs, A, p_csrsm_info[0], buffer));
        return rocsparse_status_success;
    }
#endif

    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr A_csr(A, rocsparse_format_csr, &descr_csr, A->info);

    op_A = (op_A == rocsparse_operation_none) ? rocsparse_operation_transpose
                                              : rocsparse_operation_none;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis(handle,
                                                        nrhs,
                                                        op_A,
                                                        op_B,
                                                        alpha,
                                                        &A_csr,
                                                        B,
                                                        analysis,
                                                        p_csrsm_info,
                                                        buffer_size_in_bytes,
                                                        buffer,
                                                        p_error));

    return rocsparse_status_success;
}
