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

#include "rocsparse_cscsm.hpp"

#include "rocsparse_csrsm.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"
#include <cstdio>

#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
#include "../../level2/cscsv_device.h"
#include "../../level2/rocsparse_cscsv.hpp"
#include "../../level2/rocsparse_cscsv_solve_kernel.hpp"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_scalar.hpp"

// Limited-memory solve of A * X = alpha * B with A in CSC, X overwriting B. One group of GROUP
// lanes per (task, panel), split into PANEL / Q lanes for the right-hand sides (Q each) and the
// other lanes for the entries of the task: once x(col, :) is known, lane (e, j) subtracts
// a(r, col) * x(col, j) from B(r, j) for the entries e, e + GROUP * Q / PANEL, ... of the task. The
// right-hand sides are cut into panels of PANEL columns, and each (batch, panel) has its own
// dependency counters. The (task, panel) pairs share one ticket per batch instance, task major, so
// a pair only waits on earlier tasks of its panel. B keeps b during the solve and receives the
// contributions divided by alpha, and x = alpha * B / a(col, col). With conj, the matrix values are
// conjugated (CSR matrix with conjugate transpose, see build_csc_from_csr).
namespace rocsparse
{
    // Dependency counters of each (batch, panel).
    template <uint32_t BLOCKSIZE, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsm_push_init_kernel(int64_t m,
                                int64_t nslots,
                                const J* __restrict__ dependency_count,
                                J* __restrict__ rem)
    {
        for(int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; i < m;
            i += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const J dc = dependency_count[i];
            for(int64_t slot = 0; slot < nslots; ++slot)
            {
                rem[slot * m + i] = dc;
            }
        }
    }

    // The columns split into several tasks: x(col, :) = alpha * B(col, :) / a(col, col),
    // with the zero and missing diagonal entries treated as one, like the solve. One thread per
    // owner task of such a column.
    template <uint32_t BLOCKSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsm_push_final_kernel(int64_t num_tasks,
                                 int64_t nrhs,
                                 int64_t batch_count,
                                 int64_t chunk,
                                 const I* __restrict__ csc_col_ptr,
                                 const I* __restrict__ csc_diag_ind,
                                 const T* __restrict__ csc_val,
                                 int64_t csc_val_stride,
                                 const J* __restrict__ task_col,
                                 const I* __restrict__ task_begin,
                                 T*      B,
                                 int64_t inc_i,
                                 int64_t inc_j,
                                 int64_t B_stride,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 rocsparse_index_base idx_base,
                                 rocsparse_diag_type  diag_type,
                                 bool                 conj,
                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            for(int64_t t = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
                t < num_tasks;
                t += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
            {
                const J col       = task_col[t];
                const I col_begin = csc_col_ptr[col] - idx_base;
                const I col_end   = csc_col_ptr[col + 1] - idx_base;
                if(task_begin[t] != col_begin || col_end - col_begin <= chunk)
                {
                    continue;
                }
                T s = alpha;
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    const I d = csc_diag_ind[col];
                    if(d >= 0)
                    {
                        T diag = csc_val[batch * csc_val_stride + d];
                        diag   = conj ? rocsparse::conj(diag) : diag;
                        if(diag != static_cast<T>(0))
                        {
                            s = alpha / diag;
                        }
                    }
                }
                T* b = B + batch * B_stride + col * inc_i;
                for(int64_t j = 0; j < nrhs; ++j)
                {
                    b[j * inc_j] = b[j * inc_j] * s;
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t GROUP, uint32_t Q, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void cscsm_push_solve_kernel(int64_t  num_tasks,
                                 int64_t  m,
                                 int64_t  nrhs,
                                 uint32_t panel,
                                 int64_t  npanels,
                                 int64_t  nslots,
                                 int64_t  chunk,
                                 const I* __restrict__ csc_col_ptr,
                                 const J* __restrict__ csc_row_ind,
                                 const T* __restrict__ csc_val,
                                 int64_t csc_val_stride,
                                 const I* __restrict__ csc_diag_ind,
                                 const J* __restrict__ task_col,
                                 const I* __restrict__ task_begin,
                                 T*      B,
                                 int64_t inc_i,
                                 int64_t inc_j,
                                 int64_t B_stride,
                                 J* __restrict__ rem_all,
                                 int32_t* __restrict__ ticket,
                                 J*                   zero_pivot,
                                 rocsparse_index_base idx_base,
                                 rocsparse_fill_mode  fill_mode,
                                 rocsparse_diag_type  diag_type,
                                 bool                 conj,
                                 int64_t              window,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 bool is_host_mode)
    {
        __shared__ int32_t block_id;
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        // B keeps b and receives the contributions divided by alpha, so that
        // x = alpha * B / a(col, col) needs no pass over B before the solve.
        const T inv_alpha
            = (alpha != static_cast<T>(0)) ? static_cast<T>(1) / alpha : static_cast<T>(0);

        const uint32_t lid = hipThreadIdx_x & (GROUP - 1);
        const uint32_t wid = hipThreadIdx_x / GROUP;
        // panel = R * Q: R lanes for the right-hand sides, Q right-hand sides per lane.
        const uint32_t R      = panel / Q;
        const uint32_t rid    = lid & (R - 1);
        const uint32_t eid    = lid / R;
        const uint32_t nentry = GROUP / R;
        const bool     lower  = (fill_mode == rocsparse_fill_mode_lower);

        const int64_t ny = nslots / npanels;
        for(int64_t y = hipBlockIdx_y; y < ny; y += hipGridDim_y)
        {
            if(hipThreadIdx_x == 0)
            {
                block_id = rocsparse::atomic_add(ticket + y, 1);
            }
            __syncthreads();

            const int64_t g = static_cast<int64_t>(block_id) * (BLOCKSIZE / GROUP) + wid;
            // Windows of `window` consecutive tasks, one panel after the other inside a window.
            const int64_t t    = (g / (window * npanels)) * window + g % window;
            const int64_t slot = y * npanels + (g / window) % npanels;
            if(t < num_tasks)
            {
                const int64_t batch = slot / npanels;
                const int64_t j0    = (slot % npanels) * panel + rid;
                const T*      val   = csc_val + batch * csc_val_stride;
                J*            rem   = rem_all + slot * m;
                T*            Bb    = B + batch * B_stride;
                const auto    acc   = [&](int64_t r, uint32_t q) -> T* {
                    return Bb + r * inc_i + (j0 + q * R) * inc_j;
                };

                const J    col       = task_col[t];
                const I    kbeg      = task_begin[t];
                const I    col_begin = csc_col_ptr[col] - idx_base;
                const I    col_end   = csc_col_ptr[col + 1] - idx_base;
                const I    kend      = rocsparse::min(col_end, static_cast<I>(kbeg + chunk));
                const bool owner     = (kbeg == col_begin);

                T inv_diag = static_cast<T>(1);
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    const I d = csc_diag_ind[col];
                    if(d >= 0)
                    {
                        T diag = conj ? rocsparse::conj(val[d]) : val[d];
                        if(diag == static_cast<T>(0))
                        {
                            if(owner && lid == 0)
                            {
                                rocsparse::atomic_min(zero_pivot + batch, col + idx_base);
                            }
                            diag = static_cast<T>(1);
                        }
                        inv_diag = static_cast<T>(1) / diag;
                    }
                }

                // Off the critical path: the first ITEMS entries of the lane, as in the sptrsv
                // solve; a longer task (a group clamped to the wavefront) reads the rest after
                // the wait.
                constexpr uint32_t ITEMS = rocsparse::cscsv_kernel_constants_t::s_num_items;
                J                  rows[ITEMS];
                T                  vals[ITEMS];
                bool               deps[ITEMS];
                for(uint32_t i = 0; i < ITEMS; ++i)
                {
                    const int64_t k = static_cast<int64_t>(kbeg) + eid + i * nentry;
                    rows[i]         = (k < kend) ? csc_row_ind[k] - idx_base : col;
                    vals[i]         = (k < kend) ? (conj ? rocsparse::conj(val[k]) : val[k])
                                                 : static_cast<T>(0);
                    deps[i]         = lower ? (rows[i] > col) : (rows[i] < col);
                }
                const int64_t ktail = static_cast<int64_t>(kbeg) + eid + ITEMS * nentry;

                // The work is done inside the wait loop: the groups of a wavefront that are
                // still waiting must not keep a ready group from pushing.
                bool done = false;
                while(!done)
                {
                    J r = (lid == 0) ? rocsparse::atomic_load(
                              &rem[col], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
                                     : static_cast<J>(0);
                    r   = rocsparse::cscsv_group_broadcast<GROUP>(r);
                    if(r == 0)
                    {
                        rocsparse::acquire_done_rows();

                        // A column handled by a single task writes x right away. The other
                        // tasks of a split column read the partial sum, so x is written by the
                        // last pass.
                        const bool single = owner && (kend == col_end);
                        T          xc[Q];
                        for(uint32_t q = 0; q < Q; ++q)
                        {
                            xc[q] = static_cast<T>(0);
                            if(j0 + q * R < nrhs)
                            {
                                xc[q] = alpha
                                        * rocsparse::atomic_load(
                                            acc(col, q), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                                if(diag_type == rocsparse_diag_type_non_unit)
                                {
                                    xc[q] = xc[q] * inv_diag;
                                }
                                if(single && eid == 0)
                                {
                                    Bb[col * inc_i + (j0 + q * R) * inc_j] = xc[q];
                                }
                                xc[q] = xc[q] * inv_alpha;
                            }
                        }

                        for(uint32_t i = 0; i < ITEMS; ++i)
                        {
                            if(deps[i])
                            {
                                for(uint32_t q = 0; q < Q; ++q)
                                {
                                    if(j0 + q * R < nrhs)
                                    {
                                        rocsparse::atomic_add(acc(rows[i], q), -(vals[i] * xc[q]));
                                    }
                                }
                            }
                        }
                        for(int64_t k = ktail; k < kend; k += nentry)
                        {
                            const J row = csc_row_ind[k] - idx_base;
                            if(lower ? (row > col) : (row < col))
                            {
                                const T a = conj ? rocsparse::conj(val[k]) : val[k];
                                for(uint32_t q = 0; q < Q; ++q)
                                {
                                    if(j0 + q * R < nrhs)
                                    {
                                        rocsparse::atomic_add(acc(row, q), -(a * xc[q]));
                                    }
                                }
                            }
                        }

                        // The group is in one wavefront: the release waits for the additions
                        // of all its lanes before any of them decrements a counter.
                        rocsparse::release_done_row_data();

                        // One lane per entry decrements its counter.
                        if(rid == 0)
                        {
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
                            for(int64_t k = ktail; k < kend; k += nentry)
                            {
                                const J row = csc_row_ind[k] - idx_base;
                                if(lower ? (row > col) : (row < col))
                                {
                                    rocsparse::atomic_fetch_sub(&rem[row],
                                                                static_cast<J>(1),
                                                                __ATOMIC_RELAXED,
                                                                __MEMORY_SCOPE_DEVICE);
                                }
                            }
                        }
                        done = true;
                    }
                }
            }
            // The next batch instance reuses the shared ticket of the block. Without one, the
            // waves leave as soon as their task is done instead of holding their slot.
            if(y + hipGridDim_y < ny)
            {
                __syncthreads();
            }
        }
    }

    template <uint32_t GROUP, uint32_t Q, typename I, typename J, typename T>
    static rocsparse_status cscsm_push_compute_typed(rocsparse_handle             handle,
                                                     int64_t                      panel,
                                                     int64_t                      nrhs,
                                                     int64_t                      chunk,
                                                     rocsparse_operation          op_B,
                                                     rocsparse_const_dnvec_descr  alpha,
                                                     rocsparse_const_spmat_descr  A,
                                                     rocsparse_dnmat_descr        B,
                                                     rocsparse_csrsm_info         csrsm_info,
                                                     const rocsparse::trm_info_t* trm,
                                                     void*                        buffer,
                                                     bool                         conj)
    {
        hipStream_t   stream      = handle->stream;
        const int64_t m           = A->cols;
        const int64_t batch_count = B->batch_count;
        const int64_t npanels     = (nrhs + panel - 1) / panel;
        const int64_t nslots      = npanels * batch_count;
        const int64_t num_tasks   = trm->get_max_nnz();

        // Element (i, j) is at i * inc_i + j * inc_j, with the convention of csrsm_compute: column
        // major for op_B = none in column order, row major (i * ld + j) otherwise.
        const bool col_major
            = (op_B == rocsparse_operation_none) && (B->order == rocsparse_order_column);
        const int64_t inc_i    = col_major ? 1 : B->ld;
        const int64_t inc_j    = col_major ? B->ld : 1;
        const int64_t B_stride = (batch_count > 1) ? B->batch_stride : 0;

        char*    ptr    = reinterpret_cast<char*>(buffer);
        int32_t* ticket = reinterpret_cast<int32_t*>(ptr);
        ptr += rocsparse::align_size(sizeof(int32_t) * batch_count);
        J* rem = reinterpret_cast<J*>(ptr);
        ptr += rocsparse::align_size(sizeof(J) * m * nslots);
        T* sum = reinterpret_cast<T*>(ptr);

        // Zero pivots: as in csrsm_compute.
        csrsm_info->create_singularity_numeric_exact(batch_count, A->row_type, stream);
        J* zero_pivot
            = reinterpret_cast<J*>(csrsm_info->get_singularity_numeric_exact()->get_position());
        if(A->descr->diag_type == rocsparse_diag_type_unit)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::assign_max_async(batch_count, A->row_type, zero_pivot, stream));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_device_async<J>(
                batch_count,
                zero_pivot,
                reinterpret_cast<const J*>(csrsm_info->get_position()),
                stream));
        }

        if(m == 0 || nrhs == 0)
        {
            return rocsparse_status_success;
        }

        // One right-hand side: the CSC sptrsv init and solve kernels, with the partial sums in the
        // buffer (one vector).
        if(nrhs == 1)
        {
            const uint32_t num_lanes
                = static_cast<uint32_t>(chunk / rocsparse::cscsv_kernel_constants_t::s_num_items);
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMemsetAsync(ticket, 0, sizeof(int32_t) * batch_count, stream));
            rocsparse::cscsv_init_launch_t init_launch;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::cscsv_init_launch_find(&init_launch, A->row_type, A->data_type));
            RETURN_IF_ROCSPARSE_ERROR(init_launch(handle,
                                                  m,
                                                  batch_count,
                                                  alpha->const_values,
                                                  0,
                                                  trm->get_transposed_col_ind(),
                                                  B->values,
                                                  inc_i,
                                                  B_stride,
                                                  rem,
                                                  sum));
            rocsparse::cscsv_solve_launch_t solve_launch;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_solve_launch_find(
                &solve_launch, num_lanes, A->col_type, A->row_type, A->data_type));
            RETURN_IF_ROCSPARSE_ERROR(
                solve_launch(handle,
                             batch_count,
                             m,
                             A->const_col_data,
                             A->const_row_data,
                             A->const_val_data,
                             (A->batch_stride > 0) ? A->batch_stride : static_cast<int64_t>(0),
                             trm->get_diag_ind(),
                             num_tasks,
                             trm->get_row_map(),
                             trm->get_transposed_perm(),
                             B->values,
                             inc_i,
                             B_stride,
                             rem,
                             sum,
                             ticket,
                             zero_pivot,
                             A->descr->base,
                             A->descr->fill_mode,
                             A->descr->diag_type,
                             conj));
            return rocsparse_status_success;
        }

        // PROTOTYPE: ROCSPARSE_CSCSM_WINDOW tasks per panel before the next panel (default 1).
        const char*   window_env = std::getenv("ROCSPARSE_CSCSM_WINDOW");
        const int64_t window
            = std::max(static_cast<int64_t>(1),
                       (window_env != nullptr) ? static_cast<int64_t>(std::atoll(window_env))
                                               : static_cast<int64_t>(1));
        const int64_t num_windows = (num_tasks + window - 1) / window;

        const T* alpha_ptr = reinterpret_cast<const T*>(alpha->const_values);
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(ticket, 0, sizeof(int32_t) * batch_count, stream));
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsm_push_init_kernel<rocsparse::cscsv_kernel_constants_t::s_blocksize,
                                               J>),
            dim3(rocsparse::get_grid_size_x(
                handle,
                rocsparse::ceil_div(
                    m, static_cast<int64_t>(rocsparse::cscsv_kernel_constants_t::s_blocksize)),
                rocsparse::cscsv_kernel_constants_t::s_blocksize)),
            dim3(rocsparse::cscsv_kernel_constants_t::s_blocksize),
            0,
            stream,
            m,
            nslots,
            reinterpret_cast<const J*>(trm->get_transposed_col_ind()),
            rem);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsm_push_solve_kernel<rocsparse::cscsv_kernel_constants_t::s_blocksize,
                                                GROUP,
                                                Q,
                                                I,
                                                J,
                                                T>),
            dim3(rocsparse::ceil_div(num_windows * window * npanels * GROUP,
                                     rocsparse::cscsv_kernel_constants_t::s_blocksize),
                 rocsparse::get_grid_size_y(handle, batch_count)),
            dim3(rocsparse::cscsv_kernel_constants_t::s_blocksize),
            0,
            stream,
            num_tasks,
            m,
            nrhs,
            static_cast<uint32_t>(panel),
            npanels,
            nslots,
            chunk,
            reinterpret_cast<const I*>(A->const_col_data),
            reinterpret_cast<const J*>(A->const_row_data),
            reinterpret_cast<const T*>(A->const_val_data),
            (A->batch_stride > 0) ? A->batch_stride : static_cast<int64_t>(0),
            reinterpret_cast<const I*>(trm->get_diag_ind()),
            reinterpret_cast<const J*>(trm->get_row_map()),
            reinterpret_cast<const I*>(trm->get_transposed_perm()),
            reinterpret_cast<T*>(B->values),
            inc_i,
            inc_j,
            B_stride,
            rem,
            ticket,
            zero_pivot,
            A->descr->base,
            A->descr->fill_mode,
            A->descr->diag_type,
            conj,
            window,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_ptr),
            handle->pointer_mode == rocsparse_pointer_mode_host);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsm_push_final_kernel<rocsparse::cscsv_kernel_constants_t::s_blocksize,
                                                I,
                                                J,
                                                T>),
            dim3(rocsparse::get_grid_size_x(
                     handle,
                     rocsparse::ceil_div(
                         num_tasks,
                         static_cast<int64_t>(rocsparse::cscsv_kernel_constants_t::s_blocksize)),
                     rocsparse::cscsv_kernel_constants_t::s_blocksize),
                 rocsparse::get_grid_size_y(handle, batch_count)),
            dim3(rocsparse::cscsv_kernel_constants_t::s_blocksize),
            0,
            stream,
            num_tasks,
            nrhs,
            batch_count,
            chunk,
            reinterpret_cast<const I*>(A->const_col_data),
            reinterpret_cast<const I*>(trm->get_diag_ind()),
            reinterpret_cast<const T*>(A->const_val_data),
            (A->batch_stride > 0) ? A->batch_stride : static_cast<int64_t>(0),
            reinterpret_cast<const J*>(trm->get_row_map()),
            reinterpret_cast<const I*>(trm->get_transposed_perm()),
            reinterpret_cast<T*>(B->values),
            inc_i,
            inc_j,
            B_stride,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_ptr),
            A->descr->base,
            A->descr->diag_type,
            conj,
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status cscsm_push_compute_panel(int64_t                      panel,
                                                     rocsparse_handle             handle,
                                                     int64_t                      nrhs,
                                                     int64_t                      chunk,
                                                     rocsparse_operation          op_B,
                                                     rocsparse_const_dnvec_descr  alpha,
                                                     rocsparse_const_spmat_descr  A,
                                                     rocsparse_dnmat_descr        B,
                                                     rocsparse_csrsm_info         csrsm_info,
                                                     const rocsparse::trm_info_t* trm,
                                                     void*                        buffer,
                                                     bool                         conj)
    {
        // PROTOTYPE: ROCSPARSE_CSCSM_Q right-hand sides per lane (1 to 16; default 2 from 16
        // right-hand sides, 1 below), so a group covers panel / Q lanes for the right-hand sides
        // times as many entry lanes as the analysis used for a task, at least 2 lanes and at most
        // one wavefront.
        const int64_t wf   = handle->wavefront_size;
        const char*   qenv = std::getenv("ROCSPARSE_CSCSM_Q");
        int64_t       q    = (qenv != nullptr) ? std::atoll(qenv) : ((nrhs >= 16) ? 2 : 1);
        q                  = (q >= 16) ? 16 : ((q >= 8) ? 8 : ((q >= 4) ? 4 : ((q >= 2) ? 2 : 1)));
        q                  = std::min(q, panel);
        int64_t lanes      = panel / q;
        while(lanes > wf)
        {
            q *= 2;
            lanes = panel / q;
        }
        int64_t group   = lanes * (chunk / rocsparse::cscsv_kernel_constants_t::s_num_items);
        group           = (group < 2) ? 2 : ((group > wf) ? wf : group);
        const char* env = std::getenv("ROCSPARSE_CSCSM_GROUP");
        if(env != nullptr)
        {
            group = std::atoll(env);
        }
        group = (group < lanes) ? lanes : group;
#define GROUP_CASE(N_, Q_)                                                                 \
    if(group == N_ && q == Q_)                                                             \
    {                                                                                      \
        return cscsm_push_compute_typed<N_, Q_, I, J, T>(                                  \
            handle, panel, nrhs, chunk, op_B, alpha, A, B, csrsm_info, trm, buffer, conj); \
    }
#define GROUP_CASE_Q(Q_) \
    GROUP_CASE(2, Q_)    \
    GROUP_CASE(4, Q_) GROUP_CASE(8, Q_) GROUP_CASE(16, Q_) GROUP_CASE(32, Q_) GROUP_CASE(64, Q_)
        GROUP_CASE_Q(1)
        GROUP_CASE_Q(2)
        GROUP_CASE_Q(4)
        GROUP_CASE_Q(8)
        GROUP_CASE_Q(16)
#undef GROUP_CASE_Q
#undef GROUP_CASE
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <typename I, typename J>
    static rocsparse_status cscsm_push_compute_t(rocsparse_datatype           t_type,
                                                 int64_t                      panel,
                                                 rocsparse_handle             handle,
                                                 int64_t                      nrhs,
                                                 int64_t                      chunk,
                                                 rocsparse_operation          op_B,
                                                 rocsparse_const_dnvec_descr  alpha,
                                                 rocsparse_const_spmat_descr  A,
                                                 rocsparse_dnmat_descr        B,
                                                 rocsparse_csrsm_info         csrsm_info,
                                                 const rocsparse::trm_info_t* trm,
                                                 void*                        buffer,
                                                 bool                         conj)
    {
        switch(t_type)
        {
        case rocsparse_datatype_f32_r:
            return cscsm_push_compute_panel<I, J, float>(
                panel, handle, nrhs, chunk, op_B, alpha, A, B, csrsm_info, trm, buffer, conj);
        case rocsparse_datatype_f64_r:
            return cscsm_push_compute_panel<I, J, double>(
                panel, handle, nrhs, chunk, op_B, alpha, A, B, csrsm_info, trm, buffer, conj);
        case rocsparse_datatype_f32_c:
            return cscsm_push_compute_panel<I, J, rocsparse_float_complex>(
                panel, handle, nrhs, chunk, op_B, alpha, A, B, csrsm_info, trm, buffer, conj);
        case rocsparse_datatype_f64_c:
            return cscsm_push_compute_panel<I, J, rocsparse_double_complex>(
                panel, handle, nrhs, chunk, op_B, alpha, A, B, csrsm_info, trm, buffer, conj);
        default:
            break;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    static rocsparse_status cscsm_push_compute(rocsparse_handle             handle,
                                               int64_t                      nrhs,
                                               rocsparse_operation          op_B,
                                               rocsparse_const_dnvec_descr  alpha,
                                               rocsparse_const_spmat_descr  A,
                                               rocsparse_dnmat_descr        B,
                                               rocsparse_csrsm_info         csrsm_info,
                                               const rocsparse::trm_info_t* trm,
                                               size_t                       buffer_size_in_bytes,
                                               void*                        buffer,
                                               bool                         conj)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            (buffer_size_in_bytes < rocsparse::cscsm_push_solve_buffer_size(nrhs, A, B))
                ? rocsparse_status_invalid_size
                : rocsparse_status_success,
            "the buffer of the CSC push solve is too small");

        // Size of the tasks of the analysis: same formula as in cscsm_push_analysis.
        const int64_t  m              = A->cols;
        const uint32_t analysis_lanes = rocsparse::cscsv_num_lanes(
            (m > 0) ? rocsparse::ceil_div(
                A->nnz, m * static_cast<int64_t>(rocsparse::cscsv_kernel_constants_t::s_num_items))
                    : 1,
            static_cast<uint32_t>(handle->wavefront_size));
        const int64_t chunk = static_cast<int64_t>(analysis_lanes)
                              * rocsparse::cscsv_kernel_constants_t::s_num_items;
        const int64_t panel = rocsparse::cscsm_panel_width(nrhs);

        const bool i32 = (A->col_type == rocsparse_indextype_i32);
        const bool j32 = (A->row_type == rocsparse_indextype_i32);
        if(i32 && j32)
            return cscsm_push_compute_t<int32_t, int32_t>(A->data_type,
                                                          panel,
                                                          handle,
                                                          nrhs,
                                                          chunk,
                                                          op_B,
                                                          alpha,
                                                          A,
                                                          B,
                                                          csrsm_info,
                                                          trm,
                                                          buffer,
                                                          conj);
        if(i32)
            return cscsm_push_compute_t<int32_t, int64_t>(A->data_type,
                                                          panel,
                                                          handle,
                                                          nrhs,
                                                          chunk,
                                                          op_B,
                                                          alpha,
                                                          A,
                                                          B,
                                                          csrsm_info,
                                                          trm,
                                                          buffer,
                                                          conj);
        if(j32)
            return cscsm_push_compute_t<int64_t, int32_t>(A->data_type,
                                                          panel,
                                                          handle,
                                                          nrhs,
                                                          chunk,
                                                          op_B,
                                                          alpha,
                                                          A,
                                                          B,
                                                          csrsm_info,
                                                          trm,
                                                          buffer,
                                                          conj);
        return cscsm_push_compute_t<int64_t, int64_t>(A->data_type,
                                                      panel,
                                                      handle,
                                                      nrhs,
                                                      chunk,
                                                      op_B,
                                                      alpha,
                                                      A,
                                                      B,
                                                      csrsm_info,
                                                      trm,
                                                      buffer,
                                                      conj);
    }
}
#endif

rocsparse_status rocsparse::cscsm_compute(rocsparse_handle            handle,
                                          const int64_t               nrhs,
                                          rocsparse_operation         op_A,
                                          rocsparse_operation         op_B,
                                          rocsparse_const_dnvec_descr alpha,
                                          rocsparse_const_spmat_descr A,
                                          rocsparse_dnmat_descr       B,
                                          rocsparse_csrsm_info        csrsm_info,
                                          size_t                      buffer_size_in_bytes,
                                          void*                       buffer,
                                          rocsparse_error*            p_error,
                                          bool                        limited_memory,
                                          bool                        conj)
{

    if(nrhs == 0)
    {
        return rocsparse_status_success;
    }

#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
    if(rocsparse::cscsm_use_limited_memory(op_A, op_B, alpha, A, B, limited_memory))
    {
        // The analysis of an empty matrix stores nothing.
        if(A->cols == 0)
        {
            return rocsparse_status_success;
        }
        const rocsparse::trm_info_t* trm
            = (csrsm_info != nullptr)
                  ? csrsm_info->get(rocsparse_operation_none, A->descr->fill_mode)
                  : nullptr;
        if(trm == nullptr || trm->get_row_map() == nullptr)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_pointer,
                "the limited-memory analysis was not performed for this fill mode");
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_push_compute(
            handle, nrhs, op_B, alpha, A, B, csrsm_info, trm, buffer_size_in_bytes, buffer, conj));
        return rocsparse_status_success;
    }
#endif

    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr A_csr(A, rocsparse_format_csr, &descr_csr, A->info);

    const auto A_csr_op = (op_A == rocsparse_operation_none) ? rocsparse_operation_transpose
                                                             : rocsparse_operation_none;

    const bool A_load_conjugate = (op_A == rocsparse_operation_conjugate_transpose)
                                  && rocsparse::datatype_is_complex(A->data_type);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_compute(handle,
                                                       nrhs,

                                                       A_csr_op,
                                                       A_load_conjugate,

                                                       op_B,
                                                       alpha,
                                                       &A_csr,
                                                       B,
                                                       csrsm_info,
                                                       buffer_size_in_bytes,
                                                       buffer,
                                                       p_error));

    return rocsparse_status_success;
}
