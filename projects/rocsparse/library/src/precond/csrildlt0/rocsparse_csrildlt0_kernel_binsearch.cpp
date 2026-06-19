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

#include "rocsparse_csrildlt0_kernel_binsearch.hpp"
#include "rocsparse-complex-types.h"
#include "rocsparse_common.hpp"
#include "rocsparse_floating_data_t.hpp"
#include "rocsparse_numeric_boost.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // ILDLT(0) binary-search based device kernel.
    //
    // Computes A ≈ L D L^H where L is unit lower triangular and D is real diagonal.
    //
    // For row i (Hermitian LDL^H):
    //   D_i = real(A_{ii}) - sum_{k<i} |L_{ik}|^2 * D_k        (no sqrt, D always real)
    //   L_{ij} = (A_{ij} - sum_{k<j} L_{ik} * D_k * conj(L_{jk})) / D_j
    //
    // D is stored in a separate dense array diag[] of type floating_data_t<T>.
    // csr_val stores the strictly lower-triangular entries of L.
    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csrildlt0_device_binsearch(J m,
                                                         const I* __restrict__ csr_row_ptr,
                                                         const J* __restrict__ csr_col_ind,
                                                         T* __restrict__ csr_val,
                                                         floating_data_t<T>* __restrict__ diag,
                                                         const I* __restrict__ csr_diag_ind,
                                                         int32_t* __restrict__ done,
                                                         const J* __restrict__ map,
                                                         J* __restrict__ zero_pivot,
                                                         J* __restrict__ singular_pivot,
                                                         double               tol,
                                                         rocsparse_index_base idx_base,
                                                         int                  boost,
                                                         double               boost_tol,
                                                         floating_data_t<T>   boost_val)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");
        const auto lid = hipThreadIdx_x & (WF_SIZE - 1);
        const auto wid = hipThreadIdx_x / WF_SIZE;
        const auto idx = hipBlockIdx_x * BLOCKSIZE / WF_SIZE + wid;

        if(idx >= m)
        {
            return;
        }

        // Current row this wavefront is working on
        const J row = map[idx];

        // Diagonal position in the current row
        const I row_diag = csr_diag_ind[row];

        // Row bounds
        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Accumulate: sum_{k<i} |L_{ik}|^2 * D_k  (used for D_i update)
        floating_data_t<T> diag_sum = static_cast<floating_data_t<T>>(0);

        // Loop over strictly lower-triangular columns of current row
        for(I j = row_begin; j < row_diag; ++j)
        {
            const J local_col = csr_col_ind[j] - idx_base;

            // Current L value (will be updated)
            T local_val = csr_val[j];

            const I local_begin = csr_row_ptr[local_col] - idx_base;
            I       local_diag  = csr_diag_ind[local_col];

            T local_sum = static_cast<T>(0);

            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }

            // Spin until row local_col has been computed
            (void)rocsparse::spin_loop<SLEEP>(&done[local_col], __HIP_MEMORY_SCOPE_AGENT);

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load D_{local_col}
            floating_data_t<T> d_j = diag[local_col];

            if(d_j == static_cast<floating_data_t<T>>(0))
            {
                if(lid == 0)
                {
                    rocsparse::atomic_min(zero_pivot, local_col + idx_base);
                }
                break;
            }

            if(rocsparse::abs(d_j) <= tol)
            {
                if(lid == 0)
                {
                    rocsparse::atomic_min(singular_pivot, local_col + idx_base);
                }
                // Don't skip: still update L with a near-singular pivot
            }

            floating_data_t<T> inv_d_j = static_cast<floating_data_t<T>>(1) / d_j;

            // Compute sum_{k<local_col} L_{row,k} * D_k * L_{local_col,k}
            // using binary search to find matching columns
            I l = row_begin;
            for(I k = local_begin + lid; k < local_diag; k += WF_SIZE)
            {
                I       r     = row_end - 1;
                I       m_idx = (r + l) >> 1;
                J       col_j = csr_col_ind[m_idx];
                const J col_k = csr_col_ind[k];

                while(l < r)
                {
                    if(col_j < col_k)
                    {
                        l = m_idx + 1;
                    }
                    else
                    {
                        r = m_idx;
                    }

                    m_idx = (r + l) >> 1;
                    col_j = csr_col_ind[m_idx];
                }

                if(col_j == col_k)
                {
                    // L_{row,k} * D_k * conj(L_{local_col,k})
                    floating_data_t<T> d_k = diag[col_k - idx_base];
                    local_sum              = rocsparse::fma(csr_val[m_idx],
                                               static_cast<T>(d_k) * rocsparse::conj(csr_val[k]),
                                               local_sum);
                }
            }

            local_sum = rocsparse::wfreduce_sum<WF_SIZE>(local_sum);

            if(lid == WF_SIZE - 1)
            {
                // L_{row, local_col} = (A_{row,local_col} - sum) / D_{local_col}
                local_val  = (local_val - local_sum) * inv_d_j;
                csr_val[j] = local_val;
                // Accumulate for diagonal: |L_{row,j}|^2 * D_j = (re^2 + im^2) * D_j
                const floating_data_t<T> re_l = rocsparse::real(local_val);
                const floating_data_t<T> im_l = rocsparse::imag(local_val);
                diag_sum = rocsparse::fma(rocsparse::fma(re_l, re_l, im_l * im_l), d_j, diag_sum);
            }
        }

        if(lid == WF_SIZE - 1)
        {
            if(row_diag >= 0)
            {
                // D_i = real(A_{ii}) - sum_{k<i} |L_{ik}|^2 * D_k
                floating_data_t<T> d_i = rocsparse::real(csr_val[row_diag]) - diag_sum;

                if(rocsparse::abs(d_i) <= tol)
                {
                    rocsparse::atomic_min(singular_pivot, (row + idx_base));
                }

                if(boost && rocsparse::abs(d_i) <= boost_tol)
                {
                    d_i = boost_val;
                }

                diag[row]         = d_i;
                csr_val[row_diag] = static_cast<T>(0); // unit diagonal of L (implicit)

                if(d_i == static_cast<floating_data_t<T>>(0))
                {
                    rocsparse::atomic_min(zero_pivot, (row + idx_base));
                }
            }
        }

        if(lid == WF_SIZE - 1)
        {
            __hip_atomic_store(&done[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrildlt0_kernel_binsearch(J m,
                                    const I* __restrict__ csr_row_ptr,
                                    const J* __restrict__ csr_col_ind,
                                    T* __restrict__ csr_val,
                                    int64_t csr_val_stride,
                                    floating_data_t<T>* __restrict__ diag,
                                    int64_t diag_stride,
                                    const I* __restrict__ csr_diag_ind,
                                    int32_t* __restrict__ done,
                                    int64_t done_stride,
                                    const J* __restrict__ map,
                                    J* __restrict__ zero_pivot,
                                    int64_t zero_pivot_stride,
                                    J* __restrict__ singular_pivot,
                                    int64_t            singular_pivot_stride,
                                    rocsparse_datatype tolerance_datatype,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, tolerance_32),
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, tolerance_64),
                                    bool                 is_singular_tol_host_mode,
                                    rocsparse_index_base idx_base,
                                    int                  boost,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                                    rocsparse_datatype boost_tol_datatype,
                                    bool               is_boost_tol_host_mode,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(floating_data_t<T>,
                                                                        boost_val),
                                    bool is_boost_val_host_mode)
    {
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_64);
        const double tolerance
            = (tolerance_datatype == rocsparse_datatype_f64_r) ? tolerance_64 : tolerance_32;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost
                                                && (boost_tol_datatype == rocsparse_datatype_f32_r),
                                            is_boost_tol_host_mode,
                                            boost_tol_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost
                                                && (boost_tol_datatype == rocsparse_datatype_f64_r),
                                            is_boost_tol_host_mode,
                                            boost_tol_64);
        const double b_tol
            = (boost_tol_datatype == rocsparse_datatype_f64_r) ? boost_tol_64 : boost_tol_32;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost, is_boost_val_host_mode, boost_val);

        const auto batch_index = hipBlockIdx_y;
        rocsparse::csrildlt0_device_binsearch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, J>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            diag + batch_index * diag_stride,
            csr_diag_ind,
            done + batch_index * done_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            singular_pivot + batch_index * singular_pivot_stride,
            tolerance,
            idx_base,
            boost,
            b_tol,
            boost_val);
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I, typename J>
    static rocsparse_status
        csrildlt0_kernel_binsearch_launch(rocsparse_handle         handle,
                                          rocsparse_csrildlt0_info csrildlt0_info,
                                          rocsparse_spmat_descr    A,
                                          void*                    diag,
                                          size_t                   buffer_size,
                                          void*                    buffer)
    {
        auto trm_info = csrildlt0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;

        const dim3 csrildlt0_blocks((A->rows * handle->wavefront_size - 1) / BLOCKSIZE + 1,
                                    A->batch_count);
        const dim3 csrildlt0_threads(BLOCKSIZE);

        auto numeric_exact = csrildlt0_info->get_singularity_numeric_exact();
        auto numeric_near  = csrildlt0_info->get_singularity_numeric_near();
        const rocsparse_pointer_mode tolerance_pointer_mode
            = numeric_near->get_tolerance_pointer_mode();
        const rocsparse_datatype tolerance_datatype = numeric_near->get_tolerance_datatype();
        const float*             tolerance_pointer_32
            = reinterpret_cast<const float*>(numeric_near->get_tolerance_pointer());
        const double* tolerance_pointer_64
            = reinterpret_cast<const double*>(numeric_near->get_tolerance_pointer());

        // Boost parameters
        auto                     boost              = A->info->get_boost();
        const int                boost_enable       = boost->get_enable();
        const rocsparse_datatype boost_tol_datatype = boost->get_tol_datatype();
        const auto               boost_tol_size = rocsparse::datatype_sizeof(boost_tol_datatype);
        const rocsparse_pointer_mode boost_tol_pointer_mode = boost->get_tol_pointer_mode();
        const rocsparse_pointer_mode boost_val_pointer_mode = boost->get_val_pointer_mode();
        const float*                 boost_tol_32           = (boost_tol_size == sizeof(float))
                                                                  ? reinterpret_cast<const float*>(boost->get_tol())
                                                                  : nullptr;
        const double*                boost_tol_64           = (boost_tol_size == sizeof(double))
                                                                  ? reinterpret_cast<const double*>(boost->get_tol())
                                                                  : nullptr;
        const floating_data_t<T>*    boost_val_ptr
            = reinterpret_cast<const floating_data_t<T>*>(boost->get_val());

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrildlt0_kernel_binsearch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, J>),
            csrildlt0_blocks,
            csrildlt0_threads,
            0,
            handle->stream,
            A->rows,
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<T*>(A->val_data),
            A->batch_stride,
            reinterpret_cast<floating_data_t<T>*>(diag),
            static_cast<int64_t>(A->rows),
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            done_array,
            done_array_stride,
            reinterpret_cast<const J*>(trm_info->get_row_map()),
            reinterpret_cast<J*>(numeric_exact->get_position()),
            numeric_exact->get_stride(),
            reinterpret_cast<J*>(numeric_near->get_position()),
            numeric_near->get_stride(),
            tolerance_datatype,
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_64),
            (tolerance_pointer_mode == rocsparse_pointer_mode_host),
            A->descr->base,
            boost_enable,
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_64),
            boost_tol_datatype,
            (boost_tol_pointer_mode == rocsparse_pointer_mode_host),
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_val_pointer_mode, boost_val_ptr),
            (boost_val_pointer_mode == rocsparse_pointer_mode_host));

        return rocsparse_status_success;
    }

// clang-format off
#define INST_ALL_BINSEARCH(WF, T, I, J)                                                          \
    template rocsparse_status csrildlt0_kernel_binsearch_launch<false, 256, WF, T, I, J>(        \
        rocsparse_handle, rocsparse_csrildlt0_info, rocsparse_spmat_descr, void*, size_t, void*); \
    template rocsparse_status csrildlt0_kernel_binsearch_launch<true,  256, WF, T, I, J>(        \
        rocsparse_handle, rocsparse_csrildlt0_info, rocsparse_spmat_descr, void*, size_t, void*);

    // Explicit instantiations
    INST_ALL_BINSEARCH(64, float,                    int32_t, int32_t);
    INST_ALL_BINSEARCH(32, float,                    int32_t, int32_t);
    INST_ALL_BINSEARCH(64, double,                   int32_t, int32_t);
    INST_ALL_BINSEARCH(32, double,                   int32_t, int32_t);
    INST_ALL_BINSEARCH(64, rocsparse_float_complex,  int32_t, int32_t);
    INST_ALL_BINSEARCH(32, rocsparse_float_complex,  int32_t, int32_t);
    INST_ALL_BINSEARCH(64, rocsparse_double_complex, int32_t, int32_t);
    INST_ALL_BINSEARCH(32, rocsparse_double_complex, int32_t, int32_t);
    INST_ALL_BINSEARCH(64, float,                    int64_t, int32_t);
    INST_ALL_BINSEARCH(32, float,                    int64_t, int32_t);
    INST_ALL_BINSEARCH(64, double,                   int64_t, int32_t);
    INST_ALL_BINSEARCH(32, double,                   int64_t, int32_t);
    INST_ALL_BINSEARCH(64, rocsparse_float_complex,  int64_t, int32_t);
    INST_ALL_BINSEARCH(32, rocsparse_float_complex,  int64_t, int32_t);
    INST_ALL_BINSEARCH(64, rocsparse_double_complex, int64_t, int32_t);
    INST_ALL_BINSEARCH(32, rocsparse_double_complex, int64_t, int32_t);
    INST_ALL_BINSEARCH(64, float,                    int64_t, int64_t);
    INST_ALL_BINSEARCH(32, float,                    int64_t, int64_t);
    INST_ALL_BINSEARCH(64, double,                   int64_t, int64_t);
    INST_ALL_BINSEARCH(32, double,                   int64_t, int64_t);
    INST_ALL_BINSEARCH(64, rocsparse_float_complex,  int64_t, int64_t);
    INST_ALL_BINSEARCH(32, rocsparse_float_complex,  int64_t, int64_t);
    INST_ALL_BINSEARCH(64, rocsparse_double_complex, int64_t, int64_t);
    INST_ALL_BINSEARCH(32, rocsparse_double_complex, int64_t, int64_t);

#undef INST_ALL_BINSEARCH
    // clang-format on

    // find_csrildlt0_kernel_binsearch_launch: select based on device and matrix properties
    csrildlt0_kernel_launch_t
        find_csrildlt0_kernel_binsearch_launch(rocsparse_handle            handle,
                                               rocsparse_csrildlt0_info    csrildlt0_info,
                                               rocsparse_const_spmat_descr A)
    {
        const bool sleep = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908
                            && handle->asic_rev < 2);
        const bool use_wf64 = (handle->wavefront_size == 64);

        // Helper lambda to select index-type variant given a fixed scalar type T
#define INST_ALL_BINSEARCH_IDX(SCALAR_T)                                   \
    [&]() -> csrildlt0_kernel_launch_t {                                   \
        if(A->row_type == rocsparse_indextype_i32)                         \
        {                                                                  \
            if(use_wf64)                                                   \
                return sleep ? csrildlt0_kernel_binsearch_launch<true,     \
                                                                 256,      \
                                                                 64,       \
                                                                 SCALAR_T, \
                                                                 int32_t,  \
                                                                 int32_t>  \
                             : csrildlt0_kernel_binsearch_launch<false,    \
                                                                 256,      \
                                                                 64,       \
                                                                 SCALAR_T, \
                                                                 int32_t,  \
                                                                 int32_t>; \
            else                                                           \
                return sleep ? csrildlt0_kernel_binsearch_launch<true,     \
                                                                 256,      \
                                                                 32,       \
                                                                 SCALAR_T, \
                                                                 int32_t,  \
                                                                 int32_t>  \
                             : csrildlt0_kernel_binsearch_launch<false,    \
                                                                 256,      \
                                                                 32,       \
                                                                 SCALAR_T, \
                                                                 int32_t,  \
                                                                 int32_t>; \
        }                                                                  \
        else if(A->col_type == rocsparse_indextype_i32)                    \
        {                                                                  \
            if(use_wf64)                                                   \
                return sleep ? csrildlt0_kernel_binsearch_launch<true,     \
                                                                 256,      \
                                                                 64,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int32_t>  \
                             : csrildlt0_kernel_binsearch_launch<false,    \
                                                                 256,      \
                                                                 64,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int32_t>; \
            else                                                           \
                return sleep ? csrildlt0_kernel_binsearch_launch<true,     \
                                                                 256,      \
                                                                 32,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int32_t>  \
                             : csrildlt0_kernel_binsearch_launch<false,    \
                                                                 256,      \
                                                                 32,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int32_t>; \
        }                                                                  \
        else                                                               \
        {                                                                  \
            if(use_wf64)                                                   \
                return sleep ? csrildlt0_kernel_binsearch_launch<true,     \
                                                                 256,      \
                                                                 64,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int64_t>  \
                             : csrildlt0_kernel_binsearch_launch<false,    \
                                                                 256,      \
                                                                 64,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int64_t>; \
            else                                                           \
                return sleep ? csrildlt0_kernel_binsearch_launch<true,     \
                                                                 256,      \
                                                                 32,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int64_t>  \
                             : csrildlt0_kernel_binsearch_launch<false,    \
                                                                 256,      \
                                                                 32,       \
                                                                 SCALAR_T, \
                                                                 int64_t,  \
                                                                 int64_t>; \
        }                                                                  \
    }()

        switch(A->data_type)
        {
        case rocsparse_datatype_f32_r:
            return INST_ALL_BINSEARCH_IDX(float);
        case rocsparse_datatype_f64_r:
            return INST_ALL_BINSEARCH_IDX(double);
        case rocsparse_datatype_f32_c:
            return INST_ALL_BINSEARCH_IDX(rocsparse_float_complex);
        case rocsparse_datatype_f64_c:
            return INST_ALL_BINSEARCH_IDX(rocsparse_double_complex);
        default:
            return nullptr;
        }

#undef INST_ALL_BINSEARCH_IDX
    }

} // namespace rocsparse
