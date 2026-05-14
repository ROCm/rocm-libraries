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

#include "rocsparse_csrilu0_kernel_binsearch.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csrilu0_device_binsearch(J m_,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       T* __restrict__ csr_val,
                                                       const I* __restrict__ csr_diag_ind,
                                                       int32_t* __restrict__ done,
                                                       const J* __restrict__ map,
                                                       J* __restrict__ zero_pivot,
                                                       J* __restrict__ singular_pivot,
                                                       double               tol,
                                                       rocsparse_index_base idx_base,
                                                       int                  boost,
                                                       double               boost_tol,
                                                       T                    boost_val)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        const auto lid = hipThreadIdx_x & (WFSIZE - 1);
        const auto wid = hipThreadIdx_x / WFSIZE;
        const auto idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= m_)
        {
            return;
        }

        // Current row this wavefront is working on
        J row = map[idx];

        // Diagonal entry point of the current row
        I row_diag = csr_diag_ind[row];

        // Row entry point
        I row_begin = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Loop over column of current row
        for(I j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            J local_col = csr_col_ind[j] - idx_base;

            // Corresponding value
            T local_val = csr_val[j];

            // End of the row that corresponds to local_col
            I local_end = csr_row_ptr[local_col + 1] - idx_base;

            // Diagonal entry point of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = local_end - 1;
            }

            // Spin loop until dependency has been resolved
            (void)rocsparse::spin_loop<SLEEP>(&done[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Make sure updated csr_val is visible
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load diagonal entry
            T diag_val = csr_val[local_diag];

            if(diag_val == static_cast<T>(0))
            {
                // Skip this row if it has a zero pivot
                break;
            }

            csr_val[j] = local_val = local_val / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            I l = j + 1;
            for(I k = local_diag + 1 + lid; k < local_end; k += WFSIZE)
            {
                // Perform a binary search to find matching columns
                I r     = row_end - 1;
                I m     = (r + l) >> 1;
                J col_j = csr_col_ind[m];

                J col_k = csr_col_ind[k];

                // Binary search
                while(l < r)
                {
                    if(col_j < col_k)
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m;
                    }

                    m     = (r + l) >> 1;
                    col_j = csr_col_ind[m];
                }

                // Check if a match has been found
                if(col_j == col_k)
                {
                    // If a match has been found, do ILU computation
                    csr_val[l] = rocsparse::fma(-local_val, csr_val[k], csr_val[l]);
                }
            }
        }

        __threadfence_block();

        const bool is_diag = (row_diag >= 0);
        if(is_diag)
        {
            const auto diag_val     = csr_val[row_diag];
            const auto abs_diag_val = rocsparse::abs(diag_val);
            if(boost)
            {
                const bool is_too_small = (abs_diag_val <= boost_tol);
                if(is_too_small)
                {
                    if(lid == 0)
                    {
                        csr_val[row_diag] = rocsparse::assign_ilu0_boost_value(diag_val, boost_val);
                    };
                };
            }
            else
            {

                const bool is_singular_pivot = (abs_diag_val <= tol);
                if(is_singular_pivot)
                {
                    if(lid == 0)
                    {
                        rocsparse::atomic_min(singular_pivot, (row + idx_base));
                    }
                }

                const bool is_zero_pivot = (diag_val == static_cast<T>(0));
                if(is_zero_pivot)
                {
                    if(lid == 0)
                    {
                        rocsparse::atomic_min(zero_pivot, (row + idx_base));
                    }
                }
            }
        }

        // Make sure updated csr_val is written to global memory
        __threadfence();

        if(lid == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_kernel_binsearch(J m,
                                  const I* __restrict__ csr_row_ptr,
                                  const J* __restrict__ csr_col_ind,
                                  T* __restrict__ csr_val,
                                  int64_t csr_val_stride,
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
                                  int                  boost_enable,
                                  size_t               boost_tol_size,
                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                                  bool is_tol_host_mode,
                                  ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                                  bool is_val_host_mode)
    {
        const auto i = hipBlockIdx_y;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_64);

        const double tolerance
            = (tolerance_datatype == rocsparse_datatype_f64_r) ? tolerance_64 : tolerance_32;

        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(
            boost_enable && (boost_tol_size == sizeof(float)), is_tol_host_mode, boost_tol_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(
            boost_enable && (boost_tol_size == sizeof(double)), is_tol_host_mode, boost_tol_64);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(boost_enable, is_val_host_mode, boost_val);

        const double boost_tol = (boost_tol_size == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::csrilu0_device_binsearch<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + i * csr_val_stride,
            csr_diag_ind,
            done + i * done_stride,
            map,
            zero_pivot + i * zero_pivot_stride,
            singular_pivot + i * singular_pivot_stride,
            tolerance,
            idx_base,
            boost_enable,
            boost_tol,
            boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    rocsparse_status csrilu0_kernel_binsearch_launch(rocsparse_handle          handle,
                                                     rocsparse_csrilu0_info    csrilu0_info,
                                                     rocsparse_spmat_descr     A,
                                                     rocsparse::numeric_boost* boost,

                                                     size_t buffer_size,
                                                     void*  buffer)
    {

        auto trm_info = csrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;

        auto                         numeric_exact = csrilu0_info->get_singularity_numeric_exact();
        auto                         numeric_near  = csrilu0_info->get_singularity_numeric_near();
        const rocsparse_pointer_mode tolerance_pointer_mode
            = numeric_near->get_tolerance_pointer_mode();
        const rocsparse_datatype tolerance_datatype = numeric_near->get_tolerance_datatype();
        const float*             tolerance_pointer_32
            = reinterpret_cast<const float*>(numeric_near->get_tolerance_pointer());
        const double* tolerance_pointer_64
            = reinterpret_cast<const double*>(numeric_near->get_tolerance_pointer());

        const auto boost_enable           = boost->get_enable();
        const auto boost_tol_size         = rocsparse::datatype_sizeof(boost->get_tol_datatype());
        const auto boost_tol_pointer_mode = boost->get_tol_pointer_mode();
        const auto boost_val_pointer_mode = boost->get_val_pointer_mode();

        const float*  boost_tol_32 = (boost_tol_size == sizeof(float))
                                         ? reinterpret_cast<const float*>(boost->get_tol())
                                         : nullptr;
        const double* boost_tol_64 = (boost_tol_size == sizeof(double))
                                         ? reinterpret_cast<const double*>(boost->get_tol())
                                         : nullptr;
        const T*      boost_val    = reinterpret_cast<const T*>(boost->get_val());

        dim3 csrilu0_blocks((A->rows * handle->wavefront_size - 1) / BLOCKSIZE + 1, A->batch_count);
        dim3 csrilu0_threads(BLOCKSIZE);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrilu0_kernel_binsearch<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>),
            csrilu0_blocks,
            csrilu0_threads,
            0,
            handle->stream,
            static_cast<J>(A->rows),
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<T*>(A->val_data),
            A->batch_stride,
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            done_array,
            done_array_stride,
            reinterpret_cast<const J*>(trm_info->get_row_map()),

            reinterpret_cast<J*>(numeric_exact->get_position()),
            numeric_exact->get_stride(),
            reinterpret_cast<J*>(numeric_near->get_position()),
            numeric_near->get_stride(),
            //
            tolerance_datatype,
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_64),
            (tolerance_pointer_mode == rocsparse_pointer_mode_host),
            //
            A->descr->base,
            boost_enable,
            boost_tol_size,
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_tol_pointer_mode, boost_tol_64),
            boost_tol_pointer_mode == rocsparse_pointer_mode_host,
            ROCSPARSE_SCALAR_HOST_DEVICE_PERMISSIVE_ARGUMENT(boost_val_pointer_mode, boost_val),
            boost_val_pointer_mode == rocsparse_pointer_mode_host);

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     SLEEP,
              typename T,
              typename I,
              typename... P>
    static csrilu0_kernel_launch_t transform_j_type(rocsparse_indextype j, P... p)
    {
        return (j == rocsparse_indextype_i32) ? rocsparse::
                       csrilu0_kernel_binsearch_launch<BLOCKSIZE, WF_SIZE, SLEEP, T, I, int32_t>
               : (j == rocsparse_indextype_i64) ? rocsparse::
                       csrilu0_kernel_binsearch_launch<BLOCKSIZE, WF_SIZE, SLEEP, T, I, int64_t>
                                                : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename... P>
    static csrilu0_kernel_launch_t transform_i_type(rocsparse_indextype i, P... p)
    {
        return (i == rocsparse_indextype_i32)
                   ? rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, SLEEP, T, int32_t>(p...)
               : (i == rocsparse_indextype_i64)
                   ? rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, SLEEP, T, int64_t>(p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
    static csrilu0_kernel_launch_t transform_t_type(rocsparse_datatype i, P... p)
    {
        return (i == rocsparse_datatype_f32_r)
                   ? rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, float>(p...)
               : (i == rocsparse_datatype_f32_c) ? rocsparse::
                       transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_float_complex>(p...)
               : (i == rocsparse_datatype_f64_c) ? rocsparse::
                       transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_double_complex>(p...)
               : (i == rocsparse_datatype_f64_r)
                   ? rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, double>(p...)
                   : nullptr;
    }

}

rocsparse::csrilu0_kernel_launch_t rocsparse::find_csrilu0_kernel_binsearch_launch(
    rocsparse_handle handle, rocsparse_csrilu0_info csrilu0_info, rocsparse_const_spmat_descr A)
{
    const bool sleep
        = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908 && //
           handle->asic_rev < 2);

    if((handle->wavefront_size == 32) && (sleep == false))
    {
        return rocsparse::transform_t_type<256, 32, false>(A->data_type, A->row_type, A->col_type);
    }
    else if((handle->wavefront_size == 64) && (sleep == false))
    {
        return rocsparse::transform_t_type<256, 64, false>(A->data_type, A->row_type, A->col_type);
    }
    else if((handle->wavefront_size == 64) && (sleep == true))
    {
        return rocsparse::transform_t_type<256, 64, true>(A->data_type, A->row_type, A->col_type);
    }
    else
    {
        return nullptr;
    }
}
