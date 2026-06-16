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

#include "rocsparse_csric0_kernel_binsearch.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csric0_device_binsearch(J m,
                                                      const I* __restrict__ csr_row_ptr,
                                                      const J* __restrict__ csr_col_ind,
                                                      T* __restrict__ csr_val,
                                                      const I* __restrict__ csr_diag_ind,
                                                      int32_t* __restrict__ done,
                                                      const J* __restrict__ map,
                                                      J* __restrict__ zero_pivot,
                                                      J* __restrict__ singular_pivot,
                                                      double               tol,
                                                      rocsparse_index_base idx_base)
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

        const auto tol_sq = tol * tol;

        // Current row this wavefront is working on
        const J row = map[idx];

        // Diagonal entry point of the current row
        const I row_diag = csr_diag_ind[row];

        // Row entry point
        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Row sum accumulator
        T sum = static_cast<T>(0);

        // Loop over column of current row
        for(I j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            const J local_col = csr_col_ind[j] - idx_base;

            // Corresponding value
            T local_val = csr_val[j];

            // Beginning of the row that corresponds to local_col
            const I local_begin = csr_row_ptr[local_col] - idx_base;

            // Diagonal entry point of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }

            // Spin loop until dependency has been resolved
            (void)rocsparse::spin_loop<SLEEP>(&done[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Make sure updated csr_val is visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load diagonal entry
            T diag_val = csr_val[local_diag];

            // Row has numerical zero diagonal
            if(diag_val == static_cast<T>(0))
            {
                if(lid == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, local_col + idx_base);
                }

                // Skip this row if it has a zero pivot
                break;
            }

            // Row has numerical singular diagonal
            if((rocsparse::imag(diag_val) == 0) && (rocsparse::real(diag_val) <= tol))
            {
                if(lid == 0)
                {
                    // We are looking for the first singular pivot
                    rocsparse::atomic_min(singular_pivot, local_col + idx_base);
                }

                // Don't skip this row if it has a singular pivot
            }

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            I l = row_begin;
            for(I k = local_begin + lid; k < local_diag; k += WF_SIZE)
            {
                // Perform a binary search to find matching columns
                I       r     = row_end - 1;
                I       m     = (r + l) >> 1;
                J       col_j = csr_col_ind[m];
                const J col_k = csr_col_ind[k];

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
                    // If a match has been found, do linear combination
                    local_sum = rocsparse::fma(csr_val[k], rocsparse::conj(csr_val[m]), local_sum);
                }
            }

            // Accumulate row sum
            local_sum = rocsparse::wfreduce_sum<WF_SIZE>(local_sum);

            // Last lane id computes the Cholesky factor and writes it to global memory
            if(lid == WF_SIZE - 1)
            {
                local_val = (local_val - local_sum) * diag_val;
                sum       = rocsparse::fma(local_val, rocsparse::conj(local_val), sum);

                csr_val[j] = local_val;
            }
        }

        // Last lane processes the diagonal entry
        if(lid == WF_SIZE - 1)
        {
            if((row_diag >= 0))
            {
                const T diag_val = csr_val[row_diag] - sum;

                // check for negative value and numerical small value
                if((rocsparse::imag(diag_val) == 0) && (rocsparse::real(diag_val) <= (tol_sq)))
                {
                    rocsparse::atomic_min(singular_pivot, (row + idx_base));
                }

                csr_val[row_diag] = rocsparse::sqrt(rocsparse::abs(csr_val[row_diag] - sum));
                if(csr_val[row_diag] == static_cast<T>(0))
                {
                    rocsparse::atomic_min(zero_pivot, (row + idx_base));
                }
            }
        }

        if(lid == WF_SIZE - 1)
        {
            // Last lane writes "we are done" flag
            __hip_atomic_store(&done[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csric0_kernel_binsearch(J m,
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
                                 rocsparse_index_base idx_base)
    {

        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET(is_singular_tol_host_mode, tolerance_64);
        const double tolerance
            = (tolerance_datatype == rocsparse_datatype_f64_r) ? tolerance_64 : tolerance_32;

        const auto batch_index = hipBlockIdx_y;
        rocsparse::csric0_device_binsearch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, J>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            csr_diag_ind,
            done + batch_index * done_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            singular_pivot + batch_index * singular_pivot_stride,
            tolerance,
            idx_base);
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I, typename J>
    static rocsparse_status csric0_kernel_binsearch_launch(rocsparse_handle      handle,
                                                           rocsparse_csric0_info csric0_info,
                                                           rocsparse_spmat_descr A,
                                                           size_t                buffer_size,
                                                           void*                 buffer)
    {
        auto trm_info = csric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;

        const dim3 csric0_blocks((A->rows * handle->wavefront_size - 1) / BLOCKSIZE + 1,
                                 A->batch_count);

        const dim3 csric0_threads(BLOCKSIZE);

        auto                         numeric_exact = csric0_info->get_singularity_numeric_exact();
        auto                         numeric_near  = csric0_info->get_singularity_numeric_near();
        const rocsparse_pointer_mode tolerance_pointer_mode
            = numeric_near->get_tolerance_pointer_mode();
        const rocsparse_datatype tolerance_datatype = numeric_near->get_tolerance_datatype();
        const float*             tolerance_pointer_32
            = reinterpret_cast<const float*>(numeric_near->get_tolerance_pointer());
        const double* tolerance_pointer_64
            = reinterpret_cast<const double*>(numeric_near->get_tolerance_pointer());

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csric0_kernel_binsearch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, J>),
            csric0_blocks,
            csric0_threads,
            0,
            handle->stream,
            A->rows,
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

            tolerance_datatype,
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_32),
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(tolerance_pointer_mode, tolerance_pointer_64),
            (tolerance_pointer_mode == rocsparse_pointer_mode_host),
            A->descr->base);

        return rocsparse_status_success;
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I>
    static rocsparse::csric0_kernel_launch_t transform_j_type(rocsparse_indextype value)
    {

        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::
                csric0_kernel_binsearch_launch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::
                csric0_kernel_binsearch_launch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, int64_t>;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }

        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename... P>
    static rocsparse::csric0_kernel_launch_t transform_i_type(rocsparse_indextype value, P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::transform_j_type<SLEEP, BLOCKSIZE, WF_SIZE, T, int32_t>(
                std::forward<P>(p)...);
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::transform_j_type<SLEEP, BLOCKSIZE, WF_SIZE, T, int64_t>(
                std::forward<P>(p)...);
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename... P>
    static rocsparse::csric0_kernel_launch_t transform_t_type(rocsparse_datatype value, P... p)
    {
        switch(value)
        {

        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, float>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, rocsparse_float_complex>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, double>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, rocsparse_double_complex>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
        {
            std::stringstream sstr;
            sstr << rocsparse::enum_utils::to_string(value) << " not supported";
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  sstr.str().c_str());
        }
        }

        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

}

rocsparse::csric0_kernel_launch_t rocsparse::find_csric0_kernel_binsearch_launch(
    rocsparse_handle handle, rocsparse_csric0_info csric0_info, rocsparse_const_spmat_descr A)
{

    const bool sleep
        = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908 && //
           handle->asic_rev < 2);

    if(sleep)
    {
        return rocsparse::transform_t_type<true, 256, 64>(A->data_type, A->row_type, A->col_type);
    }
    else
    {
        switch(handle->wavefront_size)
        {

        case 32:
        {
            return rocsparse::transform_t_type<false, 256, 32>(
                A->data_type, A->row_type, A->col_type);
        }

        case 64:
        {
            return rocsparse::transform_t_type<false, 256, 64>(
                A->data_type, A->row_type, A->col_type);
        }

        default:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                  "invalid wavefront size");
        }
        }
    }
}
