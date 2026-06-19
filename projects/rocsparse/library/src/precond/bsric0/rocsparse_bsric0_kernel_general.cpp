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

#include "rocsparse_bsric0_kernel_general.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsric0_device_general(rocsparse_direction direction,
                                                    J                   mb,
                                                    J                   block_dim,
                                                    const I* __restrict__ bsr_row_ptr,
                                                    const J* __restrict__ bsr_col_ind,
                                                    T* __restrict__ bsr_val,
                                                    const I* __restrict__ bsr_diag_ind,
                                                    int32_t* __restrict__ block_done,
                                                    const J* __restrict__ block_map,
                                                    J* __restrict__ zero_pivot,
                                                    rocsparse_index_base idx_base)
    {
        auto lid = hipThreadIdx_x & (WFSIZE - 1);
        auto wid = hipThreadIdx_x / WFSIZE;

        auto idx = hipBlockIdx_x + wid;

        // Current block row this wavefront is working on
        J block_row = block_map[idx];

        // Block diagonal entry point of the current block row
        I block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(lid == WFSIZE - 1)
            {
                rocsparse::atomic_min(zero_pivot, block_row + idx_base);

                // Last lane in wavefront writes "we are done" flag for its block row
                __hip_atomic_store(
                    &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }

            return;
        }

        // Block row entry point
        I block_row_begin = bsr_row_ptr[block_row] - idx_base;
        I block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

        for(J row = lid; row < block_dim; row += WFSIZE)
        {
            // Row sum accumulator
            T row_sum = static_cast<T>(0);

            // Loop over block columns of current block row
            for(I j = block_row_begin; j < block_row_diag; j++)
            {
                // Block column index currently being processes
                J block_col = bsr_col_ind[j] - idx_base;

                // Beginning of the block row that corresponds to block_col
                I local_block_begin = bsr_row_ptr[block_col] - idx_base;

                // Block diagonal entry point of block row 'block_col'
                I local_block_diag = bsr_diag_ind[block_col];

                // Structural zero pivot, do not process this block row
                if(local_block_diag == -1)
                {
                    // If one thread in the warp breaks here, then all threads in
                    // the warp break so no divergence
                    break;
                }

                // Spin loop until dependency has been resolved

                (void)rocsparse::spin_loop<SLEEP>(&block_done[block_col], __HIP_MEMORY_SCOPE_AGENT);
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                for(J k = 0; k < block_dim; k++)
                {
                    // Column index currently being processes
                    J col = block_dim * block_col + k;

                    // Load diagonal entry
                    T diag_val
                        = bsr_val[block_dim * block_dim * local_block_diag + block_dim * k + k];

                    // Row has numerical zero pivot
                    if(diag_val == static_cast<T>(0))
                    {
                        if(lid == 0)
                        {
                            // We are looking for the first zero pivot
                            rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                        }

                        // Normally would break here but to avoid divergence set diag_val to one and continue
                        // The zero pivot has already been set so further computation does not matter
                        diag_val = static_cast<T>(1);
                    }

                    T val = static_cast<T>(0);

                    // Corresponding value
                    if(direction == rocsparse_direction_row)
                    {
                        val = bsr_val[block_dim * block_dim * j + block_dim * row + k];
                    }
                    else
                    {
                        val = bsr_val[block_dim * block_dim * j + block_dim * k + row];
                    }

                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    // Loop over the row the current column index depends on
                    // Each lane processes one entry
                    for(I p = local_block_begin; p < local_block_diag + 1; p++)
                    {
                        // Perform a binary search to find matching block columns
                        I l = block_row_begin;
                        I r = block_row_end - 1;
                        I m = (r + l) >> 1;

                        J block_col_j = bsr_col_ind[m] - idx_base;
                        J block_col_p = bsr_col_ind[p] - idx_base;

                        // Binary search for block column
                        while(l < r)
                        {
                            if(block_col_j < block_col_p)
                            {
                                l = m + 1;
                            }
                            else
                            {
                                r = m;
                            }

                            m           = (r + l) >> 1;
                            block_col_j = bsr_col_ind[m] - idx_base;
                        }

                        // Check if a match has been found
                        if(block_col_j == block_col_p)
                        {
                            for(J q = 0; q < block_dim; q++)
                            {
                                if(block_dim * block_col_p + q < col)
                                {
                                    T vp = static_cast<T>(0);
                                    T vj = static_cast<T>(0);
                                    if(direction == rocsparse_direction_row)
                                    {
                                        vp = bsr_val[block_dim * block_dim * p + block_dim * k + q];
                                        vj = bsr_val[block_dim * block_dim * m + block_dim * row
                                                     + q];
                                    }
                                    else
                                    {
                                        vp = bsr_val[block_dim * block_dim * p + block_dim * q + k];
                                        vj = bsr_val[block_dim * block_dim * m + block_dim * q
                                                     + row];
                                    }

                                    // If a match has been found, do linear combination
                                    local_sum = rocsparse::fma(vj, rocsparse::conj(vp), local_sum);
                                }
                            }
                        }
                    }

                    val     = (val - local_sum) / diag_val;
                    row_sum = rocsparse::fma(val, rocsparse::conj(val), row_sum);

                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * row + k] = val;
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * k + row] = val;
                    }
                }
            }

            // Handle diagonal block column of block row
            for(J j = 0; j < block_dim; j++)
            {
                J row_diag = block_dim * block_dim * block_row_diag + block_dim * j + j;

                // Check if 'col' row is complete
                if(j == row)
                {
                    bsr_val[row_diag]
                        = rocsparse::sqrt(rocsparse::abs(bsr_val[row_diag] - row_sum));
                }

                // Ensure previous writes to global memory are seen by all threads
                __threadfence();

                // Load diagonal entry
                T diag_val = bsr_val[row_diag];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(lid == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                    }

                    // Normally would break here but to avoid divergence set diag_val to one and continue
                    // The zero pivot has already been set so further computation does not matter
                    diag_val = static_cast<T>(1);
                }

                if(j < row)
                {
                    // Current value
                    T val = static_cast<T>(0);

                    // Corresponding value
                    if(direction == rocsparse_direction_row)
                    {
                        val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * row + j];
                    }
                    else
                    {
                        val = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + row];
                    }

                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    T vk = static_cast<T>(0);
                    T vj = static_cast<T>(0);
                    for(I k = block_row_begin; k < block_row_diag; k++)
                    {
                        for(J q = 0; q < block_dim; q++)
                        {
                            if(direction == rocsparse_direction_row)
                            {
                                vk = bsr_val[block_dim * block_dim * k + block_dim * j + q];
                                vj = bsr_val[block_dim * block_dim * k + block_dim * row + q];
                            }
                            else
                            {
                                vk = bsr_val[block_dim * block_dim * k + block_dim * q + j];
                                vj = bsr_val[block_dim * block_dim * k + block_dim * q + row];
                            }

                            // If a match has been found, do linear combination
                            local_sum = rocsparse::fma(vj, rocsparse::conj(vk), local_sum);
                        }
                    }

                    for(J q = 0; q < j; q++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            vk = bsr_val[block_dim * block_dim * block_row_diag + block_dim * j
                                         + q];
                            vj = bsr_val[block_dim * block_dim * block_row_diag + block_dim * row
                                         + q];
                        }
                        else
                        {
                            vk = bsr_val[block_dim * block_dim * block_row_diag + block_dim * q
                                         + j];
                            vj = bsr_val[block_dim * block_dim * block_row_diag + block_dim * q
                                         + row];
                        }

                        // If a match has been found, do linear combination
                        local_sum = rocsparse::fma(vj, rocsparse::conj(vk), local_sum);
                    }

                    val     = (val - local_sum) / diag_val;
                    row_sum = rocsparse::fma(val, rocsparse::conj(val), row_sum);

                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * block_row_diag + block_dim * row + j] = val;
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * block_row_diag + block_dim * j + row] = val;
                    }
                }

                __threadfence();
            }
        }

        if(lid == WFSIZE - 1)
        {
            // Last lane writes "we are done" flag for current block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_kernel_general(rocsparse_direction dir,
                               J                   mb,
                               const I* __restrict__ bsr_row_ptr,
                               const J* __restrict__ bsr_col_ind,
                               T* __restrict__ bsr_val,
                               int64_t bsr_val_stride,
                               const I* __restrict__ bsr_diag_ind,
                               J bsr_dim,
                               int32_t* __restrict__ done_array,
                               int64_t done_array_stride,
                               const J* __restrict__ map,
                               J* __restrict__ zero_pivot,
                               int64_t              zero_pivot_stride,
                               rocsparse_index_base idx_base)
    {
        const auto batch_index = hipBlockIdx_y;
        rocsparse::bsric0_device_general<SLEEP, BLOCKSIZE, WFSIZE>(
            dir,
            mb,
            bsr_dim,
            bsr_row_ptr,
            bsr_col_ind,
            bsr_val + batch_index * bsr_val_stride,
            bsr_diag_ind,
            done_array + batch_index * done_array_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            idx_base);
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T, typename I, typename J>
    rocsparse_status bsric0_kernel_general_launch(rocsparse_handle      handle,
                                                  rocsparse_bsric0_info bsric0_info,
                                                  rocsparse_spmat_descr A,
                                                  size_t                buffer_size,
                                                  void*                 buffer)
    {
        auto trm_info = bsric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        auto          numeric_exact     = bsric0_info->get_singularity_numeric_exact();

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsric0_kernel_general<SLEEP, BLOCKSIZE, WFSIZE>),
            dim3(A->rows, A->batch_count),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            A->block_dir,
            static_cast<J>(A->rows),
            reinterpret_cast<const I*>(A->const_row_data),
            reinterpret_cast<const J*>(A->const_col_data),
            reinterpret_cast<T*>(A->val_data),
            A->batch_stride,
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            static_cast<J>(A->block_dim),
            done_array,
            done_array_stride,
            reinterpret_cast<const J*>(trm_info->get_row_map()),
            reinterpret_cast<J*>(numeric_exact->get_position()),
            numeric_exact->get_stride(),
            A->descr->base);

        return rocsparse_status_success;
    }

    template <bool SLEEP, uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T, typename I>
    static rocsparse::bsric0_kernel_launch_t transform_j_type(const rocsparse_indextype value)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::
                bsric0_kernel_general_launch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::
                bsric0_kernel_general_launch<SLEEP, BLOCKSIZE, WF_SIZE, T, I, int64_t>;
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
    static rocsparse::bsric0_kernel_launch_t transform_i_type(const rocsparse_indextype value,
                                                              P... p)
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
    static rocsparse::bsric0_kernel_launch_t transform_t_type(const rocsparse_datatype value,
                                                              P... p)
    {
        switch(value)
        {
        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, float>(p...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, rocsparse_float_complex>(
                p...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, double>(p...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<SLEEP, BLOCKSIZE, WF_SIZE, rocsparse_double_complex>(
                p...);
        }

        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_bf16_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_i32_r:
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

rocsparse::bsric0_kernel_launch_t rocsparse::find_bsric0_kernel_general_launch(
    rocsparse_handle handle, rocsparse_bsric0_info bsric0_info, rocsparse_const_spmat_descr A)
{
    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
    const bool sleep = (gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2);
    if(sleep)
    {
        return rocsparse::transform_t_type<true, 64, 64>(A->data_type, A->row_type, A->col_type);
    }
    else
    {
        if(handle->wavefront_size == 32)
        {
            return rocsparse::transform_t_type<false, 32, 32>(
                A->data_type, A->row_type, A->col_type);
        }
        else if(handle->wavefront_size == 64)
        {
            return rocsparse::transform_t_type<false, 64, 64>(
                A->data_type, A->row_type, A->col_type);
        }
        else
        {
            return nullptr;
        }
    }
}
