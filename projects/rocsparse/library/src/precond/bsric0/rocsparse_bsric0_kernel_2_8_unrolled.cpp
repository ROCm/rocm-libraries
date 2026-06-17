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

#include "rocsparse_bsric0_kernel_2_8_unrolled.hpp"
#include "rocsparse_bsric0_info.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    template <uint32_t BSRDIM, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsric0_device_2_8_unrolled(rocsparse_direction direction,
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
        static constexpr uint32_t MAX_NNZB = 32;

        const auto   tidx = hipThreadIdx_x;
        const auto   tidy = hipThreadIdx_y;
        const auto   tid  = BSRDIM * tidy + tidx;
        __shared__ J columns[MAX_NNZB];
        __shared__ I index[MAX_NNZB];
        __shared__ J local_index[MAX_NNZB];
        __shared__ T row_sum[BSRDIM][BSRDIM + 1];
        __shared__ T temp[BSRDIM][BSRDIM + 1];
        __shared__ T values[BSRDIM][BSRDIM + 1];
        __shared__ T local_values[BSRDIM][BSRDIM + 1];

        // Current block row this wavefront is working on
        J block_row = block_map[hipBlockIdx_x];

        // Block diagonal entry point of the current block row
        I block_row_diag = bsr_diag_ind[block_row];

        // If one thread in the warp breaks here, then all threads in
        // the warp break so no divergence
        if(block_row_diag == -1)
        {
            if(tidx == 0 && tidy == 0)
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

        // Write current block row column indices to shared memory
        for(I j = block_row_begin + tid; j < block_row_diag + 1; j += BSRDIM * BSRDIM)
        {
            columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
        }

        // Block row sum accumulator
        row_sum[tidy][tidx] = static_cast<T>(0);

        __threadfence_block();

        // Loop over non-diagonal block columns of current block row
        for(I j = block_row_begin; j < block_row_diag; j++)
        {
            // Block column index currently being processes
            J block_col = bsr_col_ind[j] - idx_base;

            // Beginning of the row that corresponds to block_col
            I local_block_begin = bsr_row_ptr[block_col] - idx_base;

            // Diagonal entry point of row block_col
            I local_block_diag = bsr_diag_ind[block_col];

            // Structural zero pivot, do not process this row
            if(local_block_diag == -1)
            {
                // If one thread in the warp breaks here, then all threads in
                // the warp break so no divergence
                break;
            }

            if(direction == rocsparse_direction_row)
            {
                values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidy + tidx];
            }
            else
            {
                values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidx + tidy];
            }

            I count = 0;
            I l     = local_block_begin;
            I k     = 0;
            J col_k = columns[k];

            while(l <= local_block_diag && col_k <= block_col)
            {
                J col_l = bsr_col_ind[l] - idx_base;
                col_k   = columns[k];

                if(col_l < col_k)
                {
                    l++;
                }
                else if(col_l > col_k)
                {
                    k++;
                }
                else
                {
                    index[count]       = BSRDIM * BSRDIM * (k + block_row_begin);
                    local_index[count] = BSRDIM * BSRDIM * l;

                    k++;
                    l++;

                    count++;
                }
            }

            __threadfence_block();

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(
                &block_done[block_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            if(direction == rocsparse_direction_row)
            {
                local_values[tidy][tidx]
                    = bsr_val[BSRDIM * BSRDIM * local_block_diag + BSRDIM * tidy + tidx];
            }
            else
            {
                local_values[tidy][tidx]
                    = bsr_val[BSRDIM * BSRDIM * local_block_diag + BSRDIM * tidx + tidy];
            }

            __threadfence_block();

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(I l = 0; l < count - 1; l++)
            {
                J idx2 = local_index[l];
                I idx  = index[l];

                if(direction == rocsparse_direction_row)
                {
                    if(BSRDIM >= 1)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 0];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 0];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 2)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 1];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 1];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 3)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 2];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 2];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 4)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 3];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 3];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 5)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 4];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 4];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 6)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 5];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 5];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 7)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 6];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 6];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 8)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * tidx + 7];
                        T v2      = bsr_val[idx + BSRDIM * tidy + 7];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                }
                else
                {
                    if(BSRDIM >= 1)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 0 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 0 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 2)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 1 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 1 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 3)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 2 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 2 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 4)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 3 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 3 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 5)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 4 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 4 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 6)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 5 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 5 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 7)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 6 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 6 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                    if(BSRDIM >= 8)
                    {
                        T v1      = bsr_val[idx2 + BSRDIM * 7 + tidx];
                        T v2      = bsr_val[idx + BSRDIM * 7 + tidy];
                        local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                    }
                }
            }

            temp[tidy][tidx] = local_sum;

            __threadfence_block();

            for(J k = 0; k < BSRDIM; k++)
            {
                // Current value
                T val = values[tidy][k];

                // Load diagonal entry
                T diag_val = local_values[k][k];

                // Row has numerical zero pivot
                if(diag_val == static_cast<T>(0))
                {
                    if(tidx == 0 && tidy == 0)
                    {
                        // We are looking for the first zero pivot
                        rocsparse::atomic_min(zero_pivot, block_col + idx_base);
                    }

                    diag_val = static_cast<T>(1);
                }

                T local_sum = temp[tidy][k];

                for(J p = 0; p < k; p++)
                {
                    T v1      = local_values[k][p];
                    T v2      = values[tidy][p];
                    local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                }

                // Compute the Cholesky factor and writes it to shared memory
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                row_sum[tidy][tidx]
                    = rocsparse::fma(val, rocsparse::conj(values[tidx][k]), row_sum[tidy][tidx]);

                __threadfence_block();
            }

            if(direction == rocsparse_direction_row)
            {
                bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidy + tidx] = values[tidy][tidx];
            }
            else
            {
                bsr_val[BSRDIM * BSRDIM * j + BSRDIM * tidx + tidy] = values[tidy][tidx];
            }

            __threadfence();
        }

        // Load current diagonal block into shared memory
        if(direction == rocsparse_direction_row)
        {
            values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidy + tidx];
        }
        else
        {
            values[tidy][tidx] = bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidx + tidy];
        }

        __threadfence_block();

        // Handle diagonal block column of block row.
        for(J k = 0; k < BSRDIM; k++)
        {
            if(k == tidy)
            {
                values[k][k] = rocsparse::sqrt(rocsparse::abs(values[k][k] - row_sum[k][k]));
            }

            __threadfence_block();

            // Load diagonal entry
            T diag_val = values[k][k];

            // Row has numerical zero pivot
            if(diag_val == static_cast<T>(0))
            {
                if(tidx == 0 && tidy == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, block_row + idx_base);
                }

                // Normally would break here but to avoid divergence set diag_val to one and continue
                // The zero pivot has already been set so further computation does not matter
                diag_val = static_cast<T>(1);
            }

            if(k < tidy)
            {
                // Load value
                T val = values[tidy][k];

                // Local row sum
                T local_sum = row_sum[tidy][k];

                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                row_sum[tidy][tidx]
                    = rocsparse::fma(val, rocsparse::conj(values[tidx][k]), row_sum[tidy][tidx]);
            }

            __threadfence_block();
        }

        if(direction == rocsparse_direction_row)
        {
            bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidy + tidx] = values[tidy][tidx];
        }
        else
        {
            bsr_val[BSRDIM * BSRDIM * block_row_diag + BSRDIM * tidx + tidy] = values[tidy][tidx];
        }

        if(tidx == 0 && tidy == 0)
        {
            // Last lane in wavefront writes "we are done" flag for its block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BSRDIM, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BSRDIM* BSRDIM)
    void bsric0_kernel_2_8_unrolled(rocsparse_direction dir,
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
        rocsparse::bsric0_device_2_8_unrolled<BSRDIM>(dir,
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

    template <uint32_t BSRDIM, typename T, typename I, typename J>
    rocsparse_status bsric0_kernel_2_8_unrolled_launch(rocsparse_handle      handle,
                                                       rocsparse_bsric0_info bsric0_info,
                                                       rocsparse_spmat_descr A,
                                                       size_t                buffer_size,
                                                       void*                 buffer)
    {

        auto trm_info = bsric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        auto          numeric_exact     = bsric0_info->get_singularity_numeric_exact();
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsric0_kernel_2_8_unrolled<BSRDIM>),
                                           dim3(A->rows, A->batch_count),
                                           dim3(BSRDIM, BSRDIM),
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

    template <uint32_t BSR_DIM, typename T, typename I>
    static rocsparse::bsric0_kernel_launch_t transform_j_type(const rocsparse_indextype value)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::bsric0_kernel_2_8_unrolled_launch<BSR_DIM, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::bsric0_kernel_2_8_unrolled_launch<BSR_DIM, T, I, int64_t>;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t BSR_DIM, typename T, typename... P>
    static rocsparse::bsric0_kernel_launch_t transform_i_type(const rocsparse_indextype value,
                                                              P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::transform_j_type<BSR_DIM, T, int32_t>(p...);
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::transform_j_type<BSR_DIM, T, int64_t>(p...);
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t BSR_DIM, typename... P>
    static rocsparse::bsric0_kernel_launch_t transform_t_type(const rocsparse_datatype value,
                                                              P... p)
    {
        switch(value)
        {
        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<BSR_DIM, float>(p...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<BSR_DIM, rocsparse_float_complex>(p...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<BSR_DIM, double>(p...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<BSR_DIM, rocsparse_double_complex>(p...);
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

rocsparse::bsric0_kernel_launch_t rocsparse::find_bsric0_kernel_2_8_unrolled_launch(
    rocsparse_handle handle, rocsparse_bsric0_info bsric0_info, rocsparse_const_spmat_descr A)
{
    switch(A->block_dim)
    {
    case 1:
    {
        return rocsparse::transform_t_type<1>(A->data_type, A->row_type, A->col_type);
    }

    case 2:
    {
        return rocsparse::transform_t_type<2>(A->data_type, A->row_type, A->col_type);
    }

    case 3:
    {
        return rocsparse::transform_t_type<3>(A->data_type, A->row_type, A->col_type);
    }

    case 4:
    {
        return rocsparse::transform_t_type<4>(A->data_type, A->row_type, A->col_type);
    }

    case 5:
    {
        return rocsparse::transform_t_type<5>(A->data_type, A->row_type, A->col_type);
    }

    case 6:
    {
        return rocsparse::transform_t_type<6>(A->data_type, A->row_type, A->col_type);
    }

    case 7:
    {
        return rocsparse::transform_t_type<7>(A->data_type, A->row_type, A->col_type);
    }

    case 8:
    {
        return rocsparse::transform_t_type<8>(A->data_type, A->row_type, A->col_type);
    }

    default:
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                              "block dimension > 8 not supported");
    }
    }
}
