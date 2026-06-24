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

#include "rocsparse_bsric0_kernel_9_16.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

#define BLOCKSIZE 64

    template <uint32_t MAX_NNZB, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsric0_device_9_16(rocsparse_direction direction,
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
        static constexpr uint32_t BSRDIM = 16;
        static constexpr uint32_t DIMX   = BLOCKSIZE / BSRDIM;
        static constexpr uint32_t DIMY   = BSRDIM;

        J tidx = hipThreadIdx_x;
        J tidy = hipThreadIdx_y;
        J tid  = DIMX * tidy + tidx;

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
        for(I j = block_row_begin + tid; j < block_row_diag + 1; j += DIMX * DIMY)
        {
            columns[j - block_row_begin] = bsr_col_ind[j] - idx_base;
        }

        // Block row sum accumulator
        for(J i = tidx; i < BSRDIM; i += DIMX)
        {
            row_sum[tidy][i] = static_cast<T>(0);
        }

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

            for(J q = tidx; q < block_dim; q += DIMX)
            {
                if(direction == rocsparse_direction_row)
                {
                    values[tidy][q]
                        = (tidy < block_dim)
                              ? bsr_val[block_dim * block_dim * j + block_dim * tidy + q]
                              : static_cast<T>(0);
                }
                else
                {
                    values[tidy][q]
                        = (tidy < block_dim)
                              ? bsr_val[block_dim * block_dim * j + block_dim * q + tidy]
                              : static_cast<T>(0);
                }

                temp[tidy][q] = static_cast<T>(0);
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
                    index[count]       = block_dim * block_dim * (k + block_row_begin);
                    local_index[count] = block_dim * block_dim * l;

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

            for(J q = tidx; q < block_dim; q += DIMX)
            {
                if(direction == rocsparse_direction_row)
                {
                    local_values[tidy][q] = (tidy < block_dim)
                                                ? bsr_val[block_dim * block_dim * local_block_diag
                                                          + block_dim * tidy + q]
                                                : static_cast<T>(0);
                }
                else
                {
                    local_values[tidy][q] = (tidy < block_dim)
                                                ? bsr_val[block_dim * block_dim * local_block_diag
                                                          + block_dim * q + tidy]
                                                : static_cast<T>(0);
                }
            }

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(I l = 0; l < count - 1; l++)
            {
                J idx2 = local_index[l];
                I idx  = index[l];

                for(J q = tidx; q < block_dim; q += DIMX)
                {
                    // Local row sum
                    T local_sum = static_cast<T>(0);

                    for(J p = 0; p < block_dim; p++)
                    {
                        if(direction == rocsparse_direction_row)
                        {
                            T v1      = bsr_val[idx2 + block_dim * q + p];
                            T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * tidy + p]
                                                           : static_cast<T>(0);
                            local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                        }
                        else
                        {
                            T v1      = bsr_val[idx2 + block_dim * p + q];
                            T v2      = (tidy < block_dim) ? bsr_val[idx + block_dim * p + tidy]
                                                           : static_cast<T>(0);
                            local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                        }
                    }

                    temp[tidy][q] += local_sum;
                }
            }

            __threadfence_block();

            for(J k = 0; k < block_dim; k++)
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

                // Local row sum
                T local_sum = temp[tidy][k];

                for(I p = 0; p < k; p++)
                {
                    T v1      = local_values[k][p];
                    T v2      = values[tidy][p];
                    local_sum = rocsparse::fma(v2, rocsparse::conj(v1), local_sum);
                }

                // Compute the Cholesky factor and writes it to global memory
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                for(J q = tidx; q < block_dim; q += DIMX)
                {
                    row_sum[tidy][q]
                        = rocsparse::fma(val, rocsparse::conj(values[q][k]), row_sum[tidy][q]);
                }

                __threadfence_block();
            }

            // Write values back to global memory
            for(J q = tidx; q < block_dim; q += DIMX)
            {
                if(tidy < block_dim)
                {
                    if(direction == rocsparse_direction_row)
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * tidy + q] = values[tidy][q];
                    }
                    else
                    {
                        bsr_val[block_dim * block_dim * j + block_dim * q + tidy] = values[tidy][q];
                    }
                }
            }

            __threadfence();
        }

        // Load current diagonal block into shared memory
        for(J q = tidx; q < block_dim; q += DIMX)
        {
            if(direction == rocsparse_direction_row)
            {
                values[tidy][q]
                    = (tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + q]
                          : static_cast<T>(0);
            }
            else
            {
                values[tidy][q]
                    = (tidy < block_dim)
                          ? bsr_val[block_dim * block_dim * block_row_diag + block_dim * q + tidy]
                          : static_cast<T>(0);
            }
        }

        __threadfence_block();

        // Handle diagonal block column of block row.
        for(J k = 0; k < block_dim; k++)
        {
            if(k == tidy)
            {
                values[k][k] = rocsparse::sqrt(rocsparse::abs(values[k][k] - row_sum[k][k]));
            }

            __threadfence_block();

            // Load value
            T val = values[tidy][k];

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

            // Local row sum
            T local_sum = row_sum[tidy][k];

            if(k < tidy)
            {
                val             = (val - local_sum) / diag_val;
                values[tidy][k] = val;

                __threadfence_block();

                for(J q = tidx; q < block_dim; q += DIMX)
                {
                    row_sum[tidy][q]
                        = rocsparse::fma(val, rocsparse::conj(values[q][k]), row_sum[tidy][q]);
                }
            }

            __threadfence_block();
        }

        // Write values back to global memory
        for(J q = tidx; q < block_dim; q += DIMX)
        {
            if(tidy < block_dim)
            {
                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * tidy + q]
                        = values[tidy][q];
                }
                else
                {
                    bsr_val[block_dim * block_dim * block_row_diag + block_dim * q + tidy]
                        = values[tidy][q];
                }
            }
        }

        if(tidx == 0 && tidy == 0)
        {
            // Last lane in wavefront writes "we are done" flag for its block row
            __hip_atomic_store(
                &block_done[block_row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t MAX_NNZB, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsric0_kernel_9_16(rocsparse_direction dir,
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
        rocsparse::bsric0_device_9_16<MAX_NNZB>(dir,
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

#undef BLOCKSIZE

    template <uint32_t MAX_NNZB, typename T, typename I, typename J>
    rocsparse_status bsric0_kernel_9_16_launch(rocsparse_handle      handle,
                                               rocsparse_bsric0_info bsric0_info,
                                               rocsparse_spmat_descr A,
                                               size_t                buffer_size,
                                               void*                 buffer)
    {

        auto trm_info = bsric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        auto          numeric_exact     = bsric0_info->get_singularity_numeric_exact();

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsric0_kernel_9_16<MAX_NNZB>),
                                           dim3(A->rows, A->batch_count),
                                           dim3(4, 16),
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

    template <uint32_t MAX_NNZB, typename T, typename I, typename... P>
    static rocsparse::bsric0_kernel_launch_t transform_j_type(const rocsparse_indextype value,
                                                              P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::bsric0_kernel_9_16_launch<MAX_NNZB, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::bsric0_kernel_9_16_launch<MAX_NNZB, T, I, int64_t>;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t MAX_NNZB, typename T, typename... P>
    static rocsparse::bsric0_kernel_launch_t transform_i_type(const rocsparse_indextype value,
                                                              P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::transform_j_type<MAX_NNZB, T, int32_t>(p...);
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::transform_j_type<MAX_NNZB, T, int64_t>(p...);
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t MAX_NNZB, typename... P>
    static rocsparse::bsric0_kernel_launch_t transform_t_type(const rocsparse_datatype value,
                                                              P... p)
    {

        switch(value)
        {
        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<MAX_NNZB, float>(p...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<MAX_NNZB, rocsparse_float_complex>(p...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<MAX_NNZB, double>(p...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<MAX_NNZB, rocsparse_double_complex>(p...);
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

rocsparse::bsric0_kernel_launch_t rocsparse::find_bsric0_kernel_9_16_launch(
    rocsparse_handle handle, rocsparse_bsric0_info bsric0_info, rocsparse_const_spmat_descr A)
{
    auto          trm_info = bsric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    const int64_t max_nnzb = trm_info->get_max_nnz();
    if(max_nnzb <= 32)
    {
        return rocsparse::transform_t_type<32>(A->data_type, A->row_type, A->col_type);
    }
    else if(max_nnzb <= 64)
    {
        return rocsparse::transform_t_type<64>(A->data_type, A->row_type, A->col_type);
    }
    else if(max_nnzb <= 128)
    {
        return rocsparse::transform_t_type<128>(A->data_type, A->row_type, A->col_type);
    }
    else
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }
}
