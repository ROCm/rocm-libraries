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

#include "rocsparse_bsrilu0_kernel_general.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsrilu0_device_general(rocsparse_direction dir,
                                                     J                   mb,
                                                     const I* __restrict__ bsr_row_ptr,
                                                     const J* __restrict__ bsr_col_ind,
                                                     T* __restrict__ bsr_val,
                                                     const I* __restrict__ bsr_diag_ind,
                                                     J block_dim,
                                                     int32_t* __restrict__ done_array,
                                                     const J* __restrict__ map,
                                                     J* __restrict__ zero_pivot,
                                                     rocsparse_index_base idx_base,
                                                     int32_t              boost,
                                                     double               boost_tol,
                                                     T                    boost_val)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        const auto lid = hipThreadIdx_x & (WFSIZE - 1);
        const auto wid = hipThreadIdx_x / WFSIZE;

        // Index
        J idx = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= mb)
        {
            return;
        }

        // Current row this wavefront is working on
        J row = map[idx];

        // Diagonal entry point of the current row
        I row_diag = bsr_diag_ind[row];

        // Row entry point
        I row_begin = bsr_row_ptr[row] - idx_base;
        I row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(I j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                J bsr_col = bsr_col_ind[j] - idx_base;

                // Process all lower matrix BSR blocks

                // Obtain corresponding row entry and exit point that corresponds with the
                // current BSR column. Actually, we skip all lower matrix column indices,
                // therefore starting with the diagonal entry.
                I diag_j    = bsr_diag_ind[bsr_col];
                I row_end_j = bsr_row_ptr[bsr_col + 1] - idx_base;

                // Check for structural pivot
                if(diag_j == -1)
                {
                    pivot = true;
                    break;
                }

                // Spin loop until dependency has been resolved
                (void)rocsparse::spin_loop<SLEEP>(&done_array[bsr_col], __HIP_MEMORY_SCOPE_AGENT);

                // Make sure dependencies are visible in global memory
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                // Loop through all rows within the BSR block
                for(J bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                    // Loop through all rows
                    for(J bk = lid; bk < block_dim; bk += WFSIZE)
                    {
                        T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                        // This has already been checked for zero by previous computations
                        val /= diag;

                        // Update
                        bsr_val[BSR_IND(j, bk, bi, dir)] = val;

                        // Do linear combination

                        // Loop through all columns above the diagonal of the BSR block
                        for(J bj = bi + 1; bj < block_dim; ++bj)
                        {
                            bsr_val[BSR_IND(j, bk, bj, dir)]
                                = rocsparse::fma(-val,
                                                 bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                                 bsr_val[BSR_IND(j, bk, bj, dir)]);
                        }
                    }
                }

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(I k = diag_j + 1; k < row_end_j; ++k)
                {
                    J bsr_col_k = bsr_col_ind[k] - idx_base;

                    // Search for matching column index in current row
                    I q         = row_begin + lid;
                    J bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int32_t match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        J m = rocsparse::shfl(q, match - 1);

                        for(J bi = lid; bi < block_dim; bi += WFSIZE)
                        {
                            for(J bj = 0; bj < block_dim; ++bj)
                            {
                                T sum = static_cast<T>(0);

                                for(J bk = 0; bk < block_dim; ++bk)
                                {
                                    sum = rocsparse::fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                                         bsr_val[BSR_IND(k, bk, bj, dir)],
                                                         sum);
                                }

                                bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                            }
                        }
                    }
                }
            }

            // Process diagonal
            if(bsr_col_ind[row_diag] - idx_base == row)
            {
                for(J bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = bsr_val[BSR_IND(row_diag, bi, bi, dir)];

                    // Numeric boost
                    if(boost)
                    {
                        diag = (boost_tol >= rocsparse::abs(diag))
                                   ? rocsparse::assign_ilu0_boost_value(diag, boost_val)
                                   : diag;

                        if(lid == 0)
                        {
                            bsr_val[BSR_IND(row_diag, bi, bi, dir)] = diag;
                        }
                    }
                    else
                    {
                        // Check for numeric pivot
                        if(diag == static_cast<T>(0))
                        {
                            pivot = true;
                            continue;
                        }
                    }

                    for(J bk = bi + 1 + lid; bk < block_dim; bk += WFSIZE)
                    {
                        // Multiplication factor
                        T val = bsr_val[BSR_IND(row_diag, bk, bi, dir)];
                        val /= diag;

                        // Update
                        bsr_val[BSR_IND(row_diag, bk, bi, dir)] = val;

                        // Do linear combination
                        for(J bj = bi + 1; bj < block_dim; ++bj)
                        {
                            bsr_val[BSR_IND(row_diag, bk, bj, dir)]
                                = rocsparse::fma(-val,
                                                 bsr_val[BSR_IND(row_diag, bi, bj, dir)],
                                                 bsr_val[BSR_IND(row_diag, bk, bj, dir)]);
                        }
                    }
                }
            }

            // Process upper diagonal BSR blocks
            for(I j = row_diag + 1; j < row_end; ++j)
            {
                for(J bi = 0; bi < block_dim; ++bi)
                {
                    for(J bk = lid; bk < block_dim; bk += WFSIZE)
                    {
                        for(J bj = bi + 1; bj < block_dim; ++bj)
                        {
                            bsr_val[BSR_IND(j, bj, bk, dir)]
                                = rocsparse::fma(-bsr_val[BSR_IND(row_diag, bj, bi, dir)],
                                                 bsr_val[BSR_IND(j, bi, bk, dir)],
                                                 bsr_val[BSR_IND(j, bj, bk, dir)]);
                        }
                    }
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        if(lid == 0)
        {
            // First lane writes "we are done" flag
            __hip_atomic_store(&done_array[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot)
            {
                // Atomically set minimum zero pivot, if found
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t SLEEP,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrilu0_kernel_general(rocsparse_direction dir,
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
                                rocsparse_index_base idx_base,
                                int32_t              enable_boost,
                                size_t               size_boost_tol,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                                bool is_tol_host_mode,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                                bool is_val_host_mode)
    {
        const auto batch_index = hipBlockIdx_y;
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(
            enable_boost && (size_boost_tol == sizeof(float)), is_tol_host_mode, boost_tol_32);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(
            enable_boost && (size_boost_tol == sizeof(double)), is_tol_host_mode, boost_tol_64);
        ROCSPARSE_SCALAR_HOST_DEVICE_GET_IF(enable_boost, is_val_host_mode, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::bsrilu0_device_general<BLOCKSIZE, WFSIZE, SLEEP>(
            dir,
            mb,
            bsr_row_ptr,
            bsr_col_ind,
            bsr_val + batch_index * bsr_val_stride,
            bsr_diag_ind,
            bsr_dim,
            done_array + batch_index * done_array_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            idx_base,
            enable_boost,
            boost_tol,
            boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    static rocsparse_status bsrilu0_kernel_general_launch(rocsparse_handle          handle,
                                                          rocsparse_bsrilu0_info    bsrilu0_info,
                                                          rocsparse_spmat_descr     A,
                                                          rocsparse::numeric_boost* boost,
                                                          size_t                    buffer_size,
                                                          void*                     buffer)
    {
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
        auto trm_info = bsrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        int32_t* done_array = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(buffer) + 256);
        const int64_t done_array_stride = A->rows;
        auto          numeric_exact     = bsrilu0_info->get_singularity_numeric_exact();
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrilu0_kernel_general<BLOCKSIZE, WFSIZE, SLEEP>),
            dim3((WFSIZE * A->rows - 1) / BLOCKSIZE + 1, A->batch_count),
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

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename I>
    static rocsparse::bsrilu0_kernel_launch_t transform_j_type(const rocsparse_indextype value)
    {

        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::
                bsrilu0_kernel_general_launch<BLOCKSIZE, WF_SIZE, SLEEP, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::
                bsrilu0_kernel_general_launch<BLOCKSIZE, WF_SIZE, SLEEP, T, I, int64_t>;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }

        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename... P>
    static rocsparse::bsrilu0_kernel_launch_t transform_i_type(const rocsparse_indextype value,
                                                               P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, SLEEP, T, int32_t>(
                std::forward<P>(p)...);
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, SLEEP, T, int64_t>(
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

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
    static rocsparse::bsrilu0_kernel_launch_t transform_t_type(const rocsparse_datatype value,
                                                               P... p)
    {
        switch(value)
        {

        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, float>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_float_complex>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, double>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_double_complex>(
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

rocsparse::bsrilu0_kernel_launch_t rocsparse::find_bsrilu0_kernel_general_launch(
    rocsparse_handle handle, rocsparse_bsrilu0_info bsrilu0_info, rocsparse_const_spmat_descr A)
{
    const bool sleep
        = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908 && //
           handle->asic_rev < 2);

    if(sleep)
    {
        return rocsparse::transform_t_type<128, 64, true>(A->data_type, A->row_type, A->col_type);
    }
    else
    {
        switch(handle->wavefront_size)
        {
        case 32:
        {
            return rocsparse::transform_t_type<128, 32, false>(
                A->data_type, A->row_type, A->col_type);
        }
        case 64:
        {
            return rocsparse::transform_t_type<128, 64, false>(
                A->data_type, A->row_type, A->col_type);
        }
        default:
        {
            THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }
        }
    }
}
