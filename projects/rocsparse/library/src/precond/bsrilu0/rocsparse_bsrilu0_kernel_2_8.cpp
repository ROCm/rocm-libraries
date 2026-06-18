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

#include "rocsparse_bsrilu0_kernel_2_8.hpp"
#include "rocsparse_bsrilu0_info.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t BSRDIM,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void bsrilu0_device_2_8(rocsparse_direction dir,
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
                                                 int                  boost,
                                                 double               boost_tol,
                                                 T                    boost_val)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");
        // Current row this wavefront is working on
        J row = map[blockIdx.x];

        // Diagonal entry point of the current row
        I row_diag = bsr_diag_ind[row];

        // Row entry point
        I row_begin = bsr_row_ptr[row] - idx_base;
        I row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Zero pivot tracker
        bool pivot = false;

        // Shared memory to cache BSR values
        __shared__ T sdata1[BSRDIM][BSRDIM + 1];
        __shared__ T sdata2[BSRDIM][BSRDIM + 1];

        // Check for structural pivot
        if(row_diag != -1)
        {
            // Process lower diagonal
            for(I j = row_begin; j < row_diag; ++j)
            {
                // Column index of current BSR block
                J bsr_col = bsr_col_ind[j] - idx_base;

                // Load row j into shared memory
                sdata2[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

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
                while(!__hip_atomic_load(
                    &done_array[bsr_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                    ;

                // Make sure dependencies are visible in global memory
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

                // Load updated BSR block into shared memory
                sdata1[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(diag_j, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                // Make sure all writes to shared memory are visible
                __threadfence_block();

                // Loop through all rows within the BSR block
                for(J bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal entry of the BSR block
                    T diag = sdata1[bi][bi];
                    T val  = sdata2[bi][threadIdx.x];

                    // This has already been checked for zero by previous computations
                    val /= diag;

                    // Make sure val has been read before updating
                    __threadfence_block();

                    // Update
                    if(threadIdx.y == 0)
                    {
                        sdata2[bi][threadIdx.x] = val;
                    }

                    // Do linear combination
                    J bj = bi + 1 + threadIdx.y;
                    if(bj < block_dim)
                    {
                        sdata2[bj][threadIdx.x]
                            = rocsparse::fma(-val, sdata1[bj][bi], sdata2[bj][threadIdx.x]);
                    }

                    __threadfence_block();
                }

                // Write row j back to global memory
                if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                {
                    bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                        = sdata2[threadIdx.y][threadIdx.x];
                }

                // Loop over upper offset pointer and do linear combination for nnz entry
                for(I k = diag_j + 1; k < row_end_j; ++k)
                {
                    J bsr_col_k = bsr_col_ind[k] - idx_base;

                    // Search for matching column index in current row
                    I q         = row_begin + threadIdx.x + threadIdx.y * blockDim.x;
                    J bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;

                    // Check if match has been found by any thread in the wavefront
                    while(bsr_col_j < bsr_col_k)
                    {
                        q += WFSIZE;
                        bsr_col_j = (q < row_end) ? bsr_col_ind[q] - idx_base : mb + 1;
                    }

                    // Check if match has been found by any thread in the wavefront
                    int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

                    // If match has been found, process it
                    if(match)
                    {
                        // Tell all other threads about the matching index
                        J m = rocsparse::shfl(q, match - 1);

                        // Load BSR block from row k into shared memory
                        sdata1[threadIdx.y][threadIdx.x]
                            = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                                  ? bsr_val[BSR_IND(k, threadIdx.x, threadIdx.y, dir)]
                                  : static_cast<T>(0);

                        // Make sure all writes to shared memory are visible
                        __threadfence_block();

                        T sum = static_cast<T>(0);

                        for(J bk = 0; bk < block_dim; ++bk)
                        {
                            sum = rocsparse::fma(
                                sdata2[bk][threadIdx.x], sdata1[threadIdx.y][bk], sum);
                        }

                        // Write back to global row m
                        if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                        {
                            // Do not pre-cache row m as we read/write only once
                            bsr_val[BSR_IND(m, threadIdx.x, threadIdx.y, dir)] -= sum;
                        }
                    }

                    __threadfence_block();
                }
            }

            // Process diagonal
            if(bsr_col_ind[row_diag] - idx_base == row)
            {
                // Load diagonal BSR block into shared memory
                sdata1[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(row_diag, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                __threadfence_block();

                for(J bi = 0; bi < block_dim; ++bi)
                {
                    // Load diagonal matrix entry
                    T diag = sdata1[bi][bi];

                    // Numeric boost
                    if(boost)
                    {

                        diag = (boost_tol >= rocsparse::abs(diag))
                                   ? rocsparse::assign_ilu0_boost_value(diag, boost_val)
                                   : diag;

                        __threadfence_block();

                        if(threadIdx.x == 0 && threadIdx.y == 0)
                        {
                            sdata1[bi][bi] = diag;
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

                    J bk = bi + 1 + threadIdx.x;
                    if(bk < block_dim)
                    {
                        // Multiplication factor
                        T val = sdata1[bi][bk];
                        val /= diag;

                        // Make sure val has been read before updating
                        __threadfence_block();

                        // Update
                        if(threadIdx.y == 0)
                        {
                            sdata1[bi][bk] = val;
                        }

                        // Do linear combination
                        J bj = bi + 1 + threadIdx.y;
                        if(bj < block_dim)
                        {
                            sdata1[bj][bk] = rocsparse::fma(-val, sdata1[bj][bi], sdata1[bj][bk]);
                        }
                    }
                }

                __threadfence_block();

                // Write diagonal BSR block back to global memory
                if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                {
                    bsr_val[BSR_IND(row_diag, threadIdx.x, threadIdx.y, dir)]
                        = sdata1[threadIdx.y][threadIdx.x];
                }
            }

            // Process upper diagonal BSR blocks
            for(I j = row_diag + 1; j < row_end; ++j)
            {
                __threadfence_block();

                // Load row j into shared memory
                sdata2[threadIdx.y][threadIdx.x]
                    = (threadIdx.x < block_dim && threadIdx.y < block_dim)
                          ? bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                          : static_cast<T>(0);

                __threadfence_block();

                for(J bi = 0; bi < block_dim; ++bi)
                {
                    J bj = bi + 1 + threadIdx.y;
                    if(bj < block_dim)
                    {
                        sdata2[threadIdx.x][bj] = rocsparse::fma(
                            -sdata1[bi][bj], sdata2[threadIdx.x][bi], sdata2[threadIdx.x][bj]);
                    }
                }

                __threadfence_block();

                // Write row j back to global memory
                if(threadIdx.x < block_dim && threadIdx.y < block_dim)
                {
                    bsr_val[BSR_IND(j, threadIdx.x, threadIdx.y, dir)]
                        = sdata2[threadIdx.y][threadIdx.x];
                }
            }
        }
        else
        {
            // Structural pivot found
            pivot = true;
        }

        if(threadIdx.x == 0 && threadIdx.y == 0)
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
              uint32_t BSRDIM,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrilu0_kernel_2_8(rocsparse_direction dir,
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
                            int                  enable_boost,
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
        const double boost_tol = (size_boost_tol == sizeof(double)) //
                                     ? boost_tol_64 //
                                     : boost_tol_32;
        rocsparse::bsrilu0_device_2_8<BLOCKSIZE, WFSIZE, BSRDIM>(
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

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t BSRDIM,
              typename T,
              typename I,
              typename J>
    rocsparse_status bsrilu0_kernel_2_8_launch(rocsparse_handle          handle,
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

        auto numeric_exact = bsrilu0_info->get_singularity_numeric_exact();

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrilu0_kernel_2_8<BLOCKSIZE, WFSIZE, BSRDIM>),
            dim3(A->rows, A->batch_count),
            dim3(8, 8),
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

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t BBDIM, typename T, typename I>
    static rocsparse::bsrilu0_kernel_launch_t transform_j_type(const rocsparse_indextype value)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::bsrilu0_kernel_2_8_launch<BLOCKSIZE, WF_SIZE, BBDIM, T, I, int32_t>;
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::bsrilu0_kernel_2_8_launch<BLOCKSIZE, WF_SIZE, BBDIM, T, I, int64_t>;
        }
        case deprecated_rocsparse_indextype_u16:
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                  "rocsparse_indextype_u16 not supported");
        }
        }

        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t BBDIM, typename T, typename... P>
    static rocsparse::bsrilu0_kernel_launch_t transform_i_type(const rocsparse_indextype value,
                                                               P... p)
    {
        switch(value)
        {
        case rocsparse_indextype_i32:
        {
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, BBDIM, T, int32_t>(
                std::forward<P>(p)...);
        }
        case rocsparse_indextype_i64:
        {
            return rocsparse::transform_j_type<BLOCKSIZE, WF_SIZE, BBDIM, T, int64_t>(
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

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t BBDIM, typename... P>
    static rocsparse::bsrilu0_kernel_launch_t transform_t_type(const rocsparse_datatype value,
                                                               P... p)
    {

        switch(value)
        {

        case rocsparse_datatype_f32_r:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, BBDIM, float>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f32_c:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, BBDIM, rocsparse_float_complex>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_r:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, BBDIM, double>(
                std::forward<P>(p)...);
        }

        case rocsparse_datatype_f64_c:
        {
            return rocsparse::transform_i_type<BLOCKSIZE, WF_SIZE, BBDIM, rocsparse_double_complex>(
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

rocsparse::bsrilu0_kernel_launch_t rocsparse::find_bsrilu0_kernel_2_8_launch(
    rocsparse_handle handle, rocsparse_bsrilu0_info bsrilu0_info, rocsparse_const_spmat_descr A)
{
    return rocsparse::transform_t_type<64, 64, 8>(A->data_type, A->row_type, A->col_type);
}
