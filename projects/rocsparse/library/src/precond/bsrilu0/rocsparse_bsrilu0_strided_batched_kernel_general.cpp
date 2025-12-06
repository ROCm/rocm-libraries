/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bsrilu0_strided_batched_kernel_general.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void bsrilu0_general_device(rocsparse_direction  dir,
                                                     J                    mb,
                                                     const I*             bsr_row_ptr,
                                                     const J*             bsr_col_ind,
                                                     T*                   bsr_val,
                                                     const I*             bsr_diag_ind,
                                                     J                    block_dim,
                                                     int*                 done_array,
                                                     const J*             map,
                                                     J*                   zero_pivot,
                                                     rocsparse_index_base idx_base,
                                                     int                  boost,
                                                     double               boost_tol,
                                                     T                    boost_val)
    {
        int lid = hipThreadIdx_x & (WFSIZE - 1);
        int wid = hipThreadIdx_x / WFSIZE;

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
                    int match = __ffsll(__ballot(bsr_col_j == bsr_col_k));

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
                        diag = (boost_tol >= rocsparse::abs(diag)) ? boost_val : diag;

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
    void bsrilu0_strided_batched_kernel_general(rocsparse_direction  dir,
                                                J                    mb,
                                                const I*             bsr_row_ptr,
                                                const J*             bsr_col_ind,
                                                T*                   bsr_val,
                                                int64_t              bsr_val_stride,
                                                const I*             bsr_diag_ind,
                                                J                    bsr_dim,
                                                int32_t*             done_array,
                                                int64_t              done_array_stride,
                                                const J*             map,
                                                J*                   zero_pivot,
                                                int64_t              zero_pivot_stride,
                                                rocsparse_index_base idx_base,
                                                int                  enable_boost,
                                                size_t               size_boost_tol,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float,
                                                                                    boost_tol_32),
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double,
                                                                                    boost_tol_64),
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                                                bool is_host_mode)
    {
        const auto batch_index = hipBlockIdx_y;
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::bsrilu0_general_device<BLOCKSIZE, WFSIZE, SLEEP>(
            dir,
            mb,
            bsr_row_ptr,
            bsr_col_ind,
            bsr_val + batch_index * bsr_val_stride,
            bsr_diag_ind,
            bsr_dim,
            done_array + batch_index * done_array_stride,
            map,
            zero_pivot + batch_index * done_array_stride,
            idx_base,
            enable_boost,
            boost_tol,
            boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    static rocsparse_status
        bsrilu0_strided_batched_kernel_general_launch(rocsparse_handle     handle,
                                                      rocsparse_direction  dir,
                                                      int64_t              batch_count,
                                                      int64_t              mb,
                                                      const void*          bsr_row_ptr,
                                                      const void*          bsr_col_ind,
                                                      void*                bsr_val,
                                                      int64_t              bsr_val_stride,
                                                      const void*          bsr_diag_ind,
                                                      int64_t              bsr_dim,
                                                      int32_t*             done_array,
                                                      int64_t              done_array_stride,
                                                      const void*          row_map,
                                                      void*                zero_pivot,
                                                      int64_t              zero_pivot_stride,
                                                      rocsparse_index_base idx_base,
                                                      int32_t              boost_enable,
                                                      size_t               boost_tol_size,
                                                      const void*          boost_tol,
                                                      const void*          boost_val_)
    {
        const float*  boost_tol_32 = reinterpret_cast<const float*>(boost_tol);
        const double* boost_tol_64 = reinterpret_cast<const double*>(boost_tol);
        const T*      boost_val    = reinterpret_cast<const T*>(boost_val_);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrilu0_strided_batched_kernel_general<BLOCKSIZE, WFSIZE, SLEEP>),
            dim3((WFSIZE * mb - 1) / BLOCKSIZE + 1, batch_count),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            dir,
            static_cast<J>(mb),
            reinterpret_cast<const I*>(bsr_row_ptr),
            reinterpret_cast<const J*>(bsr_col_ind),
            reinterpret_cast<T*>(bsr_val),
            bsr_val_stride,
            reinterpret_cast<const I*>(bsr_diag_ind),
            static_cast<J>(bsr_dim),
            reinterpret_cast<int32_t*>(done_array),
            done_array_stride,
            reinterpret_cast<const J*>(row_map),
            reinterpret_cast<J*>(zero_pivot),
            zero_pivot_stride,
            idx_base,
            boost_enable,
            boost_tol_size,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
            handle->pointer_mode == rocsparse_pointer_mode_host);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     SLEEP,
              typename T,
              typename I,
              typename... P>
    static rocsparse::bsrilu0_kernel_general_launch_t
        find_kernel_general_j(const rocsparse_indextype j, P... p)
    {
        return (j == rocsparse_indextype_i32)
                   ? rocsparse::bsrilu0_strided_batched_kernel_general_launch<BLOCKSIZE,
                                                                              WF_SIZE,
                                                                              SLEEP,
                                                                              T,
                                                                              I,
                                                                              int32_t>
               : (j == rocsparse_indextype_i64)
                   ? rocsparse::bsrilu0_strided_batched_kernel_general_launch<BLOCKSIZE,
                                                                              WF_SIZE,
                                                                              SLEEP,
                                                                              T,
                                                                              I,
                                                                              int64_t>
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename... P>
    static rocsparse::bsrilu0_kernel_general_launch_t
        find_kernel_general_i(const rocsparse_indextype i, P... p)
    {
        return (i == rocsparse_indextype_i32)
                   ? rocsparse::find_kernel_general_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int32_t>(p...)
               : (i == rocsparse_indextype_i64)
                   ? rocsparse::find_kernel_general_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int64_t>(p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
    static rocsparse::bsrilu0_kernel_general_launch_t
        find_kernel_general_t(const rocsparse_datatype i, P... p)
    {
        return (i == rocsparse_datatype_f32_r)
                   ? rocsparse::find_kernel_general_i<BLOCKSIZE, WF_SIZE, SLEEP, float>(p...)
               : (i == rocsparse_datatype_f32_c) ? rocsparse::
                       find_kernel_general_i<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_float_complex>(
                           p...)
               : (i == rocsparse_datatype_f64_c) ? rocsparse::
                       find_kernel_general_i<BLOCKSIZE, WF_SIZE, SLEEP, rocsparse_double_complex>(
                           p...)
               : (i == rocsparse_datatype_f64_r)
                   ? rocsparse::find_kernel_general_i<BLOCKSIZE, WF_SIZE, SLEEP, double>(p...)
                   : nullptr;
    }
}

rocsparse::bsrilu0_kernel_general_launch_t
    rocsparse::find_bsrilu0_strided_batched_kernel_general_launch(uint32_t            blocksize,
                                                                  uint32_t            wfsize,
                                                                  bool                sleep,
                                                                  rocsparse_datatype  t_type,
                                                                  rocsparse_indextype i_type,
                                                                  rocsparse_indextype j_type)
{
    if(blocksize == 128)
    {
        return //
            (sleep) //
                ? rocsparse::find_kernel_general_t<128, 64, true>(t_type, i_type, j_type) //
                : (wfsize == 32) //
                      ? rocsparse::find_kernel_general_t<128, 32, false>(t_type, i_type, j_type) //
                      : (wfsize == 64) //
                            ? rocsparse::find_kernel_general_t<128, 64, false>(
                                t_type, i_type, j_type) //
                            : nullptr; //
    }
    else
    {
        return nullptr;
    }
}
