/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{
    //
    // Shared grid-stride helpers for the row-parallel csritsv analysis/solve
    // kernels. Every one of those kernels maps one thread to one row and used to
    // compute the global row id as `BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x`
    // in 32-bit unsigned arithmetic, which wraps at 2^32 for a 64-bit index type
    // `J`. These helpers compute the id (and the stride) in `J` *before* the
    // multiply, so no overflow occurs, and expose a grid stride so the kernel can
    // cover an `m` larger than the (clamped) grid via a loop.
    //
    // Usage inside a kernel:
    //   for(J row = rocsparse::csritsv_grid_stride_begin<BLOCKSIZE, J>();
    //       row < m;
    //       row += rocsparse::csritsv_grid_stride_step<BLOCKSIZE, J>())
    //   { ... }
    //
    template <uint32_t BLOCKSIZE, typename J>
    __device__ __forceinline__ J csritsv_grid_stride_begin()
    {
        return static_cast<J>(BLOCKSIZE) * static_cast<J>(hipBlockIdx_x)
               + static_cast<J>(hipThreadIdx_x);
    }

    template <uint32_t BLOCKSIZE, typename J>
    __device__ __forceinline__ J csritsv_grid_stride_step()
    {
        return static_cast<J>(BLOCKSIZE) * static_cast<J>(hipGridDim_x);
    }

    //
    // Number of blocks to launch for a grid-stride kernel covering `m` rows,
    // clamped to the device's maximum grid size in the x dimension so the launch
    // is always valid even when `m` is very large.
    //
    template <uint32_t BLOCKSIZE, typename J>
    static uint32_t csritsv_grid_stride_blocks(rocsparse_handle handle, J m)
    {
        const int64_t nblocks   = (static_cast<int64_t>(m) - 1) / BLOCKSIZE + 1;
        const int64_t max_grid  = static_cast<int64_t>(handle->properties.maxGridSize[0]);
        return static_cast<uint32_t>((nblocks < max_grid) ? nblocks : max_grid);
    }

    template <typename I, typename J, typename T>
    rocsparse_status csritsv_buffer_size_template(rocsparse_handle          handle,
                                                  rocsparse_operation       trans,
                                                  J                         m,
                                                  I                         nnz,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size);

    template <typename I, typename J, typename T>
    rocsparse_status csritsv_analysis_template(rocsparse_handle          handle,
                                               rocsparse_operation       trans,
                                               J                         m,
                                               I                         nnz,
                                               const rocsparse_mat_descr descr,
                                               const T*                  csr_val,
                                               const I*                  csr_row_ptr,
                                               const J*                  csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csritsv_solve_ex_template(rocsparse_handle          handle,
                                               rocsparse_int*            host_nmaxiter,
                                               rocsparse_int             host_nfreeiter,
                                               const floating_data_t<T>* host_tol,
                                               floating_data_t<T>*       host_history,
                                               rocsparse_operation       trans,
                                               J                         m,
                                               I                         nnz,
                                               const T*                  alpha,
                                               const rocsparse_mat_descr descr,
                                               const T*                  csr_val,
                                               const I*                  csr_row_ptr,
                                               const J*                  csr_col_ind,
                                               rocsparse_mat_info        info,
                                               const T*                  x,
                                               T*                        y,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer);
}
