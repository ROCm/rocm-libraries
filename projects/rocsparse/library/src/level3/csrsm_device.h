/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"
#include "rocsparse_scalar.hpp"

namespace rocsparse
{
    // The csrsm solve flattens a two dimensional iteration space onto grid.x: one
    // thread block per (RHS panel, row) pair, where a panel is the group of
    // BLOCKSIZE right hand sides a single block carries. The launch grid and the
    // kernel's grid-stride bound both derive from this count, so it is computed in
    // 64 bit: the historical ((nrhs - 1) / blockdim + 1) * m expression evaluated
    // both operands in 32 bit and silently wrapped.
    __device__ __host__ __forceinline__ int64_t csrsm_num_blocks(int64_t m,
                                                                 int64_t nrhs,
                                                                 int64_t blockdim)
    {
        return (m > 0) ? ((nrhs - 1) / blockdim + 1) * m : 0;
    }

    // grid.x for the csrsm solve launch.
    //
    // The clamp is a one line substitution for
    //     rocsparse::get_grid_size(num_blocks, rocsparse::max_grid_size_x)
    // once PR #11512 lands; it is spelled out here so this fix does not have to
    // touch rocsparse_common.hpp.
    //
    // The clamped extent is then rounded DOWN to a whole number of RHS panels,
    // which is a correctness requirement rather than a tidiness one. The block
    // owning (panel, row) spins on done_array until the blocks owning the rows it
    // depends on have published their results; those rows are always in the same
    // panel and always earlier in the map order, i.e. at a strictly lower
    // flattened index. Keeping grid.x a multiple of m makes the kernel's stride a
    // multiple of m too, which pins every block to a fixed row and keeps each
    // dependency on a lower numbered block of the same sweep -- the dependency
    // shape the unclamped launch had. With a stride that is not a multiple of m a
    // resident block can end up waiting on a block that has not been dispatched
    // yet, and the solve hangs instead of returning.
    //
    // If m alone exceeds max_grid_x the launch fails loudly with
    // hipErrorInvalidConfiguration instead of silently computing garbage. Such a
    // matrix needs a row pointer array of more than 17 GB, so it is out of reach
    // of any current device.
    __host__ __forceinline__ int64_t csrsm_solve_grid_size(int64_t m,
                                                           int64_t nrhs,
                                                           int64_t blockdim,
                                                           int64_t max_grid_x)
    {
        if(m <= 0)
        {
            return 0;
        }

        const int64_t num_blocks = rocsparse::csrsm_num_blocks(m, nrhs, blockdim);
        const int64_t clamped    = rocsparse::min(num_blocks, max_grid_x);

        return rocsparse::max(clamped / m, static_cast<int64_t>(1)) * m;
    }

    // Body of ONE flattened block. block_id is that block's position in the
    // flattened (RHS panel, row) space and is a parameter rather than a read of
    // hipBlockIdx_x, so the grid-stride loop lives in the __global__ wrapper below
    // (the AISPARSE-666/677 idiom). Both coordinates are recovered from the same
    // induction variable: taking one from block_id and the other from
    // hipBlockIdx_x would pair the wrong row with the wrong panel on every sweep
    // after the first.
    template <uint32_t BLOCKSIZE, bool SLEEP, typename I, typename J, typename T>
    ROCSPARSE_DEVICE_ILF void csrsm_block_device(int64_t             block_id,
                                                 rocsparse_operation transB,
                                                 J                   m,
                                                 J                   nrhs,
                                                 T                   alpha,
                                                 const I* __restrict__ csr_row_ptr,
                                                 const J* __restrict__ csr_col_ind,
                                                 const T* __restrict__ csr_val,
                                                 T*      B,
                                                 int64_t ldb,
                                                 int* __restrict__ done_array,
                                                 const J* __restrict__ map,
                                                 J*                   zero_pivot,
                                                 rocsparse_index_base idx_base,
                                                 rocsparse_fill_mode  fill_mode,
                                                 rocsparse_diag_type  diag_type)
    {
        static_assert(BLOCKSIZE > 0 && (BLOCKSIZE & (BLOCKSIZE - 1)) == 0,
                      "BLOCKSIZE must be a power of two.");

        // Index into the row map
        const J idx = static_cast<J>(block_id % m);

        // RHS panel this block carries
        const int64_t panel = block_id / m;

        // Shared memory to hold columns and values
        __shared__ J scsr_col_ind[BLOCKSIZE];
        __shared__ T scsr_val[BLOCKSIZE];

        // Get the row this warp will operate on
        const J row = map[idx];

        // Current row entry point and exit point
        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Column index into B
        const J col_B = static_cast<J>(panel * BLOCKSIZE + hipThreadIdx_x);

        // Index into B (i,j)
        const int64_t idx_B = row * ldb + col_B;

        // Index into done array. 64 bit: done_array holds one flag per flattened
        // block, so the offset of the last panel exceeds 32 bit long before the
        // array itself becomes unallocatable.
        const int64_t id = panel * m;

        // Initialize local sum with alpha and X
        T local_sum = static_cast<T>(0);
        if(transB == rocsparse_operation_conjugate_transpose)
        {
            local_sum = (col_B < nrhs) ? alpha * rocsparse::conj(B[idx_B]) : static_cast<T>(0);
        }
        else
        {
            local_sum = (col_B < nrhs) ? alpha * B[idx_B] : static_cast<T>(0);
        }

        // Initialize diagonal entry
        T diagonal = static_cast<T>(1);

        for(I j = row_begin; j < row_end; ++j)
        {
            // Project j onto [0, BLOCKSIZE-1]
            const J k = (j - row_begin) & (BLOCKSIZE - 1);

            // Preload column indices and values into shared memory
            // This happens only once for each chunk of BLOCKSIZE elements
            if(k == 0)
            {
                __syncthreads();

                scsr_col_ind[hipThreadIdx_x] = (hipThreadIdx_x < row_end - j)
                                                   ? csr_col_ind[hipThreadIdx_x + j] - idx_base
                                                   : -1;
                scsr_val[hipThreadIdx_x]
                    = (hipThreadIdx_x < row_end - j) ? csr_val[hipThreadIdx_x + j] : -1;

                // Wait for preload to finish
                __syncthreads();
            }

            // Current column this lane operates on
            const J local_col = scsr_col_ind[k];

            // Local value this lane operates with
            T local_val = scsr_val[k];

            // Check for numerical zero
            if(local_val == static_cast<T>(0) && local_col == row
               && diag_type == rocsparse_diag_type_non_unit)
            {
                // Numerical zero pivot found, avoid division by 0
                // and store index for later use.
                if(hipThreadIdx_x == 0)
                {
                    rocsparse::atomic_min(zero_pivot, row + idx_base);
                }

                local_val = static_cast<T>(1);
            }

            // Differentiate upper and lower triangular mode.
            // For lower fill mode, once we pass the diagonal we must stop iterating
            // over the row, so we flag it and break out of the for loop after the switch.
            bool stop_row = false;
            switch(fill_mode)
            {
            case rocsparse_fill_mode_upper:
            {
                // Processing upper triangular

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diagonal = static_cast<T>(1) / local_val;
                    }

                    // Skip diagonal entry
                    continue;
                }
                break;
            }
            case rocsparse_fill_mode_lower:
            {
                // Processing lower triangular

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    stop_row = true;
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diagonal = static_cast<T>(1) / local_val;
                    }

                    // Skip diagonal entry
                    stop_row = true;
                    break;
                }
                break;
            }
            }

            if(stop_row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            if(hipThreadIdx_x == 0)
            {
                rocsparse::spin_loop<SLEEP>(&done_array[local_col + id], __HIP_MEMORY_SCOPE_AGENT);
            }

            // Wait for spin looping thread to finish as the whole block depends on this row
            __syncthreads();

            // Make sure updated B is visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Index into X
            const int64_t idx_X = local_col * ldb + col_B;

            // Local sum computation for each lane
            local_sum = (col_B < nrhs) ? rocsparse::fma(-local_val, B[idx_X], local_sum)
                                       : static_cast<T>(0);
        }

        // If we have non unit diagonal, take the diagonal into account
        // For unit diagonal, this would be multiplication with one
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            local_sum = local_sum * diagonal;
        }

        // Store result in B
        if(col_B < nrhs)
        {
            B[idx_B] = local_sum;
        }

        // Make sure B is written to global memory before setting row is done flag
        __threadfence();

        // Wait for all threads to finish the threadfence before we mark the row "done"
        __syncthreads();

        if(hipThreadIdx_x == 0)
        {
            // Write the "row is done" flag
            __hip_atomic_store(
                &done_array[row + id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BLOCKSIZE, bool SLEEP, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsm(rocsparse_operation transB,
               J                   m,
               J                   nrhs,
               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
               const I* __restrict__ csr_row_ptr,
               const J* __restrict__ csr_col_ind,
               const T* __restrict__ csr_val,
               T*      B,
               int64_t ldb,
               int* __restrict__ done_array,
               const J* __restrict__ map,
               J*                   zero_pivot,
               rocsparse_index_base idx_base,
               rocsparse_fill_mode  fill_mode,
               rocsparse_diag_type  diag_type,
               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        if(m <= 0)
        {
            return;
        }

        const int64_t num_blocks = rocsparse::csrsm_num_blocks(m, nrhs, BLOCKSIZE);

        // Grid-stride over the flattened (RHS panel, row) space so a grid.x clamped
        // by csrsm_solve_grid_size still covers every pair. The stride is a whole
        // number of RHS panels, never the raw grid extent: that pins this block to
        // row map[block_id % m] for its whole life and leaves every done_array
        // dependency on a lower numbered block of the same sweep, which is what
        // keeps the spin loops from deadlocking when the grid is clamped.
        //
        // Block uniform: the start, the bound and the stride read only
        // hipBlockIdx_x, hipGridDim_x, m, nrhs and BLOCKSIZE, so every thread of
        // the block runs the same number of iterations and the __syncthreads()
        // inside csrsm_block_device stay convergent. The two early returns below
        // are uniform for the same reason: the whole block takes them or none of
        // it does, so neither can strand a subset of threads at a barrier.
        const int64_t panels_per_sweep = static_cast<int64_t>(hipGridDim_x) / m;

        // Fewer than m blocks cannot be strided safely: a block would have to
        // change rows between sweeps, and the done_array flag it then waits on
        // belongs to a block that has not been dispatched. csrsm_solve_grid_size
        // never returns less than m (the launch fails with
        // hipErrorInvalidConfiguration instead), so this is a contract violation;
        // bail out rather than hang.
        if(panels_per_sweep == 0)
        {
            return;
        }

        const int64_t stride = panels_per_sweep * m;

        // Blocks past the last whole panel own no (RHS panel, row) pair. The
        // stride is rounded down to whole panels, so letting them into the loop
        // would make them repeat pairs a lower numbered block already owns, and
        // csrsm_block_device is not idempotent: it reads its own B element as the
        // right hand side and overwrites it, so a second visit corrupts the
        // result. csrsm_solve_grid_size always returns a multiple of m, so this
        // only guards direct launches (clients/unittests).
        if(hipBlockIdx_x >= stride)
        {
            return;
        }

        for(int64_t block_id = hipBlockIdx_x; block_id < num_blocks; block_id += stride)
        {
            rocsparse::csrsm_block_device<BLOCKSIZE, SLEEP>(block_id,
                                                            transB,
                                                            m,
                                                            nrhs,
                                                            alpha,
                                                            csr_row_ptr,
                                                            csr_col_ind,
                                                            csr_val,
                                                            B,
                                                            ldb,
                                                            done_array,
                                                            map,
                                                            zero_pivot,
                                                            idx_base,
                                                            fill_mode,
                                                            diag_type);
        }
    }
}
