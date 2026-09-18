/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    // The bsrsm solve flattens a two dimensional iteration space onto grid.x: one
    // thread block per (RHS panel, block row) pair, where a panel is the group of
    // NCOLS right hand sides a single block carries. The launch grid and the
    // kernels' grid-stride bound both derive from this count, so it is computed in
    // 64 bit: the historical ((nrhs - 1) / NCOL + 1) * mb expression evaluated both
    // operands as signed 32 bit and overflowed (undefined behaviour, not a defined
    // wrap) at for instance mb = 50000 with nrhs = 700000.
    __device__ __host__ __forceinline__ int64_t bsrsm_num_blocks(int64_t mb,
                                                                 int64_t nrhs,
                                                                 int64_t ncols)
    {
        return (mb > 0) ? ((nrhs - 1) / ncols + 1) * mb : 0;
    }

    // grid.x for the bsrsm solve launch.
    //
    // The clamp is a one line substitution for
    //     rocsparse::get_grid_size(num_blocks, rocsparse::max_grid_size_x)
    // once PR #11512 (AISPARSE-696) lands; it is spelled out here so this fix does
    // not have to touch rocsparse_common.hpp, which is a live conflict zone.
    //
    // The clamped extent is then rounded DOWN to a whole number of RHS panels,
    // which is a correctness requirement rather than a tidiness one, and the part
    // the ticket does not mention: maxGridSize[0] is almost never a multiple of mb,
    // so clamping alone manufactures a ragged grid. The block owning
    // (panel, block row) spins on done_array until the block rows it depends on
    // have published their results; those rows are always in the same panel and
    // always earlier in the map order, i.e. at a strictly lower flattened index.
    // Keeping grid.x a multiple of mb makes the kernels' stride a multiple of mb
    // too, which pins every block to a fixed row-map slot and keeps each dependency
    // on a lower numbered block of the same sweep -- the dependency shape the
    // unclamped launch had. With a stride that is not a multiple of mb a resident
    // block can end up waiting on a block that has not been dispatched yet, and the
    // solve hangs instead of returning (AISPARSE-669 found this in csrsm).
    //
    // If mb alone exceeds max_grid_x the launch fails loudly with
    // hipErrorInvalidConfiguration instead of silently computing garbage. Such a
    // matrix needs a block row pointer array of more than 8 GB, so it is out of
    // reach of any current device.
    __host__ __forceinline__ int64_t bsrsm_solve_grid_size(int64_t mb,
                                                           int64_t nrhs,
                                                           int64_t ncols,
                                                           int64_t max_grid_x)
    {
        if(mb <= 0)
        {
            return 0;
        }

        const int64_t num_blocks = rocsparse::bsrsm_num_blocks(mb, nrhs, ncols);
        const int64_t clamped    = rocsparse::min(num_blocks, max_grid_x);

        return rocsparse::max(clamped / mb, static_cast<int64_t>(1)) * mb;
    }

    // Body of ONE flattened block of the upper solve. block_id is that block's
    // position in the flattened (RHS panel, block row) space and is a parameter
    // rather than a read of blockIdx.x, so the grid-stride loop lives in the
    // __global__ wrapper below (the AISPARSE-666 idiom the reviewer required).
    // ALL THREE coordinates -- row-map slot, done_array panel offset and X column
    // -- are recovered from the same induction variable: taking one from block_id
    // and another from blockIdx.x would pair the wrong block row with the wrong
    // panel on every sweep after the first.
    template <uint32_t BLOCKSIZE, uint32_t NCOLS, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void bsrsm_upper_large_block_device(int64_t              block_id,
                                                             rocsparse_int        mb,
                                                             rocsparse_int        nrhs,
                                                             const rocsparse_int* bsr_row_ptr,
                                                             const rocsparse_int* bsr_col_ind,
                                                             const T*             bsr_val,
                                                             rocsparse_int        block_dim,
                                                             T*                   X,
                                                             rocsparse_int        ldx,
                                                             int*                 done_array,
                                                             const rocsparse_int* map,
                                                             rocsparse_int*       zero_pivot,
                                                             rocsparse_index_base idx_base,
                                                             rocsparse_diag_type  diag_type,
                                                             rocsparse_direction  dir)
    {
        static constexpr uint32_t WFSIZE = BLOCKSIZE / NCOLS;

        const int lid = threadIdx.x & (WFSIZE - 1);

        // Index into the row map
        const rocsparse_int idx = static_cast<rocsparse_int>(block_id % mb);

        // Get the BSR row this thread block will operate on
        const rocsparse_int row = map[idx];

        // Get the id of the rhs, this thread block will operate on. The done_array
        // offset stays 64 bit: mb * (number of RHS panels) passes INT32_MAX long
        // before the array stops being allocatable.
        const int64_t id = (block_id / mb) * mb;

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Column index (rhs) into X
        const rocsparse_int col_X
            = static_cast<rocsparse_int>((block_id / mb) * NCOLS + threadIdx.x / WFSIZE);

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Loop over current row
        rocsparse_int j;
        for(j = row_end - 1; j >= row_begin; --j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Processing upper triangular

            // Ignore all diagonal entries and below
            if(local_col <= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            if(threadIdx.x == 0)
            {
                rocsparse::spin_loop<SLEEP>(&done_array[local_col + id], __HIP_MEMORY_SCOPE_AGENT);
            }

            // Make sure updated X is visible globally
            __threadfence();

            // Wait for spin looping thread to finish as the whole block depends on this row
            __syncthreads();

            // Local sum computation

            // Do not run out of bounds
            if(col_X < nrhs)
            {
                // Loop over rows of the BSR block
                for(int bi = lid; bi < block_dim; bi += WFSIZE)
                {
                    // Local sum accumulator
                    T sum = static_cast<T>(0);

                    // Loop over columns of the BSR block
                    for(int bj = 0; bj < block_dim; ++bj)
                    {
                        sum = rocsparse::fma(bsr_val[BSR_IND(j, bi, bj, dir)],
                                             X[(block_dim * local_col + bj) * ldx + col_X],
                                             sum);
                    }

                    // Write local sum to X
                    X[(block_dim * row + bi) * ldx + col_X] -= sum;
                }
            }
        }

        bool pivot = false;

        // Process diagonal
        if(row < mb && row == local_col && col_X < nrhs)
        {
            // Loop over rows of the BSR block
            for(int bi = block_dim - 1; bi >= 0; --bi)
            {
                // Load diagonal matrix entry
                const T diag = (diag_type == rocsparse_diag_type_non_unit)
                                   ? bsr_val[BSR_IND(j, bi, bi, dir)]
                                   : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = X[(block_dim * row + bi) * ldx + col_X];

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th BSR row by diagonal entry
                    X[(block_dim * row + bi) * ldx + col_X] = val /= diag;
                }

                // Update remaining non-diagonal entries
                for(int bj = lid; bj < bi; bj += WFSIZE)
                {
                    X[(block_dim * row + bj) * ldx + col_X]
                        -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
                }
            }
        }

        // Make sure X is written to global memory before setting row is done flag
        __threadfence();

        // Wait for all threads to finish the threadfence before we mark the row "done"
        __syncthreads();

        if(row < mb && threadIdx.x == 0)
        {
            // Write "row is done" flag
            __hip_atomic_store(
                &done_array[row + id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NCOLS, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsm_upper_large_kernel(rocsparse_int        mb,
                                  rocsparse_int        nrhs,
                                  const rocsparse_int* bsr_row_ptr,
                                  const rocsparse_int* bsr_col_ind,
                                  const T*             bsr_val,
                                  rocsparse_int        block_dim,
                                  T*                   X,
                                  rocsparse_int        ldx,
                                  int*                 done_array,
                                  const rocsparse_int* map,
                                  rocsparse_int*       zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir)
    {
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % NCOLS == 0, "BLOCKSIZE must be a multiple of NCOLS.");
        static_assert((BLOCKSIZE / NCOLS) > 0
                          && ((BLOCKSIZE / NCOLS) & ((BLOCKSIZE / NCOLS) - 1)) == 0,
                      "BLOCKSIZE / NCOLS must be a power of two.");

        if(mb <= 0)
        {
            return;
        }

        const int64_t num_blocks = rocsparse::bsrsm_num_blocks(mb, nrhs, NCOLS);

        // Grid-stride over the flattened (RHS panel, block row) space so a grid.x
        // clamped by bsrsm_solve_grid_size still covers every pair. The stride is a
        // whole number of RHS panels, never the raw grid extent: that pins this
        // block to row map[block_id % mb] for its whole life and leaves every
        // done_array dependency on a lower numbered block of the same sweep, which
        // is what keeps the spin loops from deadlocking when the grid is clamped.
        //
        // Block uniform: the start, the bound and the stride read only blockIdx.x,
        // gridDim.x, mb, nrhs and NCOLS, so every thread of the block runs the same
        // number of iterations and the two __syncthreads() inside
        // bsrsm_upper_large_block_device stay convergent. The three early returns
        // below are uniform for the same reason -- the whole block takes them or
        // none of it does -- so none can strand a subset of threads at a barrier.
        const int64_t panels_per_sweep = static_cast<int64_t>(gridDim.x) / mb;

        // Fewer than mb blocks cannot be strided safely: a block would have to
        // change row-map slots between sweeps, and the done_array flag it then
        // waits on belongs to a block that has not been dispatched.
        // bsrsm_solve_grid_size never returns less than mb (the launch fails with
        // hipErrorInvalidConfiguration instead), so this is a contract violation;
        // bail out rather than hang.
        if(panels_per_sweep == 0)
        {
            return;
        }

        const int64_t stride = panels_per_sweep * mb;

        // Blocks past the last whole panel own no (RHS panel, block row) pair. The
        // stride is rounded down to whole panels, so letting them into the loop
        // would make them repeat pairs a lower numbered block already owns, and the
        // block body is not idempotent: it reads its own X elements as the right
        // hand side and overwrites them, so a second visit corrupts the result.
        // bsrsm_solve_grid_size always returns a multiple of mb, so this only
        // guards direct launches (clients/unittests).
        if(static_cast<int64_t>(blockIdx.x) >= stride)
        {
            return;
        }

        for(int64_t block_id = blockIdx.x; block_id < num_blocks; block_id += stride)
        {
            rocsparse::bsrsm_upper_large_block_device<BLOCKSIZE, NCOLS, SLEEP>(block_id,
                                                                               mb,
                                                                               nrhs,
                                                                               bsr_row_ptr,
                                                                               bsr_col_ind,
                                                                               bsr_val,
                                                                               block_dim,
                                                                               X,
                                                                               ldx,
                                                                               done_array,
                                                                               map,
                                                                               zero_pivot,
                                                                               idx_base,
                                                                               diag_type,
                                                                               dir);
        }
    }

    // Body of ONE flattened block of the lower solve. See
    // bsrsm_upper_large_block_device for why block_id is a parameter.
    template <uint32_t BLOCKSIZE, uint32_t NCOLS, bool SLEEP, typename T>
    ROCSPARSE_DEVICE_ILF void bsrsm_lower_large_block_device(int64_t              block_id,
                                                             rocsparse_int        mb,
                                                             rocsparse_int        nrhs,
                                                             const rocsparse_int* bsr_row_ptr,
                                                             const rocsparse_int* bsr_col_ind,
                                                             const T*             bsr_val,
                                                             rocsparse_int        block_dim,
                                                             T*                   X,
                                                             rocsparse_int        ldx,
                                                             int*                 done_array,
                                                             const rocsparse_int* map,
                                                             rocsparse_int*       zero_pivot,
                                                             rocsparse_index_base idx_base,
                                                             rocsparse_diag_type  diag_type,
                                                             rocsparse_direction  dir)
    {
        static constexpr uint32_t WFSIZE = BLOCKSIZE / NCOLS;

        const int lid = threadIdx.x & (WFSIZE - 1);

        // Index into the row map
        const rocsparse_int idx = static_cast<rocsparse_int>(block_id % mb);

        // Get the BSR row this thread block will operate on
        const rocsparse_int row = map[idx];

        // Get the id of the rhs, this thread block will operate on. The done_array
        // offset stays 64 bit: mb * (number of RHS panels) passes INT32_MAX long
        // before the array stops being allocatable.
        const int64_t id = (block_id / mb) * mb;

        // Current row entry and exit point
        const rocsparse_int row_begin = bsr_row_ptr[row] - idx_base;
        const rocsparse_int row_end   = bsr_row_ptr[row + 1] - idx_base;

        // Column index into X
        const rocsparse_int col_X
            = static_cast<rocsparse_int>((block_id / mb) * NCOLS + threadIdx.x / WFSIZE);

        // Initialize local_col with mb
        rocsparse_int local_col = mb;

        // Loop over current row
        rocsparse_int j;
        for(j = row_begin; j < row_end; ++j)
        {
            // Current column index
            local_col = bsr_col_ind[j] - idx_base;

            // Processing lower triangular

            // Ignore all diagonal entries and above
            if(local_col >= row)
            {
                break;
            }

            // Spin loop until dependency has been resolved
            if(threadIdx.x == 0)
            {
                rocsparse::spin_loop<SLEEP>(&done_array[local_col + id], __HIP_MEMORY_SCOPE_AGENT);
            }

            // Make sure updated X is visible globally
            __threadfence();

            // Wait for spin looping thread to finish as the whole block depends on this row
            __syncthreads();

            // Local sum computation

            // Do not run out of bounds
            if(col_X < nrhs)
            {
                // Loop over rows of the BSR block
                for(int bi = lid; bi < block_dim; bi += WFSIZE)
                {
                    // Local sum accumulator
                    T sum = static_cast<T>(0);

                    // Loop over columns of the BSR block
                    for(int bj = 0; bj < block_dim; ++bj)
                    {
                        sum = rocsparse::fma(bsr_val[BSR_IND(j, bi, bj, dir)],
                                             X[(block_dim * local_col + bj) * ldx + col_X],
                                             sum);
                    }

                    // Write local sum to X
                    X[(block_dim * row + bi) * ldx + col_X] -= sum;
                }
            }
        }

        bool pivot = false;

        // Process diagonal
        if(row < mb && row == local_col && col_X < nrhs)
        {
            // Loop over rows of the BSR block
            for(int bi = 0; bi < block_dim; ++bi)
            {
                // Load diagonal matrix entry
                T diag = (diag_type == rocsparse_diag_type_non_unit)
                             ? bsr_val[BSR_IND(j, bi, bi, dir)]
                             : static_cast<T>(1);

                // Load result of bi-th BSR row
                T val = X[(block_dim * row + bi) * ldx + col_X];

                // Check for numerical pivot
                if(diag == static_cast<T>(0))
                {
                    pivot = true;
                }
                else
                {
                    // Divide result of bi-th BSR row by diagonal entry
                    X[(block_dim * row + bi) * ldx + col_X] = val /= diag;
                }

                // Update remaining non-diagonal entries
                for(int bj = bi + lid + 1; bj < block_dim; bj += WFSIZE)
                {
                    X[(block_dim * row + bj) * ldx + col_X]
                        -= val * bsr_val[BSR_IND(j, bj, bi, dir)];
                }
            }
        }

        // Make sure X is written to global memory before setting row is done flag
        __threadfence();

        // Wait for all threads to finish the threadfence before we mark the row "done"
        __syncthreads();

        if(row < mb && threadIdx.x == 0)
        {
            // Write "row is done" flag
            __hip_atomic_store(
                &done_array[row + id], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

            if(pivot == true)
            {
                rocsparse::atomic_min(zero_pivot, row + idx_base);
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NCOLS, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsm_lower_large_kernel(rocsparse_int        mb,
                                  rocsparse_int        nrhs,
                                  const rocsparse_int* bsr_row_ptr,
                                  const rocsparse_int* bsr_col_ind,
                                  const T*             bsr_val,
                                  rocsparse_int        block_dim,
                                  T*                   X,
                                  rocsparse_int        ldx,
                                  int*                 done_array,
                                  const rocsparse_int* map,
                                  rocsparse_int*       zero_pivot,
                                  rocsparse_index_base idx_base,
                                  rocsparse_diag_type  diag_type,
                                  rocsparse_direction  dir)
    {
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % NCOLS == 0, "BLOCKSIZE must be a multiple of NCOLS.");
        static_assert((BLOCKSIZE / NCOLS) > 0
                          && ((BLOCKSIZE / NCOLS) & ((BLOCKSIZE / NCOLS) - 1)) == 0,
                      "BLOCKSIZE / NCOLS must be a power of two.");

        if(mb <= 0)
        {
            return;
        }

        const int64_t num_blocks = rocsparse::bsrsm_num_blocks(mb, nrhs, NCOLS);

        // See bsrsm_upper_large_kernel for the panel-aligned stride, the
        // block-uniformity argument and why each early return is safe.
        const int64_t panels_per_sweep = static_cast<int64_t>(gridDim.x) / mb;

        if(panels_per_sweep == 0)
        {
            return;
        }

        const int64_t stride = panels_per_sweep * mb;

        if(static_cast<int64_t>(blockIdx.x) >= stride)
        {
            return;
        }

        for(int64_t block_id = blockIdx.x; block_id < num_blocks; block_id += stride)
        {
            rocsparse::bsrsm_lower_large_block_device<BLOCKSIZE, NCOLS, SLEEP>(block_id,
                                                                               mb,
                                                                               nrhs,
                                                                               bsr_row_ptr,
                                                                               bsr_col_ind,
                                                                               bsr_val,
                                                                               block_dim,
                                                                               X,
                                                                               ldx,
                                                                               done_array,
                                                                               map,
                                                                               zero_pivot,
                                                                               idx_base,
                                                                               diag_type,
                                                                               dir);
        }
    }
}
