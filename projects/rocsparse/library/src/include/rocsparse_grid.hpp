/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <cstdint>
#include <type_traits>

// Grid-extent clamps, one per axis. Each takes the handle and reads the limit
// from the device, so a caller never writes a bound itself. Any kernel launched
// with a clamped extent MUST grid-stride over the full count, e.g.
//
//     for(int64_t i = hipBlockIdx_x; i < count; i += hipGridDim_x)
//
// or, if it spins on done flags, walk the contiguous chunk grid_x_chunk returns.
//
// The axes are not symmetric because the limits are different quantities. On y
// and z the limit is a work-group count, and maxGridSize reports it. On x the
// limit is a work-item count, and no device property exposes it.
//
// The kernel dispatch packet stores all three grid extents as 32-bit counts of
// work-items rather than of work-groups [1][2]. On y and z that never binds:
// the per-axis work-group limit in maxGridSize[1] and [2] (65535) caps them at
// 65535 * 1024 work-items at most, far below 2^32. On x the device reports no
// usable work-group limit, so the work-item count is what binds: grid.x *
// blockDim.x must fit in 32 bits, and the largest grid.x that runs every block
// is (2^32 - 1) / blockDim.x:
//
//     blockDim.x       64      128      256      512     1024
//     max grid.x 67108863 33554431 16777215  8388607  4194303
//
// against the 2147483647 that maxGridSize[0] reports.
//
// get_grid_size_x clamps to the smaller of the derived bound and the reported
// extent: the derived bound because it is the one that binds, the reported one
// because it is the documented contract.
//
// [1] hsa_kernel_dispatch_packet_t in the HSA Runtime Programmer's Reference
//     Manual, shipped with ROCm as hsa/hsa.h, which documents grid_size_x as
//     the "X dimension of grid, in work-items".
// [2] https://llvm.org/docs/AMDGPUUsage.html, "hidden_block_count_x", which
//     distinguishes the work-group count passed in the kernarg from the
//     dispatch packet value, "which has the grid size in work-items".

namespace rocsparse
{
    // Clamp a count to an axis maximum, returning a value that fits the
    // unsigned int dim3 field.
    template <typename J>
    static inline uint32_t clamp_grid_extent(J count, int64_t max_extent)
    {
        const int64_t extent = static_cast<int64_t>(count);
        return static_cast<uint32_t>((extent > max_extent) ? max_extent : extent);
    }

    // Largest grid.x extent a dispatch runs correctly for a given block size.
    // A non-positive block size is itself an invalid launch; it gets the full
    // 32-bit limit rather than a division by zero, and never a zero extent.
    static constexpr int64_t dispatch_limit_x(int64_t block_size)
    {
        return (block_size > 0) ? static_cast<int64_t>(4294967295LL) / block_size
                                : static_cast<int64_t>(4294967295LL);
    }

    // Clamp a grid.x extent. block_size must be the blockDim.x the launch uses;
    // it is a runtime argument rather than a template parameter because several
    // callers choose it at runtime, for example the csrmv LRB path picks
    // min(1 << bin, 1024) per bin.
    template <typename J>
    static inline uint32_t get_grid_size_x(rocsparse_handle handle, J count, int64_t block_size)
    {
        const int64_t device_cap = static_cast<int64_t>(handle->properties.maxGridSize[0]);
        const int64_t arch_cap   = rocsparse::dispatch_limit_x(block_size);
        return rocsparse::clamp_grid_extent(count, (device_cap < arch_cap) ? device_cap : arch_cap);
    }

    //
    // Half-open range [first, last) of the count work items owned by block
    // block_id when the items are split into grid_size contiguous chunks. The
    // range is empty when the block has nothing to do, which happens for the
    // trailing blocks when grid_size does not divide count.
    //
    // This is the alternative to the grid-stride loop for kernels launched with
    // a clamped grid.x that spin on done flags, for example the incomplete
    // factorizations. Those kernels wait on the flags of the rows they depend
    // on, and the row map is a topological order, so an item only ever waits on
    // items earlier in that order. With contiguous chunks every item a block
    // waits on belongs either to itself, earlier in its own chunk, or to a lower
    // numbered block, which the dispatcher started first. That is the same
    // property the unclamped launch relies on, and it is what keeps the spin
    // from deadlocking when the grid is larger than the device can hold
    // resident.
    //
    // A grid-stride loop does not have that property. Block 0 on its second item
    // waits on the first item of block grid_size - 1, which cannot start until
    // some block retires, and no block retires while it is spinning.
    //
    static __device__ __host__ __forceinline__ void grid_x_chunk(
        int64_t count, uint32_t grid_size, uint32_t block_id, int64_t& first, int64_t& last)
    {
        const int64_t chunk = (count - 1) / static_cast<int64_t>(grid_size) + 1;
        const int64_t start = static_cast<int64_t>(block_id) * chunk;

        //
        // Both ends are clamped, not just the upper one. grid_size does not have
        // to divide count, so a trailing block can start past the end and must
        // come back with an empty range rather than an inverted one.
        //
        first = (start < count) ? start : count;
        last  = ((first + chunk) < count) ? (first + chunk) : count;
    }

    // Clamp a grid.y extent.
    template <typename J>
    static inline uint32_t get_grid_size_y(rocsparse_handle handle, J count)
    {
        return rocsparse::clamp_grid_extent(
            count, static_cast<int64_t>(handle->properties.maxGridSize[1]));
    }

    // Clamp a grid.z extent.
    template <typename J>
    static inline uint32_t get_grid_size_z(rocsparse_handle handle, J count)
    {
        return rocsparse::clamp_grid_extent(
            count, static_cast<int64_t>(handle->properties.maxGridSize[2]));
    }

    // Launch with the grid.x extent get_grid_size_x returns, selecting the kernel
    // variant at the call site. launch(std::true_type{}, grid) runs when the clamp
    // binds and the kernel must grid-stride over count; launch(std::false_type{},
    // grid) runs otherwise, so a kernel can keep a straight-line variant for the
    // common case where every row gets its own block. The clamp binds as soon as
    // count exceeds dispatch_limit_x(block_size), 16777215 blocks at 256 threads,
    // which 32-bit indices reach as readily as 64-bit ones.
    template <typename J, typename F>
    static __forceinline__ rocsparse_status
        dispatch_grid_stride_x(rocsparse_handle handle, J count, int64_t block_size, F&& launch)
    {
        const uint32_t grid = rocsparse::get_grid_size_x(handle, count, block_size);
        if(grid < static_cast<int64_t>(count))
        {
            return launch(std::true_type{}, grid);
        }
        return launch(std::false_type{}, grid);
    }
}
