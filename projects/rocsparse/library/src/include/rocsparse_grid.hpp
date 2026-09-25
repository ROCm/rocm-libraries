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

// Grid-extent clamps, one per axis. Each takes the handle and reads the limit
// from the device, so a caller never writes a bound itself. Any kernel launched
// with a clamped extent MUST grid-stride over the full count, e.g.
//
//     for(int64_t i = hipBlockIdx_x; i < count; i += hipGridDim_x)
//
// The axes are not symmetric because the limits are different quantities. On y
// and z the limit is a work-group count, and maxGridSize reports it. On x the
// limit is a work-item count, and no device property exposes it.
//
// The kernel dispatch packet stores each grid extent as a 32-bit count of
// work-items rather than of work-groups [1][2], so the work-group count is
// grid_size_x / workgroup_size_x and it is grid.x * blockDim.x that must fit
// in 32 bits. The largest grid.x that runs every block is
// (2^32 - 1) / blockDim.x:
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
    static constexpr int64_t dispatch_limit_x(int64_t block_size)
    {
        return static_cast<int64_t>(4294967295LL) / block_size;
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
}
