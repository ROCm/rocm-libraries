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

#include <stdint.h>

//
// Kept in its own header, and deliberately free of any rocsparse or HIP
// include, so that the host-only rocsparse-unit-test binary can compile it as
// plain C++ without pulling in the HIP toolchain. Both helpers are needed on
// the host (to size the launch) and on the device (to recover the work above
// the clamp), so they are annotated for both when compiled as HIP.
//
#if defined(__HIPCC__)
#define ROCSPARSE_GRID_X_ILF static __host__ __device__ __forceinline__
#else
#define ROCSPARSE_GRID_X_ILF static inline
#endif

namespace rocsparse
{
    //
    // Largest extent the hardware accepts on grid.x. Verified on gfx1201, which
    // reports maxGridSize[0] == 2147483647 (grid.y and grid.z report 65535).
    //
    // Deliberately a function rather than a namespace scope constant. PR #11512
    // (AISPARSE-696) introduces a rocsparse::max_grid_size_x constant in
    // rocsparse_common.hpp; a namespace scope constexpr has internal linkage, so
    // the same name coming from two headers is a redeclaration error in any
    // translation unit that includes both. A function body cannot collide that
    // way, and the worst case once both land is a duplicate spelling of one
    // integer rather than a build break.
    //
    ROCSPARSE_GRID_X_ILF int64_t grid_x_max_extent()
    {
        return 2147483647;
    }

    //
    // Extent to launch on grid.x for count units of work, clamped to what the
    // hardware accepts. Work above the clamp is recovered by grid_x_chunk inside
    // the kernel.
    //
    // The clamp is the whole point: A->rows arrives from the spmat descriptor as
    // an int64_t and dim3 takes a uint32_t, so without it a row count above the
    // cap is narrowed silently. Between 2^31 and 2^32 the narrowing produces a
    // value dim3 accepts but the driver rejects; at and above 2^32 it wraps, so
    // 2^32 launches an empty grid and 2^32 + 7 launches seven blocks.
    //
    ROCSPARSE_GRID_X_ILF uint32_t get_grid_size_x(int64_t count)
    {
        const int64_t maximum = rocsparse::grid_x_max_extent();
        return static_cast<uint32_t>((count > maximum) ? maximum : count);
    }

    //
    // Half-open range [first, last) of the count work items owned by block
    // block_id when the items are split into grid_size contiguous chunks. The
    // range is empty when the block has nothing to do, which happens for the
    // trailing blocks when grid_size does not divide count.
    //
    // Contiguous chunks, not a grid-stride loop, and that is load bearing for
    // the incomplete factorizations. Those kernels spin on the done flags of the
    // rows they depend on, and the row map is a topological order, so an item
    // only ever waits on items earlier in that order. With contiguous chunks
    // every item a block waits on belongs either to itself, earlier in its own
    // chunk, or to a lower numbered block, which the dispatcher started first.
    // That is the same property the unclamped launch relies on today, and it is
    // what keeps the spin from deadlocking when the grid is larger than the
    // device can hold resident.
    //
    // A grid-stride loop does not have that property. Block 0 on its second item
    // waits on the first item of block grid_size - 1, which cannot start until
    // some block retires, and no block retires while it is spinning.
    //
    ROCSPARSE_GRID_X_ILF void grid_x_chunk(
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
}
