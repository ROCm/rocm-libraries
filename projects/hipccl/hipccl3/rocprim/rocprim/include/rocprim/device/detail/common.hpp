// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DETAIL_COMMON_HPP_
#define ROCPRIM_DEVICE_DETAIL_COMMON_HPP_

#include <hip/hip_runtime.h>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// \brief Calculate kernel grid dimensions from the total number of blocks and the block size
///
/// ROCm can launch a kernel with up to 2^32 threads in one dimension (global work size).
/// When \p number_of_blocks * \p block_size exceeds this limit, then 2-dimensional grid is used.
/// The kernel must use flat_block_id() or a similar code to calculate the 2d block id and compare
/// it with \p number_of_blocks to prevent out-of-bounds accesses.
inline dim3 calculate_grid_dim(unsigned int number_of_blocks, unsigned int block_size)
{
    const unsigned int max_blocks = std::numeric_limits<uint32_t>::max() / block_size;
    const unsigned int blocks_y   = ceiling_div(number_of_blocks, max_blocks);
    const unsigned int blocks_x   = min(number_of_blocks, max_blocks);
    return dim3(blocks_x, blocks_y);
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_COMMON_HPP_
