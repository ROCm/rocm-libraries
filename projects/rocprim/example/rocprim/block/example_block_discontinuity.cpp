// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../example_utils.hpp"

// Kernel 1: flag heads (segment starts) using not_equal_to
__global__
void flag_heads_kernel(const int* d_in, int* d_heads)
{
    // One item per thread, but API expects per-thread arrays
    constexpr int BlockSize      = 8;
    constexpr int ItemsPerThread = 1;

    using bd_t = rocprim::block_discontinuity<int, BlockSize, ItemsPerThread>;

    const int tid = threadIdx.x;

    // Per-thread input/output arrays
    int in[ItemsPerThread];
    int head_flags[ItemsPerThread];

    // Load one item per thread
    in[0] = d_in[tid];

    // Mark heads: head if current != previous (index 0 is always head)
    bd_t().flag_heads(head_flags, in, rocprim::not_equal_to<int>{});

    // Write 1 for head, 0 otherwise
    d_heads[tid] = head_flags[0] ? 1 : 0;
}

// Kernel 2: flag tails (segment ends) using not_equal_to
__global__
void flag_tails_kernel(const int* d_in, int* d_tails)
{
    constexpr int BlockSize      = 8;
    constexpr int ItemsPerThread = 1;

    using bd_t = rocprim::block_discontinuity<int, BlockSize, ItemsPerThread>;

    const int tid = threadIdx.x;

    int in[ItemsPerThread];
    int tail_flags[ItemsPerThread];

    in[0] = d_in[tid];

    // Mark tails: tail if current != next (last index is always tail)
    bd_t().flag_tails(tail_flags, in, rocprim::not_equal_to<int>{});

    d_tails[tid] = tail_flags[0] ? 1 : 0;
}

int main()
{
    // Prepare input
    // Indices: 0 1 2 3 4 5 6 7
    // Values:  1 1 2 2 2 3 1 1
    // Segments: [1, 1] [2, 2, 2] [3] [1, 1]
    std::vector<int> input = {1, 1, 2, 2, 2, 3, 1, 1};
    const int        size  = static_cast<int>(input.size()); // 8

    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_heads(size);
    common::device_ptr<int> d_tails(size);

    // 1. Flag heads
    hipLaunchKernelGGL(flag_heads_kernel, dim3(1), dim3(8), 0, 0, d_input.get(), d_heads.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Expected heads for [1, 1, 2, 2, 2, 3, 1, 1]:
    // indices 0, 2, 5, 6 are heads  -> [1, 0, 1, 0, 0, 1, 1, 0]
    const auto       heads_out      = d_heads.load();
    std::vector<int> expected_heads = {1, 0, 1, 0, 0, 1, 1, 0};

    bool pass_heads = true;
    for(int i = 0; i < size; i++)
    {
        pass_heads = pass_heads && (heads_out[i] == expected_heads[i]);
    }
    ASSERT_TRUE(pass_heads);

    // 2. Flag tails
    hipLaunchKernelGGL(flag_tails_kernel, dim3(1), dim3(8), 0, 0, d_input.get(), d_tails.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Expected tails for [1, 1, 2, 2, 2, 3, 1, 1]:
    // indices 1, 4, 5, 7 are tails -> [0, 1, 0, 0, 1, 1, 0, 1]
    const auto       tails_out      = d_tails.load();
    std::vector<int> expected_tails = {0, 1, 0, 0, 1, 1, 0, 1};

    bool pass_tails = true;
    for(int i = 0; i < size; i++)
    {
        pass_tails = pass_tails && (tails_out[i] == expected_tails[i]);
    }
    ASSERT_TRUE(pass_tails);

    return 0;
}
