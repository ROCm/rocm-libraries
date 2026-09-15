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

// Kernel 1: one item per thread, block-wide sum
__global__
void block_reduce_sum_1item(const int* d_in, int* d_out)
{
    constexpr int                                    BlockSize = 8;
    using block_reduce_t = rocprim::block_reduce<int, BlockSize>;

    __shared__ typename block_reduce_t::storage_type storage;

    const int tid = threadIdx.x;

    int x = d_in[tid];

    // Aggregate result (valid in lane 0 after reduce)
    int aggregate = 0;

    // Block-wide reduction with + operator
    block_reduce_t().reduce(x, aggregate, storage, rocprim::plus<int>());

    // Only one thread writes the block's aggregate
    if(tid == 0)
    {
        d_out[0] = aggregate;
    }
}

// Kernel 2: two items per thread, block-wide sum
__global__
void block_reduce_sum_2items(const int* d_in, int* d_out)
{
    constexpr int                                    BlockSize      = 4;
    constexpr int                                    ItemsPerThread = 2;
    using block_reduce_t = rocprim::block_reduce<int, BlockSize>;

    __shared__ typename block_reduce_t::storage_type storage;

    const int tid = threadIdx.x;

    // Gather two items per thread from global memory (blocked layout)
    int       items[ItemsPerThread];
    const int base = tid * ItemsPerThread;
    for(int i = 0; i < ItemsPerThread; i++)
    {
        items[i] = d_in[base + i];
    }
    int aggregate = 0;

    // Block-wide reduction over per-thread arrays
    block_reduce_t().reduce(items, aggregate, storage, rocprim::plus<int>());

    if(tid == 0)
        d_out[0] = aggregate;
}

int main()
{
    // Input easy to verify by hand: 1..8, sum = 36
    std::vector<int> input        = {1, 2, 3, 4, 5, 6, 7, 8};
    const int        expected_sum = 36;

    // Device buffers
    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_tmp(1);
    common::device_ptr<int> d_out(1);

    // 1. One item per thread
    hipLaunchKernelGGL(block_reduce_sum_1item, dim3(1), dim3(8), 0, 0, d_input.get(), d_tmp.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto sum1 = d_tmp.load();
    ASSERT_TRUE(sum1[0] == expected_sum);

    // 2. Two items per thread
    hipLaunchKernelGGL(block_reduce_sum_2items, dim3(1), dim3(4), 0, 0, d_input.get(), d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto sum2 = d_out.load();
    ASSERT_TRUE(sum2[0] == expected_sum);

    return 0;
}
