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
#include <rocprim/block/block_sort.hpp>

constexpr unsigned int BlockSize      = 256;
constexpr unsigned int ItemsPerThread = 8;

// Kernel: block-level key-only sort (ascending)
__global__
void block_sort_kernel(const int* d_in, int* d_out)
{
    // Specialize block_sort for int, 256 threads per block, 8 items per thread
    using block_sort_int = rocprim::block_sort<int, BlockSize, ItemsPerThread>;

    // Shared storage for block_sort
    __shared__ typename block_sort_int::storage_type storage;

    const unsigned int tid = threadIdx.x;

    // Compute per-block base (supports multiple blocks if needed)
    const unsigned int block_base = blockIdx.x * BlockSize * ItemsPerThread;

    // Load 8 items per thread from global memory
    int                items[ItemsPerThread];
    const unsigned int base = block_base + tid * ItemsPerThread;
#pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
        items[i] = d_in[base + i];

    // Execute block sort (ascending, key-only)
    block_sort_int().sort(items, storage);

// Store sorted items back to global memory
#pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
        d_out[base + i] = items[i];
}

int main()
{
    // Total elements: 256 * 8 = 2048
    constexpr int total_items = BlockSize * ItemsPerThread;

    // Host input: simple reverse order, so the expected sorted result is 0..2047
    std::vector<int> h_in(total_items);
    for(int i = 0; i < total_items; ++i)
    {
        h_in[i] = total_items - 1 - i;
    }

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(total_items);

    // Launch 1 block of 256 threads
    hipLaunchKernelGGL(block_sort_kernel, dim3(1), dim3(BlockSize), 0, 0, d_in.get(), d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out = d_out.load();

    bool passed = true;
    for(int i = 0; i < total_items; ++i)
    {
        passed = passed && (h_out[i] == i);
    }
    ASSERT_TRUE(passed);

    return 0;
}
