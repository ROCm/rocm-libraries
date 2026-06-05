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

// Kernel: block_load (direct) — 128 threads, 8 items per thread
__global__
void block_load_kernel(const int* d_in, int* d_out)
{
    constexpr int BlockSize      = 128;
    constexpr int ItemsPerThread = 8;

    using block_load_t = rocprim::
        block_load<int, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_direct>;

    // Shared storage for block_load (required by some overloads)
    __shared__ typename block_load_t::storage_type storage;

    const int tid    = threadIdx.x;
    const int offset = blockIdx.x * BlockSize * ItemsPerThread;

    // Each thread receives 8 items
    int items[ItemsPerThread];

    // Total valid elements in this block (full tile here)
    const unsigned int valid = BlockSize * ItemsPerThread;

    // Load from global memory (input + offset) into per-thread items
    // Signature: load(input_ptr, items, valid, out_of_bounds_value, storage)
    block_load_t().load(d_in + offset, items, valid, 0, storage);

    // Write back in blocked layout
    const int base = offset + tid * ItemsPerThread;
#pragma unroll
    for(int i = 0; i < ItemsPerThread; ++i)
        d_out[base + i] = items[i];
}

int main()
{
    // Host input: 0..(1 block * 128 threads * 8 items - 1) = 0..1023
    constexpr int BlockSize      = 128;
    constexpr int ItemsPerThread = 8;
    constexpr int TotalItems     = BlockSize * ItemsPerThread; // 1024

    std::vector<int> input(TotalItems);
    for(int i = 0; i < TotalItems; ++i)
    {
        input[i] = i;
    }

    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_output(TotalItems);

    // Launch one block of 128 threads
    hipLaunchKernelGGL(block_load_kernel,
                       dim3(1),
                       dim3(BlockSize),
                       0,
                       0,
                       d_input.get(),
                       d_output.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto out    = d_output.load();
    bool       passed = true;
    for(int i = 0; i < TotalItems; ++i)
    {
        passed = passed && (out[i] == input[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
