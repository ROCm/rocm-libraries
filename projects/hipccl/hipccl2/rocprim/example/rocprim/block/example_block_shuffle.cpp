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
#include <rocprim/block/block_shuffle.hpp>

constexpr unsigned int BlockSize = 192;

// Kernel: rotate values inside a block by +1 (up)
__global__
void block_shuffle_rotate_kernel(const int* d_in, int* d_out)
{
    using block_shuffle_t = rocprim::block_shuffle<int, BlockSize>;

    // Shared storage required by the storage-based overloads
    __shared__ typename block_shuffle_t::storage_type storage;

    const unsigned int tid   = threadIdx.x; // flat id (1D block)
    int                value = d_in[tid];

    // Rotate up by 1: thread i receives value from (i-1), wrapping at block edges
    block_shuffle_t().rotate(tid, value, value, 1, storage);

    d_out[tid] = value;
}

int main()
{
    // Host input: 0..191
    std::vector<int> h_in(BlockSize);
    for(unsigned int i = 0; i < BlockSize; ++i)
    {
        h_in[i] = static_cast<int>(i);
    }

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(BlockSize);

    // Launch exactly 192 threads in one block
    hipLaunchKernelGGL(block_shuffle_rotate_kernel,
                       dim3(1),
                       dim3(BlockSize),
                       0,
                       0,
                       d_in.get(),
                       d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out = d_out.load();

    // Expected: out[i] = in[(i + BlockSize - 1) % BlockSize]
    std::vector<int> expected(BlockSize);
    for(unsigned int i = 0; i < BlockSize; ++i)
    {
        expected[i] = static_cast<int>((i + 1) % BlockSize);
    }

    bool passed = true;
    for(unsigned int i = 0; i < BlockSize; ++i)
    {
        passed = passed && (h_out[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
