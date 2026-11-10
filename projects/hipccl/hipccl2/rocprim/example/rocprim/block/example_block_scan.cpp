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

// Kernel 1: inclusive block scan (sum)
__global__
void inclusive_scan_kernel(const int* d_in, int* d_out)
{
    // One item per thread.
    constexpr int                                  BlockSize = 8;
    using block_scan_t = rocprim::block_scan<int, BlockSize>;

    __shared__ typename block_scan_t::storage_type storage;

    const int tid = threadIdx.x;

    int value  = d_in[tid];
    int result = 0;

    // Perform inclusive scan across all threads in the block.
    block_scan_t().inclusive_scan(value, result, storage, rocprim::plus<int>());

    d_out[tid] = result;
}

// Kernel 2: exclusive block scan (sum)
__global__
void exclusive_scan_kernel(const int* d_in, int* d_out)
{
    // One item per thread.
    constexpr int BlockSize = 8;
    using block_scan_t      = rocprim::block_scan<int, BlockSize>;

    const int tid = threadIdx.x;

    int value  = d_in[tid];
    int result = 0;

    // Use 0 as the initial value.
    int init = 0;

    // Perform exclusive scan across all threads in the block.
    block_scan_t().exclusive_scan(value, result, init, rocprim::plus<int>());

    d_out[tid] = result;
}

int main()
{
    // Prepare input
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
    const int        size  = static_cast<int>(input.size());

    // Device buffers
    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_inclusive(size);
    common::device_ptr<int> d_exclusive(size);

    // 1. Inclusive scan
    hipLaunchKernelGGL(inclusive_scan_kernel,
                       dim3(1),
                       dim3(8),
                       0,
                       0,
                       d_input.get(),
                       d_inclusive.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Expected inclusive sums for [1, 2, 3, 4, 5, 6, 7, 8]:
    // [1, 3, 6, 10, 15, 21, 28, 36]
    const auto       inclusive_out      = d_inclusive.load();
    std::vector<int> expected_inclusive = {1, 3, 6, 10, 15, 21, 28, 36};

    bool pass_inclusive = true;
    for(int i = 0; i < size; i++)
    {
        pass_inclusive = pass_inclusive && (inclusive_out[i] == expected_inclusive[i]);
    }
    ASSERT_TRUE(pass_inclusive);

    // 2. Exclusive scan
    hipLaunchKernelGGL(exclusive_scan_kernel,
                       dim3(1),
                       dim3(8),
                       0,
                       0,
                       d_input.get(),
                       d_exclusive.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Expected exclusive sums with init=0 for [1, 2, 3, 4, 5, 6, 7, 8]:
    // [0, 1, 3, 6, 10, 15, 21, 28]
    const auto       exclusive_out      = d_exclusive.load();
    std::vector<int> expected_exclusive = {0, 1, 3, 6, 10, 15, 21, 28};

    bool pass_exclusive = true;
    for(int i = 0; i < size; i++)
    {
        pass_exclusive = pass_exclusive && (exclusive_out[i] == expected_exclusive[i]);
    }
    ASSERT_TRUE(pass_exclusive);

    return 0;
}
