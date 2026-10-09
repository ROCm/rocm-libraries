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

// Kernel 1: blocked -> striped
__global__
void blocked_to_striped_kernel(const int* d_in, int* d_out)
{
    constexpr int                                      BlockSize      = 4;
    constexpr int                                      ItemsPerThread = 2;
    using block_exchange_t = rocprim::block_exchange<int, BlockSize, ItemsPerThread>;

    __shared__ typename block_exchange_t::storage_type storage;

    const int tid = threadIdx.x;

    // Each thread holds ItemsPerThread consecutive items (blocked layout)
    int       in_items[ItemsPerThread];
    const int base = tid * ItemsPerThread;

    for(int i = 0; i < ItemsPerThread; i++)
    {
        in_items[i] = d_in[base + i];
    }

    int out_items[ItemsPerThread] = {0};

    // Convert from blocked to striped layout
    block_exchange_t().blocked_to_striped(in_items, out_items, storage);

    // Write results back to global memory
    for(int i = 0; i < ItemsPerThread; i++)
    {
        d_out[base + i] = out_items[i];
    }
}

// Kernel 2: striped -> blocked
__global__
void striped_to_blocked_kernel(const int* d_in, int* d_out)
{
    constexpr int                                      BlockSize      = 4;
    constexpr int                                      ItemsPerThread = 2;
    using block_exchange_t = rocprim::block_exchange<int, BlockSize, ItemsPerThread>;

    __shared__ typename block_exchange_t::storage_type storage;

    const int tid = threadIdx.x;

    // Each thread reads ItemsPerThread elements from striped layout
    int       in_items[ItemsPerThread];
    const int base = tid * ItemsPerThread;

    for(int i = 0; i < ItemsPerThread; i++)
    {
        in_items[i] = d_in[base + i];
    }

    int out_items[ItemsPerThread] = {0};

    // Convert from striped back to blocked layout
    block_exchange_t().striped_to_blocked(in_items, out_items, storage);

    // Write back to global memory
    for(int i = 0; i < ItemsPerThread; i++)
    {
        d_out[base + i] = out_items[i];
    }
}

int main()
{
    // Prepare input
    // Total elements = BlockSize (4) * ItemsPerThread (2) = 8
    std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7};
    const int        size  = static_cast<int>(input.size());

    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_striped(size); // striped result (blocked -> striped)
    common::device_ptr<int> d_blocked(size); // blocked result (striped -> blocked)

    // 1. blocked -> striped
    hipLaunchKernelGGL(blocked_to_striped_kernel,
                       dim3(1),
                       dim3(4),
                       0,
                       0,
                       d_input.get(),
                       d_striped.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Load and verify the blocked_to_striped result
    const auto striped = d_striped.load();

    // In a striped layout, items are interleaved across threads:
    // [0,4,
    //  1,5,
    //  2,6,
    //  3,7]
    std::vector<int> expected_striped = {0, 4, 1, 5, 2, 6, 3, 7};

    bool pass1 = true;
    for(int i = 0; i < size; i++)
    {
        pass1 = pass1 && (striped[i] == expected_striped[i]);
    }
    ASSERT_TRUE(pass1);

    // 2. striped -> blocked
    hipLaunchKernelGGL(striped_to_blocked_kernel,
                       dim3(1),
                       dim3(4),
                       0,
                       0,
                       d_striped.get(),
                       d_blocked.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto blocked = d_blocked.load();

    bool pass2 = true;
    for(int i = 0; i < size; i++)
    {
        pass2 = pass2 && (blocked[i] == input[i]);
    }
    ASSERT_TRUE(pass2);

    return 0;
}
