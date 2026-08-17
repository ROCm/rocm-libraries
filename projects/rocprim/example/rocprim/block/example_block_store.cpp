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

// Kernel 1: blocked input -> striped store
__global__
void store_striped_kernel(const int* d_in, int* d_out)
{
    constexpr int BlockSize      = 4;
    constexpr int ItemsPerThread = 2;

    // Select the striped store algorithm
    using store_t = rocprim::block_store<int,
                                         BlockSize,
                                         ItemsPerThread,
                                         rocprim::block_store_method::block_store_striped>;

    const int tid  = threadIdx.x;
    const int base = tid * ItemsPerThread;

    // Each thread gathers its blocked chunk from global memory
    int items[ItemsPerThread];
    for(int i = 0; i < ItemsPerThread; i++)
    {
        items[i] = d_in[base + i];
    }

    // Store to global memory in striped layout:
    // linear order becomes [0,2,4,6, 1,3,5,7] for input [0..7]
    store_t().store(d_out, items);
}

// Kernel 2: striped input -> direct (blocked) store
__global__
void store_direct_from_striped_kernel(const int* d_in_striped, int* d_out_blocked)
{
    constexpr int BlockSize      = 4;
    constexpr int ItemsPerThread = 2;

    using store_t = rocprim::block_store<int,
                                         BlockSize,
                                         ItemsPerThread,
                                         rocprim::block_store_method::block_store_direct>;

    const int tid = threadIdx.x;

    // Reconstruct per-thread items by reading from striped layout:
    // items[i] lives at global index i*BlockSize + tid
    int items[ItemsPerThread];
    for(int i = 0; i < ItemsPerThread; i++)
    {
        items[i] = d_in_striped[i * BlockSize + tid];
    }

    // Direct (blocked) store writes each thread's ItemsPerThread consecutively
    // back to global memory at indices tid*ItemsPerThread + i
    store_t().store(d_out_blocked, items);
}

int main()
{
    std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7};
    const int        size  = static_cast<int>(input.size());

    // Device buffers
    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_striped(size);
    common::device_ptr<int> d_output(size);

    // 1. blocked -> striped via block_store_striped
    hipLaunchKernelGGL(store_striped_kernel,
                       dim3(1),
                       dim3(4),
                       0,
                       0,
                       d_input.get(),
                       d_striped.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto striped = d_striped.load();
    // Expected striped sequence for input [0..7]:
    std::vector<int> expected_striped = {0, 2, 4, 6, 1, 3, 5, 7};

    bool pass1 = true;
    for(int i = 0; i < size; i++)
    {
        pass1 = pass1 && (striped[i] == expected_striped[i]);
    }
    ASSERT_TRUE(pass1);

    // 2. striped -> blocked via block_store_direct
    hipLaunchKernelGGL(store_direct_from_striped_kernel,
                       dim3(1),
                       dim3(4),
                       0,
                       0,
                       d_striped.get(),
                       d_output.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto recovered = d_output.load();
    // Should match the original blocked sequence
    bool pass2 = true;
    for(int i = 0; i < size; i++)
    {
        pass2 = pass2 && (recovered[i] == input[i]);
    }
    ASSERT_TRUE(pass2);

    return 0;
}
