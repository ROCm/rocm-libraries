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

constexpr unsigned int threads_per_block = 128;
constexpr unsigned int threads_per_warp  = 8;
constexpr unsigned int items_per_thread  = 4;
constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;

// Kernel: warp-level store (direct)
__global__
void warp_store_kernel(int* d_out)
{
    using warp_store_t = rocprim::warp_store<int,
                                             items_per_thread,
                                             threads_per_warp,
                                             rocprim::warp_store_method::warp_store_direct>;

    // One storage per warp
    __shared__ typename warp_store_t::storage_type storage[warps_per_block];

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / threads_per_warp;

    // Base offset for this block + this warp
    const int offset = blockIdx.x * threads_per_block * items_per_thread
                       + warp_id * threads_per_warp * items_per_thread;

    // Each thread prepares 4 items (blocked order)
    int       items[items_per_thread];
    const int base = tid * items_per_thread;
#pragma unroll
    for(unsigned int i = 0; i < items_per_thread; ++i)
    {
        items[i] = base + i; // so final array should be 0..511
    }

    // Total number of valid items this warp should write
    const unsigned int valid = threads_per_warp * items_per_thread; // 8 * 4 = 32

    // Store: ptr, items, valid, storage
    warp_store_t().store(d_out + offset, items, valid, storage[warp_id]);
}

int main()
{
    // 1 block: 128 threads * 4 items = 512 ints
    constexpr int total_items = threads_per_block * items_per_thread; // 512

    common::device_ptr<int> d_out(total_items);

    // launch 1 block
    hipLaunchKernelGGL(warp_store_kernel, dim3(1), dim3(threads_per_block), 0, 0, d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out = d_out.load();

    // Verification: should be 0..511
    bool passed = true;
    for(int i = 0; i < total_items; ++i)
    {
        passed = passed && (h_out[i] == i);
    }
    ASSERT_TRUE(passed);

    return 0;
}
