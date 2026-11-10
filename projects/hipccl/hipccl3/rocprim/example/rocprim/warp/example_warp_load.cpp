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

// Kernel: warp-level load (direct)
__global__
void warp_load_kernel(const int* d_in, int* d_out)
{
    // One warp handles threads_per_warp * items_per_thread elements
    using warp_load_t = rocprim::warp_load<int,
                                           items_per_thread,
                                           threads_per_warp,
                                           rocprim::warp_load_method::warp_load_direct>;

    // One storage per warp in this block
    __shared__ typename warp_load_t::storage_type storage[warps_per_block];

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / threads_per_warp;

    // Base offset for this warp
    const int offset = blockIdx.x * threads_per_block * items_per_thread
                       + warp_id * threads_per_warp * items_per_thread;

    // Each thread will receive 4 items
    int items[items_per_thread];

    // Total valid items for this warp = 8 threads * 4 items = 32
    const unsigned int valid = threads_per_warp * items_per_thread;

    // Load: input ptr, per-thread items, valid count, out-of-bounds value, storage
    warp_load_t().load(d_in + offset,
                       items,
                       valid,
                       0, // out-of-bounds fill value (won't be used since valid is full)
                       storage[warp_id]);

    // Write back in blocked layout
    const unsigned int base = tid * items_per_thread;
#pragma unroll
    for(unsigned int i = 0; i < items_per_thread; ++i)
    {
        d_out[base + i] = items[i];
    }
}

int main()
{
    // 1 block: 128 threads * 4 items = 512 ints
    constexpr int total_items = threads_per_block * items_per_thread;

    // Host input 0..511
    std::vector<int> h_in(total_items);
    for(int i = 0; i < total_items; ++i)
    {
        h_in[i] = i;
    }

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(total_items);

    hipLaunchKernelGGL(warp_load_kernel,
                       dim3(1),
                       dim3(threads_per_block),
                       0,
                       0,
                       d_in.get(),
                       d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out = d_out.load();

    bool passed = true;
    for(int i = 0; i < total_items; ++i)
    {
        passed = passed && (h_out[i] == h_in[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
