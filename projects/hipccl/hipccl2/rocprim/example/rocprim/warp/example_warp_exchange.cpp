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

// Kernel: warp-level blocked -> striped
__global__
void warp_exchange_kernel(const int* d_in, int* d_out)
{
    // Specialize warp_exchange for int, 4 items per thread, warp size 8
    using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;

    // Shared storage: one per warp in the block
    __shared__ typename warp_exchange_int::storage_type storage[warps_per_block];

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / threads_per_warp;

    // Load 4 items per thread from global memory (blocked layout)
    int                items[items_per_thread];
    const unsigned int base = tid * items_per_thread;
#pragma unroll
    for(unsigned int i = 0; i < items_per_thread; ++i)
    {
        items[i] = d_in[base + i];
    }

    // Do warp-level blocked -> striped exchange
    warp_exchange_int w_exchange;
    w_exchange.blocked_to_striped(items, items, storage[warp_id]);

// Store back to global memory (still 4 items per thread)
#pragma unroll
    for(unsigned int i = 0; i < items_per_thread; ++i)
    {
        d_out[base + i] = items[i];
    }
}

int main()
{
    // Total elements = 128 threads * 4 items = 512
    constexpr int total_items = threads_per_block * items_per_thread;

    // Host input: 0..511 (easy to check)
    std::vector<int> h_in(total_items);
    for(int i = 0; i < total_items; ++i)
        h_in[i] = i;

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(total_items);

    // Launch 1 block of 128 threads
    hipLaunchKernelGGL(warp_exchange_kernel,
                       dim3(1),
                       dim3(threads_per_block),
                       0,
                       0,
                       d_in.get(),
                       d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out = d_out.load();

    // Build expected output on host:
    // For each warp (32 items): striped order per thread t is
    // [base + t, base + t + 8, base + t + 16, base + t + 24]
    std::vector<int> expected(total_items);
    const int        warp_items = threads_per_warp * items_per_thread; // 8 * 4 = 32
    for(unsigned int w = 0; w < warps_per_block; ++w)
    {
        int warp_base = w * warp_items; // start index of this warp's 32 items
        for(int lane = 0; lane < (int)threads_per_warp; ++lane)
        {
            int thread_global      = w * threads_per_warp + lane;
            int out_base           = thread_global * items_per_thread;
            expected[out_base + 0] = warp_base + lane;
            expected[out_base + 1] = warp_base + lane + threads_per_warp;
            expected[out_base + 2] = warp_base + lane + 2 * threads_per_warp;
            expected[out_base + 3] = warp_base + lane + 3 * threads_per_warp;
        }
    }

    bool passed = true;
    for(int i = 0; i < total_items; ++i)
    {
        passed = passed && (h_out[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
