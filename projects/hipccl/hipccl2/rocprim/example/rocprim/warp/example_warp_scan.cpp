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

constexpr unsigned int threads_per_block = 64;
constexpr unsigned int logical_warp_size = 16;
constexpr unsigned int warps_per_block   = threads_per_block / logical_warp_size;

// Kernel: warp_scan with init via scan()
__global__
void warp_scan_kernel(const int* d_in, int* d_out_incl, int* d_out_excl)
{
    using warp_scan_int = rocprim::warp_scan<int, logical_warp_size>;
    __shared__ typename warp_scan_int::storage_type storage[warps_per_block];

    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / logical_warp_size;

    int input = d_in[tid];
    int incl  = 0;
    int excl  = 0;

    // inclusive + exclusive scan in one call (init = 0)
    warp_scan_int().scan(input, // input
                         incl, // inclusive output
                         excl, // exclusive output
                         0, // init for exclusive
                         storage[wid] // per-warp shared storage
                         // , rocprim::plus<int>()  // (optional) custom op
    );

    d_out_incl[tid] = incl;
    d_out_excl[tid] = excl;
}

int main()
{
    // One block: 64 threads, one item per thread
    constexpr int N = threads_per_block;

    // Input = all ones
    std::vector<int> h_in(N, 1);

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out_incl(N);
    common::device_ptr<int> d_out_excl(N);

    // Launch 1 block
    hipLaunchKernelGGL(warp_scan_kernel,
                       dim3(1),
                       dim3(threads_per_block),
                       0,
                       0,
                       d_in.get(),
                       d_out_incl.get(),
                       d_out_excl.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto out_incl = d_out_incl.load();
    const auto out_excl = d_out_excl.load();

    // Expected per logical warp(16):
    // inclusive: [1,2,...,16]  ; exclusive (init=0): [0,1,...,15]
    bool passed = true;
    for(int i = 0; i < N; ++i)
    {
        int lane = i % logical_warp_size;
        passed   = passed && (out_incl[i] == lane + 1) && (out_excl[i] == lane);
    }
    ASSERT_TRUE(passed);

    return 0;
}
