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
constexpr unsigned int warps_per_block   = threads_per_block / logical_warp_size; // 4

// Kernel: warp-level reduce (sum) on logical warps of 16 threads
__global__
void warp_reduce_kernel(const int* d_in, int* d_out_per_warp)
{
    // Specialize warp_reduce for int and logical warp size = 16
    using warp_reduce_int = rocprim::warp_reduce<int, logical_warp_size>;

    // One storage per logical warp in this block
    __shared__ typename warp_reduce_int::storage_type temp[warps_per_block];

    const int tid  = threadIdx.x;
    const int warp = tid / logical_warp_size; // logical warp id in this block
    const int lane = tid % logical_warp_size; // lane id in logical warp

    // Each thread loads one value (here from d_in[tid])
    int value = d_in[tid];

    // Do reduce within the logical warp; use plus<int>() explicitly
    warp_reduce_int().reduce(value, value, temp[warp], rocprim::plus<int>{});

    // Let lane 0 write the warp's reduction result
    if(lane == 0)
    {
        d_out_per_warp[warp] = value;
    }
}

int main()
{
    // Host input
    // Keep it brain-checkable: all ones -> each 16-thread warp sums to 16
    std::vector<int> h_in(threads_per_block, 1); // 64 ones
    std::vector<int> h_out(warps_per_block, 0); // 4 results

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(warps_per_block);

    // Launch one block of 64 threads
    hipLaunchKernelGGL(warp_reduce_kernel,
                       dim3(1),
                       dim3(threads_per_block),
                       0,
                       0,
                       d_in.get(),
                       d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto       result = d_out.load();
    std::vector<int> expected(warps_per_block, logical_warp_size); // {16, 16, 16, 16}

    bool passed = true;
    for(int i = 0; i < (int)warps_per_block; ++i)
    {
        passed = passed && (result[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
