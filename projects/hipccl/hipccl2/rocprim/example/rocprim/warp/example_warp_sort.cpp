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

__global__
void                   warp_sort_kernel(const int* d_in, int* d_out)
{
    // Specialize warp_sort for int and logical warp size = 16
    using warp_sort_int = rocprim::warp_sort<int, logical_warp_size>;

    // One shared storage per logical warp in this block
    __shared__ typename warp_sort_int::storage_type storage[warps_per_block];

    const int tid  = threadIdx.x;
    const int warp = tid / logical_warp_size; // logical warp id in the block

    // One key per thread
    int key = d_in[tid];

    // Sort keys within the logical warp (ascending)
    warp_sort_int().sort(key, storage[warp]);

    // Write back the sorted key
    d_out[tid] = key;
}

int main()
{
    // Host input
    // For each logical warp w, threads get values in strictly descending order:
    //   value = (15 - lane) + 100*w
    std::vector<int> h_in(threads_per_block);
    for(int i = 0; i < (int)threads_per_block; ++i)
    {
        int lane = i % logical_warp_size;
        int warp = i / logical_warp_size;
        h_in[i]  = (logical_warp_size - 1 - lane) + 100 * warp; // 15..0 (+warp offset)
    }

    common::device_ptr<int> d_in(h_in);
    common::device_ptr<int> d_out(threads_per_block);

    // Launch one block
    hipLaunchKernelGGL(warp_sort_kernel,
                       dim3(1),
                       dim3(threads_per_block),
                       0,
                       0,
                       d_in.get(),
                       d_out.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto h_out = d_out.load();

    // Expected: each logical warp becomes ascending 0..15 (+warp offset)
    // For each logical warp w, ascending 0..15 with the same 100*w offset
    std::vector<int> expected(threads_per_block);
    for(int i = 0; i < (int)threads_per_block; ++i)
    {
        int lane    = i % logical_warp_size;
        int warp    = i / logical_warp_size;
        expected[i] = 100 * warp + lane;
    }

    bool passed = true;
    for(int i = 0; i < (int)threads_per_block; ++i)
    {
        passed = passed && (h_out[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
