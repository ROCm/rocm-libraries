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

__global__
void block_histogram_kernel(const int* d_in, unsigned int* d_hist)
{
    // Configure the block histogram primitive
    constexpr unsigned int BlockSize      = 8; // 8 threads in the block
    constexpr unsigned int ItemsPerThread = 1; // one item per thread (total 8 items)
    constexpr unsigned int Bins           = 5; // histogram has 5 bins: 0..4

    // Define the block_histogram type
    using block_hist_t = rocprim::block_histogram<int, BlockSize, ItemsPerThread, Bins>;

    // Shared memory for the primitive and the histogram bins
    __shared__ typename block_hist_t::storage_type storage;
    __shared__ unsigned int hist[Bins];

    // Each thread loads its item (blocked layout, one item per thread)
    const unsigned int tid = threadIdx.x;
    int                items[ItemsPerThread];
    items[0] = d_in[tid];

    // Build the histogram in shared memory
    // - 'items' are the thread's contributions
    // - 'hist' is the shared array of bin counters
    // - 'storage' is temporary shared storage used by the primitive
    block_hist_t().histogram(items, hist, storage);

    // Make sure all bin updates are visible before writing to global memory
    __syncthreads();

    // Let the first Bins threads write the histogram to global memory
    if(tid < Bins)
    {
        d_hist[tid] = hist[tid];
    }
}

int main()
{
    // Input: numbers in the range [0, 4]
    std::vector<int> input = {0, 1, 2, 1, 0, 4, 4, 3};
    constexpr int    bins  = 5;

    common::device_ptr<int>          d_input(input);
    common::device_ptr<unsigned int> d_hist(bins);

    hipLaunchKernelGGL(block_histogram_kernel, dim3(1), dim3(8), 0, 0, d_input.get(), d_hist.get());
    HIP_CHECK(hipDeviceSynchronize());

    // Read back and verify
    const auto hist = d_hist.load();

    // Expected bin counts:
    // value : occurrences
    //   0   : 2  -> indices 0, 4
    //   1   : 2  -> indices 1, 3
    //   2   : 1  -> index   2
    //   3   : 1  -> index   7
    //   4   : 2  -> indices 5, 6
    // So expected histogram = [2, 2, 1, 1, 2]
    std::vector<unsigned int> expected = {2u, 2u, 1u, 1u, 2u};
    bool                      passed   = true;
    for(int i = 0; i < bins; ++i)
    {
        passed = passed && (hist[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
