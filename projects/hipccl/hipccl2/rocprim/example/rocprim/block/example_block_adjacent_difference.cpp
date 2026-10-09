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
void block_adj_diff_kernel(const int* d_in, int* d_out)
{
    using adjacent_diff
        = rocprim::block_adjacent_difference<int, /*BlockSize=*/8, /*ItemsPerThread=*/1>;
    __shared__ typename adjacent_diff::storage_type storage;

    const int lid = threadIdx.x;

    int in_item[1]  = {d_in[lid]};
    int out_item[1] = {0};

    // Perform adjacent difference with minus operation
    adjacent_diff().subtract_left(in_item, out_item, rocprim::minus<int>{}, storage);

    d_out[lid] = out_item[0];
}

int main()
{
    // Prepare input and output
    std::vector<int> input = {0, 1, 2, 1, 0};
    int              size  = 5;

    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_output(size);

    hipLaunchKernelGGL(block_adj_diff_kernel,
                       dim3(1),
                       dim3(8),
                       0,
                       0,
                       d_input.get(),
                       d_output.get());
    HIP_CHECK(hipDeviceSynchronize());

    const auto output = d_output.load();

    // Computed by applying minus operation to left neighbor
    // {0, 1 - 0, 2 - 1, 1 - 2, 0 - 1}
    std::vector<int> expected = {0, 1, 1, -1, -1};

    bool passed = true;
    for(int i = 0; i < size; i++)
    {
        passed = passed && (expected[i] == output[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
