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

int main()
{
    // Host input
    // segments = 3
    // input    = [4, 7, 6, 2, 5, 1, 3, 8]
    // offsets  = [0, 2, 3, 8]
    // segments:
    //   0: [4, 7]           -> min(9, 4, 7)           = 4
    //   1: [6]              -> min(9, 6)              = 6
    //   2: [2, 5, 1, 3, 8]  -> min(9, 2, 5, 1, 3, 8)  = 1
    std::vector<short> h_input   = {4, 7, 6, 2, 5, 1, 3, 8};
    std::vector<int>   h_offsets = {0, 2, 3, 8};
    const unsigned int segments  = 3;
    const int init_value         = 9;

    // Device buffers
    common::device_ptr<short> d_input(h_input);
    common::device_ptr<int>   d_offsets(h_offsets);
    common::device_ptr<int>   d_output(segments); // 3 results

    // Custom reduce op (min)
    auto min_op = [] __device__ (int a, int b) -> int {
        return a < b ? a : b;
    };

    // Get temp storage size
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    HIP_CHECK(rocprim::segmented_reduce(
        d_temp_storage,
        temp_storage_bytes,
        d_input.get(),
        d_output.get(),
        segments,
        d_offsets.get(),
        d_offsets.get() + 1,
        min_op,
        init_value
    ));

    // Allocate temp storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Actual segmented reduce
    HIP_CHECK(rocprim::segmented_reduce(
        d_temp_storage,
        temp_storage_bytes,
        d_input.get(),
        d_output.get(),
        segments,
        d_offsets.get(),
        d_offsets.get() + 1,
        min_op,
        init_value
    ));

    // Check result
    std::vector<int> h_result(segments);
    HIP_CHECK(hipMemcpy(h_result.data(), d_output.get(), segments * sizeof(int), hipMemcpyDeviceToHost));

    std::vector<int> expected = {4, 6, 1};
    bool passed = true;
    for(unsigned int i = 0; i < segments; ++i)
    {
        passed = passed && (h_result[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    // Cleanup
    HIP_CHECK(hipFree(d_temp_storage));

    return 0;
}
