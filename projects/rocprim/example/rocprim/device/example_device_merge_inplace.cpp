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
    // Host input: two sorted halves
    // left  = [1,3,5,7]
    // right = [0,2,4,6]
    std::vector<int> h_data = {1, 3, 5, 7, 0, 2, 4, 6};
    const size_t left_size  = 4;
    const size_t right_size = 4;

    common::device_ptr<int> d_data(h_data);

    // Query temp storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    HIP_CHECK(rocprim::merge_inplace(
        d_temp_storage,
        temp_storage_bytes,
        d_data.get(),
        left_size,
        right_size
    ));

    // Allocate temp storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Actual in-place merge on device
    HIP_CHECK(rocprim::merge_inplace(
        d_temp_storage,
        temp_storage_bytes,
        d_data.get(),
        left_size,
        right_size
    ));

    const auto h_out = d_data.load();
    std::vector<int> expected = {0,1,2,3,4,5,6,7};

    bool passed = true;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        passed = passed && (h_out[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    // Cleanup
    HIP_CHECK(hipFree(d_temp_storage));

    return 0;
}
