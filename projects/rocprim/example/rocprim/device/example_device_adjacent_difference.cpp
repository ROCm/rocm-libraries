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
    std::vector<int>  input = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::size_t size  = input.size();

    common::device_ptr<int> d_input(input);
    common::device_ptr<int> d_output(size);

    auto subtract_op = [](auto a, auto b) { return a - b; };

    // Query temporary storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    HIP_CHECK(rocprim::adjacent_difference(d_temp_storage,
                                           temp_storage_bytes,
                                           d_input.get(),
                                           d_output.get(),
                                           size,
                                           subtract_op));

    // Allocate temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Actual adjacent difference
    HIP_CHECK(rocprim::adjacent_difference(d_temp_storage,
                                           temp_storage_bytes,
                                           d_input.get(),
                                           d_output.get(),
                                           size,
                                           subtract_op));
    HIP_CHECK(hipDeviceSynchronize());

    const auto       output   = d_output.load();
    std::vector<int> expected = {1, 1, 1, 1, 1, 1, 1, 1};

    bool passed = true;
    for(std::size_t i = 0; i < size; ++i)
    {
        passed = passed && (output[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    HIP_CHECK(hipFree(d_temp_storage));
    return 0;
}
