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
    std::vector<float> h_input = {0.6f, 0.3f, 0.65f, 0.4f, 0.2f, 0.08f, 1.0f, 0.7f};
    const size_t size = h_input.size();

    common::device_ptr<float> d_input(h_input);
    common::device_ptr<float> d_output(size);

    // Query temporary storage
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    HIP_CHECK(rocprim::merge_sort(
        d_temp_storage,
        temp_storage_bytes,
        d_input.get(),
        d_output.get(),
        size,
        rocprim::less<float>()
    ));

    // Allocate temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Actual merge sort
    HIP_CHECK(rocprim::merge_sort(
        d_temp_storage,
        temp_storage_bytes,
        d_input.get(),
        d_output.get(),
        size,
        rocprim::less<float>()
    ));

    const auto h_output = d_output.load();
    std::vector<float> expected = {0.08f, 0.2f, 0.3f, 0.4f, 0.6f, 0.65f, 0.7f, 1.0f};

    bool passed = true;
    for(size_t i = 0; i < size; ++i)
    {
        passed = passed && (h_output[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    // Cleanup
    HIP_CHECK(hipFree(d_temp_storage));

    return 0;
}
