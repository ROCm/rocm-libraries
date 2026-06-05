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
    const size_t input_size = 8;
    const size_t middle     = 4;

    // Host input
    std::vector<unsigned int> h_input{6, 3, 5, 4, 1, 8, 2, 7};
    std::vector<unsigned int> h_output(input_size, 9u);

    common::device_ptr<unsigned int> d_input(h_input);
    common::device_ptr<unsigned int> d_output(h_output);

    size_t                   temp_storage_size = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::partial_sort_copy(d_temp_storage.get(),
                                          temp_storage_size,
                                          d_input.get(),
                                          d_output.get(),
                                          middle,
                                          input_size);
    };

    // Query required storage
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_size);

    // Run partial_sort_copy on device
    HIP_CHECK(launch());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy back result
    const auto result = d_output.load();

    // Expected: first 4 smallest elements sorted: 1, 2, 3, 4
    std::vector<unsigned int> expected_first{1, 2, 3, 4};
    bool                      passed = true;
    for(size_t i = 0; i < middle; i++)
    {
        passed = passed && (result[i] == expected_first[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
