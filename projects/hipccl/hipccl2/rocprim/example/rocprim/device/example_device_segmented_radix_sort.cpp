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
    std::vector<float> h_input    = {0.6f, 0.3f, 0.65f, 0.4f, 0.2f, 0.08f, 1.0f, 0.7f};
    const size_t       input_size = h_input.size();

    // Segments: [0,2), [2,3), [3,8)
    // So it sorts: {0.6,0.3} | {0.65} | {0.4,0.2,0.08,1,0.7}
    std::vector<int>   h_offsets = {0, 2, 3, 8};
    const unsigned int segments  = 3;

    common::device_ptr<float> d_input(h_input);
    common::device_ptr<float> d_output(input_size);
    common::device_ptr<int>   d_offsets(h_offsets);

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::segmented_radix_sort_keys(d_temp_storage.get(),
                                                  temp_storage_bytes,
                                                  d_input.get(),
                                                  d_output.get(),
                                                  input_size,
                                                  segments,
                                                  d_offsets.get(),
                                                  d_offsets.get() + 1);
    };

    // Query temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Actual segmented radix sort
    HIP_CHECK(launch());

    const auto         output   = d_output.load();
    std::vector<float> expected = {
        0.3f,
        0.6f, // segment 0
        0.65f, // segment 1
        0.08f,
        0.2f,
        0.4f,
        0.7f,
        1.0f // segment 2
    };

    bool passed = true;
    for(size_t i = 0; i < input_size; ++i)
    {
        passed = passed && (output[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
