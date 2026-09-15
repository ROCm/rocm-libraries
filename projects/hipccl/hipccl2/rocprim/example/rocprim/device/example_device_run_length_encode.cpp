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
    // Host input: 1,1,1,2,10,10,10,88
    std::vector<int> h_input{1, 1, 1, 2, 10, 10, 10, 88};
    const size_t     input_size = h_input.size(); // 8

    common::device_ptr<int> d_input(h_input);
    common::device_ptr<int> d_unique(4);
    common::device_ptr<int> d_counts(4);
    common::device_ptr<int> d_runs_count(1);

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::run_length_encode(d_temp_storage.get(),
                                          temp_storage_bytes,
                                          d_input.get(),
                                          input_size,
                                          d_unique.get(),
                                          d_counts.get(),
                                          d_runs_count.get());
    };

    // First launch: get temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual RLE
    HIP_CHECK(launch());

    // Copy back results
    const auto h_unique     = d_unique.load();
    const auto h_counts     = d_counts.load();
    const auto h_runs_count = d_runs_count.load();

    // Expected:
    // unique: [1, 2, 10, 88]
    // counts: [3, 1,  3,  1]
    // runs:   4
    std::vector<int> expected_unique{1, 2, 10, 88};
    std::vector<int> expected_counts{3, 1, 3, 1};
    int              expected_runs = 4;

    bool passed = true;
    // Check runs_count first
    passed = passed && (h_runs_count[0] == expected_runs);
    // Check only the first "runs" elements
    for(int i = 0; i < expected_runs; ++i)
    {
        passed = passed && (h_unique[i] == expected_unique[i]);
        passed = passed && (h_counts[i] == expected_counts[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
