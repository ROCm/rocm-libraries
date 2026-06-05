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
    std::vector<int> h_input{1, 2, 3, 4, 5, 6, 7, 8};
    const size_t     size = h_input.size();

    common::device_ptr<int>    d_input(h_input);
    common::device_ptr<int>    d_selected(size);
    common::device_ptr<int>    d_rejected(size);
    common::device_ptr<size_t> d_selected_count(1);

    // Predicate: keep even numbers
    auto predicate = [](int a) { return (a % 2) == 0; };

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::partition_two_way(d_temp_storage.get(),
                                          temp_storage_bytes,
                                          d_input.get(),
                                          d_selected.get(),
                                          d_rejected.get(),
                                          d_selected_count.get(),
                                          size,
                                          predicate);
    };

    // First launch: query temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual partition
    HIP_CHECK(launch());

    const auto h_selected       = d_selected.load();
    const auto h_rejected       = d_rejected.load();
    const auto h_selected_count = d_selected_count.load();

    // Expected:
    // selected: [2,4,6,8]
    // rejected: [1,3,5,7]
    // count   : 4
    std::vector<int> expected_selected{2, 4, 6, 8};
    std::vector<int> expected_rejected{1, 3, 5, 7};
    const size_t     expected_count = 4;

    bool passed = (h_selected_count[0] == expected_count);
    for(int i = 0; i < 4; ++i)
    {
        passed = passed && (h_selected[i] == expected_selected[i]);
        passed = passed && (h_rejected[i] == expected_rejected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
