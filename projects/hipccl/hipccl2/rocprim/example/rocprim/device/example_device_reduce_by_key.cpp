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
    // keys   = [1, 1, 1, 2, 10, 10, 10, 88]
    // values = [1, 2, 3, 4,  5,  6,  7,  8]
    // expect unique:     [1, 2, 10, 88]
    // expect aggregates: [6, 4, 18, 8]
    // expect unique_count: 4
    std::vector<int> h_keys{1, 1, 1, 2, 10, 10, 10, 88};
    std::vector<int> h_values{1, 2, 3, 4, 5, 6, 7, 8};
    const size_t     input_size = h_keys.size();

    common::device_ptr<int> d_keys(h_keys);
    common::device_ptr<int> d_values(h_values);
    common::device_ptr<int> d_unique(4);
    common::device_ptr<int> d_aggregates(4);
    common::device_ptr<int> d_unique_count(1);

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::reduce_by_key(d_temp_storage.get(),
                                      temp_storage_bytes,
                                      d_keys.get(),
                                      d_values.get(),
                                      input_size,
                                      d_unique.get(),
                                      d_aggregates.get(),
                                      d_unique_count.get());
    };

    // First launch: query temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual reduce_by_key
    HIP_CHECK(launch());

    const auto h_unique       = d_unique.load();
    const auto h_aggregates   = d_aggregates.load();
    const auto h_unique_count = d_unique_count.load();
    const int  unique_count   = h_unique_count[0];

    // Expected
    std::vector<int> expected_unique{1, 2, 10, 88};
    std::vector<int> expected_aggregates{6, 4, 18, 8};
    int              expected_count = 4;

    bool passed = true;
    passed      = passed && (unique_count == expected_count);
    for(int i = 0; i < expected_count; ++i)
    {
        passed = passed && (h_unique[i] == expected_unique[i]);
        passed = passed && (h_aggregates[i] == expected_aggregates[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
