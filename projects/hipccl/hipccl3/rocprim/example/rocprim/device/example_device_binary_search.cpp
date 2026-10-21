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
    // haystack must be sorted
    // {0, 1.5, 3, 4.5, 6, 7.5, 9}
    std::vector<double> h_haystack{0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0};
    const size_t        haystack_size = h_haystack.size();

    // Needles to search for
    // {1, 2, 3, 4, 5}
    std::vector<double> h_needles{1.0, 2.0, 3.0, 4.0, 5.0};
    const size_t        needles_size = h_needles.size();

    common::device_ptr<double> d_haystack(h_haystack);
    common::device_ptr<double> d_needles(h_needles);
    common::device_ptr<size_t> d_output(needles_size); // indices

    // Comparator
    rocprim::less<double> compare_op;

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::lower_bound(d_temp_storage.get(),
                                    temp_storage_bytes,
                                    d_haystack.get(),
                                    d_needles.get(),
                                    d_output.get(),
                                    haystack_size,
                                    needles_size,
                                    compare_op,
                                    0,
                                    false);
    };

    // First launch: query temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual lower_bound
    HIP_CHECK(launch());

    const auto h_result = d_output.load();

    // For haystack = [0, 1.5, 3, 4.5, 6, 7.5, 9]
    // lower_bound results for needles = [1, 2, 3, 4, 5] are:
    // 1 -> first >=1   is 1.5  -> index 1
    // 2 -> first >=2   is 3    -> index 2
    // 3 -> first >=3   is 3    -> index 2
    // 4 -> first >=4   is 4.5  -> index 3
    // 5 -> first >=5   is 6    -> index 4
    std::vector<size_t> expected{1, 2, 2, 3, 4};

    bool passed = true;
    for(size_t i = 0; i < needles_size; ++i)
    {
        passed = passed && (h_result[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
