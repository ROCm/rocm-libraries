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
    std::vector<double> h_haystack = {0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0};
    const size_t haystack_size = h_haystack.size();

    // Needles to search for
    // {1, 2, 3, 4, 5}
    std::vector<double> h_needles = {1.0, 2.0, 3.0, 4.0, 5.0};
    const size_t needles_size = h_needles.size();

    common::device_ptr<double> d_haystack(h_haystack);
    common::device_ptr<double> d_needles(h_needles);
    common::device_ptr<size_t> d_output(needles_size); // indices

    // Comparator
    rocprim::less<double> compare_op;

    // Query temporary storage size
    void*  d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;

    HIP_CHECK(rocprim::lower_bound(
        d_temp_storage,
        temp_storage_bytes,
        d_haystack.get(),
        d_needles.get(),
        d_output.get(),
        haystack_size,
        needles_size,
        compare_op
    ));

    // Allocate temporary storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Actual lower_bound
    HIP_CHECK(rocprim::lower_bound(
        d_temp_storage,
        temp_storage_bytes,
        d_haystack.get(),
        d_needles.get(),
        d_output.get(),
        haystack_size,
        needles_size,
        compare_op,
        0,
        false
    ));

    const auto h_result = d_output.load();

    // For haystack = [0, 1.5, 3, 4.5, 6, 7.5, 9]
    // lower_bound results for needles = [1, 2, 3, 4, 5] are:
    // 1 -> first >=1   is 1.5  -> index 1
    // 2 -> first >=2   is 3    -> index 2
    // 3 -> first >=3   is 3    -> index 2
    // 4 -> first >=4   is 4.5  -> index 3
    // 5 -> first >=5   is 6    -> index 4
    std::vector<size_t> expected = {1, 2, 2, 3, 4};

    bool passed = true;
    for(size_t i = 0; i < needles_size; ++i)
    {
        passed = passed && (h_result[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    HIP_CHECK(hipFree(d_temp_storage));

    return 0;
}
