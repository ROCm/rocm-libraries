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
    // Keys: same keys will be scanned together
    std::vector<int>   h_keys{1, 1, 2, 2, 3, 3, 3, 5};
    std::vector<short> h_values{1, 2, 3, 4, 5, 6, 7, 8};
    const size_t       size = h_keys.size(); // 8

    // Expected result of inclusive scan-by-key with plus<int>:
    // key=1: 1, 1+2=3
    // key=2: 3, 3+4=7
    // key=3: 5, 5+6=11, 11+7=18
    // key=5: 8
    std::vector<int> expected{1, 3, 3, 7, 5, 11, 18, 8};

    common::device_ptr<int>   d_keys(h_keys);
    common::device_ptr<short> d_values(h_values);
    common::device_ptr<int>   d_output(size);

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::inclusive_scan_by_key(d_temp_storage.get(),
                                              temp_storage_bytes,
                                              d_keys.get(),
                                              d_values.get(),
                                              d_output.get(),
                                              size,
                                              rocprim::plus<int>(),
                                              rocprim::equal_to<int>());
    };

    // First launch: query required temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual scan-by-key
    HIP_CHECK(launch());

    HIP_CHECK(hipDeviceSynchronize());

    const auto result = d_output.load();

    bool passed = true;
    for(size_t i = 0; i < size; i++)
    {
        passed = passed && (result[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
