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
    const unsigned int size = 8;
    std::vector<float> h_samples{-10.0f, 0.3f, 9.5f, 8.1f, 1.5f, 1.9f, 100.0f, 5.1f};

    // 5 bins => levels = 6, range [0, 10)
    const unsigned int levels    = 6;
    const float        lower_lvl = 0.0f;
    const float        upper_lvl = 10.0f;

    // Histogram has (levels - 1) elements
    std::vector<int> h_hist(levels - 1, 0);

    common::device_ptr<float> d_samples(h_samples);
    common::device_ptr<int>   d_hist(h_hist);

    std::size_t              temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::histogram_even(d_temp_storage.get(),
                                       temp_storage_bytes,
                                       d_samples.get(),
                                       size,
                                       d_hist.get(),
                                       levels,
                                       lower_lvl,
                                       upper_lvl);
    };

    // Query required temporary storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Actual histogram computation
    HIP_CHECK(launch());

    const auto result = d_hist.load();

    // Expected bins for range [0,10) split into 5 even bins:
    // bins: [0,2), [2,4), [4,6), [6,8), [8,10)
    // samples inside: 0.3,1.5,1.9 -> bin0
    //                 5.1         -> bin2
    //                 9.5, 8.1    -> bin4
    // so: [3, 0, 1, 0, 2]
    std::vector<int> expected{3, 0, 1, 0, 2};

    bool passed = true;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        passed = passed && (result[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
