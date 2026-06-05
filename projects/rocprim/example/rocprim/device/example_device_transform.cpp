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
    const size_t       N = 8;
    std::vector<short> h_input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int>   h_expected{6, 7, 8, 9, 10, 11, 12, 13};

    common::device_ptr<short> d_input(h_input);
    common::device_ptr<int>   d_output(N);

    // Unary transform functor: y = x + 5
    auto transform_op = [](short x) { return static_cast<int>(x) + 5; };

    // Launch transform on device
    HIP_CHECK(rocprim::transform(d_input.get(), d_output.get(), N, transform_op));

    const auto h_result = d_output.load();
    bool       passed   = true;
    for(size_t i = 0; i < N; ++i)
    {
        passed = passed && (h_result[i] == h_expected[i]);
    }
    ASSERT_TRUE(passed);

    return 0;
}
