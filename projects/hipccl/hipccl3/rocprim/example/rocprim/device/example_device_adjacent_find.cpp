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
    // Host input:
    // Pairs: (8,7)->no, (7,5)->yes, so answer should be index 1
    std::vector<int>  h_input{8, 7, 5, 4, 3, 2, 1, 0};
    const std::size_t size = h_input.size();

    common::device_ptr<int>         d_input(h_input);
    common::device_ptr<std::size_t> d_output(1);

    // Custom predicate
    auto diff_is_two = [](auto a, auto b) { return (a - b) == 2; };

    std::size_t              temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::adjacent_find(d_temp_storage.get(),
                                      temp_storage_bytes,
                                      d_input.get(),
                                      d_output.get(),
                                      size,
                                      diff_is_two);
    };

    // First launch: query temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual device adjacent_find
    HIP_CHECK(launch());

    const auto h_out = d_output.load();

    // Expected: the first matching pair is (7,5), i.e. index 1
    std::size_t expected = 1;
    ASSERT_TRUE(h_out[0] == expected);

    return 0;
}
