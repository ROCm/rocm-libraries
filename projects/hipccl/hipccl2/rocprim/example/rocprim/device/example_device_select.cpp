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
    std::vector<int>  h_input{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<char> h_flags{0, 1, 1, 0, 0, 1, 0, 1}; // select 2,3,6,8
    const size_t      input_size = h_input.size();

    common::device_ptr<int>    d_input(h_input);
    common::device_ptr<char>   d_flags(h_flags);
    common::device_ptr<int>    d_output(input_size); // max possible = input_size
    common::device_ptr<size_t> d_count(1); // number of selected items

    size_t                   temp_bytes = 0;
    common::device_ptr<void> d_temp;

    const auto launch = [&]
    {
        return rocprim::select(d_temp.get(),
                               temp_bytes,
                               d_input.get(),
                               d_flags.get(),
                               d_output.get(),
                               d_count.get(),
                               input_size);
    };

    // First launch: query temp storage size
    HIP_CHECK(launch());
    d_temp.resize(temp_bytes);

    // Second launch: actual selection
    HIP_CHECK(launch());

    // Copy back results
    const auto   h_out    = d_output.load(); // length == input_size (only first count are valid)
    const auto   h_count  = d_count.load(); // vector<size_t> of size 1
    const size_t selected = h_count[0];

    // Expected selected: [2, 3, 6, 8], count = 4
    std::vector<int> expected = {2, 3, 6, 8};
    bool             pass     = (selected == expected.size());
    for(size_t i = 0; i < selected && i < expected.size(); i++)
    {
        pass = pass && (h_out[i] == expected[i]);
    }
    ASSERT_TRUE(pass);

    return 0;
}
