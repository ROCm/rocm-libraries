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
    // Host side data
    // input:  [6, 3, 5, 4, 1, 8, 2, 5, 4, 1]
    // key:    [5, 4, 1]
    // answer: last occurrence of key in input starts at index 7
    std::vector<unsigned int> h_input{6, 3, 5, 4, 1, 8, 2, 5, 4, 1};
    std::vector<unsigned int> h_key{5, 4, 1};
    const size_t              size     = h_input.size();
    const size_t              key_size = h_key.size();

    // Device buffers
    common::device_ptr<unsigned int> d_input(h_input);
    common::device_ptr<unsigned int> d_key(h_key);
    common::device_ptr<unsigned int> d_output(1); // single index

    size_t                   temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::find_end(d_temp_storage.get(),
                                 temp_storage_bytes,
                                 d_input.get(),
                                 d_key.get(),
                                 d_output.get(),
                                 size,
                                 key_size);
    };

    // First launch: query temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual find_end
    HIP_CHECK(launch());

    std::vector<unsigned int> h_result = d_output.load();

    // Expected start index of last occurrence
    unsigned int expected = 7u;
    ASSERT_TRUE(h_result[0] == expected);

    return 0;
}
