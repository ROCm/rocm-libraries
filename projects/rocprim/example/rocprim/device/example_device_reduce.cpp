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
    std::vector<short> h_input{4, 7, 6, 2, 5, 1, 3, 8};
    const unsigned int input_size  = static_cast<unsigned int>(h_input.size());
    int                start_value = 9; // min(start_value, all elements) -> should end up as 1

    common::device_ptr<short> d_input(h_input);
    common::device_ptr<int>   d_output(1); // single result

    // Custom reduce op (min)
    auto min_op = [](int a, int b) { return a < b ? a : b; };

    std::size_t              temp_storage_bytes = 0;
    common::device_ptr<void> d_temp_storage;

    const auto launch = [&]
    {
        return rocprim::reduce(d_temp_storage.get(),
                               temp_storage_bytes,
                               d_input.get(),
                               d_output.get(),
                               start_value,
                               input_size,
                               min_op);
    };

    // First launch: get temp storage size
    HIP_CHECK(launch());
    d_temp_storage.resize(temp_storage_bytes);

    // Second launch: actual reduce
    HIP_CHECK(launch());

    // Check result
    int h_result = 0;
    HIP_CHECK(hipMemcpy(&h_result, d_output.get(), sizeof(int), hipMemcpyDeviceToHost));

    // Expected: min(9, 4, 7, 6, 2, 5, 1, 3, 8) = 1
    int expected = 1;
    ASSERT_TRUE(h_result == expected);

    return 0;
}
