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
    // Prepare input and output
    common::device_ptr<int> input(std::vector{6, 3, 5, 4, 1, 8, 2, 5, 4, 1});
    common::device_ptr<int> key(std::vector{5, 4, 1});
    common::device_ptr<int> output(1);

    // Temporary storage and its size
    std::size_t              temp_storage_size = 0;
    common::device_ptr<void> temp_storage;

    // Lambda to launch rocprim::search
    const auto launch = [&]
    {
        return rocprim::search(temp_storage.get(),
                               temp_storage_size,
                               input.get(),
                               key.get(),
                               output.get(),
                               input.size(),
                               key.size());
    };

    // First launch: query required temp storage size
    HIP_CHECK(launch());
    temp_storage.resize(temp_storage_size);

    // Second launch: actual search
    HIP_CHECK(launch());

    // Check for any errors
    HIP_CHECK(hipGetLastError());

    // Wait for the algorithm to finish
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    auto result = output.load();

    // Check that the result is correct
    ASSERT_TRUE(result[0] == 2);
}
