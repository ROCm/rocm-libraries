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
    std::vector<int> h_input = {6, 3, 5, 4, 1, 8, 2, 7};
    std::vector<int> h_keys  = {10, 5, 100};

    const size_t size      = h_input.size(); // 8
    const size_t keys_size = h_keys.size(); // 3

    // Device buffers (using the helper from example_utils.hpp)
    common::device_ptr<int> d_input(h_input);
    common::device_ptr<int> d_keys(h_keys);

    // Output is just one index (the found position)
    common::device_ptr<unsigned int> d_output(1);

    // Get temporary storage size
    void*  d_temp_storage    = nullptr;
    size_t temp_storage_size = 0;

    // First call: pointer = nullptr, get storage size
    HIP_CHECK(rocprim::find_first_of(d_temp_storage,
                                     temp_storage_size,
                                     d_input.get(),
                                     d_keys.get(),
                                     d_output.get(),
                                     size,
                                     keys_size,
                                     rocprim::equal_to<int>{}, // comparator
                                     0, // stream
                                     false // debug
                                     ));

    // Allocate much temporary storage on device
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

    // Real call
    HIP_CHECK(rocprim::find_first_of(d_temp_storage,
                                     temp_storage_size,
                                     d_input.get(),
                                     d_keys.get(),
                                     d_output.get(),
                                     size,
                                     keys_size,
                                     rocprim::equal_to<int>{},
                                     0,
                                     false));

    HIP_CHECK(hipDeviceSynchronize());

    // Copy result back and check
    std::vector<unsigned int> h_output = d_output.load();

    // Expected index is 2 (input[2] == 5 which is in keys)
    unsigned int expected = 2u;

    ASSERT_TRUE(h_output[0] == expected);

    // Clean up temp storage
    HIP_CHECK(hipFree(d_temp_storage));

    return 0;
}
