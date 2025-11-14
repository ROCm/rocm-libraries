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
    // Host inputs (both sorted)
    std::vector<int> h_input1 = {0, 2, 4, 6};
    std::vector<int> h_input2 = {1, 3, 5, 7};

    const unsigned int size1 = static_cast<unsigned int>(h_input1.size());
    const unsigned int size2 = static_cast<unsigned int>(h_input2.size());
    const unsigned int total = size1 + size2;

    // Device buffers
    common::device_ptr<int> d_input1(h_input1);
    common::device_ptr<int> d_input2(h_input2);
    common::device_ptr<int> d_output(total);

    // Temp storage ptr + size
    void*  d_temp_storage    = nullptr;
    size_t temp_storage_size = 0;

    // Query temp storage size (explicit template to avoid overload ambiguity)
    hipError_t err = rocprim::merge<rocprim::default_config,
                                    int*, // InputIterator1
                                    int*, // InputIterator2
                                    int*, // OutputIterator
                                    rocprim::less<int>>(d_temp_storage,
                                                        temp_storage_size,
                                                        d_input1.get(),
                                                        d_input2.get(),
                                                        d_output.get(),
                                                        size1,
                                                        size2,
                                                        rocprim::less<int>(),
                                                        0,
                                                        false);
    HIP_CHECK(err);

    // Allocate temp storage
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

    // Actual merge
    err = rocprim::merge<rocprim::default_config, int*, int*, int*, rocprim::less<int>>(
        d_temp_storage,
        temp_storage_size,
        d_input1.get(),
        d_input2.get(),
        d_output.get(),
        size1,
        size2,
        rocprim::less<int>(),
        0,
        false);
    HIP_CHECK(err);

    HIP_CHECK(hipDeviceSynchronize());

    // Verify (minimal)
    const auto       h_output = d_output.load();
    std::vector<int> expected = {0, 1, 2, 3, 4, 5, 6, 7};

    bool passed = true;
    for(unsigned int i = 0; i < total; i++)
    {
        passed = passed && (h_output[i] == expected[i]);
    }
    ASSERT_TRUE(passed);

    HIP_CHECK(hipFree(d_temp_storage));
    return 0;
}
