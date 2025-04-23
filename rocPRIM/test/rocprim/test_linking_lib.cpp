// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifdef TEST_LINKING_EMULATE_ANOTHER_VERSION
    // Pretend that another version of rocPRIM is used with a different implementation if scan
    #define ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
    #include "test_linking_new_scan.hpp"
#else
    #include <rocprim/device/device_scan.hpp>
#endif

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"

#include "test_utils.hpp"
#include "test_utils_types.hpp"

#include <rocprim/functional.hpp>

#ifndef TEST_FUNC
    #define TEST_FUNC test0
#endif

void TEST_FUNC(size_t size)
{
    using T = int;

    const int seed_value = 123;

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100, seed_value);

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(input.size());

    std::vector<T> expected(size);
    // Calculate expected results on host
#ifdef TEST_LINKING_EMULATE_ANOTHER_VERSION
    // The "new" version of the function fills the output with a constant value
    std::fill(expected.begin(), expected.end(), 12345);
#else
    test_utils::host_inclusive_scan(input.begin(),
                                    input.end(),
                                    expected.begin(),
                                    rocprim::plus<T>());
#endif

    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    HIP_CHECK(rocprim::inclusive_scan(nullptr,
                                      temp_storage_size_bytes,
                                      d_input.get(),
                                      d_output.get(),
                                      input.size(),
                                      rocprim::plus<T>()));

#ifdef TEST_LINKING_EMULATE_ANOTHER_VERSION
    // The "new" version of the function requests this particular size of temporary storage
    EXPECT_EQ(temp_storage_size_bytes, 12345);
#else
    EXPECT_LT(temp_storage_size_bytes, 12345);
#endif

    // allocate temporary storage
    common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

    // Run
    HIP_CHECK(rocprim::inclusive_scan(d_temp_storage.get(),
                                      temp_storage_size_bytes,
                                      d_input.get(),
                                      d_output.get(),
                                      input.size(),
                                      rocprim::plus<T>()));

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    const auto output = d_output.load();

    // Check if output values are as expected
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
}
