// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <vector>
#include <algorithm>

// Google Test
#include <gtest/gtest.h>
// HIP API
#include <hip/hip_runtime.h>
// hipCUB API
#include <hipcub/hipcub.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    class ScanOp = hipcub::Sum
>
struct DeviceScanParams
{
    using input_type = InputType;
    using output_type = OutputType;
    using scan_op_type = ScanOp;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceScanTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    using scan_op_type = typename Params::scan_op_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceScanParams<int, long>,
    DeviceScanParams<unsigned long long, unsigned long long, hipcub::Min>,
    DeviceScanParams<unsigned long>,
    DeviceScanParams<short, float, hipcub::Max>,
    DeviceScanParams<int, double>
> HipcubDeviceScanTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 18) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(HipcubDeviceScanTests, HipcubDeviceScanTestsParams);

TYPED_TEST(HipcubDeviceScanTests, InclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10);
        std::vector<U> output(input.size(), 0);

        T * d_input;
        U * d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // scan function
        scan_op_type scan_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        test_utils::host_inclusive_scan(
            input.begin(), input.end(),
            expected.begin(), scan_op
        );

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        if(std::is_same<scan_op_type, hipcub::Sum>::value)
        {
            HIP_CHECK(
                hipcub::DeviceScan::InclusiveSum(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceScan::InclusiveScan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, scan_op, input.size(),
                    stream, debug_synchronous
                )
            );
        }

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0U);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        if(std::is_same<scan_op_type, hipcub::Sum>::value)
        {
            HIP_CHECK(
                hipcub::DeviceScan::InclusiveSum(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceScan::InclusiveScan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, scan_op, input.size(),
                    stream, debug_synchronous
                )
            );
        }
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(U),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        for(size_t i = 0; i < output.size(); i++)
        {
            auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
            if(std::is_integral<U>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}

TYPED_TEST(HipcubDeviceScanTests, ExclusiveScan)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    using scan_op_type = typename TestFixture::scan_op_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 10);
        std::vector<U> output(input.size());

        T * d_input;
        U * d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
        HIP_CHECK(
            hipMemcpy(
                d_input, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // scan function
        scan_op_type scan_op;

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        const T initial_value =
            std::is_same<scan_op_type, hipcub::Sum>::value
            ? T(0)
            : test_utils::get_random_value<T>(1, 100);
        test_utils::host_exclusive_scan(
            input.begin(), input.end(),
            initial_value, expected.begin(),
            scan_op
        );

        // temp storage
        size_t temp_storage_size_bytes;
        void * d_temp_storage = nullptr;
        // Get size of d_temp_storage
        if(std::is_same<scan_op_type, hipcub::Sum>::value)
        {
            HIP_CHECK(
                hipcub::DeviceScan::ExclusiveSum(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceScan::ExclusiveScan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, scan_op, initial_value, input.size(),
                    stream, debug_synchronous
                )
            );
        }

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0U);

        // allocate temporary storage
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        if(std::is_same<scan_op_type, hipcub::Sum>::value)
        {
            HIP_CHECK(
                hipcub::DeviceScan::ExclusiveSum(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, input.size(),
                    stream, debug_synchronous
                )
            );
        }
        else
        {
            HIP_CHECK(
                hipcub::DeviceScan::ExclusiveScan(
                    d_temp_storage, temp_storage_size_bytes,
                    d_input, d_output, scan_op, initial_value, input.size(),
                    stream, debug_synchronous
                )
            );
        }
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        HIP_CHECK(
            hipMemcpy(
                output.data(), d_output,
                output.size() * sizeof(U),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Check if output values are as expected
        for(size_t i = 0; i < output.size(); i++)
        {
            auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
            if(std::is_integral<U>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff) << "where index = " << i;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(d_temp_storage);
    }
}
