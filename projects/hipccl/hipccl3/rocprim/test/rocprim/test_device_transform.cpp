// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error)         \
    ASSERT_EQ(static_cast<hipError_t>(error),hipSuccess)

namespace rp = rocprim;

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    bool UseIdentityIterator = false
>
struct DeviceTransformParams
{
    using input_type = InputType;
    using output_type = OutputType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceTransformTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool debug_synchronous = false;
};

using custom_short2 = test_utils::custom_test_type<short>;
using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    DeviceTransformParams<int, int, true>,
    DeviceTransformParams<int8_t, int8_t>,
    DeviceTransformParams<uint8_t, uint8_t>,
    DeviceTransformParams<rp::half, rp::half>,
    DeviceTransformParams<unsigned long>,
    DeviceTransformParams<short, int, true>,
    DeviceTransformParams<custom_short2, custom_int2, true>,
    DeviceTransformParams<int, float>,
    DeviceTransformParams<custom_double2, custom_double2>
> RocprimDeviceTransformTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        1, 10, 53, 211,
        1024, 2048, 5096,
        34567, (1 << 17) - 1220
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceTransformTests, RocprimDeviceTransformTestsParams);

template<class T>
struct transform
{
    __device__ __host__ inline
    T operator()(const T& a) const
    {
        return a + 5;
    }
};

template<>
struct transform<rp::half>
{
    __device__ __host__ inline
    rp::half operator()(const rp::half& a) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + rp::half(5);
        #else
        return test_utils::half_plus()(a, rp::half(5));
        #endif
    }
};


TYPED_TEST(RocprimDeviceTransformTests, Transform)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);
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

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        std::transform(input.begin(), input.end(), expected.begin(), transform<U>());

        // Run
        HIP_CHECK(
            rocprim::transform(
                d_input,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                input.size(), transform<U>(), stream, debug_synchronous
            )
        );
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        hipFree(d_input);
        hipFree(d_output);
    }
}

template<class T1, class T2, class U>
struct binary_transform
{
    __device__ __host__ inline
    constexpr U operator()(const T1& a, const T2& b) const
    {
        return a + b;
    }
};

template<>
struct binary_transform<rp::half, rp::half, rp::half>
{
    __device__ __host__ inline
    rp::half operator()(const rp::half& a, const rp::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return test_utils::half_plus()(a, b);
        #endif
    }
};

TYPED_TEST(RocprimDeviceTransformTests, BinaryTransform)
{
    using T1 = typename TestFixture::input_type;
    using T2 = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T1> input1 = test_utils::get_random_data<T1>(size, 1, 100);
        std::vector<T2> input2 = test_utils::get_random_data<T2>(size, 1, 100);
        std::vector<U> output(input1.size(), 0);

        T1 * d_input1;
        T2 * d_input2;
        U * d_output;
        HIP_CHECK(hipMalloc(&d_input1, input1.size() * sizeof(T1)));
        HIP_CHECK(hipMalloc(&d_input2, input2.size() * sizeof(T2)));
        HIP_CHECK(hipMalloc(&d_output, output.size() * sizeof(U)));
        HIP_CHECK(
            hipMemcpy(
                d_input1, input1.data(),
                input1.size() * sizeof(T1),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(
            hipMemcpy(
                d_input2, input2.data(),
                input2.size() * sizeof(T2),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate expected results on host
        std::vector<U> expected(input1.size());
        std::transform(
            input1.begin(), input1.end(), input2.begin(),
            expected.begin(), binary_transform<T1, T2, U>()
        );

        // Run
        HIP_CHECK(
            rocprim::transform(
                d_input1, d_input2,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                input1.size(), binary_transform<T1, T2, U>(), stream, debug_synchronous
            )
        );
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
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output, expected, 0.01f));

        hipFree(d_input1);
        hipFree(d_input2);
        hipFree(d_output);
    }
}
