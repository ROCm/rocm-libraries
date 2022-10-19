// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>

// required test headers
#include "test_utils_types.hpp"

// Params for tests
template<
    class InputType,
    class OutputType = InputType,
    bool UseIdentityIterator = false,
    unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct DeviceTransformParams
{
    using input_type = InputType;
    using output_type = OutputType;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr size_t size_limit = SizeLimit;
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
    static constexpr size_t size_limit = Params::size_limit;
};

using custom_short2  = test_utils::custom_test_type<short>;
using custom_int2    = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    DeviceTransformParams<int, int, true>,
    DeviceTransformParams<int8_t, int8_t>,
    DeviceTransformParams<uint8_t, uint8_t>,
    DeviceTransformParams<rocprim::half, rocprim::half>,
    DeviceTransformParams<rocprim::bfloat16, rocprim::bfloat16>,
    DeviceTransformParams<unsigned long>,
    DeviceTransformParams<short, int, true>,
    DeviceTransformParams<custom_short2, custom_int2, true>,
    DeviceTransformParams<int, float>,
    DeviceTransformParams<custom_double2, custom_double2>,
    DeviceTransformParams<int, int, false, 512>,
    DeviceTransformParams<float, float, false, 2048>,
    DeviceTransformParams<int, int, false, 4096>,
    DeviceTransformParams<int, int, false, 2097152>,
    DeviceTransformParams<int, int, false, 1073741824>
> RocprimDeviceTransformTestsParams;

template <unsigned int SizeLimit>
struct size_limit_config {
    using type = rocprim::transform_config<256, 16, SizeLimit>;
};

template <>
struct size_limit_config<ROCPRIM_GRID_SIZE_LIMIT> {
    using type = rocprim::default_config;
};

template <unsigned int SizeLimit>
using size_limit_config_t = typename size_limit_config<SizeLimit>::type;

TYPED_TEST_SUITE(RocprimDeviceTransformTests, RocprimDeviceTransformTestsParams);

template<class T>
struct transform
{
    __device__ __host__ inline
    T operator()(const T& a) const
    {
        return rocprim::plus<T>()(a, T(5));
    }
};

TYPED_TEST(RocprimDeviceTransformTests, Transform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);
            std::vector<U> output(input.size(), (U)0);

            T * d_input;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
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
                rocprim::transform<Config>(
                    d_input,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input.size(), transform<U>(), stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
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
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, test_utils::precision<U>));

            hipFree(d_input);
            hipFree(d_output);
        }
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

TYPED_TEST(RocprimDeviceTransformTests, BinaryTransform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T1 = typename TestFixture::input_type;
    using T2 = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T1> input1 = test_utils::get_random_data<T1>(size, 1, 100, seed_value);
            std::vector<T2> input2 = test_utils::get_random_data<T2>(size, 1, 100, seed_value);
            std::vector<U> output(input1.size(), (U)0);

            T1 * d_input1;
            T2 * d_input2;
            U * d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input1, input1.size() * sizeof(T1)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_input2, input2.size() * sizeof(T2)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(U)));
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
                rocprim::transform<Config>(
                    d_input1, d_input2,
                    test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output),
                    input1.size(), binary_transform<T1, T2, U>(), stream, debug_synchronous
                )
            );
            HIP_CHECK(hipGetLastError());
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
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, test_utils::precision<U>));

            hipFree(d_input1);
            hipFree(d_input2);
            hipFree(d_output);
        }
    }

}

TEST(RocprimDeviceTransformTests, LargeIndices)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = size_t;
    using InputIterator          = rocprim::counting_iterator<T>;
    using OutputIterator         = rocprim::discard_iterator;
    const bool debug_synchronous = false;

    const hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const InputIterator  input {0};
            const OutputIterator output;

            bool  flags[2] = {false, false};
            bool* d_flag   = nullptr;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_flag, sizeof(T)));
            HIP_CHECK(hipMemcpy(d_flag, flags, sizeof(flags), hipMemcpyHostToDevice));

            const auto expected = test_utils::get_random_value<T>(0, size - 1, seed_value);
            const auto limit = ROCPRIM_GRID_SIZE_LIMIT;
            const auto expected_above_limit
                = size - 1 > limit ? test_utils::get_random_value<T>(limit, size - 1, seed_value)
                                   : size - 1;

            SCOPED_TRACE(testing::Message() << "expected = " << expected);
            SCOPED_TRACE(testing::Message() << "expected_above_limit = " << expected_above_limit);

            const auto flag_expected = [=] __device__ (T value) -> int {
                if(value == expected)
                {
                    d_flag[0] = true;
                }
                if(value == expected_above_limit)
                {
                    d_flag[1] = true;
                }
                return 0;
            };

            // Run
            HIP_CHECK(
                rocprim::transform(input, output, size, flag_expected, stream, debug_synchronous));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            HIP_CHECK(hipMemcpy(flags, d_flag, sizeof(flags), hipMemcpyDeviceToHost));
            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_TRUE(flags[0]);
            ASSERT_TRUE(flags[1]);

            hipFree(d_flag);
        }
    }
}
