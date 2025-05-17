// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "identity_iterator.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_transform.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <stdint.h>
#include <vector>

// Params for tests
template<class InputType,
         class OutputType                 = InputType,
         bool         UseIdentityIterator = false,
         unsigned int SizeLimit           = ROCPRIM_GRID_SIZE_LIMIT,
         bool         UseGraphs           = false>
struct DeviceTransformParams
{
    using input_type                              = InputType;
    using output_type                             = OutputType;
    static constexpr bool   use_identity_iterator = UseIdentityIterator;
    static constexpr size_t size_limit            = SizeLimit;
    static constexpr bool   use_graphs            = UseGraphs;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceTransformTests : public ::testing::Test
{
public:
    using input_type                              = typename Params::input_type;
    using output_type                             = typename Params::output_type;
    static constexpr bool   use_identity_iterator = Params::use_identity_iterator;
    static constexpr bool   debug_synchronous     = false;
    static constexpr size_t size_limit            = Params::size_limit;
    static constexpr bool   use_graphs            = Params::use_graphs;
};

using custom_short2      = common::custom_type<short, short, true>;
using custom_int2        = common::custom_type<int, int, true>;
using custom_double2     = common::custom_type<double, double, true>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 8>;

using RocprimDeviceTransformTestsParams
    = ::testing::Types<DeviceTransformParams<int, int, true>,
                       DeviceTransformParams<int8_t, int8_t>,
                       DeviceTransformParams<uint8_t, uint8_t>,
                       DeviceTransformParams<rocprim::half, rocprim::half>,
                       DeviceTransformParams<rocprim::bfloat16, rocprim::bfloat16>,
                       DeviceTransformParams<unsigned long>,
                       DeviceTransformParams<short, int, true>,
                       DeviceTransformParams<custom_short2, custom_int2, true>,
                       DeviceTransformParams<int, float>,
                       DeviceTransformParams<uint64_t, uint64_t>,
                       DeviceTransformParams<rocprim::uint128_t, rocprim::uint128_t>,
                       DeviceTransformParams<custom_double2, custom_double2>,
                       DeviceTransformParams<custom_int64_array, custom_int64_array>,
                       DeviceTransformParams<int, int, false, 512>,
                       DeviceTransformParams<float, float, false, 2048>,
                       DeviceTransformParams<double, double, false, 4096>,
                       DeviceTransformParams<int, int, false, 2097152>,
                       DeviceTransformParams<int, int, false, 1073741824>,
                       DeviceTransformParams<int, int, false, ROCPRIM_GRID_SIZE_LIMIT, true>>;

template<unsigned int SizeLimit>
struct size_limit_config
{
    using type = rocprim::transform_config<256, 16, SizeLimit>;
};

template<>
struct size_limit_config<ROCPRIM_GRID_SIZE_LIMIT>
{
    using type = rocprim::default_config;
};

template<unsigned int SizeLimit>
using size_limit_config_t = typename size_limit_config<SizeLimit>::type;

TYPED_TEST_SUITE(RocprimDeviceTransformTests, RocprimDeviceTransformTestsParams);

template<class T>
struct transform
{
    __device__ __host__
    inline T
        operator()(const T& a) const
    {
        return rocprim::plus<T>()(a, T(5));
    }
};

TYPED_TEST(RocprimDeviceTransformTests, Transform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config                                = size_limit_config_t<TestFixture::size_limit>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 100, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(input.size());

            // Calculate expected results on host
            std::vector<U> expected(input.size());
            std::transform(input.begin(), input.end(), expected.begin(), transform<U>());

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::transform<Config>(
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                input.size(),
                transform<U>(),
                stream,
                TestFixture::debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, test_utils::precision<U>));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

template<class T1, class T2, class U>
struct binary_transform
{
    __device__ __host__
    inline constexpr U
        operator()(const T1& a, const T2& b) const
    {
        return a + b;
    }
};

TYPED_TEST(RocprimDeviceTransformTests, BinaryTransform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T1                                    = typename TestFixture::input_type;
    using T2                                    = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    using Config                                = size_limit_config_t<TestFixture::size_limit>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T1> input1
                = test_utils::get_random_data_wrapped<T1>(size, 1, 100, seed_value);
            std::vector<T2> input2
                = test_utils::get_random_data_wrapped<T2>(size, 1, 100, seed_value);

            common::device_ptr<T1> d_input1(input1);
            common::device_ptr<T2> d_input2(input2);
            common::device_ptr<U>  d_output(input1.size());

            // Calculate expected results on host
            std::vector<U> expected(input1.size());
            std::transform(input1.begin(),
                           input1.end(),
                           input2.begin(),
                           expected.begin(),
                           binary_transform<T1, T2, U>());

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::transform<Config>(
                d_input1.get(),
                d_input2.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                input1.size(),
                binary_transform<T1, T2, U>(),
                stream,
                debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, test_utils::precision<U>));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

template<class Functor, class OutputIterator, class... Inputs>
OutputIterator transform_nary(Functor f, size_t size, OutputIterator out, Inputs... inputs)
{
    for(size_t i = 0; i < size; i++)
    {
        *out++ = f(*inputs++...);
    }

    return out;
}

template<class T1, class T2, class T3, class U>
struct ternary_transform
{
    __device__ __host__
    inline constexpr U
        operator()(const T1& a, const T2& b, const T3& c) const
    {
        return a + b + c;
    }
};

TYPED_TEST(RocprimDeviceTransformTests, TernaryTransform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T1                                    = typename TestFixture::input_type;
    using T2                                    = typename TestFixture::input_type;
    using U                                     = typename TestFixture::output_type;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    using Config                                = size_limit_config_t<TestFixture::size_limit>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T1> input1
                = test_utils::get_random_data_wrapped<T1>(size, 1, 100, seed_value);
            std::vector<T2> input2
                = test_utils::get_random_data_wrapped<T2>(size, 1, 100, seed_value);
            std::vector<T2> input3
                = test_utils::get_random_data_wrapped<T1>(size, 1, 100, seed_value);

            common::device_ptr<T1> d_input1(input1);
            common::device_ptr<T2> d_input2(input2);
            common::device_ptr<T2> d_input3(input3);
            common::device_ptr<U>  d_output(input1.size());

            // Calculate expected results on host
            std::vector<U> expected(input1.size());

            transform_nary(ternary_transform<T1, T2, T1, U>(),
                           input1.size(),
                           expected.begin(),
                           input1.begin(),
                           input2.begin(),
                           input3.begin());

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::transform<Config>(
                rocprim::tuple(d_input1.get(), d_input2.get(), d_input3.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                input1.size(),
                ternary_transform<T1, T2, T1, U>(),
                stream,
                debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, test_utils::precision<U>));

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

template<class T>
struct flag_expected_op_t
{
    bool* d_flag;
    T     expected;
    T     expected_above_limit;

    __device__
    auto  operator()(const T& value) -> int
    {
        if(value == expected)
        {
            d_flag[0] = true;
        }
        if(value == expected_above_limit)
        {
            d_flag[1] = true;
        }
        return 0;
    }
};

template<bool UseGraphs = false>
void testLargeIndices()
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = size_t;
    using InputIterator          = rocprim::counting_iterator<T>;
    using OutputIterator         = rocprim::discard_iterator;
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default
    if(UseGraphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(const auto size : test_utils::get_large_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const InputIterator  input{0};
            const OutputIterator output;

            // Using char instead of bool here, since C++ vectors pack bools in single bits
            std::vector<char>            flags = {false, false};
            common::device_ptr<char>     d_flag(flags);

            const auto expected = test_utils::get_random_value<T>(0, size - 1, seed_value);
            const auto limit    = ROCPRIM_GRID_SIZE_LIMIT;
            const auto expected_above_limit
                = size - 1 > limit ? test_utils::get_random_value<T>(limit, size - 1, seed_value)
                                   : size - 1;

            SCOPED_TRACE(testing::Message() << "expected = " << expected);
            SCOPED_TRACE(testing::Message() << "expected_above_limit = " << expected_above_limit);

            const auto flag_expected
                = flag_expected_op_t<T>{(bool*)d_flag.get(), expected, expected_above_limit};

            test_utils::GraphHelper gHelper;
            if(UseGraphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(
                rocprim::transform(input, output, size, flag_expected, stream, debug_synchronous));

            if(UseGraphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            flags = d_flag.load();
            HIP_CHECK(hipDeviceSynchronize());

            ASSERT_TRUE(flags[0]);
            ASSERT_TRUE(flags[1]);

            if(UseGraphs)
                gHelper.cleanupGraphHelper();
        }
    }

    if(UseGraphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceTransformTests, LargeIndices)
{
    testLargeIndices();
}

TEST(RocprimDeviceTransformTests, LargeIndicesWithGraphs)
{
    testLargeIndices<true>();
}

TEST(RocprimDeviceTransformTests, UnalignedPointer)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = int;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input
                = test_utils::get_random_data_wrapped<T>(size + 2, 1, 100, seed_value);

            uint8_t* d_unaligned;
            HIP_CHECK(hipMalloc(&d_unaligned, (size + 3) * sizeof(T)));
            T* d_input = reinterpret_cast<T*>(d_unaligned + 1);
            HIP_CHECK(
                hipMemcpy(d_input, input.data(), (size + 2) * sizeof(T), hipMemcpyHostToDevice));

            // Calculate expected results on host
            std::vector<T> expected(input.size());
            // First and last values should be unchanged.
            expected[0]                = input[0];
            expected[input.size() - 1] = input[input.size() - 1];
            std::transform(input.begin() + 1,
                           input.end() - 1,
                           expected.begin() + 1,
                           transform<T>());

            // Run
            HIP_CHECK(rocprim::transform(d_input + 1,
                                         d_input + 1,
                                         input.size() - 2,
                                         transform<T>(),
                                         stream));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            std::vector<T> output(size + 2);
            HIP_CHECK(
                hipMemcpy(output.data(), d_input, (size + 2) * sizeof(T), hipMemcpyDeviceToHost));

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(
                test_utils::assert_near(output, expected, test_utils::precision<T>));
        }
    }
}
