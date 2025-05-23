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
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required rocprim headers
#include <rocprim/block/block_reduce.hpp>
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/thread/thread_operators.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/key_value_pair.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <vector>

using bra = ::rocprim::block_reduce_algorithm;

// Params for tests
template<class InputType,
         class OutputType           = InputType,
         bool   UseIdentityIterator = false,
         size_t SizeLimit           = ROCPRIM_GRID_SIZE_LIMIT,
         bra    Algo                = bra::default_algorithm,
         bool   UseGraphs           = false,
         bool   Deterministic       = false>
struct DeviceReduceParams
{
    static constexpr bra algo = Algo;
    using input_type = InputType;
    using output_type = OutputType;
    // Tests output iterator with void value_type (OutputIterator concept)
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr size_t size_limit = SizeLimit;
    static constexpr bool use_graphs = UseGraphs;
};

// clang-format off
#define DeviceReduceParamsList(...)                                     \
    DeviceReduceParams<__VA_ARGS__, bra::using_warp_reduce>,            \
    DeviceReduceParams<__VA_ARGS__, bra::raking_reduce>,                \
    DeviceReduceParams<__VA_ARGS__, bra::raking_reduce_commutative_only>
// clang-format on

template<unsigned int SizeLimit, bra algo = bra::default_algorithm>
struct size_limit_config
{
    using type = rocprim::reduce_config<256, 16, algo, SizeLimit>;
};

template <>
struct size_limit_config<ROCPRIM_GRID_SIZE_LIMIT> {
    using type = rocprim::default_config;
};

template <unsigned int SizeLimit>
using size_limit_config_t = typename size_limit_config<SizeLimit>::type;

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceReduceTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    const bool debug_synchronous = false;
    static constexpr bool use_identity_iterator = Params::use_identity_iterator;
    static constexpr size_t size_limit = Params::size_limit;
    const bool use_graphs = Params::use_graphs;
};

template<class Params>
class RocprimDeviceReducePrecisionTests : public RocprimDeviceReduceTests<Params>{};

using RocprimDeviceReduceTestsParams = ::testing::Types<
    DeviceReduceParams<unsigned int>,
    DeviceReduceParams<long, long, true>,
    DeviceReduceParams<short, int>,
    DeviceReduceParams<int, float>,
    DeviceReduceParamsList(int, int, false, 512),
    DeviceReduceParamsList(float, float, false, 2048),
    DeviceReduceParamsList(double, double, false, 2048),
    DeviceReduceParamsList(int, int, false, 4096),
    DeviceReduceParamsList(int, int, false, 2097152),
    DeviceReduceParamsList(int, int, false, 1073741824),
    DeviceReduceParams<int8_t, int8_t>,
    DeviceReduceParams<uint8_t, uint8_t>,
    DeviceReduceParams<rocprim::half, rocprim::half>,
    DeviceReduceParams<rocprim::bfloat16, rocprim::bfloat16>,
    DeviceReduceParams<common::custom_type<float, float, true>,
                       common::custom_type<float, float, true>>,
    DeviceReduceParams<common::custom_type<int, int, true>,
                       common::custom_type<float, float, true>>,
    DeviceReduceParams<rocprim::half,
                       rocprim::half,
                       false,
                       ROCPRIM_GRID_SIZE_LIMIT,
                       bra::default_algorithm,
                       false,
                       true>,
    DeviceReduceParams<float,
                       float,
                       false,
                       ROCPRIM_GRID_SIZE_LIMIT,
                       bra::default_algorithm,
                       false,
                       true>,
    DeviceReduceParams<double,
                       double,
                       false,
                       ROCPRIM_GRID_SIZE_LIMIT,
                       bra::default_algorithm,
                       false,
                       true>,
    DeviceReduceParams<common::custom_type<double, double, true>,
                       common::custom_type<double, double, true>,
                       false,
                       ROCPRIM_GRID_SIZE_LIMIT,
                       bra::default_algorithm,
                       false,
                       true>,
    DeviceReduceParams<int, int, false, ROCPRIM_GRID_SIZE_LIMIT, bra::default_algorithm, true>>;

using RocprimDeviceReducePrecisionTestsParams
    = ::testing::Types<DeviceReduceParams<double, double>,
                       DeviceReduceParamsList(float, float, false, 2048),
                       DeviceReduceParams<rocprim::half, rocprim::half>,
                       DeviceReduceParams<rocprim::bfloat16, rocprim::bfloat16>>;

TYPED_TEST_SUITE(RocprimDeviceReduceTests, RocprimDeviceReduceTestsParams);
TYPED_TEST_SUITE(RocprimDeviceReducePrecisionTests, RocprimDeviceReducePrecisionTestsParams);

TYPED_TEST(RocprimDeviceReduceTests, ReduceEmptyInput)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    hipStream_t stream = 0; // default stream
    if (TestFixture::use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    common::device_ptr<U> d_output(1);

    const U initial_value = U(1234);

    // Get size of d_temp_storage
    size_t temp_storage_size_bytes;
    HIP_CHECK(rocprim::reduce<Config>(nullptr,
                                      temp_storage_size_bytes,
                                      rocprim::make_constant_iterator<T>(T(345)),
                                      d_output.get(),
                                      initial_value,
                                      0,
                                      rocprim::minimum<U>(),
                                      stream,
                                      debug_synchronous));

    common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

    test_utils::GraphHelper gHelper;
    if(TestFixture::use_graphs)
    {
        gHelper.startStreamCapture(stream);
    }

    // Run
    HIP_CHECK(rocprim::reduce<Config>(d_temp_storage.get(),
                                      temp_storage_size_bytes,
                                      rocprim::make_constant_iterator<T>(T(345)),
                                      d_output.get(),
                                      initial_value,
                                      0,
                                      rocprim::minimum<U>(),
                                      stream,
                                      debug_synchronous));

    if(TestFixture::use_graphs)
    {
        gHelper.createAndLaunchGraph(stream);
    }

    HIP_CHECK(hipDeviceSynchronize());

    const auto output = d_output.load()[0];

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, initial_value));

    if (TestFixture::use_graphs)
    {
        gHelper.cleanupGraphHelper();
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TYPED_TEST(RocprimDeviceReduceTests, ReduceSum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            if(test_utils::precision<U> * size > 0.5)
            {
                std::cout << "Test is skipped from size " << size
                          << " on, potential error of summation is more than 0.5 of the result "
                             "with current or larger size"
                          << std::endl;
                break;
            }

            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(1);

            // Calculate expected results on host
            std::vector<U> expected
                = test_utils::host_reduce(input.begin(), input.end(), rocprim::plus<U>());

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::reduce<Config>(
                nullptr,
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                input.size(),
                rocprim::plus<U>(),
                stream,
                debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::reduce<Config>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                input.size(),
                rocprim::plus<U>(),
                stream,
                debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            if(size == 0)
            {
                ASSERT_NO_FATAL_FAILURE(
                    test_utils::assert_near(output[0], U{}, test_utils::precision<U> * 1));
            }
            else
            {
                for(size_t i = 0; i < output.size(); ++i)
                {
                    ASSERT_NO_FATAL_FAILURE(
                        test_utils::assert_near(output[i],
                                                expected[i],
                                                test_utils::precision<U> * (size - 1 - i)));
                }
            }

            if (TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceReduceTests, ReduceArgMinimum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using key_value = rocprim::key_value_pair<int, T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<key_value> input(size);
            for (size_t i = 0; i < size; i++)
            {
                input[i].key = (int)i;
                input[i].value = test_utils::get_random_data<T>(1, 1, 100, seed_value)[0];
            }

            common::device_ptr<key_value> d_input(input);
            common::device_ptr<key_value> d_output(1);

            rocprim::arg_min reduce_op;
            const key_value max(std::numeric_limits<int>::max(), rocprim::numeric_limits<T>::max());

            // Calculate expected results on host
            key_value expected = max;
            for(unsigned int i = 0; i < input.size(); i++)
            {
                expected = reduce_op(expected, input[i]);
            }

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::reduce<Config>(
                nullptr,
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                max,
                input.size(),
                reduce_op,
                stream,
                debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::reduce<Config>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                max,
                input.size(),
                reduce_op,
                stream,
                debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();

            // Check if output values are as expected
            test_utils::assert_eq(output[0].key, expected.key);
            test_utils::assert_eq(output[0].value, expected.value);

            if (TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

template<bool use_graphs = false>
void testLargeIndices()
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                      = size_t;
    using Iterator               = rocprim::counting_iterator<T>;
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default
    if (use_graphs)
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

            const Iterator input {0};

            common::device_ptr<T> d_output(1);

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::reduce(nullptr,
                                      temp_storage_size_bytes,
                                      input,
                                      d_output.get(),
                                      size,
                                      rocprim::plus<T>{},
                                      stream,
                                      debug_synchronous));

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::reduce(d_temp_storage.get(),
                                      temp_storage_size_bytes,
                                      input,
                                      d_output.get(),
                                      size,
                                      rocprim::plus<T>{},
                                      stream,
                                      debug_synchronous));

            if(use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load()[0];

            // Sum of numbers 0 to n - 1 is n(n - 1) / 2, note that this is correct even in case of overflow
            // The division is not integer division but either n or n - 1 has to be even.
            T expected_output = (size % 2 == 0) ? size / 2 * (size - 1) : size * ((size - 1) / 2);

            ASSERT_EQ(output, expected_output);

            if(use_graphs)
            {
                gHelper.cleanupGraphHelper();
            }
        }
    }

    if(use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceReduceTests, LargeIndices)
{
    testLargeIndices<>();
}

TEST(RocprimDeviceReduceTests, LargeIndicesWithGraphs)
{
    testLargeIndices<true>();
}

TYPED_TEST(RocprimDeviceReducePrecisionTests, ReduceSumInputEqualExponentFunction)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config                                = size_limit_config_t<TestFixture::size_limit>;

    for(auto size : test_utils::get_sizes(42))
    {
        //if(size == 0)
        //    continue;
        // as all numbers here are the same and have only 1 significant bit in matnissa the error is like this
        const float precision = std::max(0.0, test_utils::precision<U> / 2.0 * (size - 1) - 1.0);
        if(precision > 0.5)
        {
            std::cout << "Test is skipped from size " << size
                      << " on, potential error of summation is more than 0.5 of the result with "
                         "current or larger size"
                      << std::endl;
            break;
        }

        hipStream_t stream = 0; // default
        if (TestFixture::use_graphs)
        {
            // Default stream does not support hipGraph stream capture, so create one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // numeric_limits<T>::denorm_min() does not work...
        T lowest = static_cast<T>(
            -1.0
            * static_cast<double>(
                rocprim::numeric_limits<
                    T>::min())); // smallest (closest to zero) normal (negative) non-zero number

        // Generate data
        std::vector<T> input(size, lowest);

        common::device_ptr<T> d_input(input);
        common::device_ptr<U> d_output(1);

        // Calculate expected results on host mathematically (instead of using reduce on host)
        std::vector<U> expected
            = test_utils::host_reduce(input.begin(), input.end(), rocprim::plus<U>());

        // Get size of d_temp_storage
        size_t temp_storage_size_bytes;
        HIP_CHECK(rocprim::reduce<Config>(
            nullptr,
            temp_storage_size_bytes,
            d_input.get(),
            test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
            input.size(),
            rocprim::plus<U>(),
            stream,
            debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        test_utils::GraphHelper gHelper;
        if(TestFixture::use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK(rocprim::reduce<Config>(
            d_temp_storage.get(),
            temp_storage_size_bytes,
            d_input.get(),
            test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
            input.size(),
            rocprim::plus<U>(),
            stream,
            debug_synchronous));

        if(TestFixture::use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        const auto output = d_output.load();

        // Check if output values are as expected
        //ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], expected, precision));
        if(size == 0)
        {
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(output[0], U{}, precision));
        }
        else
        {
            for(size_t i = 0; i < output.size(); ++i)
            {
                ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
                    output[i],
                    expected[i],
                    test_utils::precision<U>
                        * std::max(0.0, test_utils::precision<U> / 2.0 * (size - 1 - i) - 1.0)));
            }
        }
        if (TestFixture::use_graphs)
        {
            gHelper.cleanupGraphHelper();
            HIP_CHECK(hipStreamDestroy(stream));
        }
    }
}

TYPED_TEST(RocprimDeviceReduceTests, ReduceMinimum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    using binary_op_type = rocprim::minimum<U>;

    static constexpr bool use_identity_iterator = TestFixture::use_identity_iterator;
    using Config = size_limit_config_t<TestFixture::size_limit>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if (TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100, seed_value);

            common::device_ptr<T> d_input(input);
            common::device_ptr<U> d_output(1);

            // reduce function
            binary_op_type min_op;

            // Calculate expected results on host
            U expected = U(rocprim::numeric_limits<U>::max());
            for(unsigned int i = 0; i < input.size(); i++)
            {
                expected = min_op(expected, input[i]);
            }

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::reduce<Config>(
                nullptr,
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                rocprim::numeric_limits<U>::max(),
                input.size(),
                rocprim::minimum<U>(),
                stream,
                TestFixture::debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(rocprim::reduce<Config>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                d_input.get(),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_output.get()),
                rocprim::numeric_limits<U>::max(),
                input.size(),
                rocprim::minimum<U>(),
                stream,
                TestFixture::debug_synchronous));

            // Copy output to host
            std::vector<U> output(1, U(0));
            HIP_CHECK(hipMemcpyAsync(output.data(),
                                     d_output.get(),
                                     output.size() * sizeof(U),
                                     hipMemcpyDeviceToHost,
                                     stream));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream, true, false);
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Check if output values are as expected
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_near(
                output[0],
                expected,
                std::is_same<T, U>::value
                    ? 0
                    : std::max(test_utils::precision<T>, test_utils::precision<U>)));

            if (TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}
