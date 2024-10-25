// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

// required test headers
#include "indirect_iterator.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

// required rocprim headers
#include <rocprim/device/config_types.hpp>
// #include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_find_first_of.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/counting_iterator.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include <cassert>
#include <cstddef>

// Params for tests
template<class Type,
         class KeyType            = Type,
         class OutputType         = size_t,
         class CompareFunction    = rocprim::equal_to<Type>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceFindFirstOfParams
{
    using type                                  = Type;
    using key_type                              = KeyType;
    using output_type                           = OutputType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

struct custom_compare1
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
    bool operator()(const T& a, const U& b) const
    {
        // Since data is random, the chance of equality is negligible for floating point numbers
        return static_cast<int>(a * 1.234) == static_cast<int>(b * 1.234);
    }
};

struct custom_compare2
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
    bool operator()(test_utils::custom_test_type<T> a, test_utils::custom_test_type<U> b)
    {
        return a.x == b.x;
    }
};

template<class Params>
class RocprimDeviceFindFirstOfTests : public ::testing::Test
{
public:
    using type                                  = typename Params::type;
    using key_type                              = typename Params::key_type;
    using output_type                           = typename Params::output_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    static constexpr bool debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

using RocprimDeviceFindFirstOfTestsParams
    = ::testing::Types<DeviceFindFirstOfParams<int>,
                       DeviceFindFirstOfParams<int64_t,
                                               int,
                                               unsigned int,
                                               rocprim::equal_to<int64_t>,
                                               rocprim::default_config,
                                               true,
                                               true>,
                       DeviceFindFirstOfParams<uint8_t,
                                               uint8_t,
                                               unsigned int,
                                               rocprim::equal_to<void>,
                                               rocprim::default_config,
                                               true,
                                               false>,
                       DeviceFindFirstOfParams<float,
                                               double,
                                               size_t,
                                               custom_compare1,
                                               rocprim::default_config,
                                               false,
                                               true>,
                       DeviceFindFirstOfParams<test_utils::custom_test_type<int8_t>,
                                               test_utils::custom_test_type<int8_t>,
                                               size_t,
                                               custom_compare2,
                                               rocprim::default_config,
                                               false,
                                               true>>;

TYPED_TEST_SUITE(RocprimDeviceFindFirstOfTests, RocprimDeviceFindFirstOfTestsParams);

TYPED_TEST(RocprimDeviceFindFirstOfTests, FindFirstOf)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type             = typename TestFixture::type;
    using key_type         = typename TestFixture::key_type;
    using output_type      = typename TestFixture::output_type;
    using compare_function = typename TestFixture::compare_function;
    using config           = typename TestFixture::config;

    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            const size_t keys_size
                = std::sqrt(test_utils::get_random_value<size_t>(0, size, seed_value));

            // Starting point is an appoximate position of the first match we want to test for
            for(double starting_point : {0.0, 0.234, 0.876, 1.0, 100.0})
            {
                SCOPED_TRACE(testing::Message() << "with starting_point = " << starting_point);

                hipStream_t stream = 0; // default
                if(TestFixture::use_graphs)
                {
                    // Default stream does not support hipGraph stream capture, so create one
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

                // Generate data
                auto keys = test_utils::get_random_data<key_type>(keys_size, 0, 10, seed_value + 1);

                std::vector<type> input(size);
                // Generate the input data in such a way that it does not contain any values from
                // keys before the starting point
                const size_t size1
                    = starting_point >= 1.0 ? size : static_cast<size_t>(size * starting_point);
                const size_t size2 = size - size1;
                if(size1 > 0)
                {
                    auto input1 = test_utils::get_random_data<type>(size1, 20, 100, seed_value + 2);
                    std::copy(input1.begin(), input1.end(), input.begin());
                }
                if(size2 > 0)
                {
                    auto input2 = test_utils::get_random_data<type>(size2, 0, 100, seed_value + 3);
                    std::copy(input2.begin(), input2.end(), input.begin() + size1);
                }

                // Explicitly test for boundary cases
                if(size > 0 && keys_size > 0)
                {
                    if(starting_point == 0.0)
                    {
                        input[0] = keys[keys_size - 1];
                    }
                    else if(starting_point == 1.0)
                    {
                        input[size - 1] = keys[0];
                    }
                }

                type*        d_input;
                key_type*    d_keys;
                output_type* d_output;
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(*d_keys)));
                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(*d_output)));

                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    input.size() * sizeof(*d_input),
                                    hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_keys,
                                    keys.data(),
                                    keys.size() * sizeof(*d_keys),
                                    hipMemcpyHostToDevice));

                const auto input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);
                const auto keys_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_keys);

                // compare function
                compare_function compare_op;

                // temp storage
                size_t temp_storage_size_bytes;
                void*  d_temp_storage = nullptr;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::find_first_of<config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         input_it,
                                                         keys_it,
                                                         d_output,
                                                         input.size(),
                                                         keys.size(),
                                                         compare_op,
                                                         stream,
                                                         debug_synchronous));

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                hipGraph_t graph;
                if(TestFixture::use_graphs)
                {
                    graph = test_utils::createGraphHelper(stream);
                }

                // Run
                HIP_CHECK(rocprim::find_first_of<config>(d_temp_storage,
                                                         temp_storage_size_bytes,
                                                         input_it,
                                                         keys_it,
                                                         d_output,
                                                         input.size(),
                                                         keys.size(),
                                                         compare_op,
                                                         stream,
                                                         debug_synchronous));

                hipGraphExec_t graph_instance;
                if(TestFixture::use_graphs)
                {
                    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
                }

                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());

                output_type output;

                // Copy output to host
                HIP_CHECK(hipMemcpy(&output, d_output, sizeof(*d_output), hipMemcpyDeviceToHost));

                // Check
                auto expected = std::find_first_of(input.begin(),
                                                   input.end(),
                                                   keys.begin(),
                                                   keys.end(),
                                                   compare_op)
                                - input.begin();

                ASSERT_EQ(output, expected);

                HIP_CHECK(hipFree(d_input));
                HIP_CHECK(hipFree(d_keys));
                HIP_CHECK(hipFree(d_output));
                HIP_CHECK(hipFree(d_temp_storage));

                if(TestFixture::use_graphs)
                {
                    test_utils::cleanupGraphHelper(graph, graph_instance);
                    HIP_CHECK(hipStreamDestroy(stream));
                }
            }
        }
    }
}

TEST(RocprimDeviceFindFirstOfTests, LargeIndices)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using output_type = size_t;
    using config      = rocprim::default_config;

    constexpr bool debug_synchronous = false;

    for(size_t size : test_utils::get_large_sizes(seeds[0]))
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_t keys_size = 12;

        for(double starting_point : {0.0, 0.12, 0.78, 1.1})
        {
            SCOPED_TRACE(testing::Message() << "with starting_point = " << starting_point);

            hipStream_t stream = 0; // default

            output_type* d_output;
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(*d_output)));

            const output_type expected
                = std::min(size, static_cast<output_type>(starting_point * size));

            auto input_it = rocprim::make_counting_iterator(size_t(0));
            auto keys_it  = rocprim::make_counting_iterator(expected);

            rocprim::equal_to<size_t> compare_op;

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::find_first_of<config>(d_temp_storage,
                                                     temp_storage_size_bytes,
                                                     input_it,
                                                     keys_it,
                                                     d_output,
                                                     size,
                                                     keys_size,
                                                     compare_op,
                                                     stream,
                                                     debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            // Run
            HIP_CHECK(rocprim::find_first_of<config>(d_temp_storage,
                                                     temp_storage_size_bytes,
                                                     input_it,
                                                     keys_it,
                                                     d_output,
                                                     size,
                                                     keys_size,
                                                     compare_op,
                                                     stream,
                                                     debug_synchronous));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host and check
            output_type output;
            HIP_CHECK(hipMemcpy(&output, d_output, sizeof(*d_output), hipMemcpyDeviceToHost));
            ASSERT_EQ(output, expected);

            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));
        }
    }
}
