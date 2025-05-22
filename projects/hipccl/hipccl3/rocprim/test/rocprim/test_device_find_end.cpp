// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// required test headers
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

// required common headers
#include "../../common/utils_device_ptr.hpp"

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_find_end.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <stdint.h>
#include <vector>

// Params for tests
template<class ValueType,
         class KeyType            = ValueType,
         class IndexType          = size_t,
         class CompareFunction    = rocprim::equal_to<KeyType>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceFindEndParams
{
    using value_type                            = ValueType;
    using key_type                              = KeyType;
    using index_type                            = IndexType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDeviceFindEndTests : public ::testing::Test
{
public:
    using value_type                            = typename Params::value_type;
    using key_type                              = typename Params::key_type;
    using index_type                            = typename Params::index_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

using RocprimDeviceFindEndTestsParams = ::testing::Types<
    DeviceFindEndParams<unsigned short>,
    DeviceFindEndParams<signed char>,
    DeviceFindEndParams<int, int, unsigned int>,
    DeviceFindEndParams<int, int, int>,
    DeviceFindEndParams<common::custom_type<int, int, true>>,
    DeviceFindEndParams<unsigned long>,
    DeviceFindEndParams<long long>,
    DeviceFindEndParams<float>,
    DeviceFindEndParams<int8_t>,
    DeviceFindEndParams<uint8_t>,
    DeviceFindEndParams<rocprim::half, rocprim::half, size_t, rocprim::equal_to<rocprim::half>>,
    DeviceFindEndParams<rocprim::bfloat16,
                        rocprim::bfloat16,
                        size_t,
                        rocprim::equal_to<rocprim::bfloat16>>,
    DeviceFindEndParams<short>,
    DeviceFindEndParams<double>,
    DeviceFindEndParams<common::custom_type<float, float, true>>,
    DeviceFindEndParams<test_utils::custom_float_type>,
    DeviceFindEndParams<test_utils::custom_test_array_type<int, 4>>,
    DeviceFindEndParams<int, int, size_t, rocprim::equal_to<int>, rocprim::default_config, true>,
    DeviceFindEndParams<int, int, size_t, rocprim::greater_equal<int>, rocprim::default_config>,
    DeviceFindEndParams<int, int, size_t, rocprim::greater<int>, rocprim::default_config>,
    DeviceFindEndParams<int,
                        int,
                        size_t,
                        rocprim::equal_to<int>,
                        rocprim::default_config,
                        false,
                        true>,
    DeviceFindEndParams<int,
                        int,
                        size_t,
                        rocprim::equal_to<int>,
                        rocprim::search_config<64, 16, 1024>,
                        false,
                        false>>;

TYPED_TEST_SUITE(RocprimDeviceFindEndTests, RocprimDeviceFindEndTestsParams);

TYPED_TEST(RocprimDeviceFindEndTests, FindEnd)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using value_type                     = typename TestFixture::value_type;
    using key_type                       = typename TestFixture::key_type;
    using index_type                     = typename TestFixture::index_type;
    using compare_function               = typename TestFixture::compare_function;
    using config                         = typename TestFixture::config;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    std::vector<size_t> key_sizes = {0, 1, 10, 1000, 10000};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            for(size_t key_size : key_sizes)
            {
                SCOPED_TRACE(testing::Message() << "with key size = " << key_size);

                if(TestFixture::use_graphs)
                {
                    // Default stream does not support hipGraph stream capture, so create one
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

                size_t pattern = 0;
                if(size > 0)
                {
                    pattern = test_utils::get_random_value<size_t>(0, size - 1, seed_value);
                }

                SCOPED_TRACE(testing::Message() << "with index = " << pattern);

                // Generate data
                std::vector<value_type> input;
                if(rocprim::is_floating_point<value_type>::value)
                {
                    input = test_utils::get_random_data_wrapped<value_type>(size,
                                                                            -1000,
                                                                            1000,
                                                                            seed_value);
                }
                else
                {
                    input = test_utils::get_random_data_wrapped<value_type>(
                        size,
                        rocprim::numeric_limits<value_type>::min(),
                        rocprim::numeric_limits<value_type>::max(),
                        seed_value);
                }

                std::vector<key_type> keys(key_size);
                if(pattern + key_size < size)
                {
                    keys.assign(input.begin() + pattern, input.begin() + pattern + key_size);
                }
                else
                {
                    keys.assign(input.begin() + pattern, input.end());
                }

                common::device_ptr<value_type> d_input(input);
                common::device_ptr<key_type>   d_keys(keys);
                common::device_ptr<index_type> d_output(1);

                const auto input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());
                const auto keys_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_keys.get());
                const auto output_keys
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_output.get());

                // compare function
                compare_function compare_op;

                // Get size of d_temp_storage
                size_t temp_storage_size_bytes;
                HIP_CHECK(rocprim::find_end<config>(nullptr,
                                                    temp_storage_size_bytes,
                                                    input_it,
                                                    keys_it,
                                                    output_keys,
                                                    input.size(),
                                                    keys.size(),
                                                    compare_op,
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
                HIP_CHECK(rocprim::find_end<config>(d_temp_storage.get(),
                                                    temp_storage_size_bytes,
                                                    input_it,
                                                    keys_it,
                                                    output_keys,
                                                    input.size(),
                                                    keys.size(),
                                                    compare_op,
                                                    stream,
                                                    debug_synchronous));

                if(TestFixture::use_graphs)
                {
                    gHelper.createAndLaunchGraph(stream);
                }

                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());

                // Copy output to host
                const auto output = d_output.load()[0];

                index_type expected = std::find_end(input.begin(),
                                                    input.end(),
                                                    keys.begin(),
                                                    keys.end(),
                                                    compare_op)
                                      - input.begin();

                ASSERT_EQ(output, expected);

                if(TestFixture::use_graphs)
                {
                    gHelper.cleanupGraphHelper();
                    HIP_CHECK(hipStreamDestroy(stream));
                }
            }
        }
    }
}

TYPED_TEST(RocprimDeviceFindEndTests, FindEndRepetition)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using value_type                     = typename TestFixture::value_type;
    using key_type                       = typename TestFixture::key_type;
    using index_type                     = typename TestFixture::index_type;
    using compare_function               = typename TestFixture::compare_function;
    using config                         = typename TestFixture::config;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    size_t key_size = 10;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            if(size < key_size)
            {
                continue;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            // Generate data
            std::vector<key_type> keys;
            if(rocprim::is_floating_point<value_type>::value)
            {
                keys = test_utils::get_random_data_wrapped<key_type>(key_size,
                                                                     -1000,
                                                                     1000,
                                                                     seed_value);
            }
            else
            {
                keys = test_utils::get_random_data_wrapped<key_type>(
                    key_size,
                    rocprim::numeric_limits<key_type>::min(),
                    rocprim::numeric_limits<key_type>::max(),
                    seed_value);
            }

            std::vector<value_type> input(size);
            for(size_t i = 0; i < size / key_size; i++)
            {
                std::copy(keys.begin(), keys.end(), input.begin() + i * key_size);
            }

            common::device_ptr<value_type> d_input(input);
            common::device_ptr<key_type>   d_keys(keys);
            common::device_ptr<index_type> d_output(1);

            const auto input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());

            // compare function
            compare_function compare_op;

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(rocprim::find_end<config>(nullptr,
                                                temp_storage_size_bytes,
                                                input_it,
                                                d_keys.get(),
                                                d_output.get(),
                                                input.size(),
                                                keys.size(),
                                                compare_op,
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
            HIP_CHECK(rocprim::find_end<config>(d_temp_storage.get(),
                                                temp_storage_size_bytes,
                                                input_it,
                                                d_keys.get(),
                                                d_output.get(),
                                                input.size(),
                                                keys.size(),
                                                compare_op,
                                                stream,
                                                debug_synchronous));

            if(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load()[0];

            index_type expected
                = std::find_end(input.begin(), input.end(), keys.begin(), keys.end(), compare_op)
                  - input.begin();

            ASSERT_EQ(output, expected);

            if(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}
