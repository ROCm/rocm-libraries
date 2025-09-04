// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"

#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_device_ptr.hpp"
#include "../../common/utils_half.hpp"

#include "test_seed.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"

#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_search_n.hpp>
#include <rocprim/device/device_search_n_config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <stdint.h>
#include <vector>

template<class T>
using limit_type = rocprim::numeric_limits<T>;

template<class InputIterator,
         class OutputIterator     = size_t,
         class BinaryPredicate    = rocprim::equal_to<InputIterator>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceSearchNParams
{
    using input_type                            = InputIterator;
    using output_type                           = OutputIterator;
    using op_type                               = BinaryPredicate;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDeviceSearchNTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    using op_type                               = typename Params::op_type;
    using config                                = typename Params::config;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
    static constexpr bool debug_synchronous     = false;
};

// Custom types
using custom_int2        = common::custom_type<int, int, true>;
using custom_double2     = common::custom_type<double, double, true>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 4>;

// Custom configs
using custom_config_0 = rocprim::search_n_config<256, 4, 6>;

using RocprimDeviceSearchNTestsParams = ::testing::Types<
    // Tests with default configuration
    DeviceSearchNParams<int8_t>,
    DeviceSearchNParams<int>,
    DeviceSearchNParams<rocprim::half>,
    DeviceSearchNParams<rocprim::bfloat16>,
    DeviceSearchNParams<float>,
    DeviceSearchNParams<double>,
    // Tests for custom types
    DeviceSearchNParams<custom_int2>,
    DeviceSearchNParams<custom_double2>,
    DeviceSearchNParams<custom_int64_array>,
    // Tests for supported config structs
    DeviceSearchNParams<rocprim::bfloat16,
                        size_t,
                        rocprim::equal_to<rocprim::bfloat16>,
                        custom_config_0>,
    // Tests for hipGraph support
    DeviceSearchNParams<unsigned int,
                        size_t,
                        rocprim::equal_to<unsigned int>,
                        rocprim::default_config,
                        true>,
    // Tests for when output's value_type is void
    DeviceSearchNParams<int, size_t, rocprim::equal_to<int>, rocprim::default_config, false, true>>;

TYPED_TEST_SUITE(RocprimDeviceSearchNTests, RocprimDeviceSearchNTestsParams);

TYPED_TEST(RocprimDeviceSearchNTests, RandomTest)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            size_t      count  = test_utils::get_random_value<size_t>(0, size, ++seed_value);
            size_t      temp_storage_size;
            input_type  h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type> h_input
                = test_utils::get_random_data<input_type>(size,
                                                          0,
                                                          limit_type<input_type>::max(),
                                                          ++seed_value);
            auto index = 0;
            if(size > count)
            {
                index = test_utils::get_random_value<size_t>(0, size - 1 - count, ++seed_value);
                std::fill(h_input.begin() + index, h_input.begin() + index + count, h_value);
            }

            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));
            common::device_ptr<void> d_temp_storage(temp_storage_size);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();
            h_output = d_output.load()[0];
            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, MaxCount)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            size_t      count  = size;
            size_t      temp_storage_size;
            input_type  h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type>         h_input(size, h_value);
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));
            common::device_ptr<void> d_temp_storage(temp_storage_size);

            test_utils::GraphHelper gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, MinCount)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            size_t      count  = 0;
            size_t      temp_storage_size;
            input_type  h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type>         h_input(size, h_value);
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }
            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, SmallCount)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t                     stream = 0; // default
            size_t                          count  = 0;
            size_t                          temp_storage_size;
            input_type                      h_value{1};
            input_type                      h_noise{0};
            std::vector<input_type>         h_input(size, h_noise);
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            if(size > 0 && size - 1 > 0)
            {
                count             = 1;
                size_t random_idx = test_utils::get_random_value<size_t>(0, size - 1, ++seed_value);
                h_input[random_idx] = h_noise;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }
            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, StartFromBegin)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t             stream = 0; // default
            size_t                  count  = size / 2;
            size_t                  temp_storage_size;
            input_type              h_value{1};
            std::vector<input_type> h_input(size);
            std::fill(h_input.begin(), h_input.begin() + (size - count), h_value);
            std::fill(h_input.begin() + count, h_input.end(), 0);
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, StartFromMiddle)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t             stream = 0; // default
            size_t                  count  = size / 2;
            size_t                  temp_storage_size;
            input_type              h_value{1};
            std::vector<input_type> h_input(size);
            std::fill(h_input.begin(), h_input.begin() + (size - count), 0);
            std::fill(h_input.begin() + count, h_input.end(), h_value);
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, StartFromEnd)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            size_t      count  = test_utils::get_random_value<size_t>(0, size, ++seed_value);
            size_t      temp_storage_size;
            input_type  h_value{1};
            std::vector<input_type> h_input(size);
            std::fill(h_input.begin(), h_input.begin() + (size - count), 0);
            std::fill(h_input.begin() + (size - count), h_input.end(), h_value);
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, StartFromEndButFail)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            size_t      count  = test_utils::get_random_value<size_t>(0, size, ++seed_value);
            size_t      temp_storage_size;
            input_type  h_value{1};
            std::vector<input_type> h_input(size);
            std::fill(h_input.begin(), h_input.begin() + (size - count), 0);
            std::fill(h_input.begin() + (size - count), h_input.end(), h_value);
            if(count + 2 <= size)
            {
                count += 2;
            }
            output_type                     h_output;
            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, NoiseTest_1block)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            using wrapped_config = rocprim::detail::wrapped_search_n_config<config, input_type>;
            size_t                       temp_storage_size;
            hipStream_t                  stream = 0; // default
            rocprim::detail::target_arch target_arch;
            HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
            const auto params
                = rocprim::detail::dispatch_target_arch<wrapped_config, false>(target_arch);
            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const unsigned int items_per_block  = block_size * items_per_thread;

            /// Will do test like this:
            ///     |----------------------------------- size ------------------------------------|
            ///     |----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/|
            ///     |--------count--------|
            ///     |111111111111111111110|111111111111111111110|111111111111111111111|11111111111|

            size_t                  count = 0;
            input_type              h_value{1};
            input_type              h_noise{0};
            std::vector<input_type> h_input(size, h_value);
            output_type             h_output;

            if(size > items_per_block)
            {
                count            = items_per_block;
                size_t cur_tile  = 0;
                size_t last_tile = size / count - 1;
                while(cur_tile != last_tile)
                {
                    h_input[cur_tile * count + count - 1] = h_noise;
                    ++cur_tile;
                }
            }

            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, NoiseTest_2block)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            using wrapped_config = rocprim::detail::wrapped_search_n_config<config, input_type>;
            size_t                       temp_storage_size;
            hipStream_t                  stream = 0; // default
            rocprim::detail::target_arch target_arch;
            HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
            const auto params
                = rocprim::detail::dispatch_target_arch<wrapped_config, false>(target_arch);
            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const unsigned int items_per_block  = block_size * items_per_thread;

            /// Will do test like this:
            ///     |----------------------------------- size ------------------------------------|
            ///     |----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/|
            ///     |--------count------------------------------|
            ///     |1111111111111111111111111111111111111111110|111111111111111111111111111111111|

            size_t                  count = 0;
            input_type              h_value{1};
            input_type              h_noise{0};
            std::vector<input_type> h_input(size, h_value);
            output_type             h_output;

            if(size > items_per_block * 2)
            {
                count            = items_per_block * 2;
                size_t cur_tile  = 0;
                size_t last_tile = size / count - 1;
                while(cur_tile != last_tile)
                {
                    h_input[cur_tile * count + count - 1] = h_noise;
                    ++cur_tile;
                }
            }

            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, NoiseTest_3block)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            using wrapped_config = rocprim::detail::wrapped_search_n_config<config, input_type>;
            size_t                       temp_storage_size;
            hipStream_t                  stream = 0; // default
            rocprim::detail::target_arch target_arch;
            HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
            const auto params
                = rocprim::detail::dispatch_target_arch<wrapped_config, false>(target_arch);
            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const unsigned int items_per_block  = block_size * items_per_thread;

            /// Will do test like this:
            ///     |----------------------------------- size ----------------------------------------------------------------------------------------------------------------|
            ///     |----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|
            ///     |------------------------------count------------------------------|------------------------------count------------------------------|
            ///     |11111111111111111111111111111111111111111111111111111111111111110|11111111111111111111111111111111111111111111111111111111111111111|111111111111111111111|

            size_t                  count = 0;
            input_type              h_value{1};
            input_type              h_noise{0};
            std::vector<input_type> h_input(size, h_value);
            output_type             h_output;

            if(size > items_per_block * 3)
            {
                count            = items_per_block * 3;
                size_t cur_tile  = 0;
                size_t last_tile = size / count - 1;
                while(cur_tile != last_tile)
                {
                    h_input[cur_tile * count + count - 1] = h_noise;
                    ++cur_tile;
                }
            }

            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, MultiResult1)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {
            using wrapped_config = rocprim::detail::wrapped_search_n_config<config, input_type>;
            size_t                       temp_storage_size;
            hipStream_t                  stream = 0; // default
            rocprim::detail::target_arch target_arch;
            HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
            const auto params
                = rocprim::detail::dispatch_target_arch<wrapped_config, false>(target_arch);
            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const unsigned int items_per_block  = block_size * items_per_thread;

            /// Will do test like this:
            ///     |----------------------------------- size ------------------------------------------------------------------------------------------------------------------------------------------------------------------...
            ///     |----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|----- Item/block ----|
            ///     |------------------------------count----------------------------| |------------------------------count----------------------------| |------------------------------count----------------------------|
            ///     |01111111111111111111111111111111111111111111111111111111111111110|11111111111111111111111111111111111111111111111111111111111111110|11111111111111111111111111111111111111111111111111111111111111111|11111...

            size_t                  count = 0;
            input_type              h_value{1};
            input_type              h_noise{0};
            std::vector<input_type> h_input(size, h_value);
            output_type             h_output;

            if(size > items_per_block * 3)
            {
                count            = items_per_block * 3;
                size_t cur_tile  = 0;
                size_t last_tile = size / count - 1;
                while(cur_tile != last_tile)
                {
                    h_input[cur_tile * count + count - 1] = h_noise;
                    ++cur_tile;
                }
                count -= 1;
                h_input[0] = h_noise;
            }

            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, MultiResult2)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool debug_synchronous = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        for(const auto size : test_utils::get_sizes(seed_value))
        {
            using wrapped_config = rocprim::detail::wrapped_search_n_config<config, input_type>;
            size_t                       temp_storage_size;
            hipStream_t                  stream = 0; // default
            rocprim::detail::target_arch target_arch;
            HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
            const auto params
                = rocprim::detail::dispatch_target_arch<wrapped_config, false>(target_arch);
            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const unsigned int items_per_block  = block_size * items_per_thread;

            size_t                  count = 0;
            input_type              h_value{1};
            input_type              h_noise{0};
            std::vector<input_type> h_input(size);
            output_type             h_output;

            if(size > items_per_block)
            {
                count        = items_per_block;
                size_t start = size - 1 - count;
                std::fill(h_input.begin() + start, h_input.end(), h_value);
                for(size_t i = 0; i < start; i++)
                {
                    if(!(i % 3))
                    {
                        h_input[i] = h_noise;
                    }
                }
            }

            common::device_ptr<input_type>  d_input(h_input);
            common::device_ptr<input_type>  d_value(std::vector<input_type>({h_value}));
            common::device_ptr<output_type> d_output(1);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }
            // get size
            HIP_CHECK(rocprim::search_n<config>(0,
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                nullptr));

            common::device_ptr<void> d_temp_storage(temp_storage_size);
            test_utils::GraphHelper  gHelper;
            if(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}
