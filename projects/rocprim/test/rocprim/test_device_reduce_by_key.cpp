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
#include "../../common/utils_data_generation.hpp"
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
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/constant_iterator.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

template<class Key,
         class Value,
         class ReduceOp,
         unsigned int MinSegmentLength,
         unsigned int MaxSegmentLength,
         class Aggregate          = Value,
         class KeyCompareFunction = ::rocprim::equal_to<Key>,
         // Tests output iterator with void value_type (OutputIterator concept)
         bool UseIdentityIterator = false,
         bool UseGraphs           = false,
         bool Deterministic       = false,
         typename Config          = rocprim::default_config>
struct params
{
    using key_type = Key;
    using value_type = Value;
    using reduce_op_type = ReduceOp;
    static constexpr unsigned int min_segment_length = MinSegmentLength;
    static constexpr unsigned int max_segment_length = MaxSegmentLength;
    using aggregate_type = Aggregate;
    using key_compare_op = KeyCompareFunction;
    static constexpr bool use_identity_iterator = UseIdentityIterator;
    static constexpr bool use_graphs = UseGraphs;
    static constexpr bool deterministic              = Deterministic;
    using config                                     = Config;
};

template<class Params>
class RocprimDeviceReduceByKey : public ::testing::Test {
public:
    using params = Params;
};

struct custom_reduce_op1
{
    template<class T>
    ROCPRIM_HOST_DEVICE
    T operator()(T a, T b)
    {
        return a + b;
    }
};

template<class T>
struct custom_key_compare_op1
{
    ROCPRIM_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return static_cast<int>(a / 10) == static_cast<int>(b / 10);
    }
};

using custom_int2    = common::custom_type<int, int, true>;
using custom_double2 = common::custom_type<double, double, true>;

// clang-format off
using Params = ::testing::Types<
    params<int, int, rocprim::plus<int>, 1, 1, int, rocprim::equal_to<int>, true>,
    params<double, int, rocprim::plus<int>, 3, 5, long long, custom_key_compare_op1<double>, false, false, true>,
    params<float, custom_double2, rocprim::minimum<custom_double2>, 1, 10000>,
    params<custom_double2, custom_int2, rocprim::plus<custom_int2>, 1, 10>,
    params<unsigned long long, float, rocprim::minimum<float>, 1, 30>,
    // with block size different than ROCPRIM_DEFAULT_MAX_BLOCK_SIZE and non-power-of-two items per thread
    params<int, rocprim::half, rocprim::minimum<rocprim::half>, 15, 100, rocprim::half, rocprim::equal_to<int>, false, false, true, rocprim::reduce_by_key_config<128, 3>>,
    // half should be supported, but is missing some key operators.
    // we should uncomment these, as soon as these are implemented and the tests compile and work as intended.
    //params<rocprim::half, rocprim::half, rocprim::minimum<rocprim::half>, 15, 100>,
    params<int, rocprim::bfloat16, rocprim::minimum<rocprim::bfloat16>, 15, 100>,
    params<rocprim::bfloat16, rocprim::bfloat16, rocprim::minimum<rocprim::bfloat16>, 15, 100>,
    params<int, unsigned int, rocprim::maximum<unsigned int>, 20, 100>,
    params<float, long long, rocprim::maximum<unsigned long long>, 100, 400, long long, custom_key_compare_op1<float>>,
    params<unsigned int, unsigned char, rocprim::plus<unsigned char>, 200, 600>,
    params<double, int, rocprim::plus<int>, 100, 2000, double, custom_key_compare_op1<double>, false, false, true>,
    params<int8_t, int8_t, rocprim::maximum<int8_t>, 20, 100>,
    params<uint8_t, uint8_t, rocprim::maximum<uint8_t>, 20, 100>,
    params<char, rocprim::half, rocprim::maximum<rocprim::half>, 123, 1234>,
    params<char, rocprim::bfloat16, rocprim::maximum<rocprim::bfloat16>, 123, 1234>,
    params<custom_int2, unsigned int, rocprim::plus<unsigned int>, 1000, 5000>,
    params<unsigned int, int, rocprim::plus<int>, 2048, 2048>,
    params<long long, short, rocprim::plus<long long>, 1000, 10000, long long>,
    params<unsigned int, double, rocprim::minimum<double>, 1000, 50000>,
    params<unsigned long long, unsigned long long, rocprim::plus<unsigned long long>, 100000, 100000>,
    params<test_utils::custom_test_array_type<double, 8>, unsigned long, rocprim::plus<>, 69, 420>,
    params<int, int, rocprim::plus<int>, 1, 10, int, ::rocprim::equal_to<int>, false, true>
>;
// clang-format on

template<bool Deterministic, typename Config = rocprim::default_config, typename... Args>
constexpr hipError_t invoke_reduce_by_key(Args&&... args)
{
    if(Deterministic)
    {
        return rocprim::deterministic_reduce_by_key<Config>(std::forward<Args>(args)...);
    }
    else
    {
        return rocprim::reduce_by_key<Config>(std::forward<Args>(args)...);
    }
}

TYPED_TEST_SUITE(RocprimDeviceReduceByKey, Params);

TYPED_TEST(RocprimDeviceReduceByKey, ReduceByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = typename TestFixture::params::key_type;
    using value_type = typename TestFixture::params::value_type;
    using aggregate_type = typename TestFixture::params::aggregate_type;
    using reduce_op_type = typename TestFixture::params::reduce_op_type;
    using key_compare_op_type = typename TestFixture::params::key_compare_op;
    using key_inner_type = typename test_utils::inner_type<key_type>::type;
    using key_distribution_type = typename std::conditional<
        std::is_floating_point<key_inner_type>::value,
        std::uniform_real_distribution<key_inner_type>,
        typename std::conditional<
            common::is_valid_for_int_distribution<key_inner_type>::value,
            common::uniform_int_distribution<key_inner_type>,
            typename std::conditional<rocprim::is_signed<key_inner_type>::value,
                                      common::uniform_int_distribution<int>,
                                      common::uniform_int_distribution<unsigned int>>::type>::
            type>::type;
    using config = typename TestFixture::params::config;

    constexpr bool use_identity_iterator = TestFixture::params::use_identity_iterator;
    constexpr bool deterministic         = TestFixture::params::deterministic;
    const bool debug_synchronous = false;

    reduce_op_type reduce_op;
    key_compare_op_type key_compare_op;

    const unsigned int seed = 123;
    std::default_random_engine gen(seed);

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default
            if (TestFixture::params::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            const bool use_unique_keys = bool(test_utils::get_random_value<int>(0, 1, seed_value));

            // Generate data and calculate expected results
            std::vector<key_type> unique_expected;
            std::vector<aggregate_type> aggregates_expected;
            size_t unique_count_expected = 0;

            std::vector<key_type>                    keys_input(size);
            key_distribution_type                    key_delta_dis(1, 5);
            common::uniform_int_distribution<size_t> key_count_dis(
                TestFixture::params::min_segment_length,
                TestFixture::params::max_segment_length);
            std::vector<value_type> values_input
                = test_utils::get_random_data_wrapped<value_type>(size, 0, 100, seed_value);

            size_t offset = 0;
            key_type prev_key    = static_cast<key_type>(key_distribution_type(0, 100)(gen));
            key_type current_key = static_cast<key_type>(prev_key + key_delta_dis(gen));
            while(offset < size)
            {
                const size_t key_count = key_count_dis(gen);

                const size_t end = std::min(size, offset + key_count);
                for(size_t i = offset; i < end; i++)
                {
                    keys_input[i] = current_key;
                }
                aggregate_type aggregate = values_input[offset];
                for(size_t i = offset + 1; i < end; i++)
                {
                    aggregate = reduce_op(aggregate, static_cast<aggregate_type>(values_input[i]));
                }

                // The first key of the segment must be written into unique
                // (it may differ from other keys in case of custom key compraison operators)
                if(unique_count_expected == 0 || !key_compare_op(prev_key, current_key))
                {
                    unique_expected.push_back(current_key);
                    unique_count_expected++;
                    aggregates_expected.push_back(aggregate);
                }
                else
                {
                    aggregates_expected.back() = reduce_op(aggregates_expected.back(), aggregate);
                }

                if (use_unique_keys)
                {
                    prev_key = current_key;
                    // e.g. 1,1,1,2,5,5,8,8,8
                    current_key = current_key + key_delta_dis(gen);
                }
                else
                {
                    // e.g. 1,1,5,1,5,5,5,1
                    std::swap(current_key, prev_key);
                }

                offset += key_count;
            }

            common::device_ptr<key_type>   d_keys_input(keys_input);
            common::device_ptr<value_type> d_values_input(values_input);

            common::device_ptr<key_type>       d_unique_output(unique_count_expected);
            common::device_ptr<aggregate_type> d_aggregates_output(unique_count_expected);
            common::device_ptr<unsigned int>   d_unique_count_output(1);

            size_t temporary_storage_bytes;
            HIP_CHECK((invoke_reduce_by_key<deterministic, config>(
                nullptr,
                temporary_storage_bytes,
                d_keys_input.get(),
                d_values_input.get(),
                size,
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(d_unique_output.get()),
                test_utils::wrap_in_identity_iterator<use_identity_iterator>(
                    d_aggregates_output.get()),
                d_unique_count_output.get(),
                reduce_op,
                key_compare_op,
                stream,
                debug_synchronous)));

            ASSERT_GT(temporary_storage_bytes, 0);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            test_utils::GraphHelper gHelper;
            if(TestFixture::params::use_graphs)
            {
               gHelper.startStreamCapture(stream);
            }

            HIP_CHECK((invoke_reduce_by_key<deterministic, config>(d_temporary_storage.get(),
                                                                   temporary_storage_bytes,
                                                                   d_keys_input.get(),
                                                                   d_values_input.get(),
                                                                   size,
                                                                   d_unique_output.get(),
                                                                   d_aggregates_output.get(),
                                                                   d_unique_count_output.get(),
                                                                   reduce_op,
                                                                   key_compare_op,
                                                                   stream,
                                                                   debug_synchronous)));

            if(TestFixture::params::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            const auto unique_output       = d_unique_output.load();
            const auto aggregates_output   = d_aggregates_output.load();
            const auto unique_count_output = d_unique_count_output.load();

            if (TestFixture::params::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }

            ASSERT_EQ(unique_count_output[0], unique_count_expected);

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(unique_output, unique_expected));
            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output, aggregates_expected));
        }
    }
}

template<typename value_type, bool use_graphs = false, bool Deterministic = false>
void large_indices_reduce_by_key()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type       = size_t;
    using aggregate_type = value_type;

    const bool debug_synchronous = false;

    ::rocprim::plus<value_type>   reduce_op;
    ::rocprim::equal_to<key_type> key_compare_op;

    hipStream_t stream = 0; // default
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t size : test_utils::get_large_sizes(42))
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // values in range [1, size], mapped using log2(i) to ensure non-equal groups
        // in:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        // out: 0  1  1  2  2  2  2  3  3  3  3  3  3  3  3  4
        auto d_keys_input = rocprim::make_transform_iterator(
            rocprim::make_counting_iterator(key_type(1)),
            [](key_type i)
            {
                // for i > 0, returns the position of the most significant set bit,
                // which is equal to the floor of log2
                return std::numeric_limits<size_t>::digits - 1 - __clzll(static_cast<long long>(i));
            });
        // the input values are all one, so the reduction of plus over the segments
        // results in the size of the group
        auto d_values_input = rocprim::constant_iterator<size_t>(1);

        // the count is value of the last key plus one as the value of the first key is zero
        unsigned int unique_count_expected = log2(size) + 1;

        common::device_ptr<key_type>       d_unique_output(unique_count_expected);
        common::device_ptr<aggregate_type> d_aggregates_output(unique_count_expected);
        common::device_ptr<unsigned int>   d_unique_count_output(1);

        size_t temporary_storage_bytes;
        HIP_CHECK(invoke_reduce_by_key<Deterministic>(nullptr,
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_values_input,
                                                      size,
                                                      d_unique_output.get(),
                                                      d_aggregates_output.get(),
                                                      d_unique_count_output.get(),
                                                      reduce_op,
                                                      key_compare_op,
                                                      stream,
                                                      debug_synchronous));

        ASSERT_GT(temporary_storage_bytes, 0);

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        test_utils::GraphHelper gHelper;
        if(use_graphs)
        {
           gHelper.startStreamCapture(stream);
        }

        HIP_CHECK(invoke_reduce_by_key<Deterministic>(d_temporary_storage.get(),
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_values_input,
                                                      size,
                                                      d_unique_output.get(),
                                                      d_aggregates_output.get(),
                                                      d_unique_count_output.get(),
                                                      reduce_op,
                                                      key_compare_op,
                                                      stream,
                                                      debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        const auto unique_output       = d_unique_output.load();
        const auto aggregates_output   = d_aggregates_output.load();
        const auto unique_count_output = d_unique_count_output.load();

        if(use_graphs)
        {
            gHelper.cleanupGraphHelper();
        }

        ASSERT_EQ(unique_count_output[0], unique_count_expected);

        size_t total_size = 0;
        for(size_t i = 0; i < unique_count_expected - 1; i++)
        {
            ASSERT_EQ(i, unique_output[i]);
            size_t i_size = 1ull << i;
            ASSERT_EQ(value_type(i_size), aggregates_output[i]);
            total_size += i_size;
        }
        // size of the last group may be limited by the input size
        size_t last_idx = unique_count_expected - 1;
        ASSERT_EQ(last_idx, unique_output[last_idx]);
        ASSERT_EQ(value_type(size - total_size), aggregates_output[last_idx]);
    }

    if(use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST(RocprimDeviceReduceByKey, LargeIndicesReduceByKeySmallValueType)
{
    large_indices_reduce_by_key<unsigned int>();
}

TEST(RocprimDeviceReduceByKey, LargeIndicesReduceByKeyLargeValueType)
{
    large_indices_reduce_by_key<common::custom_type<size_t, size_t, true>>();
}

TEST(RocprimDeviceReduceByKey, LargeIndicesReduceByKeyLargeValueTypeWithGraphs)
{
    large_indices_reduce_by_key<common::custom_type<size_t, size_t, true>, true>();
}

TEST(RocprimDeviceReduceByKey, LargeIndicesReduceByKeyDeterministic)
{
    large_indices_reduce_by_key<double, false, true>();
}

template<typename value_type, bool use_graphs = false, bool Deterministic = false>
void large_segment_count_reduce_by_key()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type = size_t;

    const bool debug_synchronous = false;

    ::rocprim::plus<value_type>   reduce_op;
    ::rocprim::equal_to<key_type> key_compare_op;

    hipStream_t stream = 0; // default
    if (use_graphs)
    {
        // Default stream does not support hipGraphs
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    for(size_t size : test_utils::get_large_sizes(42))
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // ensure segments of size 1
        auto d_keys_input   = rocprim::make_counting_iterator(key_type(0));
        auto d_values_input = rocprim::constant_iterator<size_t>(1);

        size_t unique_count_expected = size;

        // discard all output
        auto d_unique_output     = rocprim::make_discard_iterator();
        auto d_aggregates_output = rocprim::make_discard_iterator();

        common::device_ptr<size_t> d_unique_count_output(1);

        size_t temporary_storage_bytes;
        HIP_CHECK(invoke_reduce_by_key<Deterministic>(nullptr,
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_values_input,
                                                      size,
                                                      d_unique_output,
                                                      d_aggregates_output,
                                                      d_unique_count_output.get(),
                                                      reduce_op,
                                                      key_compare_op,
                                                      stream,
                                                      debug_synchronous));

        ASSERT_GT(temporary_storage_bytes, 0);

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        test_utils::GraphHelper gHelper;
        if(use_graphs)
        {
           gHelper.startStreamCapture(stream);
        }

        HIP_CHECK(invoke_reduce_by_key<Deterministic>(d_temporary_storage.get(),
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_values_input,
                                                      size,
                                                      d_unique_output,
                                                      d_aggregates_output,
                                                      d_unique_count_output.get(),
                                                      reduce_op,
                                                      key_compare_op,
                                                      stream,
                                                      debug_synchronous));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        const auto unique_count_output = d_unique_count_output.load()[0];

        ASSERT_EQ(unique_count_output, unique_count_expected);

        if (use_graphs)
            gHelper.cleanupGraphHelper();
    }

    if (use_graphs)
        HIP_CHECK(hipStreamDestroy(stream));
}

TEST(RocprimDeviceReduceByKey, LargeSegmentCountReduceByKeySmallValueType)
{
    large_segment_count_reduce_by_key<unsigned int>();
}

TEST(RocprimDeviceReduceByKey, LargeSegmentCountReduceByKeyLargeValueType)
{
    large_segment_count_reduce_by_key<common::custom_type<size_t, size_t, true>>();
}

TEST(RocprimDeviceReduceByKey, GraphReduceByKey)
{
    large_segment_count_reduce_by_key<unsigned int, true>();
}

TEST(RocprimDeviceReduceByKey, LargeSegmentCountReduceByKeyDeterministic)
{
    large_segment_count_reduce_by_key<float, false, true>();
}

TEST(RocprimDeviceReduceByKey, ReduceByNonEqualKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type   = size_t;
    using value_type = unsigned int;

    const bool debug_synchronous = false;
    constexpr bool deterministic     = false;

    ::rocprim::plus<value_type> reduce_op;
    auto                        key_compare_op = [](const auto&, const auto&) { return false; };

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t block_size_multiple : test_utils::get_block_size_multiples(seed_value, 256))
        {
            const size_t size = block_size_multiple + 1;

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            hipStream_t stream = 0; // default

            // Using segments of size 1.
            auto d_keys_input = rocprim::make_counting_iterator(key_type(0));

            // Setting all values to 1, so the reduction will contain the size of the input array.
            auto d_values_input = rocprim::constant_iterator<value_type>(1);

            size_t unique_count_expected = size;

            // Discard all output
            auto d_unique_output     = rocprim::make_discard_iterator();
            auto d_aggregates_output = rocprim::make_discard_iterator();

            common::device_ptr<size_t> d_unique_count_output(1);

            size_t temporary_storage_bytes;
            HIP_CHECK(invoke_reduce_by_key<deterministic>(nullptr,
                                                          temporary_storage_bytes,
                                                          d_keys_input,
                                                          d_values_input,
                                                          size,
                                                          d_unique_output,
                                                          d_aggregates_output,
                                                          d_unique_count_output.get(),
                                                          reduce_op,
                                                          key_compare_op,
                                                          stream,
                                                          debug_synchronous));

            ASSERT_GT(temporary_storage_bytes, 0);

            common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

            HIP_CHECK(invoke_reduce_by_key<deterministic>(d_temporary_storage.get(),
                                                          temporary_storage_bytes,
                                                          d_keys_input,
                                                          d_values_input,
                                                          size,
                                                          d_unique_output,
                                                          d_aggregates_output,
                                                          d_unique_count_output.get(),
                                                          reduce_op,
                                                          key_compare_op,
                                                          stream,
                                                          debug_synchronous));

            const auto unique_count_output = d_unique_count_output.load()[0];

            ASSERT_EQ(unique_count_output, unique_count_expected);
        }
    }
}
