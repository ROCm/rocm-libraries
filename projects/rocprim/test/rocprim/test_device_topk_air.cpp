// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "identity_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_get_random_data.hpp"
#include "test_utils_hipgraphs.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/detail/device_topk_air.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <stdint.h>
#include <unordered_set>
#include <vector>

template<class...>
struct MergeDeviceAirTopKTestsParams
{};

template<class... Args>
struct MergeDeviceAirTopKTestsParams<::testing::Types<Args...>>
{
    using type = ::testing::Types<Args...>;
};

template<class... Args1, class... Args2>
struct MergeDeviceAirTopKTestsParams<::testing::Types<Args1...>, ::testing::Types<Args2...>>
{
    using type = ::testing::Types<Args1..., Args2...>;
};

template<class T1, class T2, class... Ts>
struct MergeDeviceAirTopKTestsParams<T1, T2, Ts...>
{
    using type =
        typename MergeDeviceAirTopKTestsParams<typename MergeDeviceAirTopKTestsParams<T1, T2>::type,
                                               Ts...>::type;
};

struct pair_hash
{
    template<class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// This is a internal API, so here we are using custom config type
template<unsigned int RadixBit = 8, unsigned int BlockSize = 256, unsigned int ItemsPerThread = 3>
struct DeviceAirTopKConfig
{
    static constexpr unsigned int block_size        = BlockSize;
    static constexpr unsigned int items_per_threads = ItemsPerThread;
    static constexpr unsigned int radix_bits        = RadixBit;
};

template<class Key,
         bool SelectMin = true,
         class Config   = DeviceAirTopKConfig<>,
         bool UseGraphs = false,
         class Value    = std::nullptr_t,
         class SizeIn   = std::size_t,
         class SizeOut  = std::size_t>
struct DeviceAirTopKParams
{
    using key_in_t                                  = Key;
    using key_out_t                                 = Key;
    using value_in_t                                = Value;
    using value_out_t                               = Value;
    using size_in                                   = SizeIn;
    using size_out                                  = SizeOut;
    static constexpr bool         select_min        = SelectMin;
    static constexpr bool         use_graphs        = UseGraphs;
    static constexpr unsigned int block_size        = Config::block_size;
    static constexpr unsigned int items_per_threads = Config::items_per_threads;
    static constexpr unsigned int radix_bits        = Config::radix_bits;
};

template<class Params>
class RocprimDeviceAirTopKKeyTests : public ::testing::Test
{
public:
    using key_in_t                                  = typename Params::key_in_t;
    using key_out_t                                 = typename Params::key_in_t;
    using size_in                                   = typename Params::size_in;
    using size_out                                  = typename Params::size_out;
    static constexpr bool         select_min        = Params::select_min;
    static constexpr bool         use_graphs        = Params::use_graphs;
    static constexpr unsigned int block_size        = Params::block_size;
    static constexpr unsigned int items_per_threads = Params::items_per_threads;
    static constexpr unsigned int radix_bits        = Params::radix_bits;
};
template<class Key>
using BatchGenerateDeviceAirTopKKeyTestsParams
    = ::testing::Types<DeviceAirTopKParams<Key>,
                       DeviceAirTopKParams<Key, false>,
                       DeviceAirTopKParams<Key, true, DeviceAirTopKConfig<4>>,
                       DeviceAirTopKParams<Key, false, DeviceAirTopKConfig<6>>,
                       DeviceAirTopKParams<Key, true, DeviceAirTopKConfig<8>, true>,
                       DeviceAirTopKParams<Key, false, DeviceAirTopKConfig<10>, true>>;

using RocprimDeviceAirTopKKeyTestsParams = typename MergeDeviceAirTopKTestsParams<
    BatchGenerateDeviceAirTopKKeyTestsParams<uint8_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<int8_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<uint16_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<int16_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<uint32_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<int32_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<uint64_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<int64_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<rocprim::int128_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<rocprim::uint128_t>,
    BatchGenerateDeviceAirTopKKeyTestsParams<half>,
    BatchGenerateDeviceAirTopKKeyTestsParams<float>,
    BatchGenerateDeviceAirTopKKeyTestsParams<double>>::type;

TYPED_TEST_SUITE(RocprimDeviceAirTopKKeyTests, RocprimDeviceAirTopKKeyTestsParams);

TYPED_TEST(RocprimDeviceAirTopKKeyTests, AirTopKKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    constexpr unsigned int block_size        = TestFixture::block_size;
    constexpr unsigned int items_per_threads = TestFixture::items_per_threads;
    constexpr unsigned int radix_bits        = TestFixture::radix_bits;
    constexpr bool         select_min        = TestFixture::select_min;
    using key_in_t                           = typename TestFixture::key_in_t;
    using key_out_t                          = typename TestFixture::key_out_t;
    using size_in                            = typename TestFixture::size_in;
    using size_out                           = typename TestFixture::size_out;
    using decomposer_t = typename test_utils::select_decomposer<key_in_t>::type;

    using topk = rocprim::detail::device_air_topk_impl<block_size,
                                                       items_per_threads,
                                                       radix_bits,
                                                       select_min,
                                                       key_in_t*,
                                                       key_out_t*,
                                                       std::nullptr_t*,
                                                       std::nullptr_t*,
                                                       size_in,
                                                       size_out,
                                                       decomposer_t>;

    for(std::size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);
            size_out k = 0;
            if(size > 1)
            {
                k = test_utils::get_random_value<decltype(k)>(0,
                                                              static_cast<decltype(k)>(size),
                                                              seed_value);
            }
            SCOPED_TRACE(testing::Message() << "with k = " << k);

            std::vector<key_in_t> h_keys_input = test_utils::get_random_data_wrapped<key_in_t>(
                size,
                rocprim::numeric_limits<key_in_t>::min(),
                rocprim::numeric_limits<key_in_t>::max(),
                seed_value);

            auto expected_out_keys = h_keys_input;
            if constexpr(select_min)
            {
                std::nth_element(expected_out_keys.begin(),
                                 expected_out_keys.begin() + k,
                                 expected_out_keys.end(),
                                 std::less<key_in_t>());
            }
            else
            {
                std::nth_element(expected_out_keys.begin(),
                                 expected_out_keys.begin() + k,
                                 expected_out_keys.end(),
                                 std::greater<key_in_t>());
            }
            expected_out_keys.resize(k);
            std::size_t temporary_storage_size = 0;

            common::device_ptr<key_in_t>  d_keys_input(h_keys_input);
            common::device_ptr<key_out_t> d_keys_output(k);
            HIP_CHECK(topk{}(nullptr,
                             temporary_storage_size,
                             d_keys_input.get(),
                             d_keys_output.get(),
                             nullptr,
                             nullptr,
                             h_keys_input.size(),
                             k,
                             decomposer_t{},
                             stream,
                             false));

            ASSERT_GT(temporary_storage_size, 0);
            common::device_ptr<void> d_temporary_storage(temporary_storage_size);

            test_utils::GraphHelper gHelper;
            if constexpr(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(topk{}(d_temporary_storage.get(),
                             temporary_storage_size,
                             d_keys_input.get(),
                             d_keys_output.get(),
                             nullptr,
                             nullptr,
                             h_keys_input.size(),
                             k,
                             decomposer_t{},
                             stream,
                             false));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            auto out_keys = d_keys_output.load();
            std::sort(out_keys.begin(), out_keys.end());
            std::sort(expected_out_keys.begin(), expected_out_keys.end());
            ASSERT_EQ(out_keys, expected_out_keys);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

template<class Params>
class RocprimDeviceAirTopKUnstablePairTests : public ::testing::Test
{
public:
    using key_in_t                                  = typename Params::key_in_t;
    using key_out_t                                 = typename Params::key_out_t;
    using value_in_t                                = typename Params::value_in_t;
    using value_out_t                               = typename Params::value_out_t;
    using size_in                                   = typename Params::size_in;
    using size_out                                  = typename Params::size_out;
    static constexpr bool         select_min        = Params::select_min;
    static constexpr bool         use_graphs        = Params::use_graphs;
    static constexpr unsigned int block_size        = Params::block_size;
    static constexpr unsigned int items_per_threads = Params::items_per_threads;
    static constexpr unsigned int radix_bits        = Params::radix_bits;
};

template<class Key, class Val>
using BatchGenerateDeviceAirTopKPairTestsParams
    = ::testing::Types<DeviceAirTopKParams<Key, true, DeviceAirTopKConfig<4>, false, Val>,
                       DeviceAirTopKParams<Key, false, DeviceAirTopKConfig<6>, false, Val>,
                       DeviceAirTopKParams<Key, true, DeviceAirTopKConfig<8>, false, Val>,
                       DeviceAirTopKParams<Key, false, DeviceAirTopKConfig<10>, true, Val>>;

using RocprimDeviceAirTopKPairTestsParams = typename MergeDeviceAirTopKTestsParams<
    BatchGenerateDeviceAirTopKPairTestsParams<int8_t, float>,
    BatchGenerateDeviceAirTopKPairTestsParams<int64_t, int32_t>,
    BatchGenerateDeviceAirTopKPairTestsParams<float, int8_t>,
    BatchGenerateDeviceAirTopKPairTestsParams<double, uint8_t>>::type;

TYPED_TEST_SUITE(RocprimDeviceAirTopKUnstablePairTests, RocprimDeviceAirTopKPairTestsParams);

TYPED_TEST(RocprimDeviceAirTopKUnstablePairTests, AirTopKPairUnstable)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    constexpr unsigned int block_size        = TestFixture::block_size;
    constexpr unsigned int items_per_threads = TestFixture::items_per_threads;
    constexpr unsigned int radix_bits        = TestFixture::radix_bits;
    constexpr bool         select_min        = TestFixture::select_min;
    using key_in_t                           = typename TestFixture::key_in_t;
    using key_out_t                          = typename TestFixture::key_out_t;
    using value_in_t                         = typename TestFixture::value_in_t;
    using value_out_t                        = typename TestFixture::value_out_t;
    using size_in                            = typename TestFixture::size_in;
    using size_out                           = typename TestFixture::size_out;
    using decomposer_t = typename test_utils::select_decomposer<key_in_t>::type;

    using topk = rocprim::detail::device_air_topk_impl<block_size,
                                                       items_per_threads,
                                                       radix_bits,
                                                       select_min,
                                                       key_in_t*,
                                                       key_out_t*,
                                                       value_in_t*,
                                                       value_out_t*,
                                                       size_in,
                                                       size_out,
                                                       decomposer_t>;

    for(std::size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        const unsigned int seed_key
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if constexpr(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);
            size_out k = 0;
            if(size > 1)
            {
                k = test_utils::get_random_value<decltype(k)>(0,
                                                              static_cast<decltype(k)>(size),
                                                              seed_value);
            }
            SCOPED_TRACE(testing::Message() << "with k = " << k);

            std::vector<key_in_t> h_keys_input = test_utils::get_random_data_wrapped<key_in_t>(
                size,
                rocprim::numeric_limits<key_in_t>::min(),
                rocprim::numeric_limits<key_in_t>::max(),
                seed_key);
            std::vector<value_in_t> h_vals_input = test_utils::get_random_data_wrapped<value_in_t>(
                size,
                rocprim::numeric_limits<value_in_t>::min(),
                rocprim::numeric_limits<value_in_t>::max(),
                seed_value);

            std::unordered_multiset<std::pair<key_in_t, value_in_t>, pair_hash> h_input_map;
            for(size_t i = 0; i < h_keys_input.size(); ++i)
            {
                h_input_map.insert({h_keys_input[i], h_vals_input[i]});
            }

            std::size_t temporary_storage_size = 0;

            common::device_ptr<key_in_t>    d_keys_input(h_keys_input);
            common::device_ptr<key_out_t>   d_keys_output(k);
            common::device_ptr<value_in_t>  d_vals_input(h_vals_input);
            common::device_ptr<value_out_t> d_vals_output(k);
            HIP_CHECK(topk{}(nullptr,
                             temporary_storage_size,
                             d_keys_input.get(),
                             d_keys_output.get(),
                             d_vals_input.get(),
                             d_vals_output.get(),
                             h_keys_input.size(),
                             k,
                             decomposer_t{},
                             stream,
                             false));

            ASSERT_GT(temporary_storage_size, 0);
            common::device_ptr<void> d_temporary_storage;
            d_temporary_storage.resize_with_memory_check(temporary_storage_size);
            test_utils::GraphHelper gHelper;
            if constexpr(TestFixture::use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            HIP_CHECK(topk{}(d_temporary_storage.get(),
                             temporary_storage_size,
                             d_keys_input.get(),
                             d_keys_output.get(),
                             d_vals_input.get(),
                             d_vals_output.get(),
                             h_keys_input.size(),
                             k,
                             decomposer_t{},
                             stream,
                             false));

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            auto out_keys = d_keys_output.load();
            auto out_vals = d_vals_output.load();

            // Because keys are already checked in another test, here we only check if values match the keys
            for(unsigned int i = 0; i < k; ++i)
            {
                auto range = h_input_map.equal_range(
                    std::pair<key_in_t, value_in_t>{out_keys[i], out_vals[i]});
                ASSERT_NE(range.first, range.second);
            }
            // ASSERT_EQ(output, expected_out);

            if constexpr(TestFixture::use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}
