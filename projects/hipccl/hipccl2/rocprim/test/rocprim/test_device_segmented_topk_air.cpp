// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"
#include "test_utils_sort_comparator.hpp"

// required rocprim headers
#include <rocprim/device/detail/device_segmented_topk_air.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

// std library
#include <unordered_set>
#include <vector>

ROCPRIM_FORCE_INLINE bool isnan(const rocprim::half& h)
{
    const auto& bits = reinterpret_cast<const uint16_t&>(h);
    return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
}

ROCPRIM_FORCE_INLINE bool isnan(const rocprim::bfloat16& h)
{
    const auto& bits = reinterpret_cast<const uint16_t&>(h);
    return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
}

template<class OutIterable, class InIterable, class OffsetIterable, class Predicate, class KType>
void host_segmented_nth_element(OutIterable&          h_output,
                                InIterable const&     h_input,
                                OffsetIterable const& h_offsets,
                                KType                 K,
                                Predicate             predicate)
{
    const auto segments = h_offsets.size() - 1;
    h_output.clear();

    for(std::remove_const_t<decltype(segments)> i = 0; i < segments; ++i)
    {
        const auto start_pos = h_offsets[i];
        const auto N         = h_offsets[i + 1] - start_pos;
        using common_t       = typename std::common_type<decltype(K), decltype(N)>::type;
        ASSERT_GE(N, 0);
        ASSERT_GE(K, 0);
        ASSERT_GE(static_cast<common_t>(N), static_cast<common_t>(K));
        InIterable sorted_segment_input(h_input.cbegin() + start_pos,
                                        h_input.cbegin() + start_pos + N);
        std::sort(sorted_segment_input.begin(), sorted_segment_input.end(), predicate);
        std::sort(sorted_segment_input.begin(),
                  sorted_segment_input.begin() + K,
                  std::less<decltype(sorted_segment_input[0])>{});
        h_output.insert(h_output.end(),
                        sorted_segment_input.begin(),
                        sorted_segment_input.begin() + K);
    }
}

template<class Params>
void test_segmented_topk_air_keys(std::vector<typename Params::key_type> const&     h_input,
                                  std::vector<typename Params::offsets_type> const& h_offsets,
                                  typename Params::size_out_type                    K)
{
#define ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(ptr, size)                                \
    if(!ptr.resize_with_memory_check(size))                                           \
    {                                                                                 \
        std::cout << "Out of memory. Skipping test for size = " << size << std::endl; \
        return;                                                                       \
    }

    using key_type     = typename Params::key_type;
    using offsets_type = typename Params::offsets_type;
    using decomposer_t = typename Params::decomposer_t;
    using config       = typename Params::config;

    constexpr auto select_min            = !Params::descending;
    constexpr auto debug_synchronous     = Params::debug_synchronous;
    constexpr bool use_graphs            = Params::use_graphs;
    constexpr bool use_indirect_iterator = Params::use_indirect_iterator;

    const auto  size        = h_input.size();
    const auto  segments    = h_offsets.size() - 1;
    const auto  output_size = K * segments;
    const auto  decomposer  = decomposer_t{};
    hipStream_t stream      = 0;
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }
    SCOPED_TRACE(testing::Message() << "with select_min = " << select_min);
    SCOPED_TRACE(testing::Message() << "with k = " << K);

    common::device_ptr<key_type> d_input;
    common::device_ptr<key_type> d_output;
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(d_input, size);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(d_output, output_size);
    d_input.store(h_input);

    const auto input_it
        = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());

    common::device_ptr<offsets_type> d_offsets(h_offsets);
    common::device_ptr<void>         temporary_storage;
    size_t                           storage_size = 0;

    // Get size of temporary_storage
    auto ret = rocprim::detail::device_segmented_topk_air<config, select_min>(nullptr,
                                                                              storage_size,
                                                                              input_it,
                                                                              d_output.get(),
                                                                              nullptr,
                                                                              nullptr,
                                                                              size,
                                                                              K,
                                                                              segments,
                                                                              d_offsets.get(),
                                                                              d_offsets.get() + 1,
                                                                              decomposer,
                                                                              stream,
                                                                              debug_synchronous);

    HIP_CHECK(ret);
    ASSERT_GT(storage_size, 0);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(temporary_storage, storage_size);

    test_utils::GraphHelper gHelper;

    if(use_graphs)
    {
        gHelper.startStreamCapture(stream);
    }

    // Launch segmented topk
    ret = rocprim::detail::device_segmented_topk_air<config, select_min>(temporary_storage.get(),
                                                                         storage_size,
                                                                         input_it,
                                                                         d_output.get(),
                                                                         nullptr,
                                                                         nullptr,
                                                                         size,
                                                                         K,
                                                                         segments,
                                                                         d_offsets.get(),
                                                                         d_offsets.get() + 1,
                                                                         decomposer,
                                                                         stream,
                                                                         debug_synchronous);
    HIP_CHECK(ret);

    if(use_graphs)
    {
        gHelper.createAndLaunchGraph(stream);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(use_graphs)
    {
        gHelper.cleanupGraphHelper();
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipStreamCreate(&stream));
    }

    // Verify output by using segmented_radix_sort_keys
    std::vector<offsets_type> h_sort_offsets{};
    for(std::remove_const_t<decltype(segments)> i = 0; i < segments + 1; ++i)
    {
        h_sort_offsets.push_back(i * K);
    }
    common::device_ptr<offsets_type> d_sort_offsets(h_sort_offsets);

    // Get size of temporary_storage
    storage_size = 0;
    ret          = rocprim::segmented_radix_sort_keys(nullptr,
                                             storage_size,
                                             d_output.get(),
                                             d_output.get(),
                                             output_size,
                                             segments,
                                             d_sort_offsets.get(),
                                             d_sort_offsets.get() + 1,
                                             0,
                                             8 * sizeof(key_type),
                                             stream);

    HIP_CHECK(ret);
    ASSERT_GT(storage_size, 0);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(temporary_storage, storage_size);

    ret = rocprim::segmented_radix_sort_keys(temporary_storage.get(),
                                             storage_size,
                                             d_output.get(),
                                             d_output.get(),
                                             output_size,
                                             segments,
                                             d_sort_offsets.get(),
                                             d_sort_offsets.get() + 1,
                                             0,
                                             8 * sizeof(key_type),
                                             stream);
    HIP_CHECK(ret);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    const auto            h_output = d_output.load();
    std::vector<key_type> h_expected{};
    auto                  pred = [](auto _1, auto _2)
    {
        if constexpr(select_min)
        {
            return rocprim::less<decltype(_1)>{}(_1, _2);
        }
        else
        {
            return rocprim::greater<decltype(_1)>{}(_1, _2);
        }
    };
    host_segmented_nth_element(h_expected, h_input, h_offsets, K, pred);
    ASSERT_EQ(h_output.size(), output_size);
    ASSERT_EQ(h_expected.size(), output_size);

    // Check keys
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_output, h_expected, output_size));

#undef ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN
}

template<class Params>
void test_segmented_topk_air_pairs_unstable(
    std::vector<typename Params::key_type> const&     h_input_keys,
    std::vector<typename Params::value_type> const&   h_input_values,
    std::vector<typename Params::offsets_type> const& h_offsets,
    typename Params::size_out_type                    K)
{
#define ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(ptr, size)                                \
    if(!ptr.resize_with_memory_check(size))                                           \
    {                                                                                 \
        std::cout << "Out of memory. Skipping test for size = " << size << std::endl; \
        return;                                                                       \
    }

    using key_type     = typename Params::key_type;
    using value_type   = typename Params::value_type;
    using offsets_type = typename Params::offsets_type;
    using size_in_type = typename Params::size_in_type;
    using decomposer_t = typename Params::decomposer_t;
    using config       = typename Params::config;

    constexpr auto select_min            = !Params::descending;
    constexpr auto debug_synchronous     = Params::debug_synchronous;
    constexpr bool use_graphs            = Params::use_graphs;
    constexpr bool use_indirect_iterator = Params::use_indirect_iterator;

    const auto size = static_cast<size_in_type>(h_input_keys.size());
    assert(size == h_input_values.size());
    const auto  segments    = h_offsets.size() - 1;
    const auto  output_size = K * segments;
    const auto  decomposer  = decomposer_t{};
    hipStream_t stream      = 0;
    if(use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }
    SCOPED_TRACE(testing::Message() << "with select_min = " << select_min);
    SCOPED_TRACE(testing::Message() << "with k = " << K);

    common::device_ptr<key_type>   d_input_keys;
    common::device_ptr<value_type> d_input_values;
    common::device_ptr<key_type>   d_output_keys;
    common::device_ptr<value_type> d_output_values;
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(d_input_keys, size);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(d_input_values, size);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(d_output_keys, output_size);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(d_output_values, output_size);
    d_input_keys.store(h_input_keys);
    d_input_values.store(h_input_values);

    const auto input_keys_it
        = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input_keys.get());
    const auto input_values_it
        = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input_values.get());

    common::device_ptr<offsets_type> d_offsets(h_offsets);
    common::device_ptr<void>         temporary_storage;
    size_t                           storage_size = 0;

    // Get size of temporary_storage
    auto ret = rocprim::detail::device_segmented_topk_air<config, select_min>(nullptr,
                                                                              storage_size,
                                                                              input_keys_it,
                                                                              d_output_keys.get(),
                                                                              input_values_it,
                                                                              d_output_values.get(),
                                                                              size,
                                                                              K,
                                                                              segments,
                                                                              d_offsets.get(),
                                                                              d_offsets.get() + 1,
                                                                              decomposer,
                                                                              stream,
                                                                              debug_synchronous);

    HIP_CHECK(ret);
    ASSERT_GT(storage_size, 0);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(temporary_storage, storage_size);

    test_utils::GraphHelper gHelper;

    if(use_graphs)
    {
        gHelper.startStreamCapture(stream);
    }

    // Launch segmented topk
    ret = rocprim::detail::device_segmented_topk_air<config, select_min>(temporary_storage.get(),
                                                                         storage_size,
                                                                         input_keys_it,
                                                                         d_output_keys.get(),
                                                                         input_values_it,
                                                                         d_output_values.get(),
                                                                         size,
                                                                         K,
                                                                         segments,
                                                                         d_offsets.get(),
                                                                         d_offsets.get() + 1,
                                                                         decomposer,
                                                                         stream,
                                                                         debug_synchronous);
    HIP_CHECK(ret);

    if(use_graphs)
    {
        gHelper.createAndLaunchGraph(stream);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(use_graphs)
    {
        gHelper.cleanupGraphHelper();
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipStreamCreate(&stream));
    }

    // Verify output by using segmented_radix_sort_keys
    std::vector<offsets_type> h_sort_offsets{};
    for(std::remove_const_t<decltype(segments)> i = 0; i < segments + 1; ++i)
    {
        h_sort_offsets.push_back(i * K);
    }
    common::device_ptr<offsets_type> d_sort_offsets(h_sort_offsets);

    // Get size of temporary_storage
    storage_size = 0;
    ret          = rocprim::segmented_radix_sort_pairs(nullptr,
                                              storage_size,
                                              d_output_keys.get(),
                                              d_output_keys.get(),
                                              d_output_values.get(),
                                              d_output_values.get(),
                                              output_size,
                                              segments,
                                              d_sort_offsets.get(),
                                              d_sort_offsets.get() + 1,
                                              0,
                                              8 * sizeof(key_type),
                                              stream);

    HIP_CHECK(ret);
    ASSERT_GT(storage_size, 0);
    ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN(temporary_storage, storage_size);

    ret = rocprim::segmented_radix_sort_pairs(temporary_storage.get(),
                                              storage_size,
                                              d_output_keys.get(),
                                              d_output_keys.get(),
                                              d_output_values.get(),
                                              d_output_values.get(),
                                              output_size,
                                              segments,
                                              d_sort_offsets.get(),
                                              d_sort_offsets.get() + 1,
                                              0,
                                              8 * sizeof(key_type),
                                              stream);
    HIP_CHECK(ret);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(use_graphs)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }

    const auto            h_output_keys   = d_output_keys.load();
    const auto            h_output_values = d_output_values.load();
    std::vector<key_type> h_expected_keys{};
    auto                  pred = [](auto _1, auto _2)
    {
        if constexpr(select_min)
        {
            return rocprim::less<decltype(_1)>{}(_1, _2);
        }
        else
        {
            return rocprim::greater<decltype(_1)>{}(_1, _2);
        }
    };
    host_segmented_nth_element(h_expected_keys, h_input_keys, h_offsets, K, pred);
    ASSERT_EQ(d_output_keys.size(), output_size);
    ASSERT_EQ(h_output_values.size(), output_size);
    ASSERT_EQ(h_expected_keys.size(), output_size);

    // Check keys
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(h_output_keys, h_expected_keys, output_size));

    // Check values
    if constexpr(rocprim::is_integral<value_type>::value)
    {
        for(unsigned int i = 0; i < output_size; ++i)
        {
            test_utils::assert_eq(h_input_keys[h_output_values[i]], h_output_keys[i]);
        }
    }
    else
    {
        // TODO: this is slow, can use std::unordered_multimap to make it faster
        for(unsigned int i = 0; i < output_size; ++i)
        {
            const auto& key   = h_output_keys[i];
            const auto& val   = h_output_values[i];
            bool        found = false;
            for(unsigned int j = 0; j < size; ++j)
            {
                const auto& in_key = h_input_keys[j];
                const auto& in_val = h_input_values[j];
                if(in_val == val && in_key == key)
                {
                    found = true;
                    break;
                }
                if((in_key == key && isnan(val) && isnan(in_val))
                   || (val == in_val && isnan(in_key) && isnan(key))
                   || (isnan(val) && isnan(in_val) && isnan(in_key) && isnan(key)))
                {
                    found = true;
                    break;
                }
            }
            ASSERT_TRUE(found);
        }
    }

#undef ROCPRIM_TEST_RESIZE_MEMCHECK_RETURN
}
// Params for tests
template<class KeyType,
         class ValueType                  = rocprim::empty_type,
         bool Descending                  = false,
         bool UseGraphs                   = false,
         bool Ordered                     = false,
         bool Deterministic               = false,
         bool Stable                      = false,
         class Config                     = rocprim::default_config,
         class SizeInType                 = uint64_t,
         class SizeOutType                = uint64_t,
         class OffsetType                 = SizeInType,
         class Decomposer                 = rocprim::identity_decomposer,
         unsigned int MinSegmentLength    = 0,
         unsigned int MaxSegmentLength    = 1000,
         bool         UseIndirectIterator = false>
struct DeviceSegmentedTopkParams
{
    using key_type                                      = KeyType;
    using value_type                                    = ValueType;
    using offsets_type                                  = OffsetType;
    using size_out_type                                 = SizeOutType;
    using size_in_type                                  = SizeInType;
    using decomposer_t                                  = Decomposer;
    using config                                        = Config;
    static constexpr bool         descending            = Descending;
    static constexpr bool         ordered               = Ordered;
    static constexpr bool         deterministic         = Deterministic;
    static constexpr bool         stable                = Stable;
    static constexpr bool         use_graphs            = UseGraphs;
    static constexpr bool         use_indirect_iterator = UseIndirectIterator;
    static constexpr unsigned int min_segment_length    = MinSegmentLength;
    static constexpr unsigned int max_segment_length    = MaxSegmentLength;
};

template<class Params>
class RocprimDeviceSegmentedTopkTests : public ::testing::Test
{
public:
    using key_type                                      = typename Params::key_type;
    using value_type                                    = typename Params::value_type;
    using offsets_type                                  = typename Params::offsets_type;
    using size_out_type                                 = typename Params::size_out_type;
    using size_in_type                                  = typename Params::size_in_type;
    using decomposer_t                                  = typename Params::decomposer_t;
    using config                                        = typename Params::config;
    static constexpr bool         descending            = Params::descending;
    static constexpr bool         ordered               = Params::ordered;
    static constexpr bool         deterministic         = Params::deterministic;
    static constexpr bool         stable                = Params::stable;
    static constexpr bool         use_graphs            = Params::use_graphs;
    static constexpr bool         use_indirect_iterator = Params::use_indirect_iterator;
    static constexpr bool         debug_synchronous     = false;
    static constexpr unsigned int min_segment_length    = Params::min_segment_length;
    static constexpr unsigned int max_segment_length    = Params::max_segment_length;
};

// TODO: use grouped params and test_utils::merge_sequence when it's available
using RocprimDeviceSegmentedTopkTestsParams = ::testing::Types<
    // Descending
    DeviceSegmentedTopkParams<int8_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<uint8_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<int16_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<uint16_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<int32_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<uint32_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<int64_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<uint64_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<rocprim::int128_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<rocprim::uint128_t, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<rocprim::half, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<rocprim::bfloat16, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<float, rocprim::empty_type, true>,
    DeviceSegmentedTopkParams<double, rocprim::empty_type, true>,

    // Ascending
    DeviceSegmentedTopkParams<int8_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<uint8_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<int16_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<uint16_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<int32_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<uint32_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<int64_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<uint64_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<rocprim::int128_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<rocprim::uint128_t, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<rocprim::half, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<rocprim::bfloat16, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<float, rocprim::empty_type, false>,
    DeviceSegmentedTopkParams<double, rocprim::empty_type, false>,

    // Gragh
    DeviceSegmentedTopkParams<int8_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<uint8_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<int16_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<uint16_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<int32_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<uint32_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<int64_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<uint64_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<rocprim::int128_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<rocprim::uint128_t, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<rocprim::half, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<rocprim::bfloat16, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<float, rocprim::empty_type, true, true>,
    DeviceSegmentedTopkParams<double, rocprim::empty_type, true, true>,

    // Pairs
    DeviceSegmentedTopkParams<int8_t, double, true>,
    DeviceSegmentedTopkParams<uint8_t, float, false>,
    DeviceSegmentedTopkParams<int16_t, rocprim::bfloat16, true>,
    DeviceSegmentedTopkParams<uint16_t, rocprim::half, false>,
    DeviceSegmentedTopkParams<int32_t, rocprim::uint128_t, true, true>,
    DeviceSegmentedTopkParams<uint32_t, rocprim::int128_t, false, true>,
    DeviceSegmentedTopkParams<int64_t, uint64_t, true, true>>;

TYPED_TEST_SUITE(RocprimDeviceSegmentedTopkTests, RocprimDeviceSegmentedTopkTestsParams);

TYPED_TEST(RocprimDeviceSegmentedTopkTests, DeviceSegmentedTopk)
{
    using key_type      = typename TestFixture::key_type;
    using value_type    = typename TestFixture::value_type;
    using offsets_type  = typename TestFixture::offsets_type;
    using size_out_type = typename TestFixture::size_out_type;
    using size_in_type  = typename TestFixture::size_in_type;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    std::random_device                             rd;
    const size_t                                   seed = rd();
    std::default_random_engine                     gen(seed);
    common::uniform_int_distribution<offsets_type> segment_length_dis(
        TestFixture::min_segment_length,
        TestFixture::max_segment_length);
    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size_sz : test_utils::get_sizes(seed_value))
        {
            const auto size = static_cast<size_in_type>(size_sz);
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            if(size == 0)
            {
                continue;
            }
            // Generate random input
            engine_type           rng_engine(seed_value);
            std::vector<key_type> input(size);
            test_utils::generate_random_data_n(input.begin(),
                                               size,
                                               rocprim::numeric_limits<key_type>::min(),
                                               rocprim::numeric_limits<key_type>::max(),
                                               rng_engine);

            // Generate random offsets
            std::vector<offsets_type> offsets;
            offsets_type              offset             = 0;
            auto                      min_segment_length = static_cast<offsets_type>(size);
            while(offset < size)
            {
                const offsets_type segment_length = segment_length_dis(gen);
                offsets.push_back(offset);

                const auto end     = std::min<offsets_type>(size, offset + segment_length);
                min_segment_length = std::min<offsets_type>(min_segment_length, end - offset);

                offset += segment_length;
            }
            offsets.push_back(size);
            SCOPED_TRACE(testing::Message() << "with segments_count = " << (offsets.size() - 1));

            // Generate random k
            const auto k
                = test_utils::get_random_value<size_out_type>(0, min_segment_length, seed_value);
            if constexpr(std::is_same_v<value_type, rocprim::empty_type>)
            {
                test_segmented_topk_air_keys<TestFixture>(input, offsets, k);
            }
            else
            {
                std::vector<value_type> values;
                if constexpr(rocprim::is_integral<value_type>::value)
                {
                    using common_t = std::common_type_t<value_type, decltype(size)>;
                    for(common_t i = 0; i < static_cast<common_t>(size); ++i)
                    {
                        values.push_back(value_type{i});
                    }
                }
                else
                {
                    using int_value_t =
                        typename rocprim::detail::device_topk_air_helper::matched_int<
                            value_type>::type;
                    using common_t = std::common_type_t<int_value_t, decltype(size)>;
                    for(common_t i = 0; i < static_cast<common_t>(size); ++i)
                    {
                        union
                        {
                            value_type  floating_value;
                            int_value_t integral_value;
                        } value;
                        value.integral_value = i;

                        values.push_back(value.floating_value);
                    }
                }
                test_segmented_topk_air_pairs_unstable<TestFixture>(input, values, offsets, k);
            }
        }
    }
}
