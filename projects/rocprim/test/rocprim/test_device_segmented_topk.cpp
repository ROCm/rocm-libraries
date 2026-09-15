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

#include "../../common/utils_data_generation.hpp"
#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "indirect_iterator.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_hipgraphs.hpp"
#include "test_utils_memory_check.hpp"
#include "test_utils_sort_comparator.hpp"

// required rocprim headers
#include <rocprim/device/device_segmented_topk.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types.hpp>

// std library
#include <unordered_set>
#include <vector>

// Params for tests
template<class KeyType,
         class ValueType          = KeyType,
         bool Descending          = false,
         class Decomposer         = rocprim::identity_decomposer,
         class Config             = rocprim::default_config,
         bool Ordered             = false,
         bool Deterministic       = false,
         bool Stable              = false,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceSegmentedTopkParams
{
    using key_type                              = KeyType;
    using value_type                            = ValueType;
    static constexpr bool descending            = Descending;
    using decomposer_t                          = Decomposer;
    using config                                = Config;
    static constexpr bool ordered               = Ordered;
    static constexpr bool deterministic         = Deterministic;
    static constexpr bool stable                = Stable;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<bool Descending, class InputVector, class OutputVector, class SizeOut>
void inline compare_k(InputVector input, OutputVector output, SizeOut k)
{
    using key_type          = typename InputVector::value_type;
    constexpr int start_bit = 0;
    constexpr int end_bit   = sizeof(key_type) * 8;
    using comparator        = test_utils::key_comparator<key_type, Descending, start_bit, end_bit>;

    // Calculate sorted input results on host
    std::vector<key_type> sorted_input(input);
    std::stable_sort(sorted_input.begin(), sorted_input.end(), comparator());

    // Calculate sorted output results on host
    std::vector<key_type> sorted_output(output);
    std::stable_sort(sorted_output.begin(), sorted_output.end(), comparator());

    // Only check that first k coincide
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input, k));
}

namespace std
{
template<>
struct hash<rocprim::int128_t>
{
    ROCPRIM_FORCE_INLINE size_t operator()(const rocprim::int128_t& value) const
    {
        rocprim::uint128_t v    = static_cast<rocprim::uint128_t>(value);
        std::uint64_t      low  = static_cast<std::uint64_t>(v);
        std::uint64_t      high = static_cast<std::uint64_t>(v >> 64);
        return (hash<std::uint64_t>{}(low) * 31) + hash<std::uint64_t>{}(high);
    }
};
template<>
struct hash<rocprim::uint128_t>
{
    ROCPRIM_FORCE_INLINE size_t operator()(const rocprim::uint128_t& value) const
    {
        auto          v    = value;
        std::uint64_t low  = static_cast<std::uint64_t>(v);
        std::uint64_t high = static_cast<std::uint64_t>(v >> 64);
        return (hash<std::uint64_t>{}(low) * 31) + hash<std::uint64_t>{}(high);
    }
};

template<>
struct hash<rocprim::half>
{
    ROCPRIM_FORCE_INLINE size_t operator()(const rocprim::half& value) const
    {
        if(value != value)
            return 0; // NaN
        uint16_t bits;
        std::memcpy(&bits, &value, sizeof(value));
        return static_cast<size_t>(bits);
    }
};

} // namespace std

struct pair_hash
{
    template<class T1, class T2>
    ROCPRIM_FORCE_INLINE std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

ROCPRIM_FORCE_INLINE bool isnan_half(const rocprim::half& h)
{
    const auto& bits = reinterpret_cast<const uint16_t&>(h);
    return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
}

template<class T, class = void>
struct custom_equal_to
{
    ROCPRIM_FORCE_INLINE bool operator()(const T& a, const T& b) const
    {
        return rocprim::equal_to<T>{}(a, b);
    }
};

template<class T>
struct custom_equal_to<
    T,
    std::enable_if_t<rocprim::is_floating_point<T>::value && !std::is_same_v<T, rocprim::half>>>
{
    ROCPRIM_FORCE_INLINE bool operator()(const T& a, const T& b) const
    {
        return (std::isnan(a) && std::isnan(b)) || rocprim::equal_to<T>{}(a, b);
    }
};

template<>
struct custom_equal_to<rocprim::half, void>
{
    ROCPRIM_FORCE_INLINE bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        return (isnan_half(a) && isnan_half(b)) || rocprim::equal_to<rocprim::half>{}(a, b);
    }
};

template<class Pair>
struct pair_comp
{
    ROCPRIM_FORCE_INLINE bool operator()(const Pair& a, const Pair& b) const
    {
        return custom_equal_to<typename Pair::first_type>{}(a.first, b.first)
               && custom_equal_to<typename Pair::second_type>{}(a.second, b.second);
    }
};

template<bool Descending,
         class InputKeyVector,
         class InputValueVector,
         class OutputKeyVector,
         class OutputValueVector,
         class SizeOut>
ROCPRIM_FORCE_INLINE void compare_pairs_k(InputKeyVector    keys_input,
                            InputValueVector  values_input,
                            OutputKeyVector   keys_output,
                            OutputValueVector values_output,
                            SizeOut           k)
{
    using key_type          = typename InputKeyVector::value_type;
    using value_type        = typename InputValueVector::value_type;
    using key_value         = std::pair<key_type, value_type>;
    constexpr int start_bit = 0;
    constexpr int end_bit   = sizeof(key_type) * 8;
    using comparator
        = test_utils::key_value_comparator<key_type, value_type, Descending, start_bit, end_bit>;

    const std::size_t size = values_input.size();

    // Calculate sorted input results on host
    std::vector<key_value> sorted_input(size);
    for(std::size_t i = 0; i < size; i++)
    {
        sorted_input[i] = key_value(keys_input[i], values_input[i]);
    }
    std::stable_sort(sorted_input.begin(), sorted_input.end(), comparator());

    // Calculate sorted output results on host
    std::vector<key_value> sorted_output(k);
    for(std::size_t i = 0; i < k; i++)
    {
        sorted_output[i] = key_value(keys_output[i], values_output[i]);
    }
    std::stable_sort(sorted_output.begin(), sorted_output.end(), comparator());

    // Only check that first k coincide
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(sorted_output, sorted_input, k));
}

// Builds a randomized set of segment offsets covering [0, size), so that tests actually
// exercise more than a single segment. Also reports the length of the shortest generated
// segment, since K must be strictly smaller than every segment's length.
template<class OffsetType, class Engine>
void generate_segmented_offsets(size_t                   size,
                                Engine&                  rng_engine,
                                std::vector<OffsetType>& offsets,
                                unsigned int&            segments_count,
                                OffsetType&              min_segment_length)
{
    constexpr OffsetType                         min_segment_length_bound = 1;
    constexpr OffsetType                         max_segment_length_bound = 1000;
    common::uniform_int_distribution<OffsetType> segment_length_dis(min_segment_length_bound,
                                                                     max_segment_length_bound);

    offsets.clear();
    segments_count = 0;
    size_t offset  = 0;
    while(offset < size)
    {
        const size_t segment_length = segment_length_dis(rng_engine);
        offsets.push_back(static_cast<OffsetType>(offset));
        segments_count++;
        offset += segment_length;
    }
    offsets.push_back(static_cast<OffsetType>(size));

    min_segment_length = static_cast<OffsetType>(size);
    for(unsigned int i = 0; i < segments_count; ++i)
    {
        min_segment_length = std::min<OffsetType>(min_segment_length, offsets[i + 1] - offsets[i]);
    }
}

template<bool Descending, class InputVector, class OffsetVector, class OutputVector, class SizeOut>
void inline compare_k_segmented(InputVector const&  input,
                                OffsetVector const& offsets,
                                OutputVector const& output,
                                SizeOut              k)
{
    using key_type                   = typename InputVector::value_type;
    const std::size_t segments_count = offsets.size() - 1;
    for(std::size_t segment = 0; segment < segments_count; ++segment)
    {
        SCOPED_TRACE(testing::Message() << "with segment = " << segment);
        std::vector<key_type> segment_input(input.begin() + offsets[segment],
                                            input.begin() + offsets[segment + 1]);
        std::vector<key_type> segment_output(output.begin() + segment * k,
                                             output.begin() + (segment + 1) * k);
        compare_k<Descending>(segment_input, segment_output, k);
    }
}

template<bool Descending,
         class InputKeyVector,
         class InputValueVector,
         class OffsetVector,
         class OutputKeyVector,
         class OutputValueVector,
         class SizeOut>
void compare_pairs_k_segmented(InputKeyVector const&    keys_input,
                               InputValueVector const&  values_input,
                               OffsetVector const&      offsets,
                               OutputKeyVector const&   keys_output,
                               OutputValueVector const& values_output,
                               SizeOut                  k)
{
    using key_type                   = typename InputKeyVector::value_type;
    using value_type                 = typename InputValueVector::value_type;
    const std::size_t segments_count = offsets.size() - 1;
    for(std::size_t segment = 0; segment < segments_count; ++segment)
    {
        SCOPED_TRACE(testing::Message() << "with segment = " << segment);
        std::vector<key_type> segment_keys_input(keys_input.begin() + offsets[segment],
                                                 keys_input.begin() + offsets[segment + 1]);
        std::vector<value_type> segment_values_input(values_input.begin() + offsets[segment],
                                                      values_input.begin() + offsets[segment + 1]);
        std::vector<key_type> segment_keys_output(keys_output.begin() + segment * k,
                                                  keys_output.begin() + (segment + 1) * k);
        std::vector<value_type> segment_values_output(values_output.begin() + segment * k,
                                                       values_output.begin() + (segment + 1) * k);
        compare_pairs_k<Descending>(segment_keys_input,
                                    segment_values_input,
                                    segment_keys_output,
                                    segment_values_output,
                                    k);
    }
}

// ---------------------------------------------------------
// Test for ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceSegmentedTopkTests : public ::testing::Test
{
public:
    using key_type                              = typename Params::key_type;
    using value_type                            = typename Params::value_type;
    static constexpr bool descending            = Params::descending;
    using decomposer_t                          = typename Params::decomposer_t;
    using config                                = typename Params::config;
    static constexpr bool ordered               = Params::ordered;
    static constexpr bool deterministic         = Params::deterministic;
    static constexpr bool stable                = Params::stable;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

template<class T>
using grouped_params = ::testing::Types<
    // Non-ordered, non-deterministic, non-stable
    // Ascending
    DeviceSegmentedTopkParams<T, T, true>,
    // Descending
    DeviceSegmentedTopkParams<T, T, false>,

    // Ordered, non-deterministic, stable
    // Ascending
    DeviceSegmentedTopkParams<T,
                              T,
                              true,
                              rocprim::identity_decomposer,
                              rocprim::default_config,
                              true,
                              false,
                              true>,
    // Descending
    DeviceSegmentedTopkParams<T,
                              T,
                              false,
                              rocprim::identity_decomposer,
                              rocprim::default_config,
                              true,
                              false,
                              true>>;

using RocprimDeviceSegmentedTopkTestsParams =
    typename test_utils::merge_sequence<grouped_params<int8_t>,
                                        grouped_params<short>,
                                        grouped_params<int>,
                                        grouped_params<rocprim::int128_t>,
                                        grouped_params<long long>,
                                        grouped_params<rocprim::half>,
                                        grouped_params<float>,
                                        grouped_params<double>>::type;

TYPED_TEST_SUITE(RocprimDeviceSegmentedTopkTests, RocprimDeviceSegmentedTopkTestsParams);

// Basic TopK Segmented tests

TYPED_TEST(RocprimDeviceSegmentedTopkTests, SegmentedTopkKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                       = typename TestFixture::key_type;
    constexpr bool descending            = TestFixture::descending;
    using decomposer_t                   = typename TestFixture::decomposer_t;
    using config                         = typename TestFixture::config;
    using size_in_type                   = unsigned int;
    using size_out_type                  = unsigned int;
    constexpr bool ordered               = TestFixture::ordered;
    constexpr bool deterministic         = TestFixture::deterministic;
    constexpr bool stable                = TestFixture::stable;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_graphs            = TestFixture::use_graphs;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(unsigned int seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_in_type size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            engine_type rng_engine(seed_value);

            using offset_type = unsigned int;
            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            offset_type              min_segment_length = 0;
            generate_segmented_offsets(size, rng_engine, offsets, segments_count, min_segment_length);

            size_out_type k = 0;

            if(size > 0)
            {
                k = test_utils::get_random_value<size_out_type>(0,
                                                                 min_segment_length - 1,
                                                                 seed_value);
            }

            SCOPED_TRACE(testing::Message()
                         << "with k = " << k << " and segments_count = " << segments_count);

            common::device_ptr<offset_type> d_offsets(offsets);

            std::vector<key_type> input(size);
            test_utils::generate_key_input(input.begin(), size, rng_engine);

            const size_t output_size = static_cast<size_t>(k) * segments_count;

            common::device_ptr<key_type> d_input;
            common::device_ptr<key_type> d_output;

            if(!d_input.resize_with_memory_check(size)
               || !d_output.resize_with_memory_check(output_size))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }
            d_input.store(input);

            const auto input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());

            // decomposer
            decomposer_t decomposer;

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK((rocprim::segmented_topk<config, descending, ordered, deterministic, stable>(
                nullptr,
                temp_storage_size_bytes,
                input_it,
                d_output.get(),
                size,
                k,
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                decomposer,
                stream,
                debug_synchronous)));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;

            if(use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK((rocprim::segmented_topk<config, descending, ordered, deterministic, stable>(
                d_temp_storage.get(),
                temp_storage_size_bytes,
                input_it,
                d_output.get(),
                size,
                k,
                segments_count,
                d_offsets.get(),
                d_offsets.get() + 1,
                decomposer,
                stream,
                debug_synchronous)));

            if(use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto output = d_output.load();
            if(size > 0)
            {
                compare_k_segmented<descending>(input, offsets, output, k);
            }

            if(use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}

TYPED_TEST(RocprimDeviceSegmentedTopkTests, SegmentedTopkPairs)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                       = typename TestFixture::key_type;
    using value_type                     = typename TestFixture::key_type;
    constexpr bool descending            = TestFixture::descending;
    using decomposer_t                   = typename TestFixture::decomposer_t;
    using config                         = typename TestFixture::config;
    using size_in_type                   = unsigned int;
    using size_out_type                  = unsigned int;
    constexpr bool ordered               = TestFixture::ordered;
    constexpr bool deterministic         = TestFixture::deterministic;
    constexpr bool stable                = TestFixture::stable;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_graphs            = TestFixture::use_graphs;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    for(unsigned int seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_in_type size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Generate data
            engine_type rng_engine(seed_value);

            using offset_type = unsigned int;

            std::vector<offset_type> offsets;
            unsigned int             segments_count     = 0;
            offset_type              min_segment_length = 0;
            generate_segmented_offsets(size, rng_engine, offsets, segments_count, min_segment_length);

            size_out_type k = 0;

            if(size > 0)
            {
                k = test_utils::get_random_value<size_out_type>(0,
                                                                 min_segment_length - 1,
                                                                 seed_value);
            }

            SCOPED_TRACE(testing::Message()
                         << "with k = " << k << " and segments_count = " << segments_count);

            std::vector<key_type> keys_input(size);
            test_utils::generate_key_input(keys_input.begin(), size, rng_engine);
            std::vector<value_type> values_input(size);
            test_utils::iota(values_input.begin(), values_input.end(), 0);

            const size_t output_size = static_cast<size_t>(k) * segments_count;

            common::device_ptr<key_type>   d_keys_input;
            common::device_ptr<value_type> d_values_input;
            common::device_ptr<key_type>   d_keys_output;
            common::device_ptr<value_type> d_values_output;

            if(!d_keys_input.resize_with_memory_check(size)
               || !d_values_input.resize_with_memory_check(size)
               || !d_keys_output.resize_with_memory_check(output_size)
               || !d_values_output.resize_with_memory_check(output_size))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }
            d_keys_input.store(keys_input);
            d_values_input.store(values_input);

            const auto keys_input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_keys_input.get());
            const auto values_input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(
                    d_values_input.get());

            common::device_ptr<offset_type> d_offsets(offsets);

            // decomposer
            decomposer_t decomposer;

            // Get size of d_temp_storage
            size_t temp_storage_size_bytes;
            HIP_CHECK(
                (rocprim::segmented_topk_pairs<config, descending, ordered, deterministic, stable>(
                    nullptr,
                    temp_storage_size_bytes,
                    keys_input_it,
                    d_keys_output.get(),
                    values_input_it,
                    d_values_output.get(),
                    size,
                    k,
                    segments_count,
                    d_offsets.get(),
                    d_offsets.get() + 1,
                    decomposer,
                    stream,
                    debug_synchronous)));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            common::device_ptr<void> d_temp_storage;

            if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
            {
                std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
                break;
            }

            test_utils::GraphHelper gHelper;

            if(use_graphs)
            {
                gHelper.startStreamCapture(stream);
            }

            // Run
            HIP_CHECK(
                (rocprim::segmented_topk_pairs<config, descending, ordered, deterministic, stable>(
                    d_temp_storage.get(),
                    temp_storage_size_bytes,
                    keys_input_it,
                    d_keys_output.get(),
                    values_input_it,
                    d_values_output.get(),
                    size,
                    k,
                    segments_count,
                    d_offsets.get(),
                    d_offsets.get() + 1,
                    decomposer,
                    stream,
                    debug_synchronous)));

            if(use_graphs)
            {
                gHelper.createAndLaunchGraph(stream);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Copy output to host
            const auto keys_output   = d_keys_output.load();
            const auto values_output = d_values_output.load();

            if(use_graphs)
            {
                gHelper.cleanupGraphHelper();
                HIP_CHECK(hipStreamDestroy(stream));
            }
            if(size <= 0)
            {
                continue;
            }

            if constexpr(stable)
            {
                compare_pairs_k_segmented<descending>(keys_input,
                                                       values_input,
                                                       offsets,
                                                       keys_output,
                                                       values_output,
                                                       k);
            }
            else
            {
                for(std::size_t segment = 0; segment < segments_count; ++segment)
                {
                    SCOPED_TRACE(testing::Message() << "with segment = " << segment);

                    // Create input pairs for this segment
                    std::unordered_multiset<std::pair<key_type, value_type>,
                                            pair_hash,
                                            pair_comp<std::pair<key_type, value_type>>>
                        h_input_map;
                    for(size_t i = offsets[segment]; i < offsets[segment + 1]; ++i)
                    {
                        h_input_map.insert({keys_input[i], values_input[i]});
                    }

                    // Varify keys & values are matching each other
                    for(unsigned int i = 0; i < k; ++i)
                    {
                        const auto output_index = segment * k + i;
                        auto       range        = h_input_map.equal_range(
                            std::pair<key_type, value_type>{keys_output[output_index],
                                                            values_output[output_index]});
                        ASSERT_NE(range.first, range.second);
                    }

                    // Check keys are correct for this segment
                    std::vector<key_type> segment_keys_input(keys_input.begin()
                                                                  + offsets[segment],
                                                              keys_input.begin()
                                                                  + offsets[segment + 1]);
                    std::vector<key_type> segment_keys_output(keys_output.begin()
                                                                   + segment * k,
                                                               keys_output.begin()
                                                                   + (segment + 1) * k);
                    compare_k<descending>(segment_keys_input, segment_keys_output, k);
                }
            }
        }
    }
}

// Large sizes Segmented TopK tests

template<class config,
         bool descending,
         bool stable,
         class decomposer_t,
         bool use_graphs,
         bool use_indirect_iterator,
         class size_in_type  = unsigned int,
         class size_out_type = unsigned int>
void segmented_topk_large_sizes_test(bool debug_synchronous)
{
    using key_type = size_in_type;

    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    unsigned int seed_value = seeds[1];
    SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

    std::vector<size_in_type> sizes = {(size_in_type{1} << 30) - 1, (size_in_type{1} << 32) + 15};

    for(auto size : sizes)
    {
        hipStream_t stream = 0; // default
        if(use_graphs)
        {
            // Default stream does not support hipGraph stream capture, so create one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }

        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_out_type k = 1;

        SCOPED_TRACE(testing::Message() << "with k = " << k);

        using offset_type = unsigned int;
        std::vector<offset_type> offsets;
        unsigned int             segments_count = 0;
        size_t                   offset         = 0;
        while(offset < size)
        {
            const size_t segment_length = size;
            offsets.push_back(offset);
            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        common::device_ptr<offset_type> d_offsets(offsets);

        SCOPED_TRACE(testing::Message() << "with segments = " << segments_count);

        test_utils::MemCheck memcheck;

        // Generate data
        MEMCHECK_OR_BREAK_ALLOC_HOST(key_type, size)
        std::vector<key_type> input(size);
        test_utils::iota(input.begin(), input.end(), 0);

        common::device_ptr<key_type> d_input;
        common::device_ptr<key_type> d_output;

        MEMCHECK_OR_BREAK_ALLOC_DEVICE(key_type, size)
        MEMCHECK_OR_BREAK_ALLOC_DEVICE(key_type, k)
        if(!d_input.resize_with_memory_check(size) || !d_output.resize_with_memory_check(k))
        {
            std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
            break;
        }
        d_input.store(input);

        const auto input_it
            = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input.get());

        // decomposer
        decomposer_t decomposer;

        // Get size of d_temp_storage
        size_t temp_storage_size_bytes;
        HIP_CHECK((rocprim::segmented_topk<config, descending, true, false>(nullptr,
                                                                            temp_storage_size_bytes,
                                                                            input_it,
                                                                            d_output.get(),
                                                                            size,
                                                                            k,
                                                                            segments_count,
                                                                            d_offsets.get(),
                                                                            d_offsets.get() + 1,
                                                                            decomposer,
                                                                            stream,
                                                                            debug_synchronous)));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);

        // allocate temporary storage
        common::device_ptr<void> d_temp_storage;

        MEMCHECK_OR_BREAK_ALLOC_DEVICE_BYTES(temp_storage_size_bytes)
        if(!d_temp_storage.resize_with_memory_check(temp_storage_size_bytes))
        {
            std::cout << "Out of memory. Skipping test for size = " << size << std::endl;
            break;
        }

        test_utils::GraphHelper gHelper;

        if(use_graphs)
        {
            gHelper.startStreamCapture(stream);
        }

        // Run
        HIP_CHECK((rocprim::segmented_topk<config, descending, true, false, stable>(
            d_temp_storage.get(),
            temp_storage_size_bytes,
            input_it,
            d_output.get(),
            size,
            k,
            segments_count,
            d_offsets.get(),
            d_offsets.get() + 1,
            decomposer,
            stream,
            debug_synchronous)));

        if(use_graphs)
        {
            gHelper.createAndLaunchGraph(stream);
        }

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Validate output
        const auto output = d_output.load()[0];
        if constexpr(descending)
        {
            test_utils::assert_eq(output, size_in_type{size - 1});
        }
        else
        {
            test_utils::assert_eq(output, size_in_type{0});
        }

        if(use_graphs)
        {
            gHelper.cleanupGraphHelper();
            HIP_CHECK(hipStreamDestroy(stream));
        }
    }
}

TEST(RocprimDeviceSegmentedTopkTests, SegmentedTopkLargeSizesStable)
{
    constexpr bool descending            = false;
    constexpr bool stable                = true;
    using decomposer_t                   = rocprim::identity_decomposer;
    using config                         = rocprim::default_config;
    using size_in_type                   = std::size_t;
    using size_out_type                  = std::size_t;
    const bool     debug_synchronous     = false;
    constexpr bool use_graphs            = false;
    constexpr bool use_indirect_iterator = false;

    // Radix Segmented TopK
    // Same data type for 'size' and 'k'
    segmented_topk_large_sizes_test<config,
                                    descending,
                                    stable,
                                    decomposer_t,
                                    use_graphs,
                                    use_indirect_iterator,
                                    size_in_type,
                                    size_out_type>(debug_synchronous);
}

TEST(RocprimDeviceSegmentedTopkTests, TopkLargeSizesUnstable)
{    
    constexpr bool descending            = false;
    constexpr bool stable                = false;
    using decomposer_t                   = rocprim::identity_decomposer;
    using config                         = rocprim::default_config;
    using size_in_type                   = std::size_t;
    using size_out_type                  = std::size_t;
    const bool     debug_synchronous     = false;
    constexpr bool use_graphs            = false;
    constexpr bool use_indirect_iterator = false;

    // Radix Segmented TopK
    // Same data type for 'size' and 'k'
    segmented_topk_large_sizes_test<config,
                                    descending,
                                    stable,
                                    decomposer_t,
                                    use_graphs,
                                    use_indirect_iterator,
                                    size_in_type,
                                    size_out_type>(debug_synchronous);
}
