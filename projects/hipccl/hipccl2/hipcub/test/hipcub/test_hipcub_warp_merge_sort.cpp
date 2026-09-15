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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"
#include "test_utils.hpp"

// hipcub API
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/warp/warp_merge_sort.hpp>

#include <limits>
#include <type_traits>
#include <utility>

#include <cstdio>

template<
    typename Key,
    typename Value,
    unsigned int LogicalWarpSize,
    unsigned int ItemsPerThread = 1u,
    unsigned int BlockSize = 256u,
    typename CompareFunction = test_utils::less,
    bool Stable = false
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr unsigned int logical_warp_size = LogicalWarpSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int block_size = BlockSize;
    using compare_function = CompareFunction;
    static constexpr bool stable = Stable;
};

template<class Params>
class HipcubWarpMergeSort : public ::testing::Test {
public:
    using params = Params;
};

using Params = ::testing::Types<
    params<unsigned int, int, 2u>,
    params<float, char, 16u, 1u, 64u>,
    params<double, int, 32u, 1u, 32u>,
    params<short, int, 64u, 1u, 512u>,
    params<test_utils::half, int, 32u, 1u, 32u>,
    params<test_utils::bfloat16, int, 32u, 1u, 32u>,

    params<test_utils::custom_test_type<float>, unsigned long long, 32u, 2u>,
    params<test_utils::custom_test_type<float>, int, 32u, 4u>,
    params<test_utils::custom_test_type<float>, int, 32u, 8u>,
    params<long long, short, 8u, 4u, 256u, test_utils::less, true>,
    params<int, test_utils::custom_test_type<short>, 32u, 7u, 256u, test_utils::greater>>;

TYPED_TEST_SUITE(HipcubWarpMergeSort, Params);

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Compare>
__device__ auto sort_keys_full_test(Key* keys, Compare compare_op)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_tid     = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * items_per_block;
    Key thread_keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(flat_tid, keys + block_offset, thread_keys);

    constexpr unsigned int warps_per_block = BlockSize / LogicalWarpSize;
    const unsigned int     warp_id         = threadIdx.x / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<Key, ItemsPerThread, LogicalWarpSize>;
    __shared__ typename warp_merge_sort::TempStorage storage[warps_per_block];

    warp_merge_sort wsort{storage[warp_id]};
    if HIPCUB_IF_CONSTEXPR(Stable)
    {
        wsort.StableSort(thread_keys, compare_op);
    } else
    {
        wsort.Sort(thread_keys, compare_op);
    }

    hipcub::StoreDirectBlocked(flat_tid, keys + block_offset, thread_keys);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Compare>
__device__ auto sort_keys_full_test(Key* /*keys*/, Compare /*compare_op*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Compare>
__global__ __launch_bounds__(BlockSize) void sort_keys_full(Key* keys, Compare compare_op)
{
    sort_keys_full_test<BlockSize, LogicalWarpSize, ItemsPerThread, Stable>(keys, compare_op);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Value,
         typename Compare>
__device__ auto sort_keys_values_full_test(Key* keys, Value* values, Compare compare_op)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_tid     = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * items_per_block;
    Key   thread_keys  [ItemsPerThread];
    Value thread_values[ItemsPerThread];
    hipcub::LoadDirectBlocked(flat_tid, keys + block_offset, thread_keys);
    hipcub::LoadDirectBlocked(flat_tid, values + block_offset, thread_values);

    constexpr unsigned int warps_per_block = BlockSize / LogicalWarpSize;
    const unsigned int     warp_id         = threadIdx.x / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<Key, ItemsPerThread, LogicalWarpSize, Value>;
    __shared__ typename warp_merge_sort::TempStorage storage[warps_per_block];

    warp_merge_sort wsort{storage[warp_id]};
    if HIPCUB_IF_CONSTEXPR(Stable)
    {
        wsort.StableSort(thread_keys, thread_values, compare_op);
    } else
    {
        wsort.Sort(thread_keys, thread_values, compare_op);
    }

    hipcub::StoreDirectBlocked(flat_tid, keys + block_offset, thread_keys);
    hipcub::StoreDirectBlocked(flat_tid, values + block_offset, thread_values);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Value,
         typename Compare>
__device__ auto sort_keys_values_full_test(Key* /*keys*/, Value* /*values*/, Compare /*compare_op*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Value,
         typename Compare>
__global__ __launch_bounds__(BlockSize) void sort_keys_values_full(Key*    keys,
                                                                   Value*  values,
                                                                   Compare compare_op)
{
    sort_keys_values_full_test<BlockSize, LogicalWarpSize, ItemsPerThread, Stable>(keys,
                                                                                   values,
                                                                                   compare_op);
}

// Provides the value that would be sorted last according to the comparison function
template <typename Compare, typename T>
struct sort_last;

template <typename T>
struct sort_last<test_utils::less, T> {
    static constexpr T value = std::numeric_limits<T>::max();
};

template <typename T>
struct sort_last<test_utils::greater, T> {
    static constexpr T value = std::numeric_limits<T>::lowest();
};

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Compare>
__device__ auto
    sort_keys_segmented_test(Key* keys, const unsigned int* segment_sizes, Compare compare)
        -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int max_segment_size = LogicalWarpSize * ItemsPerThread;
    constexpr unsigned int segments_per_block = BlockSize / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<Key, ItemsPerThread, LogicalWarpSize>;
    __shared__ typename warp_merge_sort::TempStorage storage[segments_per_block];

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    warp_merge_sort wsort{storage[warp_id]};

    const unsigned int segment_id = blockIdx.x * segments_per_block + warp_id;

    const unsigned int segment_size = segment_sizes[segment_id];
    const unsigned int warp_offset = segment_id * max_segment_size;
    Key thread_keys[ItemsPerThread];

    const unsigned int flat_tid = wsort.get_linear_tid();
    hipcub::LoadDirectBlocked(flat_tid, keys + warp_offset, thread_keys, segment_size);

    const Key oob_default = sort_last<Compare, Key>::value;
    if HIPCUB_IF_CONSTEXPR(Stable)
    {
        wsort.StableSort(thread_keys, compare, segment_size, oob_default);
    } else
    {
        wsort.Sort(thread_keys, compare, segment_size, oob_default);
    }

    hipcub::StoreDirectBlocked(flat_tid, keys + warp_offset, thread_keys, segment_size);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Compare>
__device__ auto sort_keys_segmented_test(Key* /*keys*/,
                                         const unsigned int* /*segment_sizes*/,
                                         Compare /*compare*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Compare>
__global__ __launch_bounds__(BlockSize) void sort_keys_segmented(Key*                keys,
                                                                 const unsigned int* segment_sizes,
                                                                 Compare             compare)
{
    sort_keys_segmented_test<BlockSize, LogicalWarpSize, ItemsPerThread, Stable>(keys,
                                                                                 segment_sizes,
                                                                                 compare);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Value,
         typename Compare>
__device__ auto sort_keys_values_segmented_test(Key*                keys,
                                                Value*              values,
                                                const unsigned int* segment_sizes,
                                                Compare             compare)
    -> std::enable_if_t<test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    constexpr unsigned int max_segment_size = LogicalWarpSize * ItemsPerThread;
    constexpr unsigned int segments_per_block = BlockSize / LogicalWarpSize;

    using warp_merge_sort = hipcub::WarpMergeSort<Key, ItemsPerThread, LogicalWarpSize, Value>;
    __shared__ typename warp_merge_sort::TempStorage storage[segments_per_block];

    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    warp_merge_sort wsort{storage[warp_id]};

    const unsigned int segment_id = blockIdx.x * segments_per_block + warp_id;

    const unsigned int segment_size = segment_sizes[segment_id];
    const unsigned int warp_offset = segment_id * max_segment_size;
    Key   thread_keys[ItemsPerThread];
    Value thread_values[ItemsPerThread];

    const unsigned int flat_tid = wsort.get_linear_tid();
    hipcub::LoadDirectBlocked(flat_tid, keys + warp_offset, thread_keys, segment_size);
    hipcub::LoadDirectBlocked(flat_tid, values + warp_offset, thread_values, segment_size);

    const Key oob_default = sort_last<Compare, Key>::value;
    if HIPCUB_IF_CONSTEXPR(Stable)
    {
        wsort.StableSort(thread_keys, thread_values, compare, segment_size, oob_default);
    } else
    {
        wsort.Sort(thread_keys, thread_values, compare, segment_size, oob_default);
    }

    hipcub::StoreDirectBlocked(flat_tid, keys + warp_offset, thread_keys, segment_size);
    hipcub::StoreDirectBlocked(flat_tid, values + warp_offset, thread_values, segment_size);
}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Value,
         typename Compare>
__device__ auto sort_keys_values_segmented_test(Key* /*keys*/,
                                                Value* /*values*/,
                                                const unsigned int* /*segment_sizes*/,
                                                Compare /*compare*/)
    -> std::enable_if_t<!test_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int LogicalWarpSize,
         unsigned int ItemsPerThread,
         bool         Stable,
         typename Key,
         typename Value,
         typename Compare>
__global__ __launch_bounds__(BlockSize) void sort_keys_values_segmented(
    Key* keys, Value* values, const unsigned int* segment_sizes, Compare compare)
{
    sort_keys_values_segmented_test<BlockSize, LogicalWarpSize, ItemsPerThread, Stable>(
        keys,
        values,
        segment_sizes,
        compare);
}

TYPED_TEST(HipcubWarpMergeSort, SortKeysSegmented)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using params = typename TestFixture::params;
    using key_type = typename params::key_type;

    constexpr auto block_size = params::block_size;
    constexpr auto warp_size =  params::logical_warp_size;
    constexpr auto warps_per_block =  block_size / warp_size;
    constexpr auto items_per_thread = params::items_per_thread;

    constexpr auto items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const auto current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    // Check if warp size is supported
    if(warp_size > current_device_warp_size ||
       (current_device_warp_size != HIPCUB_WARP_SIZE_32 && current_device_warp_size != HIPCUB_WARP_SIZE_64))
    {
        GTEST_SKIP() << "Unsupported test warp size / computed block size: " << warp_size << "/"
                     << block_size << ". Current device warp size: " << current_device_warp_size;
    }

    constexpr auto num_blocks = 97;
    constexpr auto num_warps  = num_blocks * warps_per_block;
    constexpr auto max_segment_size = warp_size * items_per_thread;
    constexpr size_t size = num_blocks * items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data, not const because sorted results are copied back to it
        using wrapped_type = typename test_utils::inner_type<key_type>::type;

        auto keys = test_utils::is_floating_point<wrapped_type>::value
                        ? test_utils::get_random_data<key_type>(
                            size,
                            test_utils::convert_to_device<wrapped_type>(-1000),
                            test_utils::convert_to_device<wrapped_type>(1000),
                            seed_value)
                        : test_utils::get_random_data<key_type>(
                            size,
                            std::numeric_limits<wrapped_type>::lowest(),
                            std::numeric_limits<wrapped_type>::max(),
                            seed_value);

        const auto segment_sizes = test_utils::get_random_data<unsigned int>(
            num_warps, 0u, max_segment_size, ~seed_value);

        const auto compare = typename params::compare_function{};

        // Calculate expected results on host
        const auto expected = [&]{;
            auto result = keys;
            unsigned int segment = 0;
            for(const auto segment_size : segment_sizes) {
                std::stable_sort(result.begin() + segment * max_segment_size,
                                 result.begin() + segment * max_segment_size + segment_size, compare);
                ++segment;
            }
            return result;
        }();

        key_type* device_keys = nullptr;
        unsigned int* device_segment_sizes = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys, keys.size() * sizeof(keys[0])));
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_segment_sizes,
            segment_sizes.size() * sizeof(segment_sizes[0])));
        HIP_CHECK(hipMemcpy(device_keys, keys.data(),
                            keys.size() * sizeof(keys[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_segment_sizes, segment_sizes.data(),
                            segment_sizes.size() * sizeof(segment_sizes[0]),
                            hipMemcpyHostToDevice));

        sort_keys_segmented<block_size, warp_size, items_per_thread, params::stable>
            <<<dim3(num_blocks), dim3(block_size), 0, 0>>>(device_keys,
                                                           device_segment_sizes,
                                                           compare);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(
            hipMemcpy(
                keys.data(), device_keys,
                keys.size() * sizeof(keys[0]),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys[i]),
                      test_utils::convert_to_native(expected[i]));
        }

        HIP_CHECK(hipFree(device_keys));
        HIP_CHECK(hipFree(device_segment_sizes));
    }
}

TYPED_TEST(HipcubWarpMergeSort, SortKeysValuesSegmented)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using params = typename TestFixture::params;
    using key_type = typename params::key_type;
    using value_type = typename params::value_type;

    constexpr auto block_size = params::block_size;
    constexpr auto warp_size =  params::logical_warp_size;
    constexpr auto warps_per_block =  block_size / warp_size;
    constexpr auto items_per_thread = params::items_per_thread;

    constexpr auto items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const auto current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    // Check if warp size is supported
    if(warp_size > current_device_warp_size ||
       (current_device_warp_size != HIPCUB_WARP_SIZE_32 && current_device_warp_size != HIPCUB_WARP_SIZE_64))
    {
        GTEST_SKIP() << "Unsupported test warp size / computed block size: " << warp_size << "/"
                     << block_size << ". Current device warp size: " << current_device_warp_size;
    }

    constexpr auto num_blocks = 97;
    constexpr auto num_warps  = num_blocks * warps_per_block;
    constexpr auto max_segment_size = warp_size * items_per_thread;
    constexpr size_t size = num_blocks * items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data, not const because sorted results are copied back to it
        using wrapped_type = typename test_utils::inner_type<key_type>::type;

        auto keys = test_utils::is_floating_point<wrapped_type>::value
                        ? test_utils::get_random_data<key_type>(
                            size,
                            test_utils::convert_to_device<wrapped_type>(-1000),
                            test_utils::convert_to_device<wrapped_type>(1000),
                            seed_value)
                        : test_utils::get_random_data<key_type>(
                            size,
                            std::numeric_limits<wrapped_type>::lowest(),
                            std::numeric_limits<wrapped_type>::max(),
                            seed_value);

        using value_wrapped_type = typename test_utils::inner_type<value_type>::type;

        auto values = test_utils::is_floating_point<value_wrapped_type>::value
                          ? test_utils::get_random_data<value_type>(
                              size,
                              test_utils::convert_to_device<value_wrapped_type>(-1000),
                              test_utils::convert_to_device<value_wrapped_type>(1000),
                              seed_value)
                          : test_utils::get_random_data<value_type>(
                              size,
                              std::numeric_limits<value_wrapped_type>::lowest(),
                              std::numeric_limits<value_wrapped_type>::max(),
                              seed_value ^ (seed_value >> 1ul));

        const auto segment_sizes = test_utils::get_random_data<unsigned int>(
            num_warps, 0u, max_segment_size, ~seed_value);

        const auto compare = typename params::compare_function{};

        // Calculate expected results on host
        const auto expected = [&]{
            using pair = std::pair<key_type, value_type>;
            auto result = std::vector<pair>(keys.size());
            for(size_t i = 0; i < keys.size(); ++i) {
                result[i].first  = keys[i];
                result[i].second = values[i];
            }
            unsigned int segment = 0;
            for(const auto segment_size : segment_sizes) {
                std::stable_sort(result.begin() + segment * max_segment_size,
                                 result.begin() + segment * max_segment_size + segment_size,
                                 [&compare](const pair &lhs, const pair &rhs)
                                 { return compare(lhs.first, rhs.first); });
                ++segment;
            }
            return result;
        }();

        key_type* device_keys              = nullptr;
        value_type* device_values          = nullptr;
        unsigned int* device_segment_sizes = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys, keys.size() * sizeof(keys[0])));
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_values, values.size() * sizeof(values[0])));
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_segment_sizes,
            segment_sizes.size() * sizeof(segment_sizes[0])));
        HIP_CHECK(hipMemcpy(device_keys, keys.data(),
                            keys.size() * sizeof(keys[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_values, values.data(),
                            values.size() * sizeof(values[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_segment_sizes, segment_sizes.data(),
                            segment_sizes.size() * sizeof(segment_sizes[0]),
                            hipMemcpyHostToDevice));

        sort_keys_values_segmented<block_size, warp_size, items_per_thread, params::stable>
            <<<dim3(num_blocks), dim3(block_size), 0, 0>>>(device_keys,
                                                           device_values,
                                                           device_segment_sizes,
                                                           compare);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(
            hipMemcpy(
                keys.data(), device_keys,
                keys.size() * sizeof(keys[0]),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                values.data(), device_values,
                values.size() * sizeof(values[0]),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys[i]),
                      test_utils::convert_to_native(expected[i].first));
            ASSERT_EQ(values[i], expected[i].second);
        }

        HIP_CHECK(hipFree(device_keys));
        HIP_CHECK(hipFree(device_values));
        HIP_CHECK(hipFree(device_segment_sizes));
    }
}

TYPED_TEST(HipcubWarpMergeSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using params = typename TestFixture::params;
    using key_type = typename params::key_type;

    constexpr auto block_size = params::block_size;
    constexpr auto warp_size =  params::logical_warp_size;
    constexpr auto warps_per_block =  block_size / warp_size;
    constexpr auto items_per_thread = params::items_per_thread;

    constexpr auto items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const auto current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    // Check if warp size is supported
    if(warp_size > current_device_warp_size ||
       (current_device_warp_size != HIPCUB_WARP_SIZE_32 && current_device_warp_size != HIPCUB_WARP_SIZE_64))
    {
        GTEST_SKIP() << "Unsupported test warp size / computed block size: " << warp_size << "/"
                     << block_size << ". Current device warp size: " << current_device_warp_size;
    }

    constexpr auto num_blocks = 337;
    constexpr auto num_warps  = num_blocks * warps_per_block;
    constexpr auto items_per_warp = warp_size * items_per_thread;
    constexpr auto size = num_blocks * items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        using wrapped_type = typename test_utils::inner_type<key_type>::type;

        auto keys = test_utils::is_floating_point<wrapped_type>::value
                        ? test_utils::get_random_data<key_type>(
                            size,
                            test_utils::convert_to_device<wrapped_type>(-1000),
                            test_utils::convert_to_device<wrapped_type>(1000),
                            seed_value)
                        : test_utils::get_random_data<key_type>(
                            size,
                            std::numeric_limits<wrapped_type>::lowest(),
                            std::numeric_limits<wrapped_type>::max(),
                            seed_value);

        const auto compare = typename params::compare_function{};

        // Calculate expected results on host
        const auto expected = [&]() {
            auto result = keys;
            for (unsigned int warp = 0; warp < num_warps; ++warp) {
                std::stable_sort(result.begin() + warp * items_per_warp,
                                 result.begin() + warp * items_per_warp + items_per_warp,
                                 compare);
            }
            return result;
        }();

        key_type*   device_keys   = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys, keys.size() * sizeof(keys[0])));
        HIP_CHECK(hipMemcpy(device_keys, keys.data(),
                            keys.size() * sizeof(keys[0]),
                            hipMemcpyHostToDevice));

        sort_keys_full<block_size, warp_size, items_per_thread, params::stable>
            <<<dim3(num_blocks), dim3(block_size), 0, 0>>>(device_keys, compare);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(
            hipMemcpy(
                keys.data(), device_keys,
                keys.size() * sizeof(keys[0]),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys[i]),
                      test_utils::convert_to_native(expected[i]));
        }

        HIP_CHECK(hipFree(device_keys));
    }
}

TYPED_TEST(HipcubWarpMergeSort, SortKeysValues)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using params = typename TestFixture::params;
    using key_type = typename params::key_type;
    using value_type = typename params::value_type;

    constexpr auto block_size = params::block_size;
    constexpr auto warp_size =  params::logical_warp_size;
    constexpr auto warps_per_block =  block_size / warp_size;
    constexpr auto items_per_thread = params::items_per_thread;

    constexpr auto items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const auto current_device_warp_size = HIPCUB_HOST_WARP_THREADS;

    // Check if warp size is supported
    if(warp_size > current_device_warp_size ||
       (current_device_warp_size != HIPCUB_WARP_SIZE_32 && current_device_warp_size != HIPCUB_WARP_SIZE_64))
    {
        GTEST_SKIP() << "Unsupported test warp size / computed block size: " << warp_size << "/"
                     << block_size << ". Current device warp size: " << current_device_warp_size;
    }

    constexpr auto num_blocks = 269;
    constexpr auto num_warps  = num_blocks * warps_per_block;
    constexpr auto items_per_warp = warp_size * items_per_thread;
    constexpr auto size = num_blocks * items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        using wrapped_type = typename test_utils::inner_type<key_type>::type;

        auto keys = test_utils::is_floating_point<wrapped_type>::value
                        ? test_utils::get_random_data<key_type>(
                            size,
                            test_utils::convert_to_device<wrapped_type>(-1000),
                            test_utils::convert_to_device<wrapped_type>(1000),
                            seed_value)
                        : test_utils::get_random_data<key_type>(
                            size,
                            std::numeric_limits<wrapped_type>::lowest(),
                            std::numeric_limits<wrapped_type>::max(),
                            seed_value);

        using value_wrapped_type = typename test_utils::inner_type<value_type>::type;

        auto values = test_utils::is_floating_point<value_wrapped_type>::value
                          ? test_utils::get_random_data<value_type>(
                              size,
                              test_utils::convert_to_device<value_wrapped_type>(-1000),
                              test_utils::convert_to_device<value_wrapped_type>(1000),
                              seed_value)
                          : test_utils::get_random_data<value_type>(
                              size,
                              std::numeric_limits<value_wrapped_type>::lowest(),
                              std::numeric_limits<value_wrapped_type>::max(),
                              seed_value ^ (seed_value >> 1ul));

        const auto compare = typename params::compare_function{};

        // Calculate expected results on host
        const auto expected = [&]() {
            using pair = std::pair<key_type, value_type>;
            auto result = std::vector<pair>{size};
            for(size_t i = 0; i < keys.size(); ++i) {
                result[i].first = keys[i];
                result[i].second = values[i];
            }

            for (unsigned int warp = 0; warp < num_warps; ++warp) {
                std::stable_sort(result.begin() + warp * items_per_warp,
                                 result.begin() + warp * items_per_warp + items_per_warp,
                                 [&compare](const pair &lhs, const pair &rhs) {
                                   return compare(lhs.first, rhs.first);
                                 });
            }
            return result;
        }();

        key_type*   device_keys   = nullptr;
        value_type* device_values = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys, keys.size() * sizeof(keys[0])));
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_values,
            values.size() * sizeof(values[0])));
        HIP_CHECK(hipMemcpy(device_keys, keys.data(),
                            keys.size() * sizeof(keys[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_values, values.data(),
                            values.size() * sizeof(values[0]),
                            hipMemcpyHostToDevice));

        sort_keys_values_full<block_size, warp_size, items_per_thread, params::stable>
            <<<dim3(num_blocks), dim3(block_size), 0, 0>>>(device_keys, device_values, compare);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(
            hipMemcpy(
                keys.data(), device_keys,
                keys.size() * sizeof(keys[0]),
                hipMemcpyDeviceToHost
            )
        );
        HIP_CHECK(
            hipMemcpy(
                values.data(), device_values,
                values.size() * sizeof(values[0]),
                hipMemcpyDeviceToHost
            )
        );

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys[i]),
                      test_utils::convert_to_native(expected[i].first));
            ASSERT_EQ(values[i], expected[i].second);
        }

        HIP_CHECK(hipFree(device_keys));
        HIP_CHECK(hipFree(device_values));
    }
}
