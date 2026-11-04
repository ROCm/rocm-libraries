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

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/thread/thread_sort.hpp>
#include <hipcub/util_type.hpp>

#include <hip/hip_runtime.h>

template<
    typename Key,
    typename Value,
    unsigned int ItemsPerThread,
    typename CompareFunction = test_utils::less
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    using compare_function = CompareFunction;
};

template<class Params>
class HipcubThreadSort : public ::testing::Test {
public:
    using params = Params;
};

using Params = ::testing::Types<
    // Test that it does nothing
    params<int, int, 1U>,
    params<unsigned int, int, 2U>,
    params<int, int, 3U>,
    params<unsigned int, int, 4U>,
    params<unsigned short, char, 5U>,
    params<float, char, 6U, test_utils::greater>,
    params<int, short, 7U>,
    params<unsigned short, char, 8U>,
    params<unsigned long long, char, 9U>,
    params<unsigned char, int, 10U>,
    params<double, long long, 11U>,
    params<test_utils::custom_test_type<int>, test_utils::custom_test_type<char>, 4U>,
    params<test_utils::bfloat16, test_utils::bfloat16, 7U>,
    params<test_utils::half, test_utils::half, 6U>,
    //params<test_utils::half, long long, 6U>,
    params<test_utils::bfloat16, int, 7U>>;

TYPED_TEST_SUITE(HipcubThreadSort, Params);

template <unsigned int BlockSize, unsigned int ItemsPerThread, typename Key, typename Compare>
__global__
__launch_bounds__(BlockSize)
void sort_keys(Key* keys, Compare compare) {
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Key thread_keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(threadIdx.x, keys + block_offset, thread_keys);

    hipcub::NullType ignored_values[ItemsPerThread];
    hipcub::StableOddEvenSort(thread_keys, ignored_values, compare);

    hipcub::StoreDirectBlocked(threadIdx.x, keys + block_offset, thread_keys);
}

template <unsigned int BlockSize, unsigned int ItemsPerThread, typename Key, typename Value, typename Compare>
__global__
__launch_bounds__(BlockSize)
void sort_keys_values(Key* keys, Value* values, Compare compare) {
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Key   thread_keys[ItemsPerThread];
    Value thread_values[ItemsPerThread];
    hipcub::LoadDirectBlocked(threadIdx.x, keys + block_offset, thread_keys);
    hipcub::LoadDirectBlocked(threadIdx.x, values + block_offset, thread_values);

    hipcub::StableOddEvenSort(thread_keys, thread_values, compare);

    hipcub::StoreDirectBlocked(threadIdx.x, keys + block_offset, thread_keys);
    hipcub::StoreDirectBlocked(threadIdx.x, values + block_offset, thread_values);
}

TYPED_TEST(HipcubThreadSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using params = typename TestFixture::params;
    using key_type = typename params::key_type;

    constexpr unsigned int block_size = 256;
    constexpr auto items_per_thread   = params::items_per_thread;

    constexpr auto items_per_block = block_size * items_per_thread;

    constexpr auto num_blocks  = 337;
    constexpr auto num_threads = num_blocks * block_size;
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
                            test_utils::numeric_limits<wrapped_type>::lowest(),
                            test_utils::numeric_limits<wrapped_type>::max(),
                            seed_value);

        const auto compare = typename params::compare_function{};

        // Calculate expected results on host
        const auto expected = [&]() {
            auto result = keys;
            for (unsigned int thread = 0; thread < num_threads; ++thread) {
                std::stable_sort(result.begin() + thread * items_per_thread,
                                 result.begin() + thread * items_per_thread + items_per_thread,
                                 compare);
            }
            return result;
        }();

        key_type* device_keys = nullptr;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys, keys.size() * sizeof(keys[0])));
        HIP_CHECK(hipMemcpy(device_keys, keys.data(),
                            keys.size() * sizeof(keys[0]),
                            hipMemcpyHostToDevice));

        hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_keys<block_size, items_per_thread>), dim3(num_blocks),
                           dim3(block_size), 0, 0, device_keys, compare);
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

TYPED_TEST(HipcubThreadSort, SortKeysValues)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using params = typename TestFixture::params;
    using key_type = typename params::key_type;
    using value_type = typename params::value_type;

    constexpr unsigned int block_size = 256;
    constexpr auto items_per_thread   = params::items_per_thread;

    constexpr auto items_per_block = block_size * items_per_thread;

    constexpr auto num_blocks  = 269;
    constexpr auto num_threads = num_blocks * block_size;
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
                            test_utils::numeric_limits<wrapped_type>::lowest(),
                            test_utils::numeric_limits<wrapped_type>::max(),
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
                              test_utils::numeric_limits<value_wrapped_type>::lowest(),
                              test_utils::numeric_limits<value_wrapped_type>::max(),
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

            for (unsigned int thread = 0; thread < num_threads; ++thread) {
                std::stable_sort(result.begin() + thread * items_per_thread,
                                 result.begin() + thread * items_per_thread + items_per_thread,
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

        hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_keys_values<block_size, items_per_thread>), dim3(num_blocks),
                           dim3(block_size), 0, 0, device_keys, device_values,
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

            ASSERT_EQ(test_utils::convert_to_native(values[i]),
                      test_utils::convert_to_native(expected[i].second));
        }

        HIP_CHECK(hipFree(device_keys));
        HIP_CHECK(hipFree(device_values));
    }
}
