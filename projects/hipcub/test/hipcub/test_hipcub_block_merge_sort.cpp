// MIT License
//
// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// hipcub API
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_merge_sort.hpp>
#include <hipcub/block/block_store.hpp>

#include <algorithm>
#include <string>

#define ull unsigned long long

template<class Key,
         class Value,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class CompareFunction = test_utils::less,
         bool ToStriped        = false>
struct params
{
    using key_type                                 = Key;
    using value_type                               = Value;
    static constexpr unsigned int block_size       = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    using compare_function                         = CompareFunction;
    static constexpr bool to_striped               = ToStriped;
};

template<class Params>
class HipcubBlockMergeSort : public ::testing::Test
{
public:
    using params = Params;
};

using Params = ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1>,
    params<int, int, 128U, 1>,
    params<unsigned int, int, 256U, 1>,
    params<unsigned short, char, 1024U, 1, test_utils::greater>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<float, char, 64U, 2, test_utils::greater>,
    params<test_utils::half, test_utils::half, 64U, 2, test_utils::greater>,
    params<test_utils::bfloat16, test_utils::bfloat16, 64U, 2, test_utils::greater>,
    params<int, short, 128U, 4>,
    params<unsigned short, char, 256U, 7>,

    params<unsigned long long, char, 64U, 1, test_utils::less, false>,

    // Stability (a number of key values is lower than BlockSize * ItemsPerThread: some keys appear
    // multiple times with different values
    params<unsigned char, int, 512U, 2, test_utils::less, true>>;

TYPED_TEST_SUITE(HipcubBlockMergeSort, Params);

template<unsigned int BlockSize, unsigned int ItemsPerThread, class key_type, typename CompareOp>
__global__
__launch_bounds__(BlockSize)
void sort_key_kernel(key_type* device_keys_output, CompareOp compare_op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     lid             = hipThreadIdx_x;
    const unsigned int     block_offset    = hipBlockIdx_x * items_per_block;

    key_type keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);

    hipcub::BlockMergeSort<key_type, BlockSize, ItemsPerThread> bsort;
    bsort.Sort(keys, compare_op);

    hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
}

TYPED_TEST(HipcubBlockMergeSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using compare_function            = typename TestFixture::params::compare_function;
    constexpr size_t items_per_block  = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    const size_t size      = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_output;
        keys_output = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value);

        // Calculate expected results on host
        std::vector<key_type> expected(keys_output);
        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(expected.begin() + (i * items_per_block),
                             expected.begin() + ((i + 1) * items_per_block),
                             compare_function());
        }

        // Preparing device
        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output,
                                                     keys_output.size() * sizeof(key_type)));

        HIP_CHECK(hipMemcpy(device_keys_output,
                            keys_output.data(),
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_key_kernel<block_size, items_per_thread, key_type>),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           0,
                           device_keys_output,
                           compare_function());

        // Getting results to host
        HIP_CHECK(hipMemcpy(keys_output.data(),
                            device_keys_output,
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyDeviceToHost));

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys_output[i]),
                      test_utils::convert_to_native(expected[i]));
        }

        HIP_CHECK(hipFree(device_keys_output));
    }
}

template<typename T, size_t items_per_thread, size_t block_size, class CompareOp>
__global__
void sort_key_with_valid_items_kernel(T*        device_input,
                                      CompareOp compare_op,
                                      int       valid_items,
                                      T         default_val)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t     offset = (blockIdx.x * items_per_block) + (threadIdx.x * items_per_thread);

    T input[items_per_thread];

    for(size_t i = 0; i < items_per_thread; i++)
        input[i] = device_input[offset + i];

    hipcub::BlockMergeSort<T, block_size, items_per_thread> bsort;

    bsort.Sort(input, compare_op, valid_items, default_val);

    for(size_t i = 0; i < items_per_thread; i++)
        device_input[offset + i] = input[i];
}

TYPED_TEST(HipcubBlockMergeSort, SortKeysWithValidItems)
{

    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using compare_function            = typename TestFixture::params::compare_function;
    using T                           = typename TestFixture::params::key_type;
    constexpr int items_per_block     = items_per_thread * block_size;
    constexpr int grid_size           = 113;

    auto compare_op = compare_function();

    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    constexpr size_t size = grid_size * items_per_block;

    // minus|plus two to prevent overflow weirdness
    const T mini = std::numeric_limits<T>::min() + static_cast<T>(2);
    const T maxi = std::numeric_limits<T>::max() - static_cast<T>(2);

    const T   default_val        = static_cast<T>(compare_op(mini, maxi) ? maxi : mini);
    const int valid_items_arr[8] = {items_per_block / 2,
                                    items_per_block / 3,
                                    items_per_block / 4,
                                    items_per_block / 5,
                                    items_per_block - 10,
                                    items_per_block - 5,
                                    items_per_block - 2,
                                    items_per_block - 1};

    T* host_keys_input    = new T[size];
    T* host_keys_output   = new T[size];
    T* host_keys_expected = new T[size];

    T* device_keys_input;
    HIP_CHECK(hipMalloc(&device_keys_input, sizeof(T) * size));

    for(size_t it = 0; it < 8; it++)
    {
        int valid_items = valid_items_arr[it];

        // need to cast the 0 because of __half and bfloat16 types
        T elem = static_cast<T>(0);
        for(size_t i = 0; i < size; i++)
        {
            if(elem > maxi)
                elem = static_cast<T>(0);
            host_keys_input[i] = host_keys_expected[i] = elem++;
        }

        // filling in the default_val
        for(size_t bI = 0; bI < grid_size; bI++)
        {
            size_t offset = (bI * items_per_block);
            for(size_t i = valid_items; i < items_per_block; i++)
                host_keys_expected[offset + i] = default_val;
        }

        // sorting the values
        for(size_t bI = 0; bI < grid_size; bI++)
        {
            size_t offset = (bI * items_per_block);
            std::sort(host_keys_expected + offset,
                      host_keys_expected + offset + items_per_block,
                      compare_op);
        }

        HIP_CHECK(
            hipMemcpy(device_keys_input, host_keys_input, sizeof(T) * size, hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(sort_key_with_valid_items_kernel<T, items_per_thread, block_size>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_keys_input,
            compare_op,
            valid_items,
            default_val);

        HIP_CHECK(hipMemcpy(host_keys_output,
                            device_keys_input,
                            sizeof(T) * size,
                            hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
            ASSERT_EQ(host_keys_expected[i], host_keys_output[i]);
    }

    delete[] host_keys_input;
    delete[] host_keys_output;
    delete[] host_keys_expected;

    HIP_CHECK(hipFree(device_keys_input));
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class key_type,
         class value_type,
         class CompareOp>
__global__
    __launch_bounds__(BlockSize)
void sort_key_value_kernel(key_type*   device_keys_output,
                           value_type* device_values_output,
                           CompareOp   compare_op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     lid             = hipThreadIdx_x;
    const unsigned int     block_offset    = hipBlockIdx_x * items_per_block;

    key_type   keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::LoadDirectBlocked(lid, device_values_output + block_offset, values);

    hipcub::BlockMergeSort<key_type, BlockSize, ItemsPerThread, value_type> bsort;
    bsort.Sort(keys, values, compare_op);

    hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::StoreDirectBlocked(lid, device_values_output + block_offset, values);
}

TYPED_TEST(HipcubBlockMergeSort, SortKeysValues)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using compare_function            = typename TestFixture::params::compare_function;
    constexpr size_t items_per_block  = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size      = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_output;
        keys_output = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value);

        std::vector<value_type> values_output;
        values_output
            = test_utils::get_random_data<value_type>(size,
                                                      std::numeric_limits<value_type>::min(),
                                                      std::numeric_limits<value_type>::max(),
                                                      seed_value + seed_value_addition);

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_output[i], values_output[i]);
        }

        compare_function compare_op;
        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(expected.begin() + (i * items_per_block),
                             expected.begin() + ((i + 1) * items_per_block),
                             [compare_op](const key_value& a, const key_value& b)
                             { return compare_op(a.first, b.first); });
        }

        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output,
                                                     keys_output.size() * sizeof(key_type)));
        value_type* device_values_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_values_output,
                                                     values_output.size() * sizeof(value_type)));

        HIP_CHECK(hipMemcpy(device_keys_output,
                            keys_output.data(),
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyHostToDevice));

        HIP_CHECK(
            hipMemcpy(device_values_output,
                      values_output.data(),
                      values_output.size() * sizeof(typename decltype(values_output)::value_type),
                      hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                sort_key_value_kernel<block_size, items_per_thread, key_type, value_type>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_keys_output,
            device_values_output,
            compare_op);

        // Getting results to host
        HIP_CHECK(hipMemcpy(keys_output.data(),
                            device_keys_output,
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(
            hipMemcpy(values_output.data(),
                      device_values_output,
                      values_output.size() * sizeof(typename decltype(values_output)::value_type),
                      hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys_output[i]),
                      test_utils::convert_to_native(expected[i].first));
            ASSERT_EQ(test_utils::convert_to_native(values_output[i]),
                      test_utils::convert_to_native(expected[i].second));
        }

        HIP_CHECK(hipFree(device_keys_output));
        HIP_CHECK(hipFree(device_values_output));
    }
}

template<typename T, size_t items_per_thread, size_t block_size, class CompareOp>
__global__
void stable_sort_kernel(T* device_input, CompareOp compare_op)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t     offset = (blockIdx.x * items_per_block) + (threadIdx.x * items_per_thread);

    T input[items_per_thread];

    for(size_t i = 0; i < items_per_thread; i++)
        input[i] = device_input[offset + i];

    hipcub::BlockMergeSort<T, block_size, items_per_thread, int> bsort;

    bsort.StableSort(input,
                     [&](const T& lhs, const T& rhs) { return compare_op(lhs.elem, rhs.elem); });

    for(size_t i = 0; i < items_per_thread; i++)
        device_input[offset + i] = input[i];
}

template<class T>
struct custom_type_elem_id
{
    T      elem;
    size_t id;
};

TYPED_TEST(HipcubBlockMergeSort, StableSort)
{
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using T                           = typename TestFixture::params::key_type;
    using compare_function            = typename TestFixture::params::compare_function;
    using custom_type                 = custom_type_elem_id<T>;
    constexpr size_t items_per_block  = items_per_thread * block_size;
    constexpr size_t grid_size        = 113;
    const size_t     size             = grid_size * items_per_block;

    auto compare_op = compare_function();
    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    custom_type* host_input    = new custom_type[size];
    custom_type* host_expected = new custom_type[size];

    //populate the inputs
    for(size_t i = 0; i < size; i++)
    {
        if(i % 2)
            host_expected[i] = host_input[i] = {static_cast<T>(i - 1), i};
        else
            host_expected[i] = host_input[i] = {static_cast<T>(i), i};
    }

    // get the expected result
    for(size_t bI = 0; bI < grid_size; bI++)
    {
        size_t offset = (bI * items_per_block);
        std::stable_sort(host_expected + offset,
                         host_expected + offset + items_per_block,
                         [&](const custom_type& lhs, const custom_type& rhs)
                         { return compare_op(lhs.elem, rhs.elem); });
    }
    custom_type* device_input;

    HIP_CHECK(hipMalloc(&device_input, sizeof(custom_type) * size));
    HIP_CHECK(
        hipMemcpy(device_input, host_input, sizeof(custom_type) * size, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(stable_sort_kernel<custom_type, items_per_thread, block_size>),
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        device_input,
        compare_op);

    HIP_CHECK(
        hipMemcpy(host_input, device_input, sizeof(custom_type) * size, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(host_input[i].elem, host_expected[i].elem);
        ASSERT_EQ(host_input[i].id, host_expected[i].id);
    }

    delete[] host_input;
    delete[] host_expected;

    HIP_CHECK(hipFree(device_input));
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class key_type,
         class value_type,
         class CompareOp>
__global__
    __launch_bounds__(BlockSize)
void stable_sort_key_value_kernel(key_type*   device_keys_output,
                                  value_type* device_values_output,
                                  CompareOp   compare_op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     lid             = hipThreadIdx_x;
    const unsigned int     block_offset    = hipBlockIdx_x * items_per_block;

    key_type   keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::LoadDirectBlocked(lid, device_values_output + block_offset, values);

    hipcub::BlockMergeSort<key_type, BlockSize, ItemsPerThread, value_type> bsort;
    bsort.StableSort(keys, values, compare_op);

    hipcub::StoreDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::StoreDirectBlocked(lid, device_values_output + block_offset, values);
}

TYPED_TEST(HipcubBlockMergeSort, StableSortKeysValues)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                    = typename TestFixture::params::key_type;
    using value_type                  = typename TestFixture::params::value_type;
    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using compare_function            = typename TestFixture::params::compare_function;
    constexpr size_t items_per_block  = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size      = items_per_block * 1134;
    const size_t grid_size = size / items_per_block;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<key_type> keys_output;
        keys_output = test_utils::get_random_data<key_type>(size,
                                                            std::numeric_limits<key_type>::min(),
                                                            std::numeric_limits<key_type>::max(),
                                                            seed_value);

        std::vector<value_type> values_output;
        values_output
            = test_utils::get_random_data<value_type>(size,
                                                      std::numeric_limits<value_type>::min(),
                                                      std::numeric_limits<value_type>::max(),
                                                      seed_value + seed_value_addition);

        // Set some keys to be the same, but have different values to test stability
        for(size_t i = 0; i < 10; i++)
        {
            keys_output[i]   = static_cast<key_type>(0);
            values_output[i] = static_cast<value_type>(i);
        }

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_output[i], values_output[i]);
        }

        compare_function compare_op;
        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(expected.begin() + (i * items_per_block),
                             expected.begin() + ((i + 1) * items_per_block),
                             [compare_op](const key_value& a, const key_value& b)
                             { return compare_op(a.first, b.first); });
        }

        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output,
                                                     keys_output.size() * sizeof(key_type)));
        value_type* device_values_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_values_output,
                                                     values_output.size() * sizeof(value_type)));

        HIP_CHECK(hipMemcpy(device_keys_output,
                            keys_output.data(),
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyHostToDevice));

        HIP_CHECK(
            hipMemcpy(device_values_output,
                      values_output.data(),
                      values_output.size() * sizeof(typename decltype(values_output)::value_type),
                      hipMemcpyHostToDevice));

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                stable_sort_key_value_kernel<block_size, items_per_thread, key_type, value_type>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_keys_output,
            device_values_output,
            compare_op);

        // Getting results to host
        HIP_CHECK(hipMemcpy(keys_output.data(),
                            device_keys_output,
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(
            hipMemcpy(values_output.data(),
                      device_values_output,
                      values_output.size() * sizeof(typename decltype(values_output)::value_type),
                      hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(keys_output[i]),
                      test_utils::convert_to_native(expected[i].first));
            ASSERT_EQ(test_utils::convert_to_native(values_output[i]),
                      test_utils::convert_to_native(expected[i].second));
        }

        HIP_CHECK(hipFree(device_keys_output));
        HIP_CHECK(hipFree(device_values_output));
    }
}

template<typename T, size_t items_per_thread, size_t block_size, class CompareOp>
__global__
void stable_sort_key_with_valid_items_kernel(T*        device_input,
                                             CompareOp compare_op,
                                             int       valid_items,
                                             T         default_val)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t     offset = (blockIdx.x * items_per_block) + (threadIdx.x * items_per_thread);

    T input[items_per_thread];

    for(size_t i = 0; i < items_per_thread; i++)
        input[i] = device_input[offset + i];

    hipcub::BlockMergeSort<T, block_size, items_per_thread> bsort;

    bsort.StableSort(
        input,
        [&](const T& lhs, const T& rhs) { return compare_op(lhs.elem, rhs.elem); },
        valid_items,
        default_val);

    for(size_t i = 0; i < items_per_thread; i++)
        device_input[offset + i] = input[i];
}

TYPED_TEST(HipcubBlockMergeSort, StableSortKeysWithValidItems)
{

    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using compare_function            = typename TestFixture::params::compare_function;
    using T                           = typename TestFixture::params::key_type;
    using custom_type                 = custom_type_elem_id<T>;
    constexpr int items_per_block     = items_per_thread * block_size;
    constexpr int grid_size           = 113;

    auto compare_op = compare_function();

    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    constexpr size_t size = grid_size * items_per_block;

    // minus|plus two to prevent overflow weirdness
    const T mini = std::numeric_limits<T>::min() + static_cast<T>(2);
    const T maxi = std::numeric_limits<T>::max() - static_cast<T>(2);

    const custom_type default_val = {static_cast<T>(compare_op(mini, maxi) ? maxi : mini), 0};
    const int         valid_items_arr[8] = {items_per_block / 2,
                                            items_per_block / 3,
                                            items_per_block / 4,
                                            items_per_block / 5,
                                            items_per_block - 10,
                                            items_per_block - 5,
                                            items_per_block - 2,
                                            items_per_block - 1};

    custom_type* host_keys_input    = new custom_type[size];
    custom_type* host_keys_output   = new custom_type[size];
    custom_type* host_keys_expected = new custom_type[size];

    custom_type* device_keys_input;
    HIP_CHECK(hipMalloc(&device_keys_input, sizeof(custom_type) * size));

    for(size_t it = 0; it < 8; it++)
    {
        int valid_items = valid_items_arr[it];

        // need to cast 0 because of __half and bfloat16 types
        T elem = static_cast<T>(0);
        for(size_t i = 0; i < size; i++)
        {
            if(elem > maxi)
                elem = static_cast<T>(0);

            host_keys_input[i] = host_keys_expected[i] = {elem++, i};
        }

        // filling in the default_val
        for(size_t bI = 0; bI < grid_size; bI++)
        {
            size_t offset = (bI * items_per_block);
            for(size_t i = valid_items; i < items_per_block; i++)
            {
                host_keys_expected[offset + i] = default_val;
            }
        }

        // sorting the values
        for(size_t bI = 0; bI < grid_size; bI++)
        {
            size_t offset = (bI * items_per_block);
            std::stable_sort(host_keys_expected + offset,
                             host_keys_expected + offset + items_per_block,
                             [&](const custom_type& lhs, const custom_type& rhs)
                             { return compare_op(lhs.elem, rhs.elem); });
        }

        HIP_CHECK(hipMemcpy(device_keys_input,
                            host_keys_input,
                            sizeof(custom_type) * size,
                            hipMemcpyHostToDevice));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                stable_sort_key_with_valid_items_kernel<custom_type, items_per_thread, block_size>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_keys_input,
            compare_op,
            valid_items,
            default_val);

        HIP_CHECK(hipMemcpy(host_keys_output,
                            device_keys_input,
                            sizeof(custom_type) * size,
                            hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(host_keys_expected[i].elem, host_keys_output[i].elem);
            ASSERT_EQ(host_keys_expected[i].id, host_keys_output[i].id);
        }
    }

    delete[] host_keys_input;
    delete[] host_keys_output;
    delete[] host_keys_expected;

    HIP_CHECK(hipFree(device_keys_input));
}

template<typename T, size_t items_per_thread, size_t block_size, class CompareOp>
__global__
void stable_sort_key_value_with_valid_items_kernel(T*        device_key_input,
                                                   T*        device_value_input,
                                                   CompareOp compare_op,
                                                   int       valid_items,
                                                   T         default_val)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t     offset = (blockIdx.x * items_per_block) + (threadIdx.x * items_per_thread);

    T key_input[items_per_thread];
    T value_input[items_per_thread];

    for(size_t i = 0; i < items_per_thread; i++)
    {
        key_input[i]   = device_key_input[offset + i];
        value_input[i] = device_value_input[offset + i];
    }

    hipcub::BlockMergeSort<T, block_size, items_per_thread, T> bsort;

    bsort.StableSort(key_input, value_input, compare_op, valid_items, default_val);

    for(size_t i = 0; i < items_per_thread; i++)
    {
        device_key_input[offset + i]   = key_input[i];
        device_value_input[offset + i] = value_input[i];
    }
}

TYPED_TEST(HipcubBlockMergeSort, StableSortKeysValuesWithValidItems)
{

    constexpr size_t block_size       = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    using compare_function            = typename TestFixture::params::compare_function;
    using T                           = typename TestFixture::params::key_type;
    constexpr int items_per_block     = items_per_thread * block_size;
    constexpr int grid_size           = 113;

    auto compare_op = compare_function();

    if(block_size > test_utils::get_max_block_size())
    {
        GTEST_SKIP();
    }

    struct custom_type
    {
        T key;
        T value;
    };

    constexpr size_t size = grid_size * items_per_block;

    // minus|plus two to prevent overflow weirdness
    const T mini = std::numeric_limits<T>::min() + static_cast<T>(2);
    const T maxi = std::numeric_limits<T>::max() - static_cast<T>(2);

    T         default_val        = static_cast<T>(compare_op(mini, maxi) ? maxi : mini);
    const int valid_items_arr[8] = {items_per_block / 2,
                                    items_per_block / 3,
                                    items_per_block / 4,
                                    items_per_block / 5,
                                    items_per_block - 10,
                                    items_per_block - 5,
                                    items_per_block - 2,
                                    items_per_block - 1};

    custom_type* host_side_sort    = new custom_type[size];
    T*           host_keys_input   = new T[size];
    T*           host_values_input = new T[size];

    T* device_keys_input;
    T* device_values_input;
    HIP_CHECK(hipMalloc(&device_keys_input, sizeof(T) * size));
    HIP_CHECK(hipMalloc(&device_values_input, sizeof(T) * size));

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(static_cast<double>(mini) + 2,
                                               static_cast<double>(maxi) - 2);

    for(size_t it = 0; it < 8; it++)
    {
        int valid_items = valid_items_arr[it];

        // need to cast the 0 because of __half and bfloat16 types
        T rIndex = static_cast<T>(0);
        for(size_t i = 0; i < size; i++)
        {
            if(rIndex > maxi)
                rIndex = static_cast<T>(0);

            if(i % 2)
            {
                T oIndex          = rIndex - static_cast<T>(1);
                host_side_sort[i] = {oIndex, static_cast<T>(dis(gen))};
            }
            else
                host_side_sort[i] = {rIndex, static_cast<T>(dis(gen))};
            host_keys_input[i]   = host_side_sort[i].key;
            host_values_input[i] = host_side_sort[i].value;
            rIndex++;
        }

        // filling in the default_val
        for(size_t bI = 0; bI < grid_size; bI++)
        {
            size_t offset = (bI * items_per_block);
            for(size_t i = valid_items; i < items_per_block; i++)
            {
                host_side_sort[offset + i].key = default_val;
            }
        }

        // sorting the values
        for(size_t bI = 0; bI < grid_size; bI++)
        {
            size_t offset = (bI * items_per_block);
            std::stable_sort(host_side_sort + offset,
                             host_side_sort + offset + items_per_block,
                             [&](const custom_type& lhs, const custom_type& rhs)
                             { return compare_op(lhs.key, rhs.key); });
        }

        HIP_CHECK(
            hipMemcpy(device_keys_input, host_keys_input, sizeof(T) * size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_values_input,
                            host_values_input,
                            sizeof(T) * size,
                            hipMemcpyHostToDevice));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                stable_sort_key_value_with_valid_items_kernel<T, items_per_thread, block_size>),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_keys_input,
            device_values_input,
            compare_op,
            valid_items,
            default_val);

        HIP_CHECK(
            hipMemcpy(host_keys_input, device_keys_input, sizeof(T) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(host_values_input,
                            device_values_input,
                            sizeof(T) * size,
                            hipMemcpyDeviceToHost));

        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(host_side_sort[i].key, host_keys_input[i]);
            ASSERT_EQ(host_side_sort[i].value, host_values_input[i]);
        }
    }

    delete[] host_keys_input;
    delete[] host_values_input;
    delete[] host_side_sort;

    HIP_CHECK(hipFree(device_keys_input));
    HIP_CHECK(hipFree(device_values_input));
}