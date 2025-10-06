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

#include "common_test_header.hpp"

// hipcub API
#include "test_utils_custom_test_types.hpp"
#include "test_utils_sort_comparator.hpp"
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_radix_sort.hpp>
#include <hipcub/block/block_store.hpp>

#include <cstdint>

template<class Key,
         class Value,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         Descending = false,
         bool         ToStriped  = false,
         unsigned int StartBit   = 0,
         unsigned int EndBit     = sizeof(Key) * 8>
struct params
{
    using key_type                                 = Key;
    using value_type                               = Value;
    static constexpr unsigned int block_size       = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool         descending       = Descending;
    static constexpr bool         to_striped       = ToStriped;
    static constexpr unsigned int start_bit        = StartBit;
    static constexpr unsigned int end_bit          = EndBit;
};

template<class Params>
class HipcubBlockRadixSort : public ::testing::Test
{
public:
    using params = Params;
};

using Params = ::testing::Types<
// Power of 2 BlockSize
#if HIPCUB_IS_INT128_ENABLED
    params<__int128_t, __int128_t, 64U, 1>,
    params<__uint128_t, __uint128_t, 64U, 1>,
#endif
    params<char, int, 64U, 1>,
    params<unsigned char, int, 64U, 1>,
    params<int, int, 64U, 1>,
    params<unsigned int, int, 64U, 1>,
    params<long, int, 64U, 1>,
    params<unsigned long, int, 64U, 1>,
    params<float, int, 64U, 1>,
    params<double, int, 64U, 1>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 1>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 1>,
    params<test_utils::custom_test_type<float>, int, 64U, 1>,

    // Non-power of 2 BlockSize
    params<char, int, 63U, 1>,
    params<unsigned char, int, 63U, 1>,
    params<int, int, 63U, 1>,
    params<unsigned int, int, 63U, 1>,
    params<long, int, 63U, 1>,
    params<unsigned long, int, 63U, 1>,
    params<float, int, 63U, 1>,
    params<double, int, 63U, 1>,
    params<test_utils::custom_test_type<uint8_t>, float, 63U, 1>,
    params<test_utils::custom_test_type<int16_t>, int, 63U, 1>,
    params<test_utils::custom_test_type<float>, int, 63U, 1>,

    // Power of 2 BlockSize and ItemsPerThread > 1 and ItemsPerThread Power of 2
    params<char, int, 64U, 4>,
    params<unsigned char, int, 64U, 4>,
    params<int, int, 64U, 4>,
    params<unsigned int, int, 64U, 4>,
    params<long, int, 64U, 4>,
    params<unsigned long, int, 64U, 4>,
    params<float, int, 64U, 4>,
    params<double, int, 64U, 4>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4>,
    params<test_utils::custom_test_type<float>, int, 64U, 4>,

    // Power of 2 BlockSize and ItemsPerThread > 1 and ItemsPerThread Non-power of 2
    params<char, int, 64U, 3>,
    params<unsigned char, int, 64U, 3>,
    params<int, int, 64U, 3>,
    params<unsigned int, int, 64U, 3>,
    params<long, int, 64U, 3>,
    params<unsigned long, int, 64U, 3>,
    params<float, int, 64U, 3>,
    params<double, int, 64U, 3>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 3>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 3>,
    params<test_utils::custom_test_type<float>, int, 64U, 3>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1 and ItemsPerThread Power of 2
    params<char, int, 63U, 4>,
    params<unsigned char, int, 63U, 4>,
    params<int, int, 63U, 4>,
    params<unsigned int, int, 63U, 4>,
    params<long, int, 63U, 4>,
    params<unsigned long, int, 63U, 4>,
    params<float, int, 63U, 4>,
    params<double, int, 63U, 4>,
    params<test_utils::custom_test_type<uint8_t>, float, 63U, 4>,
    params<test_utils::custom_test_type<int16_t>, int, 63U, 4>,
    params<test_utils::custom_test_type<float>, int, 63U, 4>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1 and ItemsPerThread Non-power of 2
    params<char, int, 63U, 3>,
    params<unsigned char, int, 63U, 3>,
    params<int, int, 63U, 3>,
    params<unsigned int, int, 63U, 3>,
    params<long, int, 63U, 3>,
    params<unsigned long, int, 63U, 3>,
    params<float, int, 63U, 3>,
    params<double, int, 63U, 3>,
    params<test_utils::custom_test_type<uint8_t>, float, 63U, 3>,
    params<test_utils::custom_test_type<int16_t>, int, 63U, 3>,
    params<test_utils::custom_test_type<float>, int, 63U, 3>,

    // Sort with Striped arangement
    params<char, int, 64U, 4, false, true>,
    params<unsigned char, int, 64U, 4, false, true>,
    params<int, int, 64U, 4, false, true>,
    params<unsigned int, int, 64U, 4, false, true>,
    params<long, int, 64U, 4, false, true>,
    params<unsigned long, int, 64U, 4, false, true>,
    params<float, int, 64U, 4, false, true>,
    params<double, int, 64U, 4, false, true>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4, false, true>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4, false, true>,
    params<test_utils::custom_test_type<float>, int, 64U, 4, false, true>,

    // Sort in Descending order
    params<char, int, 64U, 4, true>,
    params<unsigned char, int, 64U, 4, true>,
    params<int, int, 64U, 4, true>,
    params<unsigned int, int, 64U, 4, true>,
    params<long, int, 64U, 4, true>,
    params<unsigned long, int, 64U, 4, true>,
    params<float, int, 64U, 4, true>,
    params<double, int, 64U, 4, true>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4, true>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4, true>,
    params<test_utils::custom_test_type<float>, int, 64U, 4, true>,
    params<char, int, 64U, 4, true, true>,
    params<unsigned char, int, 64U, 4, true, true>,
    params<int, int, 64U, 4, true, true>,
    params<unsigned int, int, 64U, 4, true, true>,
    params<long, int, 64U, 4, true, true>,
    params<unsigned long, int, 64U, 4, true, true>,
    params<float, int, 64U, 4, true, true>,
    params<double, int, 64U, 4, true, true>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4, true, true>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4, true, true>,
    params<test_utils::custom_test_type<float>, int, 64U, 4, true, true>,

    // Sort with
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 3, false, false, 1, 7>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 3, false, false, 1, 7>,
    params<test_utils::custom_test_type<float>, int, 64U, 3, false, false, 1, 7>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4, false, true, 1, 7>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4, false, true, 1, 7>,
    params<test_utils::custom_test_type<float>, int, 64U, 4, false, true, 1, 7>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4, true, false, 1, 7>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4, true, false, 1, 7>,
    params<test_utils::custom_test_type<float>, int, 64U, 4, true, false, 1, 7>,
    params<test_utils::custom_test_type<uint8_t>, float, 64U, 4, true, true, 1, 7>,
    params<test_utils::custom_test_type<int16_t>, int, 64U, 4, true, true, 1, 7>,
    params<test_utils::custom_test_type<float>, int, 64U, 4, true, true, 1, 7>>;

TYPED_TEST_SUITE(HipcubBlockRadixSort, Params);

template<bool Striped, bool Descending>
struct SortDispatch;

template<>
struct SortDispatch<false, false>
{
    template<class BlockSort, class... Args>
    __device__
    static void sort(BlockSort&& block_sort, Args&&... args)
    {
        block_sort.Sort(std::forward<Args>(args)...);
    }
};

template<>
struct SortDispatch<false, true>
{
    template<class BlockSort, class... Args>
    __device__
    static void sort(BlockSort&& block_sort, Args&&... args)
    {
        block_sort.SortDescending(std::forward<Args>(args)...);
    }
};

template<>
struct SortDispatch<true, false>
{
    template<class BlockSort, class... Args>
    __device__
    static void sort(BlockSort&& block_sort, Args&&... args)
    {
        block_sort.SortBlockedToStriped(std::forward<Args>(args)...);
    }
};

template<>
struct SortDispatch<true, true>
{
    template<class BlockSort, class... Args>
    __device__
    static void sort(BlockSort&& block_sort, Args&&... args)
    {
        block_sort.SortDescendingBlockedToStriped(std::forward<Args>(args)...);
    }
};

template<unsigned int BlockSize, unsigned int ItemsPerThread, bool Striped, bool Descending>
struct SortOp
{
    using dispatch_t = SortDispatch<Striped, Descending>;

    template<class Key>
    __device__
    void operator()(Key (&keys)[ItemsPerThread], int start_bit, int end_bit) const
    {
        hipcub::BlockRadixSort<Key, BlockSize, ItemsPerThread> block_sort;
        if(start_bit == 0 && end_bit == sizeof(Key) * 8)
        {
            dispatch_t::sort(block_sort, keys);
        }
        else
        {
            dispatch_t::sort(block_sort, keys, start_bit, end_bit);
        }
    }

    template<class InnerT>
    __device__
    void operator()(test_utils::custom_test_type<InnerT> (&keys)[ItemsPerThread],
                    int start_bit,
                    int end_bit) const
    {
        using custom_test_t = test_utils::custom_test_type<InnerT>;
        hipcub::BlockRadixSort<custom_test_t, BlockSize, ItemsPerThread> block_sort;
        test_utils::custom_test_type_decomposer<custom_test_t>           decomposer;
        if(start_bit == 0 && end_bit == sizeof(custom_test_t) * 8)
        {
            dispatch_t::sort(block_sort, keys, decomposer);
        }
        else
        {
            dispatch_t::sort(block_sort, keys, decomposer, start_bit, end_bit);
        }
    }

    template<class Key, class Value>
    __device__
    void operator()(Key (&keys)[ItemsPerThread],
                    Value (&values)[ItemsPerThread],
                    int start_bit,
                    int end_bit) const
    {
        hipcub::BlockRadixSort<Key, BlockSize, ItemsPerThread, Value> block_sort;
        if(start_bit == 0 && end_bit == sizeof(Key) * 8)
        {
            dispatch_t::sort(block_sort, keys, values);
        }
        else
        {
            dispatch_t::sort(block_sort, keys, values, start_bit, end_bit);
        }
    }

    template<class InnerT, class Value>
    __device__
    void operator()(test_utils::custom_test_type<InnerT> (&keys)[ItemsPerThread],
                    Value (&values)[ItemsPerThread],
                    int start_bit,
                    int end_bit) const
    {
        using custom_test_t = test_utils::custom_test_type<InnerT>;
        hipcub::BlockRadixSort<custom_test_t, BlockSize, ItemsPerThread, Value> block_sort;
        test_utils::custom_test_type_decomposer<custom_test_t>                  decomposer;
        if(start_bit == 0 && end_bit == sizeof(custom_test_t) * 8)
        {
            dispatch_t::sort(block_sort, keys, values, decomposer);
        }
        else
        {
            dispatch_t::sort(block_sort, keys, values, decomposer, start_bit, end_bit);
        }
    }
};

template<unsigned int BlockSize, unsigned int ItemsPerThread, bool Striped>
struct StoreOp;

template<unsigned int BlockSize, unsigned int ItemsPerThread>
struct StoreOp<BlockSize, ItemsPerThread, false>
{
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    template<class Key>
    __device__
    void operator()(Key (&keys)[ItemsPerThread], Key* keys_output) const
    {
        const unsigned int block_offset = blockIdx.x * items_per_block;
        hipcub::StoreDirectBlocked(threadIdx.x, keys_output + block_offset, keys);
    }

    template<class Key, class Value>
    __device__
    void operator()(Key (&keys)[ItemsPerThread],
                    Value (&values)[ItemsPerThread],
                    Key*   keys_output,
                    Value* values_output) const
    {
        const unsigned int block_offset = blockIdx.x * items_per_block;
        hipcub::StoreDirectBlocked(threadIdx.x, keys_output + block_offset, keys);
        hipcub::StoreDirectBlocked(threadIdx.x, values_output + block_offset, values);
    }
};

template<unsigned int BlockSize, unsigned int ItemsPerThread>
struct StoreOp<BlockSize, ItemsPerThread, true>
{
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    template<class Key>
    __device__
    void operator()(Key (&keys)[ItemsPerThread], Key* keys_output) const
    {
        const unsigned int block_offset = blockIdx.x * items_per_block;
        hipcub::StoreDirectStriped<BlockSize>(threadIdx.x, keys_output + block_offset, keys);
    }

    template<class Key, class Value>
    __device__
    void operator()(Key (&keys)[ItemsPerThread],
                    Value (&values)[ItemsPerThread],
                    Key*   keys_output,
                    Value* values_output) const
    {
        const unsigned int block_offset = blockIdx.x * items_per_block;
        hipcub::StoreDirectStriped<BlockSize>(threadIdx.x, keys_output + block_offset, keys);
        hipcub::StoreDirectStriped<BlockSize>(threadIdx.x, values_output + block_offset, values);
    }
};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         Striped,
         bool         Descending,
         class key_type>
__global__ __launch_bounds__(BlockSize)
void sort_key_kernel(key_type* device_keys_output, unsigned int start_bit, unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     block_offset    = blockIdx.x * items_per_block;

    key_type keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(threadIdx.x, device_keys_output + block_offset, keys);

    SortOp<BlockSize, ItemsPerThread, Striped, Descending>{}(keys, start_bit, end_bit);
    StoreOp<BlockSize, ItemsPerThread, Striped>{}(keys, device_keys_output);
}

// GTest's ASSERT_EQ prints the values if the test fails. On Windows, GTest doesn't currently provide overloads for
// printing 128 bit types, resulting in linker errors.
// Check if we're testing with 128 bit types. If so, test using bools so GTest doesn't try to print them on failure.
#define ASSERT_EQ_128BIT_WORKAROUND(output, expected, index, type)                  \
    if(test_utils::is_int128<type>::value || test_utils::is_uint128<type>::value)   \
    {                                                                               \
        const bool values_equal = (output == expected);                             \
        ASSERT_EQ(values_equal, true) << "at index: " << index;                     \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        ASSERT_EQ(output, expected) << "at index: " << index;                       \
    }    

TYPED_TEST(HipcubBlockRadixSort, SortKeys)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                          = typename TestFixture::params::key_type;
    constexpr size_t       block_size       = TestFixture::params::block_size;
    constexpr size_t       items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool         descending       = TestFixture::params::descending;
    constexpr bool         to_striped       = TestFixture::params::to_striped;
    constexpr unsigned int start_bit        = TestFixture::params::start_bit;
    constexpr unsigned int end_bit          = TestFixture::params::end_bit;
    constexpr size_t       items_per_block  = block_size * items_per_thread;
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
        using limits_t = typename test_utils::inner_type<key_type>::type;
        if(test_utils::is_floating_point<key_type>::value)
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                test_utils::convert_to_device<limits_t>(-1000),
                test_utils::convert_to_device<limits_t>(+1000),
                seed_value);
        }
        else
        {
            keys_output
                = test_utils::get_random_data<key_type>(size,
                                                        std::numeric_limits<limits_t>::min(),
                                                        std::numeric_limits<limits_t>::max(),
                                                        seed_value);
        }

        // Calculate expected results on host
        std::vector<key_type> expected(keys_output);
        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>());
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
        sort_key_kernel<block_size, items_per_thread, to_striped, descending>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(device_keys_output, start_bit, end_bit);

        // Getting results to host
        HIP_CHECK(hipMemcpy(keys_output.data(),
                            device_keys_output,
                            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
                            hipMemcpyDeviceToHost));

        // Verifying results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ_128BIT_WORKAROUND(test_utils::convert_to_native(keys_output[i]), 
                                        test_utils::convert_to_native(expected[i]), 
                                        i, 
                                        key_type);
        }

        HIP_CHECK(hipFree(device_keys_output));
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         Striped,
         bool         Descending,
         class key_type,
         class value_type>
__global__ __launch_bounds__(BlockSize)
void sort_key_value_kernel(key_type*    device_keys_output,
                           value_type*  device_values_output,
                           unsigned int start_bit,
                           unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     lid             = threadIdx.x;
    const unsigned int     block_offset    = blockIdx.x * items_per_block;

    key_type   keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, device_keys_output + block_offset, keys);
    hipcub::LoadDirectBlocked(lid, device_values_output + block_offset, values);

    SortOp<BlockSize, ItemsPerThread, Striped, Descending>{}(keys, values, start_bit, end_bit);
    StoreOp<BlockSize, ItemsPerThread, Striped>{}(keys,
                                                  values,
                                                  device_keys_output,
                                                  device_values_output);
}

TYPED_TEST(HipcubBlockRadixSort, SortKeysValues)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type                          = typename TestFixture::params::key_type;
    using value_type                        = typename TestFixture::params::value_type;
    constexpr size_t       block_size       = TestFixture::params::block_size;
    constexpr size_t       items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool         descending       = TestFixture::params::descending;
    constexpr bool         to_striped       = TestFixture::params::to_striped;
    constexpr unsigned int start_bit        = TestFixture::params::start_bit;
    constexpr unsigned int end_bit          = TestFixture::params::end_bit;
    constexpr size_t       items_per_block  = block_size * items_per_thread;
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
        using limits_t = typename test_utils::inner_type<key_type>::type;
        if(test_utils::is_floating_point<key_type>::value)
        {
            keys_output = test_utils::get_random_data<key_type>(
                size,
                test_utils::convert_to_device<limits_t>(-1000),
                test_utils::convert_to_device<limits_t>(+1000),
                seed_value);
        }
        else
        {
            keys_output
                = test_utils::get_random_data<key_type>(size,
                                                        std::numeric_limits<limits_t>::min(),
                                                        std::numeric_limits<limits_t>::max(),
                                                        seed_value);
        }

        std::vector<value_type> values_output;
        if(test_utils::is_floating_point<value_type>::value)
        {
            values_output = test_utils::get_random_data<value_type>(
                size,
                test_utils::convert_to_device<value_type>(-1000),
                test_utils::convert_to_device<value_type>(+1000),
                seed_value + seed_value_addition);
        }
        else
        {
            values_output
                = test_utils::get_random_data<value_type>(size,
                                                          std::numeric_limits<value_type>::min(),
                                                          std::numeric_limits<value_type>::max(),
                                                          seed_value + seed_value_addition);
        }

        using key_value = std::pair<key_type, value_type>;

        // Calculate expected results on host
        std::vector<key_value> expected(size);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = key_value(keys_output[i], values_output[i]);
        }

        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                test_utils::
                    key_value_comparator<key_type, value_type, descending, start_bit, end_bit>());
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
        sort_key_value_kernel<block_size, items_per_thread, to_striped, descending>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(device_keys_output,
                                                          device_values_output,
                                                          start_bit,
                                                          end_bit);

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
            ASSERT_EQ_128BIT_WORKAROUND(test_utils::convert_to_native(keys_output[i]), 
                                        test_utils::convert_to_native(expected[i].first), 
                                        i, 
                                        key_type);   
                                        
            ASSERT_EQ_128BIT_WORKAROUND(test_utils::convert_to_native(values_output[i]), 
                                        test_utils::convert_to_native(expected[i].second), 
                                        i, 
                                        value_type);                    
        }

        HIP_CHECK(hipFree(device_keys_output));
        HIP_CHECK(hipFree(device_values_output));
    }
}
