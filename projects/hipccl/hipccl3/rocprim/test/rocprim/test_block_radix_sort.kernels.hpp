// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_BLOCK_RADIX_SORT_KERNELS_HPP_
#define TEST_BLOCK_RADIX_SORT_KERNELS_HPP_

static constexpr size_t n_sizes = 12;
static constexpr unsigned int items_radix[n_sizes] = {
    1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3
};
static constexpr bool desc_radix[n_sizes] = {
    false, false, false, false, false, false, true, true, true, true, true, true
};
static constexpr bool striped_radix[n_sizes] = {
    false, false, false, true, true, true, false, false, false, true, true, true
};
static constexpr unsigned int start_radix[n_sizes] = {
    0, 0, 0, 3, 4, 8, 0, 0, 0, 3, 4, 8
};
static constexpr unsigned int end_radix[n_sizes] = {
    0, 0, 0, 10, 11, 12, 0, 0, 0, 10, 11, 12
};

static constexpr unsigned int bits_per_pass_radix[n_sizes] = {4, 3, 1, 1, 3, 4, 4, 3, 1, 1, 3, 4};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBitsPerPass,
         class key_type>
__global__ __launch_bounds__(BlockSize) void sort_key_kernel(key_type*    device_keys_output,
                                                             bool         to_striped,
                                                             bool         descending,
                                                             unsigned int start_bit,
                                                             unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    key_type keys[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_keys_output + block_offset, keys);

    rocprim::block_radix_sort<key_type,
                              BlockSize,
                              ItemsPerThread,
                              rocprim::empty_type,
                              1,
                              1,
                              RadixBitsPerPass>
        bsort;

    test_utils::select_decomposer_t<key_type> decomposer{};

    if(to_striped)
    {
        if(descending)
        {
            bsort.sort_desc_to_striped(keys, start_bit, end_bit, decomposer);
        }
        else
        {
            bsort.sort_to_striped(keys, start_bit, end_bit, decomposer);
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, device_keys_output + block_offset, keys);
    }
    else
    {
        if(descending)
        {
            bsort.sort_desc(keys, start_bit, end_bit, decomposer);
        }
        else
        {
            bsort.sort(keys, start_bit, end_bit, decomposer);
        }

        rocprim::block_store_direct_blocked(lid, device_keys_output + block_offset, keys);
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBitsPerPass,
         class key_type,
         class value_type>
__global__ __launch_bounds__(BlockSize) void sort_key_value_kernel(key_type*   device_keys_output,
                                                                   value_type* device_values_output,
                                                                   bool        to_striped,
                                                                   bool        descending,
                                                                   unsigned int start_bit,
                                                                   unsigned int end_bit)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    key_type keys[ItemsPerThread];
    value_type values[ItemsPerThread];
    rocprim::block_load_direct_blocked(lid, device_keys_output + block_offset, keys);
    rocprim::block_load_direct_blocked(lid, device_values_output + block_offset, values);

    rocprim::
        block_radix_sort<key_type, BlockSize, ItemsPerThread, value_type, 1, 1, RadixBitsPerPass>
                                              bsort;
    test_utils::select_decomposer_t<key_type> decomposer{};
    if(to_striped)
    {
        if(descending)
        {
            bsort.sort_desc_to_striped(keys, values, start_bit, end_bit, decomposer);
        }
        else
        {
            bsort.sort_to_striped(keys, values, start_bit, end_bit, decomposer);
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, device_keys_output + block_offset, keys);
        rocprim::block_store_direct_striped<BlockSize>(lid, device_values_output + block_offset, values);
    }
    else
    {
        if(descending)
        {
            bsort.sort_desc(keys, values, start_bit, end_bit, decomposer);
        }
        else
        {
            bsort.sort(keys, values, start_bit, end_bit, decomposer);
        }

        rocprim::block_store_direct_blocked(lid, device_keys_output + block_offset, keys);
        rocprim::block_store_direct_blocked(lid, device_values_output + block_offset, values);
    }
}

// Test for radix sort with keys only
template<class Key,
         class Value,
         unsigned int Method,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBitsPerPass,
         bool         Descending = false,
         bool         ToStriped  = false,
         unsigned int StartBit   = 0,
         unsigned int EndBit     = sizeof(Key) * 8>
auto test_block_radix_sort() -> typename std::enable_if<Method == 0>::type
{
    using key_type                                    = Key;
    static constexpr size_t       block_size          = BlockSize;
    static constexpr size_t       items_per_thread    = ItemsPerThread;
    static constexpr unsigned     radix_bits_per_pass = RadixBitsPerPass;
    static constexpr bool         descending          = Descending;
    static constexpr bool         to_striped          = ToStriped;
    static constexpr unsigned int start_bit           = StartBit;
    static constexpr unsigned int end_bit             = EndBit;
    static constexpr size_t       items_per_block     = block_size * items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        engine_type rng_engine(seed_value);

        // Generate data
        auto keys_output = std::make_unique<key_type[]>(size);
        if(rocprim::is_floating_point<key_type>::value)
        {
            test_utils::generate_random_data_n(keys_output.get(), size, -100, +100, rng_engine);
        }
        else
        {
            test_utils::generate_random_data_n(keys_output.get(),
                                               size,
                                               test_utils::numeric_limits<key_type>::min(),
                                               test_utils::numeric_limits<key_type>::max(),
                                               rng_engine);
        }

        // Calculate expected results on host
        std::vector<key_type> expected(keys_output.get(), keys_output.get() + size);
        for(size_t i = 0; i < size / items_per_block; i++)
        {
            std::stable_sort(
                expected.begin() + (i * items_per_block),
                expected.begin() + ((i + 1) * items_per_block),
                test_utils::key_comparator<key_type, descending, start_bit, end_bit>()
            );
        }

        // Preparing device
        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output, size * sizeof(key_type)));

        HIP_CHECK(hipMemcpy(device_keys_output,
                            keys_output.get(),
                            size * sizeof(keys_output[0]),
                            hipMemcpyHostToDevice));

        sort_key_kernel<block_size, items_per_thread, radix_bits_per_pass, key_type>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(device_keys_output,
                                                          to_striped,
                                                          descending,
                                                          start_bit,
                                                          end_bit);
        HIP_CHECK(hipGetLastError());

        // Getting results to host
        HIP_CHECK(hipMemcpy(keys_output.get(),
                            device_keys_output,
                            size * sizeof(keys_output[0]),
                            hipMemcpyDeviceToHost));

        // Verifying results
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output.get(),
                                                      keys_output.get() + size,
                                                      expected.begin(),
                                                      expected.end()));

        HIP_CHECK(hipFree(device_keys_output));
    }

}

// Test for radix_sort with keys and values. Also ensures that (block) radix_sort is stable
template<class Key,
         class Value,
         unsigned int Method,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBitsPerPass,
         bool         Descending = false,
         bool         ToStriped  = false,
         unsigned int StartBit   = 0,
         unsigned int EndBit     = sizeof(Key) * 8>
auto test_block_radix_sort() -> typename std::enable_if<Method == 1>::type
{
    using key_type                                    = Key;
    using value_type                                  = Value;
    static constexpr size_t       block_size          = BlockSize;
    static constexpr size_t       items_per_thread    = ItemsPerThread;
    static constexpr unsigned     radix_bits_per_pass = RadixBitsPerPass;
    static constexpr bool         descending          = Descending;
    static constexpr bool         to_striped          = ToStriped;
    static constexpr unsigned int start_bit
        = (rocprim::is_unsigned<Key>::value == false) ? 0 : StartBit;
    static constexpr unsigned int end_bit
        = (rocprim::is_unsigned<Key>::value == false) ? sizeof(Key) * 8 : EndBit;
    static constexpr size_t items_per_block = block_size * items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 19;
    const size_t grid_size = size / items_per_block;

    SCOPED_TRACE(testing::Message() << "with items_per_block = " << items_per_block);
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    SCOPED_TRACE(testing::Message() << "with grid_size = " << grid_size);

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        seed_type seed_value = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        engine_type rng_engine(seed_value);

        // Generate data
        auto keys_output = std::make_unique<key_type[]>(size);
        if(rocprim::is_floating_point<key_type>::value)
        {
            test_utils::generate_random_data_n(keys_output.get(), size, -100, +100, rng_engine);
        }
        else
        {
            test_utils::generate_random_data_n(keys_output.get(),
                                               size,
                                               test_utils::numeric_limits<key_type>::min(),
                                               test_utils::numeric_limits<key_type>::max(),
                                               rng_engine);
        }

        std::vector<value_type> values_output(size);
        std::iota(values_output.begin(), values_output.end(), 0u);

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
                test_utils::key_value_comparator<key_type, value_type, descending, start_bit, end_bit>()
            );
        }

        std::vector<key_type> keys_expected(size);
        std::vector<value_type> values_expected(size);
        for(size_t i = 0; i < size; i++)
        {
            keys_expected[i] = expected[i].first;
            values_expected[i] = expected[i].second;
        }

        key_type* device_keys_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_keys_output, size * sizeof(key_type)));
        value_type* device_values_output;
        HIP_CHECK(test_common_utils::hipMallocHelper(&device_values_output, values_output.size() * sizeof(value_type)));

        HIP_CHECK(hipMemcpy(device_keys_output,
                            keys_output.get(),
                            size * sizeof(keys_output[0]),
                            hipMemcpyHostToDevice));

        HIP_CHECK(
            hipMemcpy(
                device_values_output, values_output.data(),
                values_output.size() * sizeof(typename decltype(values_output)::value_type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        sort_key_value_kernel<block_size,
                              items_per_thread,
                              radix_bits_per_pass,
                              key_type,
                              value_type>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(device_keys_output,
                                                          device_values_output,
                                                          to_striped,
                                                          descending,
                                                          start_bit,
                                                          end_bit);
        HIP_CHECK(hipGetLastError());

        // Getting results to host
        HIP_CHECK(hipMemcpy(keys_output.get(),
                            device_keys_output,
                            size * sizeof(keys_output[0]),
                            hipMemcpyDeviceToHost));

        HIP_CHECK(
            hipMemcpy(
                values_output.data(), device_values_output,
                values_output.size() * sizeof(typename decltype(values_output)::value_type),
                hipMemcpyDeviceToHost
            )
        );

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(keys_output.get(),
                                                      keys_output.get() + size,
                                                      keys_expected.begin(),
                                                      keys_expected.end()));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(values_output, values_expected));

        HIP_CHECK(hipFree(device_keys_output));
        HIP_CHECK(hipFree(device_values_output));
    }

}

// Static for-loop
template<unsigned int First,
         unsigned int Last,
         class T,
         class U,
         int          Method,
         unsigned int BlockSize = 256U>
struct static_for
{
    static constexpr unsigned int end = (end_radix[First] == 0) ? sizeof(T) * 8 : end_radix[First];

    static void run()
    {
        {
            SCOPED_TRACE(testing::Message() << "TestID = " << First);
            test_block_radix_sort<T,
                                  U,
                                  Method,
                                  BlockSize,
                                  items_radix[First],
                                  bits_per_pass_radix[First],
                                  desc_radix[First],
                                  striped_radix[First],
                                  start_radix[First],
                                  end>();
        }
        static_for<First + 1, Last, T, U, Method, BlockSize>::run();
    }
};

template<unsigned int N, class T, class U, int Method, unsigned int BlockSize>
struct static_for<N, N, T, U, Method, BlockSize>
{
    static void run() {}
};

#endif // TEST_BLOCK_RADIX_SORT_KERNELS_HPP_
