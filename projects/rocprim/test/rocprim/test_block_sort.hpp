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
#include "test_utils.hpp"

#include "../../common/utils.hpp"
#include "../../common/utils_device_ptr.hpp"
#include "test_seed.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/block/block_sort.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>

block_sort_test_suite_type_def(suite_name, name_suffix);

typed_test_suite_def(suite_name, name_suffix, block_params);

// using header guards for these test functions because this file is included multiple times:
// once for the integrals test suite and once for the floating point test suite.
#ifndef TEST_ROCPRIM_TEST_BLOCK_SORT_HPP_
    #define TEST_ROCPRIM_TEST_BLOCK_SORT_HPP_

template<unsigned int block_size,
         unsigned int items_per_thread,
         class key_type,
         class value_type,
         rocprim::block_sort_algorithm algo,
         class binary_op_type>
void TestSortKeyValue()
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    static constexpr const unsigned int items_per_block = block_size * items_per_thread;
    static constexpr const size_t       grid_size       = 1134;
    static constexpr const size_t       size            = items_per_block * grid_size;
    hipStream_t                         stream          = 0; // default

    if(!is_buildable(block_size, items_per_thread, algo))
    {
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<key_type> output_key
            = test_utils::get_random_data_wrapped<key_type>(size, 0, 100, seed_value);
        std::vector<value_type> output_value
            = test_utils::get_random_data_wrapped<value_type>(size, -100, 100, seed_value);

        // Combine vectors to form pairs with key and value
        std::vector<std::pair<key_type, value_type>> target(size);
        for(unsigned i = 0; i < target.size(); i++)
            target[i] = std::make_pair(output_key[i], output_value[i]);

        // Calculate expected results on host
        using key_value = std::pair<key_type, value_type>;
        std::vector<key_value> expected(target);
        constexpr bool descending = !std::is_same<binary_op_type, rocprim::less<key_type>>::value;
        for(size_t i = 0; i < expected.size() / items_per_block; i++)
        {
            std::sort(expected.begin() + (i * items_per_block),
                      expected.begin() + ((i + 1) * items_per_block),
                      test_utils::key_value_comparator<key_type,
                                                       value_type,
                                                       descending,
                                                       0,
                                                       sizeof(key_type) * 8>());
        }

        // Preparing device
        common::device_ptr<key_type>   device_key_output(output_key);
        common::device_ptr<value_type> device_value_output(output_value);

        // Running kernel, ignored if invalid size
        if(size > 0)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_pairs_kernel<block_size,
                                                                 items_per_thread,
                                                                 key_type,
                                                                 value_type,
                                                                 algo,
                                                                 binary_op_type>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               device_key_output.get(),
                               device_value_output.get(),
                               size);
            HIP_CHECK(hipGetLastError());
        }

        // Reading results back
        output_key   = device_key_output.load();
        output_value = device_value_output.load();

        std::vector<key_type>   expected_key(expected.size());
        std::vector<value_type> expected_value(expected.size());
        for(size_t i = 0; i < expected.size(); i++)
        {
            expected_key[i]   = expected[i].first;
            expected_value[i] = expected[i].second;
        }

        // Keys are sorted, Values order not guaranteed
        // Sort subsets where key was the same to make sure all values are still present
        using value_op_type = rocprim::less<value_type>;
        using eq_op_type    = rocprim::equal_to<key_type>;
        value_op_type value_op;
        eq_op_type    eq_op;
        for(size_t i = 0; i < output_key.size();)
        {
            auto j = i;
            for(; j < output_key.size() && eq_op(output_key[j], output_key[i]); ++j)
            {}
            std::sort(output_value.begin() + i, output_value.begin() + j, value_op);
            std::sort(expected_value.begin() + i, expected_value.begin() + j, value_op);
            i = j;
        }

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_key, expected_key));
        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output_value, expected_value));
    }
}

template<unsigned int block_size,
         unsigned int items_per_thread,
         class key_type,
         class value_type,
         rocprim::block_sort_algorithm algo,
         class binary_op_type>
void TestSortKey(std::vector<size_t> sizes)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    static constexpr const unsigned int items_per_block = block_size * items_per_thread;
    hipStream_t                         stream          = 0; // default

    if(!is_buildable(block_size, items_per_thread, algo))
    {
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            if(size == 0 && common::use_hmm())
            {
                // hipMallocManaged() currently doesnt support zero byte allocation
                continue;
            }
            // Generate data
            std::vector<key_type> output
                = test_utils::get_random_data_wrapped<key_type>(size, -100, 100, seed_value);

            // Calculate expected results on host
            std::vector<key_type> expected(output);
            binary_op_type        binary_op;
            for(size_t i = 0; i < rocprim::detail::ceiling_div(output.size(), items_per_block); i++)
            {
                std::sort(expected.begin() + (i * items_per_block),
                          expected.begin() + std::min(size, ((i + 1) * items_per_block)),
                          binary_op);
            }

            // Preparing device
            common::device_ptr<key_type> device_key_output(output);

            const unsigned int grid_size = rocprim::detail::ceiling_div(size, items_per_block);
            // Running kernel, ignored if invalid size
            if(size > 0)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_keys_kernel<block_size,
                                                                    items_per_thread,
                                                                    key_type*,
                                                                    algo,
                                                                    binary_op_type>),
                                   dim3(grid_size),
                                   dim3(block_size),
                                   0,
                                   stream,
                                   device_key_output.get(),
                                   size);
            }

            // Reading results back
            output = device_key_output.load();

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
        }
    }
}

struct less_tuple
{
    template<class A, class B>
    ROCPRIM_HOST_DEVICE inline constexpr bool operator()(const rocprim::tuple<A, B>& a,
                                                         const rocprim::tuple<A, B>& b) const
    {
        return rocprim::get<0>(a) < rocprim::get<0>(b);
    }
};

template<unsigned int block_size, unsigned int items_per_thread, rocprim::block_sort_algorithm algo>
void TestSortStableKey(std::vector<size_t> sizes)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using key_type       = int8_t;
    using index_type     = unsigned int;
    using binary_op_type = less_tuple;

    static constexpr const unsigned int items_per_block = block_size * items_per_thread;
    hipStream_t                         stream          = 0; // default

    if(!is_buildable(block_size, items_per_thread, algo))
    {
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : sizes)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);
            if(size == 0 && common::use_hmm())
            {
                // hipMallocManaged() currently doesnt support zero byte allocation
                continue;
            }

            // Generate data
            std::vector<key_type> keys
                = test_utils::get_random_data_wrapped<key_type>(size, -10, 10, seed_value);

            std::vector<index_type> values(size);
            std::iota(values.begin(), values.end(), 0u);
            using tuple_type = rocprim::tuple<key_type, index_type>;
            std::vector<tuple_type> tuples(size);
            std::vector<tuple_type> expected(size);
            for(size_t i = 0; i < expected.size(); i++)
            {
                tuples[i]   = rocprim::make_tuple(keys[i], values[i]);
                expected[i] = rocprim::make_tuple(keys[i], values[i]);
            }

            // Preparing device
            common::device_ptr<tuple_type> device_tuples_output(tuples);

            const unsigned int grid_size = rocprim::detail::ceiling_div(size, items_per_block);

            // Running kernel
            hipLaunchKernelGGL(HIP_KERNEL_NAME(sort_keys_kernel<block_size,
                                                                items_per_thread,
                                                                tuple_type*,
                                                                algo,
                                                                binary_op_type>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               device_tuples_output.get(),
                               size);

            // Reading results back
            tuples = device_tuples_output.load();

            // Calculate expected results on host
            binary_op_type binary_op;
            for(size_t i = 0; i < rocprim::detail::ceiling_div(expected.size(), items_per_block);
                i++)
            {
                std::stable_sort(expected.begin() + (i * items_per_block),
                                 expected.begin() + std::min(size, (i + 1) * items_per_block),
                                 binary_op);
            }

            ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(tuples, expected));
        }
    }
}
#endif // TEST_ROCPRIM_TEST_BLOCK_SORT_HPP_
// This file is included multiple times in the test_block_sort_[algo].cpp file, because
// the test definitions below this header guard need to be compiled for each test suites:
// integrals and floating point types.
typed_test_def(suite_name, name_suffix, SortKeys)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 1;

    std::vector<size_t> sizes = {1134 * items_per_thread * block_size};
    TestSortKey<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>(sizes);
}

typed_test_def(suite_name, name_suffix, SortKeyValue)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = typename rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 1;
    TestSortKeyValue<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>();
}

typed_test_def(suite_name, name_suffix, SortKeysMultipleItemsPerThread)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 4;

    std::vector<size_t> sizes = {1134 * items_per_thread * block_size};
    TestSortKey<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>(sizes);
}

typed_test_def(suite_name, name_suffix, SortKeysStable)
{
    using key_type                                            = typename TestFixture::key_type;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    if(std::is_same<key_type, int>::value
       && algo == rocprim::block_sort_algorithm::stable_merge_sort)
    {
        static constexpr const unsigned int block_size       = TestFixture::block_size;
        static constexpr const unsigned int items_per_thread = 4;

        std::vector<size_t> sizes = {1134 * items_per_thread * block_size};
        TestSortStableKey<block_size, items_per_thread, algo>(sizes);
        TestSortStableKey<block_size, 1, algo>(sizes);
    }
    else
    {
        GTEST_SKIP();
    }
}

typed_test_def(suite_name, name_suffix, SortKeyValueMultipleItemsPerThread)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = typename rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 4;
    TestSortKeyValue<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>();
}

typed_test_def(suite_name, name_suffix, SortKeyInputSizeNotMultipleOfBlockSize)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = typename rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 1;
    std::vector<size_t>                                  sizes
        = {0, 53, 512, 5000, 34567, (1 << 17) - 1220, 1134 * 256, (1 << 20) - 123};
    TestSortKey<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>(sizes);
}

typed_test_def(suite_name,
               name_suffix,
               SortKeyInputSizeNotMultipleOfBlockSizeMultipleItemsPerThread)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = typename rocprim::less<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 4;
    std::vector<size_t> sizes
        = {0, 53, 512, 5000, 34567, (1 << 17) - 1220, 1134 * 256, (1 << 20) - 123};
    TestSortKey<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>(sizes);
}

typed_test_def(suite_name, name_suffix, SortKeyValueDesc)
{
    using key_type                                            = typename TestFixture::key_type;
    using value_type                                          = typename TestFixture::value_type;
    using binary_op_type                                      = typename rocprim::greater<key_type>;
    static constexpr const rocprim::block_sort_algorithm algo = TEST_BLOCK_SORT_ALGORITHM;
    static constexpr const unsigned int                  block_size       = TestFixture::block_size;
    static constexpr const unsigned int                  items_per_thread = 1;
    TestSortKeyValue<block_size, items_per_thread, key_type, value_type, algo, binary_op_type>();
}
