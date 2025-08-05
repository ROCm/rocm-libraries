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

#include "../../common/utils_device_ptr.hpp"

// required test headers
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_data_generation.hpp"

// required rocprim headers
#include <rocprim/device/device_reduce_by_key.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/discard_iterator.hpp>

#include <cstddef>

TEST(RocprimDiscardIteratorTests, Basic)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Iterator = rocprim::discard_iterator;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        const size_t base_index
            = static_cast<size_t>(test_utils::get_random_value<int>(1, 100, seed_value));
        Iterator it = rocprim::make_discard_iterator(base_index);

        // Check dereferencing returns discard_value (should be callable)
        auto discard_val = *it;
        static_assert(std::is_same<decltype(discard_val), typename Iterator::value_type>::value,
                      "Dereferencing should yield discard_value.");

        // Post-increment
        Iterator post = it++;
        ASSERT_EQ((it - post), 1);

        // Pre-increment
        ++it;
        ASSERT_EQ((it - post), 2);

        // Post-decrement
        Iterator post_dec = it--;
        ASSERT_EQ((post_dec - it), 1);

        // Pre-decrement
        --it;
        ASSERT_EQ((post_dec - it), 2);

        // Arithmetic operations
        Iterator it_plus_5 = it + 5;
        ASSERT_EQ(it_plus_5 - it, 5);

        Iterator it_5_plus = 5 + it;
        ASSERT_EQ(it_5_plus - it, 5);

        Iterator it_minus_3 = it_plus_5 - 3;
        ASSERT_EQ(it_minus_3 - it, 2);

        it += 12;
        ASSERT_EQ(it - post, 12);

        it -= 4;
        ASSERT_EQ(it - post, 8);

        // Comparison checks
        Iterator a(100);
        Iterator b(105);
        ASSERT_TRUE(a < b);
        ASSERT_TRUE(b > a);
        ASSERT_TRUE(a <= b);
        ASSERT_TRUE(b >= a);
        ASSERT_TRUE(a != b);
        ASSERT_TRUE(a == a);

        // Operator[] returns discard_value
        auto val = a[3];
        static_assert(std::is_same<decltype(val), typename Iterator::value_type>::value,
                      "operator[] should return discard_value");
    }
}

TEST(RocprimDiscardIteratorTests, ReduceByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default

    // host input
    std::vector<int> keys_input = {0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0};
    std::vector<int> values_input(keys_input.size(), 1);

    // expected output
    std::vector<int> aggregates_expected = {3, 2, 2, 4};

    // device input/output
    common::device_ptr<int> d_keys_input(keys_input);
    common::device_ptr<int> d_values_input(values_input);
    common::device_ptr<int> d_aggregates_output(aggregates_expected.size());

    test_utils::test_kernel_wrapper(
        [&](void* temp_storage, size_t& storage_bytes)
        {
            return rocprim::reduce_by_key(temp_storage,
                                          storage_bytes,
                                          d_keys_input.get(),
                                          d_values_input.get(),
                                          values_input.size(),
                                          rocprim::make_discard_iterator(),
                                          d_aggregates_output.get(),
                                          rocprim::make_discard_iterator(),
                                          rocprim::plus<int>(),
                                          rocprim::equal_to<int>(),
                                          stream,
                                          debug_synchronous);
        },
        stream);

    // Check if output values are as expected
    std::vector<int> aggregates_output = d_aggregates_output.load();

    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(aggregates_output, aggregates_expected));
}
