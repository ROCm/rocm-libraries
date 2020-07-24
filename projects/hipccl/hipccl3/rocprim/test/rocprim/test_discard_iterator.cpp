// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

// required rocprim headers
#include <rocprim/functional.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>

// required test headers
#include "test_seed.hpp"
#include "test_utils_types.hpp"

TEST(RocprimDiscardIteratorTests, Equal)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Iterator = typename rocprim::discard_iterator;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        Iterator x(test_utils::get_random_value<size_t>(0, 200, seed_value));
            Iterator y = x;
            ASSERT_EQ(x, y);

            x += 100;
            for(size_t i = 0; i < 100; i++)
            {
                y++;
            }
            ASSERT_EQ(x, y);

            y--;
            ASSERT_NE(x, y);
    }
}

TEST(RocprimDiscardIteratorTests, Less)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using Iterator = typename rocprim::discard_iterator;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        Iterator x(test_utils::get_random_value<size_t>(0, 200, seed_value));
        Iterator y = x + 1;
        ASSERT_LT(x, y);

        x += 100;
        for(size_t i = 0; i < 100; i++)
        {
            y++;
        }
        ASSERT_LT(x, y);
    }
}

TEST(RocprimDiscardIteratorTests, ReduceByKey)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    const bool debug_synchronous = false;

    hipStream_t stream = 0; // default

    // host input
    std::vector<int> keys_input = {
        0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0
    };
    std::vector<int> values_input(keys_input.size(), 1);

    // expected output
    std::vector<int> aggregates_expected = { 3, 2, 2, 4 };

    // device input/output
    int * d_keys_input;
    int * d_values_input;
    HIP_CHECK(hipMalloc(&d_keys_input, keys_input.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_values_input, values_input.size() * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            keys_input.size() * sizeof(int),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            values_input.size() * sizeof(int),
            hipMemcpyHostToDevice
        )
    );
    int * d_aggregates_output;
    HIP_CHECK(hipMalloc(&d_aggregates_output, aggregates_expected.size() * sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    // Get temporary storage size
    size_t temporary_storage_bytes;
    HIP_CHECK(
        rocprim::reduce_by_key(
            nullptr, temporary_storage_bytes,
            d_keys_input,
            d_values_input, values_input.size(),
            rocprim::make_discard_iterator(),
            d_aggregates_output,
            rocprim::make_discard_iterator(),
            rocprim::plus<int>(), rocprim::equal_to<int>(),
            stream, debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    ASSERT_GT(temporary_storage_bytes, 0);

    void * d_temporary_storage;
    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

    HIP_CHECK(
        rocprim::reduce_by_key(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input,
            d_values_input, values_input.size(),
            rocprim::make_discard_iterator(),
            d_aggregates_output,
            rocprim::make_discard_iterator(),
            rocprim::plus<int>(), rocprim::equal_to<int>(),
            stream, debug_synchronous
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Check if output values are as expected
    std::vector<int> aggregates_output(aggregates_expected.size());
    HIP_CHECK(
        hipMemcpy(
            aggregates_output.data(), d_aggregates_output,
            aggregates_expected.size() * sizeof(int),
            hipMemcpyDeviceToHost
        )
    );
    for(size_t i = 0; i < aggregates_output.size(); i++)
    {
        ASSERT_EQ(aggregates_output[i], aggregates_expected[i]);
    }

    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_aggregates_output));
}
