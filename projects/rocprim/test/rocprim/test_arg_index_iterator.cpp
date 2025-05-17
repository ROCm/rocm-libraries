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

// required test headers
#include "../../common/utils_device_ptr.hpp"
#include "test_seed.hpp"
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

// required rocprim headers
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/arg_index_iterator.hpp>
#include <rocprim/thread/thread_operators.hpp>
#include <rocprim/type_traits.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

// Params for tests
template<class InputType>
struct RocprimArgIndexIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimArgIndexIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    const bool debug_synchronous = false;
};

using RocprimArgIndexIteratorTestsParams
    = ::testing::Types<RocprimArgIndexIteratorParams<int>,
                       RocprimArgIndexIteratorParams<unsigned int>,
                       RocprimArgIndexIteratorParams<unsigned long>,
                       RocprimArgIndexIteratorParams<float>>;

TYPED_TEST_SUITE(RocprimArgIndexIteratorTests, RocprimArgIndexIteratorTestsParams);

TYPED_TEST(RocprimArgIndexIteratorTests, Equal)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::arg_index_iterator<T*>;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        std::vector<T> input = test_utils::get_random_data_wrapped<T>(5, 1, 200, seed_value);

        Iterator x(input.data());
        Iterator y = x;
        for(size_t i = 0; i < 5; i++)
        {
            ASSERT_EQ(x[i].key, i);
            ASSERT_EQ(x[i].value, input[i]);
        }
        ASSERT_EQ(x[2].value, input[2]);

        x += 2;
        for(size_t i = 0; i < 2; i++)
        {
            y++;
        }
        ASSERT_TRUE(x == y);
    }
}

TYPED_TEST(RocprimArgIndexIteratorTests, ReduceArgMinimum)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::arg_index_iterator<T*>;
    using key_value = typename Iterator::value_type;
    using difference_type = typename Iterator::difference_type;
    const bool debug_synchronous = false;

    const size_t size = 1024;

    hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<T> input = test_utils::get_random_data_wrapped<T>(size, 1, 200, seed_value);
        std::vector<key_value> output(1);

        common::device_ptr<T>         d_input(input);
        common::device_ptr<key_value> d_output(output.size());

        Iterator d_iter(d_input.get());

        rocprim::arg_min reduce_op;
        const key_value  max(rocprim::numeric_limits<difference_type>::max(),
                            rocprim::numeric_limits<T>::max());

        // Calculate expected results on host
        Iterator x(input.data());
        key_value expected = std::accumulate(x, x + size, max, reduce_op);

        // temp storage
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        HIP_CHECK(rocprim::reduce(nullptr,
                                  temp_storage_size_bytes,
                                  d_iter,
                                  d_output.get(),
                                  max,
                                  input.size(),
                                  reduce_op,
                                  stream,
                                  debug_synchronous));

        // temp_storage_size_bytes must be >0
        ASSERT_GT(temp_storage_size_bytes, 0);
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        // Run
        HIP_CHECK(rocprim::reduce(d_temp_storage.get(),
                                  temp_storage_size_bytes,
                                  d_iter,
                                  d_output.get(),
                                  max,
                                  input.size(),
                                  reduce_op,
                                  stream,
                                  debug_synchronous));
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        output = d_output.load();

        // Check if output values are as expected
        test_utils::assert_eq(output[0].key, expected.key);
        test_utils::assert_eq(output[0].value, expected.value);
    }
}
