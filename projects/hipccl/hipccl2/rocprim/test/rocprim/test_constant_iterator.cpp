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
#include "test_utils_types.hpp"

// required rocprim headers
#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/constant_iterator.hpp>

#include <cstddef>
#include <vector>

// Params for tests
template<class InputType>
struct RocprimConstantIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimConstantIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    const bool debug_synchronous = false;
};

using RocprimConstantIteratorTestsParams
    = ::testing::Types<RocprimConstantIteratorParams<int>,
                       RocprimConstantIteratorParams<unsigned int>,
                       RocprimConstantIteratorParams<unsigned long>,
                       RocprimConstantIteratorParams<float>>;

TYPED_TEST_SUITE(RocprimConstantIteratorTests, RocprimConstantIteratorTestsParams);

TYPED_TEST(RocprimConstantIteratorTests, Basic)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T          = typename TestFixture::input_type;
    using Iterator   = rocprim::constant_iterator<T>;
    using value_type = typename Iterator::value_type;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        T        value = test_utils::get_random_value<T>(1, 100, seed_value);
        Iterator begin = rocprim::make_constant_iterator<T>(value, 0);
        Iterator mid   = begin + 5;
        Iterator end   = begin + 10;

        // Pre-increment
        Iterator it = begin;
        ++it;
        ASSERT_EQ(*it, value);

        // Post-increment
        Iterator post = it++;
        ASSERT_EQ(*post, value);
        ASSERT_EQ(*it, value);

        // Pre-decrement
        --it;
        ASSERT_EQ(*it, value);

        // Post-decrement
        post = it--;
        ASSERT_EQ(*post, value);
        ASSERT_EQ(*it, value);

        // operator+
        Iterator plus_it = begin + 3;
        ASSERT_EQ(*plus_it, value);
        Iterator plus_it_rev = 3 + begin;
        ASSERT_EQ(*plus_it_rev, value);

        // operator-
        Iterator minus_it = end - 3;
        ASSERT_EQ(*minus_it, value);

        // compound assignment +=
        Iterator a = begin;
        a += 4;
        ASSERT_EQ(*a, value);

        // compound assignment -=
        a -= 2;
        ASSERT_EQ(*a, value);

        // Subtraction of iterators (distance)
        ASSERT_EQ(end - begin, 10);
        ASSERT_EQ(mid - begin, 5);
        ASSERT_EQ(begin - mid, -5);

        // Indexing operator[]
        for(int i = 0; i < 10; i++)
        {
            ASSERT_EQ(begin[i], value);
        }

        // Comparisons
        ASSERT_TRUE(begin == begin);
        ASSERT_TRUE(begin != end);
        ASSERT_TRUE(begin < end);
        ASSERT_TRUE(end > begin);
        ASSERT_TRUE(begin <= begin);
        ASSERT_TRUE(begin <= end);
        ASSERT_TRUE(end >= begin);
        ASSERT_TRUE(end >= end);

        // Arrow operator
        const value_type* ptr = begin.operator->();
        ASSERT_EQ(*ptr, value);
    }
}

template<class T>
struct transform
{
    __device__ __host__
    constexpr T operator()(const T& a) const
    {
        return 5 + a;
    }
};

TYPED_TEST(RocprimConstantIteratorTests, Transform)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::constant_iterator<T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const size_t size = 1024;

    hipStream_t stream = 0; // default

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Create constant_iterator<U> with random starting point
        const auto value = test_utils::get_random_value<T>(0, 200, seed_value);
        Iterator input_begin(value);

        std::vector<T> output(size);
        common::device_ptr<T> d_output(output.size());

        // Calculate expected results on host
        std::vector<T> expected(size, transform<T>()(value));

        // Run
        HIP_CHECK(rocprim::transform(input_begin,
                                     d_output.get(),
                                     size,
                                     transform<T>(),
                                     stream,
                                     debug_synchronous));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy output to host
        output = d_output.load();

        // Validating results
        test_utils::assert_near(output, expected, test_utils::precision<T>);
    }

}
