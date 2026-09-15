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
#include "test_utils.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
// #include "test_utils_types.hpp"

// required rocprim headers
#include <rocprim/config.hpp>
#include <rocprim/device/device_reduce.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/iterator/transform_output_iterator.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

template<class T>
struct times_two
{
    ROCPRIM_HOST_DEVICE
    T operator()(const T& value) const
    {
        return 2 * value;
    }
};

template<class T>
struct plus_ten
{
    ROCPRIM_HOST_DEVICE
    T operator()(const T& value) const
    {
        return value + 10;
    }
};

// Params for tests
template<class InputType, class UnaryFunction = times_two<InputType>, class ValueType = InputType>
struct RocprimTransformIteratorParams
{
    using input_type     = InputType;
    using value_type     = ValueType;
    using unary_function = UnaryFunction;
};

template<class Params>
class RocprimTransformIteratorTests : public ::testing::Test
{
public:
    using input_type             = typename Params::input_type;
    using value_type             = typename Params::value_type;
    using unary_function         = typename Params::unary_function;
    const bool debug_synchronous = false;
};

using RocprimTransformIteratorTestsParams
    = ::testing::Types<RocprimTransformIteratorParams<int, plus_ten<long>>,
                       RocprimTransformIteratorParams<unsigned int>,
                       RocprimTransformIteratorParams<unsigned long>,
                       RocprimTransformIteratorParams<float, plus_ten<double>, double>>;

TYPED_TEST_SUITE(RocprimTransformIteratorTests, RocprimTransformIteratorTestsParams);

TYPED_TEST(RocprimTransformIteratorTests, Basic)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::input_type;
    using value_type     = typename TestFixture::value_type;
    using unary_function = typename TestFixture::unary_function;
    using iterator_type =
        typename rocprim::transform_iterator<input_type*, unary_function, value_type>;
    using difference_type = typename iterator_type::difference_type;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        std::vector<input_type> input
            = test_utils::get_random_data_wrapped<input_type>(10, 1, 200, seed_value);

        unary_function transform;

        auto begin = rocprim::make_transform_iterator<input_type*, unary_function>(input.data(),
                                                                                   transform);
        auto mid   = begin + 5;
        auto end   = begin + 10;

        // Pre-increment
        auto it = begin;
        ++it;
        ASSERT_EQ(*it, transform(input[1]));

        // Post-increment
        auto post = it++;
        ASSERT_EQ(*post, transform(input[1]));
        ASSERT_EQ(*it, transform(input[2]));

        // Pre-decrement
        it = it + 2;
        --it;
        ASSERT_EQ(*it, transform(input[3]));

        // Post-decrement
        post = it--;
        ASSERT_EQ(*post, transform(input[3]));
        ASSERT_EQ(*it, transform(input[2]));

        // Pre-decrement via -=
        it -= 2;
        ASSERT_EQ(*it, transform(input[0]));

        // operator+
        auto plus_it = begin + 3;
        ASSERT_EQ(*plus_it, transform(input[3]));
        auto plus_it_rev = 3 + begin;
        ASSERT_EQ(*plus_it_rev, transform(input[3]));

        // operator-
        auto minus_it = end - 3;
        ASSERT_EQ(*minus_it, transform(input[7]));

        // compound assignment +=
        auto a = begin;
        a += 4;
        ASSERT_EQ(*a, transform(input[4]));

        // compound assignment -=
        a -= 2;
        ASSERT_EQ(*a, transform(input[2]));

        // Subtraction of iterators (distance)
        ASSERT_EQ(end - begin, difference_type(10));
        ASSERT_EQ(mid - begin, difference_type(5));
        ASSERT_EQ(begin - mid, difference_type(-5));

        // Indexing operator[]
        for(int i = 0; i < 10; i++)
        {
            ASSERT_EQ(begin[i], transform(input[i]));
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

        struct Wrapper
        {
            value_type value;
        };

        auto transform_wrap = [&](const value_type& value) -> Wrapper
        { return Wrapper{static_cast<value_type>(transform(value))}; };
        auto test_transform_wrap = rocprim::make_transform_iterator(input.data(), transform_wrap);

        ASSERT_EQ(test_transform_wrap->value, transform(input[0]));
        ASSERT_EQ((*test_transform_wrap).value, transform(input[0]));
        ASSERT_EQ((++test_transform_wrap)->value, transform(input[1]));

        struct WrapperPointer
        {
            value_type* a;
        };

        std::vector<WrapperPointer> input_test;
        input_test.reserve(input.size());

        for(const auto& val : input)
        {
            value_type* copy = new value_type(val);
            input_test.push_back(WrapperPointer{copy});
        }

        auto func = [&](const WrapperPointer& wrapper) -> const WrapperPointer&
        {
            *(wrapper.a) = transform(static_cast<value_type>(*(wrapper.a)));
            return wrapper;
        };

        auto test_transform_wrap_ref = rocprim::make_transform_iterator(input_test.data(), func);

        ASSERT_EQ(*((*test_transform_wrap_ref).a), transform(input[0]));
        ASSERT_EQ(*(test_transform_wrap_ref->a), transform(transform(input[0])));
        ASSERT_EQ(*((++test_transform_wrap_ref)->a), transform(input[1]));

        for(auto& wp : input_test)
        {
            delete wp.a;
        }
    }
}

TYPED_TEST(RocprimTransformIteratorTests, BasicOutput)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    // Use value type over input type due to precision/casting shenanigans.
    using value_type     = typename TestFixture::value_type;
    using unary_function = typename TestFixture::unary_function;
    using iterator_type  = typename rocprim::transform_output_iterator<value_type*, unary_function>;
    using difference_type = typename iterator_type::difference_type;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        std::vector<value_type> input
            = test_utils::get_random_data_wrapped<value_type>(10, 1, 200, seed_value);

        std::vector<value_type> data = std::vector<value_type>(10, value_type{});

        unary_function transform;

        auto output
            = rocprim::make_transform_output_iterator<value_type*, unary_function>(data.data(),
                                                                                   transform);

        auto output_begin = output;
        auto output_mid   = output_begin + 5;
        auto output_end   = output_begin + 10;

        auto result_begin = data.begin();
        auto result_end   = result_begin + 10;

        // Pre-increment
        auto it_o = output_begin;
        auto it_r = result_begin;
        ++it_o;
        ++it_r;
        *it_o = input[1];
        ASSERT_EQ(*it_r, transform(input[1]));

        // Post-increment
        auto post_o = it_o++;
        auto post_r = it_r++;
        *it_o       = input[2];
        ASSERT_EQ(*post_r, transform(input[1]));
        ASSERT_EQ(*it_r, transform(input[2]));

        // Pre-decrement
        it_o = it_o + 2;
        it_r = it_r + 2;
        --it_o;
        --it_r;
        *it_o = input[3];
        ASSERT_EQ(*it_r, transform(input[3]));

        // Post-decrement
        post_o = it_o--;
        post_r = it_r--;
        *it_o  = input[2];
        ASSERT_EQ(*post_r, transform(input[3]));
        ASSERT_EQ(*it_r, transform(input[2]));

        // Pre-decrement via -=
        it_o -= 2;
        it_r -= 2;
        *it_o = input[0];
        ASSERT_EQ(*it_r, transform(input[0]));

        // operator+
        auto plus_it_o = output_begin + 3;
        auto plus_it_r = result_begin + 3;
        *plus_it_o     = input[3];
        ASSERT_EQ(*plus_it_r, transform(input[3]));

        auto plus_it_rev_o = 3 + output_begin;
        auto plus_it_rev_r = 3 + result_begin;
        *plus_it_rev_o     = input[3];
        ASSERT_EQ(*plus_it_rev_r, transform(input[3]));

        // operator-
        auto minus_it_o = output_end - 3;
        auto minus_it_r = result_end - 3;
        *minus_it_o     = input[7];
        ASSERT_EQ(*minus_it_r, transform(input[7]));

        // compound assignment +=
        auto a_o = output_begin;
        auto a_r = result_begin;
        a_o += 4;
        a_r += 4;
        *a_o = input[4];
        ASSERT_EQ(*a_r, transform(input[4]));

        // compound assignment -=
        a_r -= 2;
        a_o -= 2;
        *a_o = input[2];
        ASSERT_EQ(*a_r, transform(input[2]));

        // Subtraction of iterators (distance)
        ASSERT_EQ(output_end - output_begin, difference_type(10));
        ASSERT_EQ(output_mid - output_begin, difference_type(5));
        ASSERT_EQ(output_begin - output_mid, difference_type(-5));

        // Indexing operator[]
        for(int i = 0; i < 10; i++)
        {
            output_begin[i] = input[i];
            ASSERT_EQ(result_begin[i], transform(input[i]));
        }

        // Comparisons
        ASSERT_TRUE(output_begin == output_begin);
        ASSERT_TRUE(output_begin != output_end);
        ASSERT_TRUE(output_begin < output_end);
        ASSERT_TRUE(output_end > output_begin);
        ASSERT_TRUE(output_begin <= output_begin);
        ASSERT_TRUE(output_begin <= output_end);
        ASSERT_TRUE(output_end >= output_begin);
        ASSERT_TRUE(output_end >= output_end);
    }
}

TYPED_TEST(RocprimTransformIteratorTests, TransformReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using input_type     = typename TestFixture::input_type;
    using value_type     = typename TestFixture::value_type;
    using unary_function = typename TestFixture::unary_function;
    using iterator_type =
        typename rocprim::transform_iterator<input_type*, unary_function, value_type>;

    hipStream_t stream = 0; // default

    const size_t size = 1024;

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        // Generate data
        std::vector<input_type> input
            = test_utils::get_random_data_wrapped<input_type>(size, 1, 200, seed_value);
        std::vector<value_type> output(1);

        common::device_ptr<input_type> d_input(input);
        common::device_ptr<value_type> d_output(output.size());

        auto           reduce_op = rocprim::plus<value_type>();
        unary_function transform;

        // Calculate expected results on host
        iterator_type x(input.data(), transform);
        value_type    expected = std::accumulate(x, x + size, value_type(0), reduce_op);

        auto d_iter = iterator_type(d_input.get(), transform);

        test_utils::test_kernel_wrapper(
            [&](void* temp_storage, size_t& storage_bytes)
            {
                return rocprim::reduce(temp_storage,
                                       storage_bytes,
                                       d_iter,
                                       d_output.get(),
                                       value_type(0),
                                       input.size(),
                                       reduce_op,
                                       stream);
            },
            stream);

        output = d_output.load();

        // Check if output values are as expected
        test_utils::assert_near(output[0], expected, test_utils::precision<value_type> * size);
    }
}
