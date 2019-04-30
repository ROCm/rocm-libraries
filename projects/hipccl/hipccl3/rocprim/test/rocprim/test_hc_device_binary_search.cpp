// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>
#include <utility>

// Google Test
#include <gtest/gtest.h>

// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

template<
    class Haystack,
    class Needle,
    class Output = size_t,
    class CompareFunction = rocprim::less<>
>
struct params
{
    using haystack_type = Haystack;
    using needle_type = Needle;
    using output_type = Output;
    using compare_op_type = CompareFunction;
};

template<class Params>
class RocprimDeviceBinarySearch : public ::testing::Test {
public:
    using params = Params;
};

using custom_int2 = test_utils::custom_test_type<int>;
using custom_double2 = test_utils::custom_test_type<double>;

typedef ::testing::Types<
    params<int, int>,
    params<unsigned long long, unsigned long long, size_t, rocprim::greater<unsigned long long> >,
    params<float, double, unsigned int, rocprim::greater<double> >,
    params<double, int>,
    params<custom_int2, custom_int2>,
    params<custom_double2, custom_double2, unsigned int, rocprim::greater<custom_double2> >
> Params;

TYPED_TEST_CASE(RocprimDeviceBinarySearch, Params);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = { 1, 10, 53, 211, 1024, 2345, 4096, 34567, (1 << 16) - 1220, (1 << 22) - 76543 };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(5, 1, 100000);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

TYPED_TEST(RocprimDeviceBinarySearch, LowerBound)
{
    using haystack_type = typename TestFixture::params::haystack_type;
    using needle_type = typename TestFixture::params::needle_type;
    using output_type = typename TestFixture::params::output_type;
    using compare_op_type = typename TestFixture::params::compare_op_type;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const bool debug_synchronous = false;

    compare_op_type compare_op;

    for(size_t size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_t haystack_size = size;
        const size_t needles_size = std::sqrt(size);
        const size_t d = haystack_size / 100;

        // Generate data
        std::vector<haystack_type> haystack = test_utils::get_random_data<haystack_type>(
            haystack_size, 0, haystack_size + 2 * d
        );
        std::sort(haystack.begin(), haystack.end(), compare_op);

        // Use a narrower range for needles for checking out-of-haystack cases
        std::vector<needle_type> needles = test_utils::get_random_data<needle_type>(
            needles_size, d, haystack_size + d
        );

        hc::array<haystack_type> d_haystack(hc::extent<1>(haystack_size), haystack.begin(), acc_view);
        hc::array<needle_type> d_needles(hc::extent<1>(needles_size), needles.begin(), acc_view);
        hc::array<output_type> d_output(needles_size, acc_view);

        // Calculate expected results on host
        std::vector<output_type> expected(needles_size);
        for(size_t i = 0; i < needles_size; i++)
        {
            expected[i] =
                std::lower_bound(haystack.begin(), haystack.end(), needles[i], compare_op) -
                haystack.begin();
        }

        size_t temporary_storage_bytes;
        rocprim::lower_bound(
            nullptr, temporary_storage_bytes,
            d_haystack.accelerator_pointer(), d_needles.accelerator_pointer(), d_output.accelerator_pointer(),
            haystack_size, needles_size,
            compare_op,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0);

        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rocprim::lower_bound(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_haystack.accelerator_pointer(), d_needles.accelerator_pointer(), d_output.accelerator_pointer(),
            haystack_size, needles_size,
            compare_op,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<output_type> output = d_output;

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}

TYPED_TEST(RocprimDeviceBinarySearch, UpperBound)
{
    using haystack_type = typename TestFixture::params::haystack_type;
    using needle_type = typename TestFixture::params::needle_type;
    using output_type = typename TestFixture::params::output_type;
    using compare_op_type = typename TestFixture::params::compare_op_type;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const bool debug_synchronous = false;

    compare_op_type compare_op;

    for(size_t size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_t haystack_size = size;
        const size_t needles_size = std::sqrt(size);
        const size_t d = haystack_size / 100;

        // Generate data
        std::vector<haystack_type> haystack = test_utils::get_random_data<haystack_type>(
            haystack_size, 0, haystack_size + 2 * d
        );
        std::sort(haystack.begin(), haystack.end(), compare_op);

        // Use a narrower range for needles for checking out-of-haystack cases
        std::vector<needle_type> needles = test_utils::get_random_data<needle_type>(
            needles_size, d, haystack_size + d
        );

        hc::array<haystack_type> d_haystack(hc::extent<1>(haystack_size), haystack.begin(), acc_view);
        hc::array<needle_type> d_needles(hc::extent<1>(needles_size), needles.begin(), acc_view);
        hc::array<output_type> d_output(needles_size, acc_view);

        // Calculate expected results on host
        std::vector<output_type> expected(needles_size);
        for(size_t i = 0; i < needles_size; i++)
        {
            expected[i] =
                std::upper_bound(haystack.begin(), haystack.end(), needles[i], compare_op) -
                haystack.begin();
        }

        size_t temporary_storage_bytes;
        rocprim::upper_bound(
            nullptr, temporary_storage_bytes,
            d_haystack.accelerator_pointer(), d_needles.accelerator_pointer(), d_output.accelerator_pointer(),
            haystack_size, needles_size,
            compare_op,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0);

        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rocprim::upper_bound(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_haystack.accelerator_pointer(), d_needles.accelerator_pointer(), d_output.accelerator_pointer(),
            haystack_size, needles_size,
            compare_op,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<output_type> output = d_output;

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}

TYPED_TEST(RocprimDeviceBinarySearch, BinarySearch)
{
    using haystack_type = typename TestFixture::params::haystack_type;
    using needle_type = typename TestFixture::params::needle_type;
    using output_type = typename TestFixture::params::output_type;
    using compare_op_type = typename TestFixture::params::compare_op_type;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const bool debug_synchronous = false;

    compare_op_type compare_op;

    for(size_t size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        const size_t haystack_size = size;
        const size_t needles_size = std::sqrt(size);
        const size_t d = haystack_size / 100;

        // Generate data
        std::vector<haystack_type> haystack = test_utils::get_random_data<haystack_type>(
            haystack_size, 0, haystack_size + 2 * d
        );
        std::sort(haystack.begin(), haystack.end(), compare_op);

        // Use a narrower range for needles for checking out-of-haystack cases
        std::vector<needle_type> needles = test_utils::get_random_data<needle_type>(
            needles_size, d, haystack_size + d
        );

        hc::array<haystack_type> d_haystack(hc::extent<1>(haystack_size), haystack.begin(), acc_view);
        hc::array<needle_type> d_needles(hc::extent<1>(needles_size), needles.begin(), acc_view);
        hc::array<output_type> d_output(needles_size, acc_view);

        // Calculate expected results on host
        std::vector<output_type> expected(needles_size);
        for(size_t i = 0; i < needles_size; i++)
        {
            expected[i] = std::binary_search(haystack.begin(), haystack.end(), needles[i], compare_op);
        }

        size_t temporary_storage_bytes;
        rocprim::binary_search(
            nullptr, temporary_storage_bytes,
            d_haystack.accelerator_pointer(), d_needles.accelerator_pointer(), d_output.accelerator_pointer(),
            haystack_size, needles_size,
            compare_op,
            acc_view, debug_synchronous
        );

        ASSERT_GT(temporary_storage_bytes, 0);

        hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);

        rocprim::binary_search(
            d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
            d_haystack.accelerator_pointer(), d_needles.accelerator_pointer(), d_output.accelerator_pointer(),
            haystack_size, needles_size,
            compare_op,
            acc_view, debug_synchronous
        );
        acc_view.wait();

        std::vector<output_type> output = d_output;

        ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(output, expected));
    }
}
