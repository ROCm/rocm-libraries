/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/unique.h>

#include "test_header.hpp"

TESTS_DEFINE(UniqueTests, FullTestsParams);

TESTS_DEFINE(UniqueIntegralTests, IntegerTestsParams);

template <typename ForwardIterator>
ForwardIterator unique(my_system& system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

TEST(UniqueTests, TestUniqueDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique(sys, vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator unique(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

TEST(UniqueTests, TestUniqueDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::unique(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator unique_copy(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(UniqueTests, TestUniqueCopyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique_copy(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator unique_copy(my_tag, InputIterator, InputIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(UniqueTests, TestUniqueCopyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::unique_copy(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename T>
struct is_equal_div_10_unique
{
    __host__ __device__ bool operator()(const T x, const T& y) const
    {
        return ((int)x / 10) == ((int)y / 10);
    }
};

TYPED_TEST(UniqueTests, TestUniqueSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(10);
    data[0] = 11;
    data[1] = 11;
    data[2] = 12;
    data[3] = 20;
    data[4] = 29;
    data[5] = 21;
    data[6] = 21;
    data[7] = 31;
    data[8] = 31;
    data[9] = 37;

    typename Vector::iterator new_last;

    new_last = thrust::unique(data.begin(), data.end());

    ASSERT_EQ(new_last - data.begin(), 7);
    ASSERT_EQ(data[0], 11);
    ASSERT_EQ(data[1], 12);
    ASSERT_EQ(data[2], 20);
    ASSERT_EQ(data[3], 29);
    ASSERT_EQ(data[4], 21);
    ASSERT_EQ(data[5], 31);
    ASSERT_EQ(data[6], 37);

    new_last = thrust::unique(data.begin(), new_last, is_equal_div_10_unique<T>());

    ASSERT_EQ(new_last - data.begin(), 3);
    ASSERT_EQ(data[0], 11);
    ASSERT_EQ(data[1], 20);
    ASSERT_EQ(data[2], 31);
}

TYPED_TEST(UniqueIntegralTests, TestUnique)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        h_new_last = thrust::unique(h_data.begin(), h_data.end());
        d_new_last = thrust::unique(d_data.begin(), d_data.end());

        ASSERT_EQ(h_new_last - h_data.begin(), d_new_last - d_data.begin());

        h_data.resize(h_new_last - h_data.begin());
        d_data.resize(d_new_last - d_data.begin());

        ASSERT_EQ(h_data, d_data);
    }
}

TYPED_TEST(UniqueTests, TestUniqueCopySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(10);
    data[0] = 11;
    data[1] = 11;
    data[2] = 12;
    data[3] = 20;
    data[4] = 29;
    data[5] = 21;
    data[6] = 21;
    data[7] = 31;
    data[8] = 31;
    data[9] = 37;

    Vector output(10, -1);

    typename Vector::iterator new_last;

    new_last = thrust::unique_copy(data.begin(), data.end(), output.begin());

    ASSERT_EQ(new_last - output.begin(), 7);
    ASSERT_EQ(output[0], 11);
    ASSERT_EQ(output[1], 12);
    ASSERT_EQ(output[2], 20);
    ASSERT_EQ(output[3], 29);
    ASSERT_EQ(output[4], 21);
    ASSERT_EQ(output[5], 31);
    ASSERT_EQ(output[6], 37);

    new_last
        = thrust::unique_copy(output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>());

    ASSERT_EQ(new_last - data.begin(), 3);
    ASSERT_EQ(data[0], 11);
    ASSERT_EQ(data[1], 20);
    ASSERT_EQ(data[2], 31);
}

TYPED_TEST(UniqueIntegralTests, TestUniqueCopy)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        h_new_last = thrust::unique_copy(h_data.begin(), h_data.end(), h_output.begin());
        d_new_last = thrust::unique_copy(d_data.begin(), d_data.end(), d_output.begin());

        ASSERT_EQ(h_new_last - h_output.begin(), d_new_last - d_output.begin());

        h_data.resize(h_new_last - h_output.begin());
        d_data.resize(d_new_last - d_output.begin());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(UniqueIntegralTests, TestUniqueCopyToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T> h_unique = h_data;
        h_unique.erase(thrust::unique(h_unique.begin(), h_unique.end()), h_unique.end());

        thrust::discard_iterator<> reference(h_unique.size());

        typename thrust::host_vector<T>::iterator   h_new_last;
        typename thrust::device_vector<T>::iterator d_new_last;

        thrust::discard_iterator<> h_result
            = thrust::unique_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> d_result
            = thrust::unique_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator());

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}
