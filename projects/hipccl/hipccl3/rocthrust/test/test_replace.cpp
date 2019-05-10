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

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/replace.h>

#include "test_header.hpp"

TESTS_DEFINE(ReplaceTests, FullTestsParams);

TESTS_DEFINE(PrimitiveReplaceTests, NumericalTestsParams);

TEST(ReplaceTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(ReplaceTests, SimpleReplace)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    thrust::replace(data.begin(), data.end(), (T)1, (T)4);
    thrust::replace(data.begin(), data.end(), (T)2, (T)5);

    Vector result(5);
    result[0] = 4;
    result[1] = 5;
    result[2] = 4;
    result[3] = 3;
    result[4] = 5;

    ASSERT_EQ(data, result);
}

template <typename ForwardIterator, typename T>
void replace(my_system& system, ForwardIterator, ForwardIterator, const T&, const T&)
{
    system.validate_dispatch();
}

TEST(ReplaceTests, ValidateDispatch)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace(sys, vec.begin(), vec.begin(), 0, 0);

    ASSERT_EQ(sys.is_valid(), true);
}

template <typename ForwardIterator, typename T>
void replace(my_tag, ForwardIterator first, ForwardIterator, const T&, const T&)
{
    *first = 13;
}

TEST(ReplaceTests, ValidateDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::replace(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceWithRandomDataAndDifferentSizes)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        T new_value = (T)0;
        T old_value = (T)1;

        thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
        thrust::replace(d_data.begin(), d_data.end(), old_value, new_value);

        ASSERT_EQ(h_data.size(), size);
        ASSERT_EQ(d_data.size(), size);

        thrust::host_vector<T> h_data_d(d_data);
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
        }
    }
}

TYPED_TEST(ReplaceTests, SimpleCopyReplace)
{
    using Vector      = typename TestFixture::input_type;
    using T           = typename Vector::value_type;
    const size_t size = 5;

    Vector data(size);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    Vector dest(size);

    thrust::replace_copy(data.begin(), data.end(), dest.begin(), (T)1, (T)4);
    thrust::replace_copy(dest.begin(), dest.end(), dest.begin(), (T)2, (T)5);

    Vector result(size);
    result[0] = 4;
    result[1] = 5;
    result[2] = 4;
    result[3] = 3;
    result[4] = 5;

    thrust::host_vector<T> h_dest(dest);
    thrust::host_vector<T> h_result(result);
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_NEAR(h_dest[i], h_result[i], T(0.1));
    }
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(
    my_system& system, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
    system.validate_dispatch();
    return result;
}

TEST(ReplaceTests, ReplaceCopyValidateDispatch)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_copy(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator
    replace_copy(my_tag, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
    *result = 13;
    return result;
}

TEST(ReplaceTests, ReplaceCopyValidateDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::replace_copy(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         0,
                         0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceCopyWithRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {

        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        T old_value = (T)0;
        T new_value = (T)1;

        thrust::host_vector<T>   h_dest(size);
        thrust::device_vector<T> d_dest(size);

        thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);
        thrust::replace_copy(d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);

        thrust::host_vector<T> h_data_d(d_data);
        thrust::host_vector<T> h_dest_d(d_dest);
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
            ASSERT_NEAR(h_dest[i], h_dest_d[i], T(0.1));
        }
    }
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceCopyToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        T old_value = 0;
        T new_value = 1;

        thrust::discard_iterator<> h_result = thrust::replace_copy(
            h_data.begin(), h_data.end(), thrust::make_discard_iterator(), old_value, new_value);

        thrust::discard_iterator<> d_result = thrust::replace_copy(
            d_data.begin(), d_data.end(), thrust::make_discard_iterator(), old_value, new_value);

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ(reference, d_result);
        ASSERT_EQ(reference, h_result);
    }
}

template <typename T>
struct less_than_five
{
    __host__ __device__ bool operator()(const T& val) const
    {
        return val < 5;
    }
};

TYPED_TEST(ReplaceTests, ReplaceIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;
    size_t size  = 5;

    Vector data(size);
    data[0] = 1;
    data[1] = 3;
    data[2] = 4;
    data[3] = 6;
    data[4] = 5;

    thrust::replace_if(data.begin(), data.end(), less_than_five<T>(), (T)0);

    Vector result(size);
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 6;
    result[4] = 5;

    thrust::host_vector<T> h_data(data);
    thrust::host_vector<T> h_result(result);
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_NEAR(h_data[i], h_result[i], T(0.1));
    }
}

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, Predicate, const T&)
{
    system.validate_dispatch();
}

TEST(ReplaceTests, ValidateDispatchReplaceIf)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_if(sys, vec.begin(), vec.begin(), less_than_five<int>(), 0);

    ASSERT_EQ(sys.is_valid(), true);
}

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, Predicate, const T&)
{
    *first = 13;
}

template <class T>
struct always_true
{
    __host__ __device__ bool operator()(const T&) const
    {
        return true;
    }
};

TEST(ReplaceTests, ValidateDispatchImplicitReplaceIf)
{
    thrust::device_vector<int> vec(1);

    thrust::replace_if(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       always_true<int>(),
                       0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ReplaceTests, ReplaceIfStencilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;
    size_t size  = 5;

    Vector data(5);
    data[0] = 1;
    data[1] = 3;
    data[2] = 4;
    data[3] = 6;
    data[4] = 5;

    Vector stencil(5);
    stencil[0] = 5;
    stencil[1] = 4;
    stencil[2] = 6;
    stencil[3] = 3;
    stencil[4] = 7;

    thrust::replace_if(data.begin(), data.end(), stencil.begin(), less_than_five<T>(), (T)0);

    Vector result(5);
    result[0] = 1;
    result[1] = 0;
    result[2] = 4;
    result[3] = 0;
    result[4] = 5;

    thrust::host_vector<T> h_data(data);
    thrust::host_vector<T> h_result(result);
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_NEAR(h_data[i], h_result[i], T(0.1));
    }
}

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(
    my_system& system, ForwardIterator, ForwardIterator, InputIterator, Predicate, const T&)
{
    system.validate_dispatch();
}

TEST(ReplaceTests, ReplaceIfStencilDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_if(sys, vec.begin(), vec.begin(), vec.begin(), less_than_five<int>(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate, const T&)
{
    *first = 13;
}

TEST(ReplaceTests, ReplaceIfStencilDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::replace_if(thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       thrust::retag<my_tag>(vec.begin()),
                       always_true<int>(),
                       0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceIfWithRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T)0);
        thrust::replace_if(d_data.begin(), d_data.end(), less_than_five<T>(), (T)0);

        thrust::host_vector<T> h_data_d(d_data);
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
        }
    }
}

TYPED_TEST(ReplaceTests, ReplaceCopyIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;
    size_t size  = 5;

    Vector data(5);
    data[0] = 1;
    data[1] = 3;
    data[2] = 4;
    data[3] = 6;
    data[4] = 5;

    Vector dest(5);
    thrust::replace_copy_if(data.begin(), data.end(), dest.begin(), less_than_five<T>(), (T)0);

    Vector result(5);
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 6;
    result[4] = 5;

    thrust::host_vector<T> h_dest(dest);
    thrust::host_vector<T> h_result(result);
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_NEAR(h_dest[i], h_result[i], T(0.1));
    }
}

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
    my_system& system, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
    system.validate_dispatch();
    return result;
}

TEST(ReplaceTests, ReplaceCopyIfDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), always_true<int>(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
    my_tag, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
    *result = 13;
    return result;
}

TEST(ReplaceTests, ReplaceCopyIfDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::replace_copy_if(thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            always_true<int>(),
                            0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ReplaceTests, ReplaceCopyIfStencilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;
    size_t size  = 5;

    Vector data(5);
    data[0] = 1;
    data[1] = 3;
    data[2] = 4;
    data[3] = 6;
    data[4] = 5;

    Vector stencil(5);
    stencil[0] = 1;
    stencil[1] = 5;
    stencil[2] = 4;
    stencil[3] = 7;
    stencil[4] = 8;

    Vector dest(5);
    thrust::replace_copy_if(
        data.begin(), data.end(), stencil.begin(), dest.begin(), less_than_five<T>(), (T)0);

    Vector result(5);
    result[0] = 0;
    result[1] = 3;
    result[2] = 0;
    result[3] = 6;
    result[4] = 5;

    thrust::host_vector<T> h_dest(dest);
    thrust::host_vector<T> h_result(result);
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_NEAR(h_dest[i], h_result[i], T(0.1));
    }
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate,
          typename T>
OutputIterator replace_copy_if(my_system& system,
                               InputIterator1,
                               InputIterator1,
                               InputIterator2,
                               OutputIterator result,
                               Predicate,
                               const T&)
{
    system.validate_dispatch();
    return result;
}

TEST(ReplaceTests, TestReplaceCopyIfStencilDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::replace_copy_if(
        sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), always_true<int>(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate,
          typename T>
OutputIterator replace_copy_if(my_tag,
                               InputIterator1,
                               InputIterator1,
                               InputIterator2,
                               OutputIterator result,
                               Predicate,
                               const T&)
{
    *result = 13;
    return result;
}

TEST(ReplaceTests, ReplaceCopyIfStencilDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::replace_copy_if(thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            thrust::retag<my_tag>(vec.begin()),
                            always_true<int>(),
                            0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceCopyIfWithRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {

        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_dest(size);
        thrust::device_vector<T> d_dest(size);

        thrust::replace_copy_if(
            h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<T>(), 0);
        thrust::replace_copy_if(
            d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<T>(), 0);

        thrust::host_vector<T> h_data_d(d_data);
        thrust::host_vector<T> h_dest_d(d_dest);
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
            ASSERT_NEAR(h_dest[i], h_dest_d[i], T(0.1));
        }
    }
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceCopyIfToDiscardIteratorRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        thrust::discard_iterator<> h_result = thrust::replace_copy_if(
            h_data.begin(), h_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), 0);

        thrust::discard_iterator<> d_result = thrust::replace_copy_if(
            d_data.begin(), d_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), 0);

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ(reference, h_result);
        ASSERT_EQ(reference, d_result);
    }
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceCopyIfStencil)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_stencil = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_stencil = h_stencil;

        thrust::host_vector<T>   h_dest(size);
        thrust::device_vector<T> d_dest(size);

        thrust::replace_copy_if(h_data.begin(),
                                h_data.end(),
                                h_stencil.begin(),
                                h_dest.begin(),
                                less_than_five<T>(),
                                0);
        thrust::replace_copy_if(d_data.begin(),
                                d_data.end(),
                                d_stencil.begin(),
                                d_dest.begin(),
                                less_than_five<T>(),
                                0);

        thrust::host_vector<T> h_data_d(d_data);
        thrust::host_vector<T> h_dest_d(d_dest);
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
            ASSERT_NEAR(h_dest[i], h_dest_d[i], T(0.1));
        }
    }
}

TYPED_TEST(PrimitiveReplaceTests, ReplaceCopyIfStencilToDiscardIteratorRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {

        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_stencil = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_stencil = h_stencil;

        thrust::discard_iterator<> h_result
            = thrust::replace_copy_if(h_data.begin(),
                                      h_data.end(),
                                      h_stencil.begin(),
                                      thrust::make_discard_iterator(),
                                      less_than_five<T>(),
                                      0);

        thrust::discard_iterator<> d_result
            = thrust::replace_copy_if(d_data.begin(),
                                      d_data.end(),
                                      d_stencil.begin(),
                                      thrust::make_discard_iterator(),
                                      less_than_five<T>(),
                                      0);

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ(reference, h_result);
        ASSERT_EQ(reference, d_result);
    }
}
