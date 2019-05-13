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

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>

#include "test_header.hpp"

TESTS_DEFINE(RemoveTests, FullTestsParams);

TESTS_DEFINE(RemoveVariableTests, NumericalTestsParams);

template <typename T>
struct is_even : thrust::unary_function<T, bool>
{
    __host__ __device__ bool operator()(T x)
    {
        return (static_cast<unsigned int>(x) & 1) == 0;
    }
};

template <typename T>
struct is_true : thrust::unary_function<T, bool>
{
    __host__ __device__ bool operator()(T x)
    {
        return x ? true : false;
    }
};

TYPED_TEST(RemoveTests, TestRemoveSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = T(1);
    data[1] = T(2);
    data[2] = T(1);
    data[3] = T(3);
    data[4] = T(2);

    typename Vector::iterator end = thrust::remove(data.begin(), data.end(), (T)2);

    ASSERT_EQ(end - data.begin(), 3);

    ASSERT_EQ(data[0], T(1));
    ASSERT_EQ(data[1], T(1));
    ASSERT_EQ(data[2], T(3));
}

template <typename ForwardIterator, typename T>
ForwardIterator remove(my_system& system, ForwardIterator first, ForwardIterator, const T&)
{
    system.validate_dispatch();
    return first;
}

TEST(RemoveTests, TestRemoveDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename T>
ForwardIterator remove(my_tag, ForwardIterator first, ForwardIterator, const T&)
{
    *first = 13;
    return first;
}

TEST(RemoveTests, TestRemoveDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::remove(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(RemoveTests, TestRemoveCopySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = T(1);
    data[1] = T(2);
    data[2] = T(1);
    data[3] = T(3);
    data[4] = T(2);

    Vector result(5);

    typename Vector::iterator end
        = thrust::remove_copy(data.begin(), data.end(), result.begin(), T(2));

    ASSERT_EQ(end - result.begin(), 3);

    ASSERT_EQ(result[0], T(1));
    ASSERT_EQ(result[1], T(1));
    ASSERT_EQ(result[2], T(3));
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator
    remove_copy(my_system& system, InputIterator, InputIterator, OutputIterator result, const T&)
{
    system.validate_dispatch();
    return result;
}

TEST(RemoveTests, TestRemoveCopyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_copy(sys, vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator remove_copy(my_tag, InputIterator, InputIterator, OutputIterator result, const T&)
{
    *result = 13;
    return result;
}

TEST(RemoveTests, TestRemoveCopyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::remove_copy(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(RemoveTests, TestRemoveIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    typename Vector::iterator end = thrust::remove_if(data.begin(), data.end(), is_even<T>());

    ASSERT_EQ(end - data.begin(), 3);

    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 1);
    ASSERT_EQ(data[2], 3);
}

template <typename ForwardIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    remove_if(my_system& system, ForwardIterator first, ForwardIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(RemoveTests, TestRemoveIfDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_if(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    remove_if(my_tag, ForwardIterator first, ForwardIterator, Predicate)
{
    *first = 13;
    return first;
}

TEST(RemoveTests, TestRemoveIfDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::remove_if(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(RemoveTests, TestRemoveIfStencilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    Vector stencil(5);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 0;
    stencil[3] = 0;
    stencil[4] = 1;

    typename Vector::iterator end
        = thrust::remove_if(data.begin(), data.end(), stencil.begin(), thrust::identity<T>());

    ASSERT_EQ(end - data.begin(), 3);

    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 1);
    ASSERT_EQ(data[2], 3);
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    remove_if(my_system& system, ForwardIterator first, ForwardIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(RemoveTests, TestRemoveIfStencilDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_if(sys, vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    remove_if(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate)
{
    *first = 13;
    return first;
}

TEST(RemoveTests, TestRemoveIfStencilDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::remove_if(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(RemoveTests, TestRemoveCopyIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    Vector result(5);

    typename Vector::iterator end
        = thrust::remove_copy_if(data.begin(), data.end(), result.begin(), is_even<T>());

    ASSERT_EQ(end - result.begin(), 3);

    ASSERT_EQ(result[0], 1);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 3);
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__ InputIterator
                    remove_copy_if(my_system& system, InputIterator first, InputIterator, OutputIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(RemoveTests, TestRemoveCopyIfDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__ InputIterator
                    remove_copy_if(my_tag, InputIterator first, InputIterator, OutputIterator, Predicate)
{
    *first = 13;
    return first;
}

TEST(RemoveTests, TestRemoveCopyIfDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::remove_copy_if(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(RemoveTests, TestRemoveCopyIfStencilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(5);
    data[0] = T(1);
    data[1] = T(2);
    data[2] = T(1);
    data[3] = T(3);
    data[4] = T(2);

    Vector stencil(5);
    stencil[0] = T(0);
    stencil[1] = T(1);
    stencil[2] = T(0);
    stencil[3] = T(0);
    stencil[4] = T(1);

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy_if(
        data.begin(), data.end(), stencil.begin(), result.begin(), thrust::identity<T>());

    ASSERT_EQ(end - result.begin(), 3);

    ASSERT_EQ(result[0], T(1));
    ASSERT_EQ(result[1], T(1));
    ASSERT_EQ(result[2], T(3));
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
__host__ __device__ OutputIterator remove_copy_if(my_system& system,
                                                  InputIterator1,
                                                  InputIterator1,
                                                  InputIterator2,
                                                  OutputIterator result,
                                                  Predicate)
{
    system.validate_dispatch();
    return result;
}

TEST(RemoveTests, TestRemoveCopyIfStencilDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::remove_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
__host__ __device__ OutputIterator remove_copy_if(
    my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
    *result = 13;
    return result;
}

TEST(RemoveTests, TestRemoveCopyIfStencilDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::remove_copy_if(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(RemoveVariableTests, TestRemove)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data = h_data;

        size_t h_size = thrust::remove(h_data.begin(), h_data.end(), T(0)) - h_data.begin();
        size_t d_size = thrust::remove(d_data.begin(), d_data.end(), T(0)) - d_data.begin();

        ASSERT_EQ(h_size, d_size);

        h_data.resize(h_size);
        d_data.resize(d_size);

        ASSERT_EQ(h_data, d_data);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveIf)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data = h_data;

        size_t h_size
            = thrust::remove_if(h_data.begin(), h_data.end(), is_true<T>()) - h_data.begin();
        size_t d_size
            = thrust::remove_if(d_data.begin(), d_data.end(), is_true<T>()) - d_data.begin();

        ASSERT_EQ(h_size, d_size);

        h_data.resize(h_size);
        d_data.resize(d_size);

        ASSERT_EQ(h_data, d_data);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveIfStencil)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<bool> h_stencil = get_random_data<bool>(
            size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max());
        thrust::device_vector<T> d_stencil = h_stencil;

        size_t h_size
            = thrust::remove_if(h_data.begin(), h_data.end(), h_stencil.begin(), is_true<T>())
              - h_data.begin();
        size_t d_size
            = thrust::remove_if(d_data.begin(), d_data.end(), d_stencil.begin(), is_true<T>())
              - d_data.begin();

        ASSERT_EQ(h_size, d_size);

        h_data.resize(h_size);
        d_data.resize(d_size);

        ASSERT_EQ(h_data, d_data);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopy)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(size);
        thrust::device_vector<T> d_result(size);

        size_t h_size = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), T(0))
                        - h_result.begin();
        size_t d_size = thrust::remove_copy(d_data.begin(), d_data.end(), d_result.begin(), T(0))
                        - d_result.begin();

        ASSERT_EQ(h_size, d_size);

        h_data.resize(h_size);
        d_data.resize(d_size);

        ASSERT_EQ(h_data, d_data);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopyToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data = h_data;

        size_t num_zeros    = thrust::count(h_data.begin(), h_data.end(), T(0));
        size_t num_nonzeros = h_data.size() - num_zeros;

        thrust::discard_iterator<> h_result = thrust::remove_copy(
            h_data.begin(), h_data.end(), thrust::make_discard_iterator(), T(0));

        thrust::discard_iterator<> d_result = thrust::remove_copy(
            d_data.begin(), d_data.end(), thrust::make_discard_iterator(), T(0));

        thrust::discard_iterator<> reference(num_nonzeros);

        ASSERT_EQ(reference, h_result);
        ASSERT_EQ(reference, d_result);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopyToDiscardIteratorZipped)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        size_t num_zeros    = thrust::count(h_data.begin(), h_data.end(), T(0));
        size_t num_nonzeros = h_data.size() - num_zeros;

        typedef thrust::tuple<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
            Tuple1;
        typedef thrust::tuple<typename thrust::device_vector<T>::iterator,
                              thrust::discard_iterator<>>
            Tuple2;

        typedef thrust::zip_iterator<Tuple1> ZipIterator1;
        typedef thrust::zip_iterator<Tuple2> ZipIterator2;

        ZipIterator1 h_result = thrust::remove_copy(
            thrust::make_zip_iterator(thrust::make_tuple(h_data.begin(), h_data.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(h_data.end(), h_data.end())),
            thrust::make_zip_iterator(
                thrust::make_tuple(h_output.begin(), thrust::make_discard_iterator())),
            thrust::make_tuple(T(0), T(0)));

        ZipIterator2 d_result = thrust::remove_copy(
            thrust::make_zip_iterator(thrust::make_tuple(d_data.begin(), d_data.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_data.end(), d_data.end())),
            thrust::make_zip_iterator(
                thrust::make_tuple(d_output.begin(), thrust::make_discard_iterator())),
            thrust::make_tuple(T(0), T(0)));

        thrust::discard_iterator<> reference(num_nonzeros);

        ASSERT_EQ(h_output, d_output);
        ASSERT_EQ_QUIET(reference, thrust::get<1>(h_result.get_iterator_tuple()));
        ASSERT_EQ_QUIET(reference, thrust::get<1>(d_result.get_iterator_tuple()));
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopyIf)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(size);
        thrust::device_vector<T> d_result(size);

        size_t h_size
            = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>())
              - h_result.begin();
        size_t d_size
            = thrust::remove_copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>())
              - d_result.begin();

        ASSERT_EQ(h_size, d_size);

        h_result.resize(h_size);
        d_result.resize(d_size);

        ASSERT_EQ(h_result, d_result);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopyIfToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        size_t num_false
            = thrust::count_if(h_data.begin(), h_data.end(), thrust::not1(is_true<T>()));

        thrust::discard_iterator<> h_result = thrust::remove_copy_if(
            h_data.begin(), h_data.end(), thrust::make_discard_iterator(), is_true<T>());

        thrust::discard_iterator<> d_result = thrust::remove_copy_if(
            d_data.begin(), d_data.end(), thrust::make_discard_iterator(), is_true<T>());

        thrust::discard_iterator<> reference(num_false);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopyIfStencil)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<bool> h_stencil = get_random_data<bool>(
            size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max());
        thrust::device_vector<bool> d_stencil = h_stencil;

        thrust::host_vector<T>   h_result(size);
        thrust::device_vector<T> d_result(size);

        size_t h_size
            = thrust::remove_copy_if(
                  h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_true<T>())
              - h_result.begin();
        size_t d_size
            = thrust::remove_copy_if(
                  d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_true<T>())
              - d_result.begin();

        ASSERT_EQ(h_size, d_size);

        h_result.resize(h_size);
        d_result.resize(d_size);

        ASSERT_EQ(h_result, d_result);
    }
}

TYPED_TEST(RemoveVariableTests, TestRemoveCopyIfStencilToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<bool> h_stencil = get_random_data<bool>(
            size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max());
        thrust::device_vector<bool> d_stencil = h_stencil;

        size_t num_false
            = thrust::count_if(h_stencil.begin(), h_stencil.end(), thrust::not1(is_true<T>()));

        thrust::discard_iterator<> h_result
            = thrust::remove_copy_if(h_data.begin(),
                                     h_data.end(),
                                     h_stencil.begin(),
                                     thrust::make_discard_iterator(),
                                     is_true<T>());

        thrust::discard_iterator<> d_result
            = thrust::remove_copy_if(d_data.begin(),
                                     d_data.end(),
                                     d_stencil.begin(),
                                     thrust::make_discard_iterator(),
                                     is_true<T>());

        thrust::discard_iterator<> reference(num_false);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}
