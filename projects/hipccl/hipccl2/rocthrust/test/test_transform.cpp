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
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "test_header.hpp"

TESTS_INOUT_DEFINE(TransformTests, AllInOutTestsParams);

TESTS_DEFINE(TransformVectorTests, FullTestsParams);

template <class T>
struct unary_transform
{
    __device__ __host__ inline constexpr T operator()(const T& a) const
    {
        return a + 5;
    }
};

template <class T>
struct binary_transform
{
    __device__ __host__ inline constexpr T operator()(const T& a, const T& b) const
    {
        return a * 2 + b * 5;
    }
};

TYPED_TEST(TransformTests, UnaryTransform)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_input(size);
        for(size_t i = 0; i < size; i++)
        {
            h_input[i] = i;
        }

        // Calculate expected results on host
        thrust::host_vector<U> expected(size);
        thrust::transform(h_input.begin(), h_input.end(), expected.begin(), unary_transform<U>());

        thrust::device_vector<T> d_input(h_input);
        thrust::device_vector<U> d_output(size);
        thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), unary_transform<U>());

        thrust::host_vector<U> h_output = d_output;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_output[i], expected[i]) << "where index = " << i;
        }
    }
}

TYPED_TEST(TransformTests, BinaryTransform)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_input1(size);
        thrust::host_vector<T> h_input2(size);
        for(size_t i = 0; i < size; i++)
        {
            h_input1[i] = i * 3;
            h_input2[i] = i;
        }

        // Calculate expected results on host
        thrust::host_vector<U> expected(size);
        thrust::transform(h_input1.begin(),
                          h_input1.end(),
                          h_input2.begin(),
                          expected.begin(),
                          binary_transform<U>());

        thrust::device_vector<T> d_input1(h_input1);
        thrust::device_vector<T> d_input2(h_input2);
        thrust::device_vector<U> d_output(size);
        thrust::transform(d_input1.begin(),
                          d_input1.end(),
                          d_input2.begin(),
                          d_output.begin(),
                          binary_transform<U>());

        thrust::host_vector<U> h_output = d_output;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_output[i], expected[i]) << "where index = " << i;
        }
    }
}

TYPED_TEST(TransformVectorTests, TestTransformUnarySimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Iterator iter;

    Vector input(3);
    Vector output(3);
    Vector result(3);
    input[0]  = T(1);
    input[1]  = T(-2);
    input[2]  = T(3);
    result[0] = T(-1);
    result[1] = T(2);
    result[2] = T(-3);

    iter = thrust::transform(input.begin(), input.end(), output.begin(), thrust::negate<T>());

    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(output, result);
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
__host__ __device__ OutputIterator
                    transform(my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformVectorTests, TestTransformUnaryDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform(sys, vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
__host__ __device__ OutputIterator
                    transform(my_tag, InputIterator, InputIterator, OutputIterator result, UnaryFunction)
{
    *result = 13;
    return result;
}

TEST(TransformVectorTests, TestTransformUnaryDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformVectorTests, TestTransformIfUnaryNoStencilSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Iterator iter;

    Vector input(3);
    Vector output(3);
    Vector result(3);

    input[0]  = T(0);
    input[1]  = T(-2);
    input[2]  = T(0);
    output[0] = T(-1);
    output[1] = T(-2);
    output[2] = T(-3);
    result[0] = T(-1);
    result[1] = T(2);
    result[2] = T(-3);

    iter = thrust::transform_if(
        input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::identity<T>());

    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(output, result);
}

template <typename InputIterator,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
__host__ __device__ ForwardIterator transform_if(my_system& system,
                                                 InputIterator,
                                                 InputIterator,
                                                 ForwardIterator result,
                                                 UnaryFunction,
                                                 Predicate)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformVectorTests, TestTransformIfUnaryNoStencilDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
__host__ __device__ ForwardIterator transform_if(
    my_tag, InputIterator, InputIterator, ForwardIterator result, UnaryFunction, Predicate)
{
    *result = 13;
    return result;
}

TEST(TransformVectorTests, TestTransformIfUnaryNoStencilDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform_if(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformVectorTests, TestTransformIfUnarySimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Iterator iter;

    Vector input(3);
    Vector stencil(3);
    Vector output(3);
    Vector result(3);

    input[0]   = T(1);
    input[1]   = T(-2);
    input[2]   = T(3);
    output[0]  = T(1);
    output[1]  = T(2);
    output[2]  = T(3);
    stencil[0] = T(1);
    stencil[1] = T(0);
    stencil[2] = T(1);
    result[0]  = T(-1);
    result[1]  = T(2);
    result[2]  = T(-3);

    iter = thrust::transform_if(input.begin(),
                                input.end(),
                                stencil.begin(),
                                output.begin(),
                                thrust::negate<T>(),
                                thrust::identity<T>());

    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(output, result);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
__host__ __device__ ForwardIterator transform_if(my_system& system,
                                                 InputIterator1,
                                                 InputIterator1,
                                                 ForwardIterator result,
                                                 UnaryFunction,
                                                 Predicate)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformVectorTests, TestTransformIfUnaryDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
__host__ __device__ ForwardIterator transform_if(
    my_tag, InputIterator1, InputIterator1, ForwardIterator result, UnaryFunction, Predicate)
{
    *result = 13;
    return result;
}

TEST(TransformVectorTests, TestTransformIfUnaryDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform_if(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         0,
                         0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformVectorTests, TestTransformBinarySimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Iterator iter;

    Vector input1(3);
    Vector input2(3);
    Vector output(3);
    Vector result(3);
    input1[0] = T(1);
    input1[1] = T(-2);
    input1[2] = T(3);
    input2[0] = T(-4);
    input2[1] = T(5);
    input2[2] = T(6);
    result[0] = T(5);
    result[1] = T(-7);
    result[2] = T(-3);

    iter = thrust::transform(
        input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>());

    ASSERT_EQ(iter - output.begin(), input1.size());
    ASSERT_EQ(output, result);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename UnaryFunction>
__host__ __device__ OutputIterator transform(my_system& system,
                                             InputIterator1,
                                             InputIterator1,
                                             InputIterator2,
                                             OutputIterator result,
                                             UnaryFunction)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformVectorTests, TestTransformBinaryDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename UnaryFunction>
__host__ __device__ OutputIterator transform(
    my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, UnaryFunction)
{
    *result = 13;
    return result;
}

TEST(TransformVectorTests, TestTransformBinaryDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformVectorTests, TestTransformIfBinarySimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Iterator iter;

    Vector input1(3);
    Vector input2(3);
    Vector stencil(3);
    Vector output(3);
    Vector result(3);

    input1[0]  = T(1);
    input1[1]  = T(-2);
    input1[2]  = T(3);
    input2[0]  = T(-4);
    input2[1]  = T(5);
    input2[2]  = T(6);
    stencil[0] = T(0);
    stencil[1] = T(1);
    stencil[2] = T(0);
    output[0]  = T(1);
    output[1]  = T(2);
    output[2]  = T(3);
    result[0]  = T(5);
    result[1]  = T(2);
    result[2]  = T(-3);

    thrust::identity<T> identity;

    iter = thrust::transform_if(input1.begin(),
                                input1.end(),
                                input2.begin(),
                                stencil.begin(),
                                output.begin(),
                                thrust::minus<T>(),
                                thrust::not1(identity));

    ASSERT_EQ(iter - output.begin(), input1.size());
    ASSERT_EQ(output, result);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
__host__ __device__ ForwardIterator transform_if(my_system& system,
                                                 InputIterator1,
                                                 InputIterator1,
                                                 InputIterator2,
                                                 InputIterator3,
                                                 ForwardIterator result,
                                                 BinaryFunction,
                                                 Predicate)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformVectorTests, TestTransformIfBinaryDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform_if(
        sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0, 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename ForwardIterator,
          typename BinaryFunction,
          typename Predicate>
__host__ __device__ ForwardIterator transform_if(my_tag,
                                                 InputIterator1,
                                                 InputIterator1,
                                                 InputIterator2,
                                                 InputIterator3,
                                                 ForwardIterator result,
                                                 BinaryFunction,
                                                 Predicate)
{
    *result = 13;
    return result;
}

TEST(TransformVectorTests, TestTransformIfBinaryDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform_if(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.begin()),
                         0,
                         0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformTests, TestTransformUnary)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<U>   h_output(size);
        thrust::device_vector<U> d_output(size);

        thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>());
        thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(TransformTests, TestTransformUnaryToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> h_result = thrust::transform(
            h_input.begin(), h_input.end(), thrust::make_discard_iterator(), thrust::negate<T>());

        thrust::discard_iterator<> d_result = thrust::transform(
            d_input.begin(), d_input.end(), thrust::make_discard_iterator(), thrust::negate<T>());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

struct repeat2
{
    template <typename T>
    __host__ __device__ thrust::pair<T, T> operator()(T x)
    {
        return thrust::make_pair(x, x);
    }
};

TYPED_TEST(TransformTests, TestTransformUnaryToDiscardIteratorZipped)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<U>   h_output(size);
        thrust::device_vector<U> d_output(size);

        using Iterator1 = typename thrust::host_vector<U>::iterator;
        using Iterator2 = typename thrust::device_vector<U>::iterator;

        using Tuple1 = thrust::tuple<Iterator1, thrust::discard_iterator<>>;
        using Tuple2 = thrust::tuple<Iterator2, thrust::discard_iterator<>>;

        using ZipIterator1 = thrust::zip_iterator<Tuple1>;
        using ZipIterator2 = thrust::zip_iterator<Tuple2>;

        ZipIterator1 z1(thrust::make_tuple(h_output.begin(), thrust::make_discard_iterator()));
        ZipIterator2 z2(thrust::make_tuple(d_output.begin(), thrust::make_discard_iterator()));

        ZipIterator1 h_result = thrust::transform(h_input.begin(), h_input.end(), z1, repeat2());

        ZipIterator2 d_result = thrust::transform(d_input.begin(), d_input.end(), z2, repeat2());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ(h_output, d_output);

        ASSERT_EQ_QUIET(reference, thrust::get<1>(h_result.get_iterator_tuple()));
        ASSERT_EQ_QUIET(reference, thrust::get<1>(d_result.get_iterator_tuple()));
    }
}

struct is_positive
{
    template <typename T>
    __host__ __device__ bool operator()(T& x)
    {
        return x > 0;
    } // end operator()()
}; // end is_positive

TYPED_TEST(TransformTests, TestTransformIfUnaryNoStencil)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<U> h_output = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input  = h_input;
        thrust::device_vector<U> d_output = h_output;

        thrust::transform_if(
            h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), is_positive());

        thrust::transform_if(
            d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), is_positive());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(TransformTests, TestTransformIfUnary)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_stencil = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<U> h_output = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input   = h_input;
        thrust::device_vector<T> d_stencil = h_stencil;
        thrust::device_vector<U> d_output  = h_output;

        thrust::transform_if(h_input.begin(),
                             h_input.end(),
                             h_stencil.begin(),
                             h_output.begin(),
                             thrust::negate<T>(),
                             is_positive());

        thrust::transform_if(d_input.begin(),
                             d_input.end(),
                             d_stencil.begin(),
                             d_output.begin(),
                             thrust::negate<T>(),
                             is_positive());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(TransformTests, TestTransformIfUnaryToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_stencil = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input   = h_input;
        thrust::device_vector<T> d_stencil = h_stencil;

        thrust::discard_iterator<> h_result = thrust::transform_if(h_input.begin(),
                                                                   h_input.end(),
                                                                   h_stencil.begin(),
                                                                   thrust::make_discard_iterator(),
                                                                   thrust::negate<T>(),
                                                                   is_positive());

        thrust::discard_iterator<> d_result = thrust::transform_if(d_input.begin(),
                                                                   d_input.end(),
                                                                   d_stencil.begin(),
                                                                   thrust::make_discard_iterator(),
                                                                   thrust::negate<T>(),
                                                                   is_positive());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(TransformTests, TestTransformBinary)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_input2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input1 = h_input1;
        thrust::device_vector<T> d_input2 = h_input2;

        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        thrust::transform(h_input1.begin(),
                          h_input1.end(),
                          h_input2.begin(),
                          h_output.begin(),
                          thrust::minus<T>());
        thrust::transform(d_input1.begin(),
                          d_input1.end(),
                          d_input2.begin(),
                          d_output.begin(),
                          thrust::minus<T>());

        ASSERT_EQ(h_output, d_output);

        thrust::transform(h_input1.begin(),
                          h_input1.end(),
                          h_input2.begin(),
                          h_output.begin(),
                          thrust::multiplies<T>());
        thrust::transform(d_input1.begin(),
                          d_input1.end(),
                          d_input2.begin(),
                          d_output.begin(),
                          thrust::multiplies<T>());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(TransformTests, TestTransformBinaryToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_input2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input1 = h_input1;
        thrust::device_vector<T> d_input2 = h_input2;

        thrust::discard_iterator<> h_result = thrust::transform(h_input1.begin(),
                                                                h_input1.end(),
                                                                h_input2.begin(),
                                                                thrust::make_discard_iterator(),
                                                                thrust::minus<T>());
        thrust::discard_iterator<> d_result = thrust::transform(d_input1.begin(),
                                                                d_input1.end(),
                                                                d_input2.begin(),
                                                                thrust::make_discard_iterator(),
                                                                thrust::minus<T>());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(TransformTests, TestTransformIfBinary)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_input2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<T> h_stencil = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_output = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input1  = h_input1;
        thrust::device_vector<T> d_input2  = h_input2;
        thrust::device_vector<T> d_stencil = h_stencil;
        thrust::device_vector<T> d_output  = h_output;

        thrust::transform_if(h_input1.begin(),
                             h_input1.end(),
                             h_input2.begin(),
                             h_stencil.begin(),
                             h_output.begin(),
                             thrust::minus<T>(),
                             is_positive());

        thrust::transform_if(d_input1.begin(),
                             d_input1.end(),
                             d_input2.begin(),
                             d_stencil.begin(),
                             d_output.begin(),
                             thrust::minus<T>(),
                             is_positive());

        ASSERT_EQ(h_output, d_output);

        h_stencil = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        d_stencil = h_stencil;

        thrust::transform_if(h_input1.begin(),
                             h_input1.end(),
                             h_input2.begin(),
                             h_stencil.begin(),
                             h_output.begin(),
                             thrust::multiplies<T>(),
                             is_positive());

        thrust::transform_if(d_input1.begin(),
                             d_input1.end(),
                             d_input2.begin(),
                             d_stencil.begin(),
                             d_output.begin(),
                             thrust::multiplies<T>(),
                             is_positive());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(TransformTests, TestTransformIfBinaryToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        thrust::host_vector<T> h_input1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_input2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<T> h_stencil = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_input1  = h_input1;
        thrust::device_vector<T> d_input2  = h_input2;
        thrust::device_vector<T> d_stencil = h_stencil;

        thrust::discard_iterator<> h_result = thrust::transform_if(h_input1.begin(),
                                                                   h_input1.end(),
                                                                   h_input2.begin(),
                                                                   h_stencil.begin(),
                                                                   thrust::make_discard_iterator(),
                                                                   thrust::minus<T>(),
                                                                   is_positive());

        thrust::discard_iterator<> d_result = thrust::transform_if(d_input1.begin(),
                                                                   d_input1.end(),
                                                                   d_input2.begin(),
                                                                   d_stencil.begin(),
                                                                   thrust::make_discard_iterator(),
                                                                   thrust::minus<T>(),
                                                                   is_positive());

        thrust::discard_iterator<> reference(size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(TransformTests, TestTransformUnaryCountingIterator)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        size = thrust::min<size_t>(size, std::numeric_limits<T>::max());

        thrust::counting_iterator<T, thrust::host_system_tag> h_first
            = thrust::make_counting_iterator<T>(0);
        thrust::counting_iterator<T, thrust::device_system_tag> d_first
            = thrust::make_counting_iterator<T>(0);

        thrust::host_vector<U>   h_result(size);
        thrust::device_vector<U> d_result(size);

        thrust::transform(h_first, h_first + size, h_result.begin(), thrust::identity<T>());
        thrust::transform(d_first, d_first + size, d_result.begin(), thrust::identity<T>());

        ASSERT_EQ(h_result, d_result);
    }
}

TYPED_TEST(TransformTests, TestTransformBinaryCountingIterators)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;

    for(auto size : get_sizes())
    {
        size = thrust::min<size_t>(size, std::numeric_limits<T>::max());

        thrust::counting_iterator<T, thrust::host_system_tag> h_first
            = thrust::make_counting_iterator<T>(0);
        thrust::counting_iterator<T, thrust::device_system_tag> d_first
            = thrust::make_counting_iterator<T>(0);

        thrust::host_vector<U>   h_result(size);
        thrust::device_vector<U> d_result(size);

        thrust::transform(h_first, h_first + size, h_first, h_result.begin(), thrust::plus<T>());
        thrust::transform(d_first, d_first + size, d_first, d_result.begin(), thrust::plus<T>());

        ASSERT_EQ(h_result, d_result);
    }
}

template <typename T>
struct plus_mod3
{
    T* table;

    plus_mod3(T* table)
        : table(table)
    {
    }

    __host__ __device__ T operator()(T a, T b)
    {
        return table[(int)(a + b)];
    }
};

TYPED_TEST(TransformVectorTests, TestTransformWithIndirection)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector input1(7);
    Vector input2(7);
    Vector output(7, T(0));
    input1[0] = T(0);
    input2[0] = T(2);
    input1[1] = T(1);
    input2[1] = T(2);
    input1[2] = T(2);
    input2[2] = T(2);
    input1[3] = T(1);
    input2[3] = T(0);
    input1[4] = T(2);
    input2[4] = T(2);
    input1[5] = T(0);
    input2[5] = T(1);
    input1[6] = T(1);
    input2[6] = T(0);

    Vector table(6);
    table[0] = T(0);
    table[1] = T(1);
    table[2] = T(2);
    table[3] = T(0);
    table[4] = T(1);
    table[5] = T(2);

    thrust::transform(input1.begin(),
                      input1.end(),
                      input2.begin(),
                      output.begin(),
                      plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

    ASSERT_EQ(output[0], T(2));
    ASSERT_EQ(output[1], T(0));
    ASSERT_EQ(output[2], T(1));
    ASSERT_EQ(output[3], T(1));
    ASSERT_EQ(output[4], T(1));
    ASSERT_EQ(output[5], T(1));
    ASSERT_EQ(output[6], T(1));
}
