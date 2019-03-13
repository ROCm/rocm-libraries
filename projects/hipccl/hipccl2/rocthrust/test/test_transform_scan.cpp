// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

// Google Test
#include <gtest/gtest.h>

#include "test_utils.hpp"

#include <thrust/transform_scan.h>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "test_assertions.hpp"
#include "test_utils.hpp"

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<class InputType> struct Params
{
    using input_type = InputType;
};

template<class Params> class TransformScanTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<thrust::host_vector<short>>,
        Params<thrust::host_vector<int>>,
        Params<thrust::host_vector<long long>>,
        Params<thrust::host_vector<unsigned short>>,
        Params<thrust::host_vector<unsigned int>>,
        Params<thrust::host_vector<unsigned long long>>,
        Params<thrust::host_vector<float>>,
        Params<thrust::host_vector<double>>,
        Params<thrust::device_vector<short>>,
        Params<thrust::device_vector<int>>,
        Params<thrust::device_vector<long long>>,
        Params<thrust::device_vector<unsigned short>>,
        Params<thrust::device_vector<unsigned int>>,
        Params<thrust::device_vector<unsigned long long>>,
        Params<thrust::device_vector<float>>,
        Params<thrust::device_vector<double>>
> TestParams;

TYPED_TEST_CASE(TransformScanTests, TestParams);

template<class Params> class TransformScanVariablesTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<char>,
        Params<unsigned char>,
        Params<short>,
        Params<unsigned short>,
        Params<int>,
        Params<unsigned int>,
        Params<float>
> TestVariableParams;

TYPED_TEST_CASE(TransformScanVariablesTests, TestVariableParams);

template<class Params> class TransformScanVectorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<thrust::device_vector<short>>,
        Params<thrust::device_vector<int>>,
        Params<thrust::host_vector<short>>,
        Params<thrust::host_vector<int>>
> VectorParams;

TYPED_TEST_CASE(TransformScanVectorTests, VectorParams);

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
__host__ __device__
OutputIterator transform_inclusive_scan(my_system &system,
                                        InputIterator,
                                        InputIterator,
                                        OutputIterator result,
                                        UnaryFunction,
                                        AssociativeOperator)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformScanTests, TestTransformInclusiveScanDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform_inclusive_scan(sys,
                                     vec.begin(),
                                     vec.begin(),
                                     vec.begin(),
                                     0,
                                     0);

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename AssociativeOperator>
__host__ __device__
OutputIterator transform_inclusive_scan(my_tag,
                                        InputIterator,
                                        InputIterator,
                                        OutputIterator result,
                                        UnaryFunction,
                                        AssociativeOperator)
{
    *result = 13;
    return result;
}

TEST(TransformScanTests, TestTransformInclusiveScanDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform_inclusive_scan(thrust::retag<my_tag>(vec.begin()),
                                     thrust::retag<my_tag>(vec.begin()),
                                     thrust::retag<my_tag>(vec.begin()),
                                     0,
                                     0);

    ASSERT_EQ(13, vec.front());
}

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
__host__ __device__
OutputIterator transform_exclusive_scan(my_system &system,
                                        InputIterator,
                                        InputIterator,
                                        OutputIterator result,
                                        UnaryFunction,
                                        T,
                                        AssociativeOperator)
{
    system.validate_dispatch();
    return result;
}

TEST(TransformScanTests, TestTransformExclusiveScanDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::transform_exclusive_scan(sys,
                                     vec.begin(),
                                     vec.begin(),
                                     vec.begin(),
                                     0,
                                     0,
                                     0);

    ASSERT_EQ(true, sys.is_valid());
}


template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction,
         typename T,
         typename AssociativeOperator>
__host__ __device__
OutputIterator transform_exclusive_scan(my_tag,
                                        InputIterator,
                                        InputIterator,
                                        OutputIterator result,
                                        UnaryFunction,
                                        T,
                                        AssociativeOperator)
{
    *result = 13;
    return result;
}

TEST(TransformScanTests, TestTransformExclusiveScanDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::transform_exclusive_scan(thrust::retag<my_tag>(vec.begin()),
                                     thrust::retag<my_tag>(vec.begin()),
                                     thrust::retag<my_tag>(vec.begin()),
                                     0,
                                     0,
                                     0);

    ASSERT_EQ(13, vec.front());
}


TYPED_TEST(TransformScanVectorTests, TestTransformScanSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typename Vector::iterator iter;

    Vector input(5);
    Vector result(5);
    Vector output(5);

    input[0] = 1; input[1] = 3; input[2] = -2; input[3] = 4; input[4] = -5;

    Vector input_copy(input);

    // inclusive scan
    iter = thrust::transform_inclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>());
    result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input,  input_copy);
    ASSERT_EQ(output, result);

    // exclusive scan with 0 init
    iter = thrust::transform_exclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>());
    result[0] = 0; result[1] = -1; result[2] = -4; result[3] = -2; result[4] = -6;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input,  input_copy);
    ASSERT_EQ(output, result);

    // exclusive scan with nonzero init
    iter = thrust::transform_exclusive_scan(input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
    ASSERT_EQ(iter - output.begin(), input.size());
    ASSERT_EQ(input,  input_copy);
    ASSERT_EQ(output, result);

    // inplace inclusive scan
    input = input_copy;
    iter = thrust::transform_inclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>());
    result[0] = -1; result[1] = -4; result[2] = -2; result[3] = -6; result[4] = -1;
    ASSERT_EQ(iter - input.begin(), input.size());
    ASSERT_EQ(input, result);

    // inplace exclusive scan with init
    input = input_copy;
    iter = thrust::transform_exclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
    result[0] = 3; result[1] = 2; result[2] = -1; result[3] = 1; result[4] = -3;
    ASSERT_EQ(iter - input.begin(), input.size());
    ASSERT_EQ(input, result);
}

TYPED_TEST(TransformScanVariablesTests, TestTransformScan)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_input = get_random_data<T>(size,
                                                            std::numeric_limits<T>::min(),
                                                            std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<T>   h_output(size);
        thrust::device_vector<T> d_output(size);

        thrust::transform_inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        thrust::transform_inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        ASSERT_EQ(d_output, h_output);

        thrust::transform_exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        thrust::transform_exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        ASSERT_EQ(d_output, h_output);

        // in-place scans
        h_output = h_input;
        d_output = d_input;
        thrust::transform_inclusive_scan(h_output.begin(), h_output.end(), h_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        thrust::transform_inclusive_scan(d_output.begin(), d_output.end(), d_output.begin(), thrust::negate<T>(), thrust::plus<T>());
        ASSERT_EQ(d_output, h_output);

        h_output = h_input;
        d_output = d_input;
        thrust::transform_exclusive_scan(h_output.begin(), h_output.end(), h_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        thrust::transform_exclusive_scan(d_output.begin(), d_output.end(), d_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
        ASSERT_EQ(d_output, h_output);
    }
};

TYPED_TEST(TransformScanVectorTests, TestTransformScanCountingIterator)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typedef typename thrust::iterator_system<typename Vector::iterator>::type space;

    thrust::counting_iterator<T, space> first(1);

    Vector result(3);

    thrust::transform_inclusive_scan(first, first + 3, result.begin(), thrust::negate<T>(), thrust::plus<T>());

    ASSERT_EQ(result[0], -1);
    ASSERT_EQ(result[1], -3);
    ASSERT_EQ(result[2], -6);
}

TYPED_TEST(TransformScanVariablesTests, TestTransformScanToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_input = get_random_data<T>(size,
                                                            std::numeric_limits<T>::min(),
                                                            std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> reference(size);

        thrust::discard_iterator<> h_result =
          thrust::transform_inclusive_scan(h_input.begin(),
                                           h_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           thrust::plus<T>());

        thrust::discard_iterator<> d_result =
          thrust::transform_inclusive_scan(d_input.begin(),
                                           d_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           thrust::plus<T>());
        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);

        h_result =
          thrust::transform_exclusive_scan(h_input.begin(),
                                           h_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           (T) 11,
                                           thrust::plus<T>());

        d_result =
          thrust::transform_exclusive_scan(d_input.begin(),
                                           d_input.end(),
                                           thrust::make_discard_iterator(),
                                           thrust::negate<T>(),
                                           (T) 11,
                                           thrust::plus<T>());

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
