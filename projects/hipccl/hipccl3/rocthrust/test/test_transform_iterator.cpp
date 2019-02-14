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

#include <vector>
#include <list>
#include <limits>
#include <utility>

// Google Test
#include <gtest/gtest.h>
#include "test_utils.hpp"

// Thrust

#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

template< class InputType >
struct Params
{
    using input_type = InputType;
};

template<class Params>
class TransformIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

template<class Params>
class PrimitiveTransformIteratorTests : public ::testing::Test
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
> TransformIteratorTestsParams;

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> TransformIteratorTestsPrimitiveParams;

TYPED_TEST_CASE(TransformIteratorTests, TransformIteratorTestsParams);
TYPED_TEST_CASE(PrimitiveTransformIteratorTests, TransformIteratorTestsPrimitiveParams);

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
TEST(TransformIteratorTests, UsingHip)
{
  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(TransformIteratorTests, TransformIterator)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typedef thrust::negate<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);

    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);

    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(iter, iter + 4, output.begin());

    ASSERT_EQ(output[0], (T)-1);
    ASSERT_EQ(output[1], (T)-2);
    ASSERT_EQ(output[2], (T)-3);
    ASSERT_EQ(output[3], (T)-4);

}

TYPED_TEST(TransformIteratorTests, MakeTransformIterator)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typedef thrust::negate<T> UnaryFunction;
    typedef typename Vector::iterator Iterator;

    Vector input(4);
    Vector output(4);

    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);

    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(thrust::make_transform_iterator(input.begin(), UnaryFunction()),
                 thrust::make_transform_iterator(input.end(), UnaryFunction()),
                 output.begin());

    ASSERT_EQ(output[0], (T)-1);
    ASSERT_EQ(output[1], (T)-2);
    ASSERT_EQ(output[2], (T)-3);
    ASSERT_EQ(output[3], (T)-4);

}

TYPED_TEST(PrimitiveTransformIteratorTests, TransformIteratorReduce)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        T error_margin = (T) 0.01 * size;
        thrust::host_vector<T>   h_data = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data = h_data;

        // run on host
        T h_result = thrust::reduce( thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                                     thrust::make_transform_iterator(h_data.end(),   thrust::negate<T>()) );

        // run on device
        T d_result = thrust::reduce( thrust::make_transform_iterator(d_data.begin(), thrust::negate<T>()),
                                     thrust::make_transform_iterator(d_data.end(),   thrust::negate<T>()) );

        ASSERT_NEAR(h_result, d_result, error_margin);

    }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
