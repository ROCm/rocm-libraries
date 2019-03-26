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

// Thrust
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(TransformReduceTests, FullTestsParams);
TESTS_DEFINE(TransformReduceIntegerTests, VectorSignedIntegerTestsParams);

template<typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
OutputType transform_reduce(my_system &system,
                            InputIterator,
                            InputIterator,
                            UnaryFunction,
                            OutputType init,
                            BinaryFunction)
{
    system.validate_dispatch();
    return init;
}

TEST(TransformReduceTests, TestTransformReduceDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_reduce(sys,
                           vec.begin(),
                           vec.begin(),
                           0,
                           0,
                           0);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
OutputType transform_reduce(my_tag,
                            InputIterator first,
                            InputIterator,
                            UnaryFunction,
                            OutputType init,
                            BinaryFunction)
{
    *first = 13;
    return init;
}

TEST(TransformReduceTests, TestTransformReduceDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::transform_reduce(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0,
                           0,
                           0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformReduceTests, TestTransformReduceSimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector data(3);
  data[0] = T(1); data[1] = T(-2); data[2] = T(3);

  T init = T(10);
  T result = thrust::transform_reduce(data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>());

  ASSERT_EQ(result, T(8));
}

TYPED_TEST(TransformReduceIntegerTests, TestTransformReduce)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<T> h_data = get_random_data<T>(size,
                                                       std::numeric_limits<T>::min(),
                                                       std::numeric_limits<T>::max());

    thrust::device_vector<T> d_data = h_data;

    T init = T(13);

    T cpu_result = thrust::transform_reduce(h_data.begin(), h_data.end(), thrust::negate<T>(), init, thrust::plus<T>());
    T gpu_result = thrust::transform_reduce(d_data.begin(), d_data.end(), thrust::negate<T>(), init, thrust::plus<T>());

    ASSERT_NEAR(cpu_result, gpu_result, std::abs(T(0.01 * cpu_result)) );
  }
}

TYPED_TEST(TransformReduceIntegerTests, TestTransformReduceFromConst)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<T> h_data = get_random_data<T>(size,
                                                       std::numeric_limits<T>::min(),
                                                       std::numeric_limits<T>::max());

    thrust::device_vector<T> d_data = h_data;

    T init = T(13);

    T cpu_result = thrust::transform_reduce(h_data.cbegin(), h_data.cend(), thrust::negate<T>(), init, thrust::plus<T>());
    T gpu_result = thrust::transform_reduce(d_data.cbegin(), d_data.cend(), thrust::negate<T>(), init, thrust::plus<T>());

    ASSERT_NEAR(cpu_result, gpu_result, std::abs(T(0.01 * cpu_result)) );
  }
}

TYPED_TEST(TransformReduceTests, TestTransformReduceCountingIterator)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using space = typename thrust::iterator_system<typename Vector::iterator>::type;
  if( std::is_signed<T>::value )
  {
    thrust::counting_iterator<T, space> first(1);

    T result = thrust::transform_reduce(first, first + 3, thrust::negate<short>(), 0, thrust::plus<short>());

    ASSERT_EQ(result, -6);
  }
}
