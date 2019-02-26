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

// Thrust
#include <thrust/tuple.h>
#include <thrust/transform.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"
#include "test_assertions.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class TupleTransformTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<short>,
    Params<int>,
    Params<long long>
> TupleTransformTestsParams;

TYPED_TEST_CASE(TupleTransformTests, TupleTransformTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

struct MakeTupleFunctor
{
  template<typename T1, typename T2>
  __host__ __device__
  thrust::tuple<T1,T2> operator()(T1 &lhs, T2 &rhs)
  {
    return thrust::make_tuple(lhs, rhs);
  }
};

template<int N>
struct GetFunctor
{
  template<typename Tuple>
  __host__ __device__
  typename thrust::access_traits<
    typename thrust::tuple_element<N, Tuple>::type
  >::const_type
  operator()(const Tuple &t)
  {
    return thrust::get<N>(t);
  }
};

TYPED_TEST(TupleTransformTests, TestTupleTransform)
{
  using T = typename TestFixture::input_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<T> h_t1 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

    thrust::host_vector<T> h_t2 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

    // zip up the data
    thrust::host_vector< thrust::tuple<T,T> > h_tuples(size);
    thrust::transform(h_t1.begin(), h_t1.end(),
                      h_t2.begin(), h_tuples.begin(),
                      MakeTupleFunctor());

    // copy to device
    thrust::device_vector< thrust::tuple<T,T> > d_tuples = h_tuples;

    thrust::device_vector<T> d_t1(size), d_t2(size);

    // select 0th
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_t1.begin(), GetFunctor<0>());

    // select 1st
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_t2.begin(), GetFunctor<1>());

    ASSERT_EQ(h_t1, d_t1);
    ASSERT_EQ(h_t2, d_t2);

    ASSERT_EQ_QUIET(h_tuples, d_tuples);
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
