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
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "test_header.hpp"

TESTS_DEFINE(TupleReduceTests, IntegerTestsParams);

struct SumTupleFunctor
{
  template <typename Tuple>
  __host__ __device__
  Tuple operator()(const Tuple &lhs, const Tuple &rhs)
  {
    using thrust::get;

    return thrust::make_tuple(get<0>(lhs) + get<0>(rhs),
                              get<1>(lhs) + get<1>(rhs));
  }
};

struct MakeTupleFunctor
{
  template<typename T1, typename T2>
  __host__ __device__
  thrust::tuple<T1,T2> operator()(T1 &lhs, T2 &rhs)
  {
    return thrust::make_tuple(lhs, rhs);
  }
};

TYPED_TEST(TupleReduceTests, TestTupleReduce)
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
    thrust::transform(h_t1.begin(), h_t1.end(), h_t2.begin(), h_tuples.begin(), MakeTupleFunctor());

    // copy to device
    thrust::device_vector< thrust::tuple<T,T> > d_tuples = h_tuples;

    thrust::tuple<T,T> zero(0,0);

    // sum on host
    thrust::tuple<T,T> h_result = thrust::reduce(h_tuples.begin(), h_tuples.end(), zero, SumTupleFunctor());

    // sum on device
    thrust::tuple<T,T> d_result = thrust::reduce(d_tuples.begin(), d_tuples.end(), zero, SumTupleFunctor());

    ASSERT_EQ_QUIET(h_result, d_result);
  }
}
