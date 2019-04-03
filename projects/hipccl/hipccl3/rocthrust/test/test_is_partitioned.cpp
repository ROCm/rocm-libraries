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

#include <thrust/partition.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>

#include "test_header.hpp"
TESTS_DEFINE(IsPartitionedTests, FullTestsParams);
TESTS_DEFINE(IsPartitionedVectorTests, VectorSignedIntegerTestsParams);

template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};

TYPED_TEST(IsPartitionedVectorTests, TestIsPartitionedSimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector v(4);
  v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 0;

  // empty partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin(), thrust::identity<T>()));

  // one element true partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 1, thrust::identity<T>()));

  // just true partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 2, thrust::identity<T>()));

  // both true & false partitions
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.end(), thrust::identity<T>()));

  // one element false partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin() + 3, v.end(), thrust::identity<T>()));

  v[0] = 1; v[1] = 0; v[2] = 1; v[3] = 1;

  // not partitioned
  ASSERT_EQ_QUIET(false, thrust::is_partitioned(v.begin(), v.end(), thrust::identity<T>()));
}

TYPED_TEST(IsPartitionedVectorTests, TestIsPartitioned)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  const size_t n = (1 << 16) + 13;

  Vector v = get_random_data<T>(n,
                                std::numeric_limits<T>::min(),
                                std::numeric_limits<T>::max());

  v[0] = 1;
  v[1] = 0;

  ASSERT_EQ(false, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));

  thrust::partition(v.begin(), v.end(), is_even<T>());

  ASSERT_EQ(true, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));
}


template<typename InputIterator, typename Predicate>
__host__ __device__
bool is_partitioned(my_system &system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TYPED_TEST(IsPartitionedTests, TestIsPartitionedDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_partitioned(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}


template<typename InputIterator, typename Predicate>
__host__ __device__
bool is_partitioned(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TYPED_TEST(IsPartitionedTests, TestIsPartitionedDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::is_partitioned(thrust::retag<my_tag>(vec.begin()),
                         thrust::retag<my_tag>(vec.end()),
                         0);

  ASSERT_EQ(13, vec.front());
}

