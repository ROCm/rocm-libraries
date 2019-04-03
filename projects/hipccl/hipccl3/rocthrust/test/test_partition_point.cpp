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
TESTS_DEFINE(PartitionPointTests, FullTestsParams);
TESTS_DEFINE(PartitionPointVectorTests, VectorSignedIntegerTestsParams);

template<typename T>
struct is_even
{
  __host__ __device__
  bool operator()(T x) const { return ((int) x % 2) == 0; }
};

TYPED_TEST(PartitionPointVectorTests, TestPartitionPointSimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector v(4);
  v[0] = 1; v[1] = 1; v[2] = 1; v[3] = 0;

  Iterator first = v.begin();

  Iterator last = v.begin() + 4;
  Iterator ref = first + 3;
  ASSERT_EQ_QUIET(ref, thrust::partition_point(first, last, thrust::identity<T>()));

  last = v.begin() + 3;
  ref = last;
  ASSERT_EQ_QUIET(ref, thrust::partition_point(first, last, thrust::identity<T>()));
}

TYPED_TEST(PartitionPointVectorTests, TestPartitionPoint)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  const size_t n = (1 << 16) + 13;

  Vector v = get_random_data<T>(n,
                                std::numeric_limits<T>::min(),
                                std::numeric_limits<T>::max());

  Iterator ref = thrust::stable_partition(v.begin(), v.end(), is_even<T>());

  ASSERT_EQ(ref - v.begin(), thrust::partition_point(v.begin(), v.end(), is_even<T>()) - v.begin());
}

template<typename ForwardIterator, typename Predicate>
__host__ __device__
ForwardIterator partition_point(my_system &system,
                                ForwardIterator first,
                                ForwardIterator,
                                Predicate)
{
  system.validate_dispatch();
  return first;
}

TYPED_TEST(PartitionPointTests, TestPartitionPointDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::partition_point(sys,
                          vec.begin(),
                          vec.begin(),
                          0);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator, typename Predicate>
__host__ __device__
ForwardIterator partition_point(my_tag,
                                ForwardIterator first,
                                ForwardIterator,
                                Predicate)
{
  *first = 13;
  return first;
}

TYPED_TEST(PartitionPointTests, TestPartitionPointDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::partition_point(thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.begin()),
                          0);

  ASSERT_EQ(13, vec.front());
}

