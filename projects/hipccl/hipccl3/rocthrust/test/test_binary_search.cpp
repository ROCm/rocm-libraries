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

// Thrust
#include <thrust/binary_search.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(BinarySearchTests, FullTestsParams);

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

TYPED_TEST(BinarySearchTests, TestScalarLowerBoundSimple)
{
  using Vector = typename TestFixture::input_type;
  Vector vec(5);

  vec[0] = 0;
  vec[1] = 2;
  vec[2] = 5;
  vec[3] = 7;
  vec[4] = 8;

  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 0) - vec.begin(), 0);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 1) - vec.begin(), 1);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 2) - vec.begin(), 1);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 3) - vec.begin(), 2);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 4) - vec.begin(), 2);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 5) - vec.begin(), 2);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 6) - vec.begin(), 3);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 7) - vec.begin(), 3);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 8) - vec.begin(), 4);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), 9) - vec.begin(), 5);
}

template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(my_system &system, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    system.validate_dispatch();
    return first;
}

TEST(BinarySearchTests, TestScalarLowerBoundDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::lower_bound(sys,
                      vec.begin(),
                      vec.end(),
                      0);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    *first = 13;
    return first;
}

TEST(BinarySearchTests, TestScalarLowerBoundDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::lower_bound(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.end()),
                      0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchTests, TestScalarUpperBoundSimple)
{
  using Vector = typename TestFixture::input_type;
  Vector vec(5);

  vec[0] = 0;
  vec[1] = 2;
  vec[2] = 5;
  vec[3] = 7;
  vec[4] = 8;

  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 0) - vec.begin(), 1);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 1) - vec.begin(), 1);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 2) - vec.begin(), 2);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 3) - vec.begin(), 2);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 4) - vec.begin(), 2);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 5) - vec.begin(), 3);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 6) - vec.begin(), 3);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 7) - vec.begin(), 4);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 8) - vec.begin(), 5);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), 9) - vec.begin(), 5);
}

template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(my_system &system, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    system.validate_dispatch();
    return first;
}

TEST(BinarySearchTests, TestScalarUpperBoundDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::upper_bound(sys,
                      vec.begin(),
                      vec.end(),
                      0);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    *first = 13;
    return first;
}

TEST(BinarySearchTests, TestScalarUpperBoundDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::upper_bound(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.end()),
                      0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchTests, TestScalarBinarySearchSimple)
{
  using Vector = typename TestFixture::input_type;
  Vector vec(5);

  vec[0] = 0;
  vec[1] = 2;
  vec[2] = 5;
  vec[3] = 7;
  vec[4] = 8;

  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 0),  true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 1), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 2),  true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 3), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 4), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 5),  true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 6), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 7),  true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 8),  true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), 9), false);
}

template<typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_system &system, ForwardIterator, ForwardIterator, const LessThanComparable &)
{
    system.validate_dispatch();
    return false;
}

TEST(BinarySearchTests, TestScalarBinarySearchDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::binary_search(sys,
                        vec.begin(),
                        vec.end(),
                        0);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    *first = 13;
    return false;
}

TEST(BinarySearchTests, TestScalarBinarySearchDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::binary_search(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchTests, TestScalarEqualRangeSimple)
{
  using Vector = typename TestFixture::input_type;
  Vector vec(5);

  vec[0] = 0;
  vec[1] = 2;
  vec[2] = 5;
  vec[3] = 7;
  vec[4] = 8;

  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 0).first - vec.begin(), 0);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 1).first - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 2).first - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 3).first - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 4).first - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 5).first - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 6).first - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 7).first - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 8).first - vec.begin(), 4);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 9).first - vec.begin(), 5);

  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 0).second - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 1).second - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 2).second - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 3).second - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 4).second - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 5).second - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 6).second - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 7).second - vec.begin(), 4);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 8).second - vec.begin(), 5);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 9).second - vec.begin(), 5);
}

template<typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator,ForwardIterator> equal_range(my_system &system, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    system.validate_dispatch();
    return thrust::make_pair(first,first);
}

TEST(BinarySearchTests, TestScalarEqualRangeDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::equal_range(sys,
                      vec.begin(),
                      vec.end(),
                      0);

  ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator,ForwardIterator> equal_range(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable &)
{
    *first = 13;
    return thrust::make_pair(first,first);
}

TEST(BinarySearchTests, TestScalarEqualRangeDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::binary_search(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        0);

  ASSERT_EQ(13, vec.front());
}

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
