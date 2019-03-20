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
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(MismatchTests, FullTestsParams);

TYPED_TEST(MismatchTests, TestMismatchSimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector a(4); Vector b(4);
  a[0] = T(1); b[0] = T(1);
  a[1] = T(2); b[1] = T(2);
  a[2] = T(3); b[2] = T(4);
  a[3] = T(4); b[3] = T(3);

  ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 2);
  ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 2);

  b[2] = T(3);

  ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 3);
  ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 3);

  b[3] = T(4);

  ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).first  - a.begin(), 4);
  ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 4);
}

template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(my_system &system,
                                                      InputIterator1 first,
                                                      InputIterator1,
                                                      InputIterator2)
{
    system.validate_dispatch();
    return thrust::make_pair(first,first);
}

TEST(MismatchTests, TestMismatchDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::mismatch(sys,
                   vec.begin(),
                   vec.begin(),
                   vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2> mismatch(my_tag,
                                                      InputIterator1 first,
                                                      InputIterator1,
                                                      InputIterator2)
{
    *first = 13;
    return thrust::make_pair(first,first);
}

TEST(MismatchTests, TestMismatchDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::mismatch(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
