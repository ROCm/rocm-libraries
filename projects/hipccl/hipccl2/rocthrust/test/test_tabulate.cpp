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

TESTS_DEFINE(TabulateTests, FullTestsParams);

template<typename ForwardIterator, typename UnaryOperation>
void tabulate(my_system &system, ForwardIterator, ForwardIterator, UnaryOperation)
{
  system.validate_dispatch();
}

TEST(TabulateTests, TestTabulateDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::tabulate(sys, vec.begin(), vec.end(), thrust::identity<int>());

  ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator, typename UnaryOperation>
void tabulate(my_tag, ForwardIterator first, ForwardIterator, UnaryOperation)
{
  *first = 13;
}

TEST(TabulateTests, TestTabulateDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::tabulate(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   thrust::identity<int>());

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TabulateTests, TestTabulateSimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using namespace thrust::placeholders;

  Vector v(5);

  thrust::tabulate(v.begin(), v.end(), thrust::identity<T>());

  ASSERT_EQ(v[0], T(0));
  ASSERT_EQ(v[1], T(1));
  ASSERT_EQ(v[2], T(2));
  ASSERT_EQ(v[3], T(3));
  ASSERT_EQ(v[4], T(4));

  thrust::tabulate(v.begin(), v.end(), -_1);

  ASSERT_EQ(v[0], T( 0));
  ASSERT_EQ(v[1], T(-1));
  ASSERT_EQ(v[2], T(-2));
  ASSERT_EQ(v[3], T(-3));
  ASSERT_EQ(v[4], T(-4));

  thrust::tabulate(v.begin(), v.end(), _1 * _1 * _1);

  ASSERT_EQ(v[0], T(0));
  ASSERT_EQ(v[1], T(1));
  ASSERT_EQ(v[2], T(8));
  ASSERT_EQ(v[3], T(27));
  ASSERT_EQ(v[4], T(64));
}

TYPED_TEST(TabulateTests, TestTabulate)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using namespace thrust::placeholders;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T>   h_data(size);
    thrust::device_vector<T> d_data(size);

    thrust::tabulate(h_data.begin(), h_data.end(), _1 * _1 + T(13));
    thrust::tabulate(d_data.begin(), d_data.end(), _1 * _1 + T(13));

    thrust::host_vector<T> h_result = d_data;
    for (size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_data[i], h_result[i]) << "where index = " << i;
    }

    thrust::tabulate(h_data.begin(), h_data.end(), (_1 - T(7)) * _1);
    thrust::tabulate(d_data.begin(), d_data.end(), (_1 - T(7)) * _1);

    ASSERT_EQ(h_data, d_data);
  }
}

TEST(TabulateTests, TestTabulateToDiscardIterator)
{
  for (auto size : get_sizes())
  {
    thrust::tabulate(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(size),
                     thrust::identity<int>());
  }
  // nothing to check -- just make sure it compiles
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
