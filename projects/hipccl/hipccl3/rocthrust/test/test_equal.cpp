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

#include "test_header.hpp"

// Thrust
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

TESTS_DEFINE(EqualTests, FullTestsParams)

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(EqualTests, TestEqualSimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector v1(5);
  Vector v2(5);
  v1[0] = T(5); v1[1] = T(2); v1[2] = T(0); v1[3] = T(0); v1[4] = T(0);
  v2[0] = T(5); v2[1] = T(2); v2[2] = T(0); v2[3] = T(6); v2[4] = T(1);

  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v1.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin()), false);
  ASSERT_EQ(thrust::equal(v2.begin(), v2.end(), v2.begin()), true);

  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 0, v1.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 1, v1.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 3, v2.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 4, v2.begin()), false);

  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::less_equal<T>()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::greater<T>()),    false);
}

TYPED_TEST(EqualTests, TestEqual)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<T> h_data1 = get_random_data<T>(size,
                                                        std::numeric_limits<T>::min(),
                                                        std::numeric_limits<T>::max());
    thrust::host_vector<T> h_data2 = get_random_data<T>(size,
                                                        std::numeric_limits<T>::min(),
                                                        std::numeric_limits<T>::max());

    thrust::device_vector<T> d_data1 = h_data1;
    thrust::device_vector<T> d_data2 = h_data2;

    //empty ranges
    ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin(), h_data1.begin()), true);
    ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin(), d_data1.begin()), true);

    //symmetric cases
    ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.end(), h_data1.begin()), true);
    ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.end(), d_data1.begin()), true);

    if (size > 0)
    {
        h_data1[0] = 0; h_data2[0] = 1;
        d_data1[0] = 0; d_data2[0] = 1;

        //different vectors
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.end(), h_data2.begin()), false);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.end(), d_data2.begin()), false);

        //different predicates
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::less<T>()), true);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::less<T>()), true);
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::greater<T>()), false);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::greater<T>()), false);
    }
  }
}

template<typename InputIterator1, typename InputIterator2>
bool equal(my_system &system, InputIterator1, InputIterator1, InputIterator2)
{
    system.validate_dispatch();
    return false;
}

TEST(EqualTests, TestEqualDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::equal(sys,
                vec.begin(),
                vec.end(),
                vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator1, typename InputIterator2>
bool equal(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
    *first = 13;
    return false;
}

TEST(EqualTests, TestEqualDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::equal(thrust::retag<my_tag>(vec.begin()),
                thrust::retag<my_tag>(vec.end()),
                thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
