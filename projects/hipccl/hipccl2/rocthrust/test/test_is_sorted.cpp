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
#include <thrust/sort.h>
#include <thrust/iterator/retag.h>

#include "test_header.hpp"

TESTS_DEFINE(IsSortedTests, FullTestsParams);
TESTS_DEFINE(IsSortedVectorTests, VectorSignedIntegerTestsParams);

TYPED_TEST(IsSortedVectorTests, TestIsSortedSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v(4);
    v[0] = 0; v[1] = 5; v[2] = 8; v[3] = 0;

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 0), true);
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 1), true);

    // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
    // do nothing
#else
    // compile this line on other compilers
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 2), true);
#endif // GCC

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 3), true);
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 4), false);

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 3, thrust::less<T>()),    true);

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 1, thrust::greater<T>()), true);
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 4, thrust::greater<T>()), false);

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.end()), false);
}

TYPED_TEST(IsSortedVectorTests, TestIsSortedRepeatedElements)
{
  using Vector = typename TestFixture::input_type;
  Vector v(10);

  v[0] = 0;
  v[1] = 1;
  v[2] = 1;
  v[3] = 2;
  v[4] = 3;
  v[5] = 4;
  v[6] = 5;
  v[7] = 5;
  v[8] = 5;
  v[9] = 6;

  ASSERT_EQ(true, thrust::is_sorted(v.begin(), v.end()));
}


TYPED_TEST(IsSortedVectorTests, TestIsSorted)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    const size_t n = (1 << 16) + 13;

    Vector v = get_random_data<T>(n,
                                  std::numeric_limits<T>::min(),
                                  std::numeric_limits<T>::max());

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.end()), false);

    thrust::sort(v.begin(), v.end());

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.end()), true);
}

template<typename InputIterator>
bool is_sorted(my_system &system, InputIterator, InputIterator)
{
  system.validate_dispatch();
  return false;
}

TEST(IsSortedTests, TestIsSortedDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_sorted(sys,
                    vec.begin(),
                    vec.end());

  ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator>
bool is_sorted(my_tag, InputIterator first, InputIterator)
{
  *first = 13;
  return false;
}

TEST(IsSortedTests, TestIsSortedDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::is_sorted(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()));

  ASSERT_EQ(13, vec.front());
}
