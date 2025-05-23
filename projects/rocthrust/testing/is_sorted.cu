/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <class Vector>
void TestIsSortedSimple(void)
{
  using T = typename Vector::value_type;

  Vector v(4);
  v[0] = 0;
  v[1] = 5;
  v[2] = 8;
  v[3] = 0;

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 0), true);
  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 1), true);

  // the following line crashes gcc 4.3
#if (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
  // do nothing
#else
  // compile this line on other compilers
  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 2), true);
#endif // GCC

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 3), true);
  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 4), false);

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 3, thrust::less<T>()), true);

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 1, thrust::greater<T>()), true);
  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.begin() + 4, thrust::greater<T>()), false);

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.end()), false);
}
DECLARE_VECTOR_UNITTEST(TestIsSortedSimple);

template <class Vector>
void TestIsSortedRepeatedElements(void)
{
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

  ASSERT_EQUAL(true, thrust::is_sorted(v.begin(), v.end()));
}
DECLARE_VECTOR_UNITTEST(TestIsSortedRepeatedElements);

template <class Vector>
void TestIsSorted(void)
{
  using T = typename Vector::value_type;

  const size_t n = (1 << 16) + 13;

  Vector v = unittest::random_integers<T>(n);

  v[0] = 1;
  v[1] = 0;

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.end()), false);

  thrust::sort(v.begin(), v.end());

  ASSERT_EQUAL(thrust::is_sorted(v.begin(), v.end()), true);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestIsSorted);

template <typename InputIterator>
bool is_sorted(my_system& system, InputIterator /*first*/, InputIterator)
{
  system.validate_dispatch();
  return false;
}

void TestIsSortedDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_sorted(sys, vec.begin(), vec.end());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestIsSortedDispatchExplicit);

template <typename InputIterator>
bool is_sorted(my_tag, InputIterator first, InputIterator)
{
  *first = 13;
  return false;
}

void TestIsSortedDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::is_sorted(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestIsSortedDispatchImplicit);
