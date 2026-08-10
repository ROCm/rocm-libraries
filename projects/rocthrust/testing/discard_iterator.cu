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

#include <thrust/iterator/discard_iterator.h>

#include <unittest/unittest.h>

void TestDiscardIteratorIncrement(void)
{
  thrust::discard_iterator<> lhs(0);
  thrust::discard_iterator<> rhs(0);

  ASSERT_EQUAL(0, lhs - rhs);

  lhs++;

  ASSERT_EQUAL(1, lhs - rhs);

  lhs++;
  lhs++;

  ASSERT_EQUAL(3, lhs - rhs);

  lhs += 5;

  ASSERT_EQUAL(8, lhs - rhs);

  lhs -= 10;

  ASSERT_EQUAL(-2, lhs - rhs);
}
DECLARE_UNITTEST(TestDiscardIteratorIncrement);
static_assert(std::is_trivially_copy_constructible<thrust::discard_iterator<>>::value, "");
static_assert(std::is_trivially_copyable<thrust::discard_iterator<>>::value, "");

void TestDiscardIteratorComparison(void)
{
  thrust::discard_iterator<> iter1(0);
  thrust::discard_iterator<> iter2(0);

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);

  iter1++;

  ASSERT_EQUAL(1, iter1 - iter2);
  ASSERT_EQUAL(false, iter1 == iter2);

  iter2++;

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);
}
DECLARE_UNITTEST(TestDiscardIteratorComparison);

void TestMakeDiscardIterator(void)
{
  thrust::discard_iterator<> iter0 = thrust::make_discard_iterator(13);

  *iter0 = 7;

  thrust::discard_iterator<> iter1 = thrust::make_discard_iterator(7);

  *iter1 = 13;

  ASSERT_EQUAL(6, iter0 - iter1);
}
DECLARE_UNITTEST(TestMakeDiscardIterator);

void TestZippedDiscardIterator(void)
{
  using namespace thrust;

  using IteratorTuple1 = tuple<discard_iterator<>>;
  using ZipIterator1   = zip_iterator<IteratorTuple1>;

  IteratorTuple1 t = thrust::make_tuple(thrust::make_discard_iterator());

  ZipIterator1 z_iter1_first = thrust::make_zip_iterator(t);
  ZipIterator1 z_iter1_last  = z_iter1_first + 10;
  for (; z_iter1_first != z_iter1_last; ++z_iter1_first)
  {
    ;
  }

  ASSERT_EQUAL(10, thrust::get<0>(z_iter1_first.get_iterator_tuple()) - thrust::make_discard_iterator());

  using IteratorTuple2 = tuple<int*, discard_iterator<>>;
  using ZipIterator2   = zip_iterator<IteratorTuple2>;

  ZipIterator2 z_iter_first = thrust::make_zip_iterator(thrust::make_tuple((int*) 0, thrust::make_discard_iterator()));
  ZipIterator2 z_iter_last  = z_iter_first + 10;

  for (; z_iter_first != z_iter_last; ++z_iter_first)
  {
    ;
  }

  ASSERT_EQUAL(10, thrust::get<1>(z_iter_first.get_iterator_tuple()) - thrust::make_discard_iterator());
}
DECLARE_UNITTEST(TestZippedDiscardIterator);
