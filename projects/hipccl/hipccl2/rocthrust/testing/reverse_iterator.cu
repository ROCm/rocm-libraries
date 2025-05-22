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

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <unittest/unittest.h>

void TestReverseIteratorCopyConstructor(void)
{
  thrust::host_vector<int> h_v(1, 13);

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter0(h_v.end());
  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter1(h_iter0);

  ASSERT_EQUAL_QUIET(h_iter0, h_iter1);
  ASSERT_EQUAL(*h_iter0, *h_iter1);

  thrust::device_vector<int> d_v(1, 13);

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter2(d_v.end());
  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter3(d_iter2);

  ASSERT_EQUAL_QUIET(d_iter2, d_iter3);
  ASSERT_EQUAL(*d_iter2, *d_iter3);
}
DECLARE_UNITTEST(TestReverseIteratorCopyConstructor);
static_assert(std::is_trivially_copy_constructible<thrust::reverse_iterator<int*>>::value, "");
static_assert(std::is_trivially_copyable<thrust::reverse_iterator<int*>>::value, "");

void TestReverseIteratorIncrement(void)
{
  thrust::host_vector<int> h_v(4);
  thrust::sequence(h_v.begin(), h_v.end());

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter(h_v.end());

  ASSERT_EQUAL(*h_iter, 3);

  h_iter++;
  ASSERT_EQUAL(*h_iter, 2);

  h_iter++;
  ASSERT_EQUAL(*h_iter, 1);

  h_iter++;
  ASSERT_EQUAL(*h_iter, 0);

  thrust::device_vector<int> d_v(4);
  thrust::sequence(d_v.begin(), d_v.end());

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter(d_v.end());

  ASSERT_EQUAL(*d_iter, 3);

  d_iter++;
  ASSERT_EQUAL(*d_iter, 2);

  d_iter++;
  ASSERT_EQUAL(*d_iter, 1);

  d_iter++;
  ASSERT_EQUAL(*d_iter, 0);
}
DECLARE_UNITTEST(TestReverseIteratorIncrement);

template <typename Vector>
void TestReverseIteratorCopy(void)
{
  Vector source(4);
  source[0] = 10;
  source[1] = 20;
  source[2] = 30;
  source[3] = 40;

  Vector destination(4, 0);

  thrust::copy(
    thrust::make_reverse_iterator(source.end()), thrust::make_reverse_iterator(source.begin()), destination.begin());

  ASSERT_EQUAL(destination[0], 40);
  ASSERT_EQUAL(destination[1], 30);
  ASSERT_EQUAL(destination[2], 20);
  ASSERT_EQUAL(destination[3], 10);
}
DECLARE_VECTOR_UNITTEST(TestReverseIteratorCopy);

void TestReverseIteratorExclusiveScanSimple(void)
{
  using T        = int;
  const size_t n = 10;

  thrust::host_vector<T> h_data(n);
  thrust::sequence(h_data.begin(), h_data.end());

  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_result(h_data.size());
  thrust::device_vector<T> d_result(d_data.size());

  thrust::exclusive_scan(
    thrust::make_reverse_iterator(h_data.end()), thrust::make_reverse_iterator(h_data.begin()), h_result.begin());

  thrust::exclusive_scan(
    thrust::make_reverse_iterator(d_data.end()), thrust::make_reverse_iterator(d_data.begin()), d_result.begin());

  ASSERT_EQUAL_QUIET(h_result, d_result);
}
DECLARE_UNITTEST(TestReverseIteratorExclusiveScanSimple);

template <typename T>
struct TestReverseIteratorExclusiveScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data = unittest::random_samples<T>(n);

    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    thrust::exclusive_scan(
      thrust::make_reverse_iterator(h_data.end()), thrust::make_reverse_iterator(h_data.begin()), h_result.begin());

    thrust::exclusive_scan(
      thrust::make_reverse_iterator(d_data.end()), thrust::make_reverse_iterator(d_data.begin()), d_result.begin());

    ASSERT_EQUAL_QUIET(h_result, d_result);
  }
};
VariableUnitTest<TestReverseIteratorExclusiveScan, IntegralTypes> TestReverseIteratorExclusiveScanInstance;
