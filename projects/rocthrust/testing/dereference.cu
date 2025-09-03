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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <unittest/unittest.h>

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename Iterator1, typename Iterator2>
__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void
simple_copy_on_device(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  while (first1 != last1)
  {
    *(first2++) = *(first1++);
  }
}

template <typename Iterator1, typename Iterator2>
void simple_copy(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  simple_copy_on_device<<<1, 1>>>(first1, last1, first2);
}

void TestDeviceDereferenceDeviceVectorIterator()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<int> output(input.size(), 0);

  simple_copy(input.begin(), input.end(), output.begin());

  ASSERT_EQUAL(input, output);
}
DECLARE_UNITTEST(TestDeviceDereferenceDeviceVectorIterator);

void TestDeviceDereferenceDevicePtr()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<int> output(input.size(), 0);

  thrust::device_ptr<int> _first1 = &input[0];
  thrust::device_ptr<int> _last1  = _first1 + input.size();
  thrust::device_ptr<int> _first2 = &output[0];

  simple_copy(_first1, _last1, _first2);

  ASSERT_EQUAL(input, output);
}
DECLARE_UNITTEST(TestDeviceDereferenceDevicePtr);

void TestDeviceDereferenceTransformIterator()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<int> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), ::internal::identity{}),
              thrust::make_transform_iterator(input.end(), ::internal::identity{}),
              output.begin());

  ASSERT_EQUAL(input, output);
}
DECLARE_UNITTEST(TestDeviceDereferenceTransformIterator);

void TestDeviceDereferenceTransformIteratorInputConversion()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<double> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), ::internal::identity{}),
              thrust::make_transform_iterator(input.end(), ::internal::identity{}),
              output.begin());

  ASSERT_EQUAL(input == output, true);
}
DECLARE_UNITTEST(TestDeviceDereferenceTransformIteratorInputConversion);

void TestDeviceDereferenceTransformIteratorOutputConversion()
{
  thrust::device_vector<int> input = unittest::random_integers<int>(100);
  thrust::device_vector<double> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), ::internal::identity{}),
              thrust::make_transform_iterator(input.end(), ::internal::identity{}),
              output.begin());

  ASSERT_EQUAL(input == output, true);
}
DECLARE_UNITTEST(TestDeviceDereferenceTransformIteratorOutputConversion);

void TestDeviceDereferenceCountingIterator()
{
  thrust::counting_iterator<int> first(1);
  thrust::counting_iterator<int> last(6);

  thrust::device_vector<int> output(5);

  simple_copy(first, last, output.begin());

  thrust::device_vector<int> ref{1, 2, 3, 4, 5};
  ASSERT_EQUAL(output, ref);
}
DECLARE_UNITTEST(TestDeviceDereferenceCountingIterator);

void TestDeviceDereferenceTransformedCountingIterator()
{
  thrust::counting_iterator<int> first(1);
  thrust::counting_iterator<int> last(6);

  thrust::device_vector<int> output(5);

  simple_copy(thrust::make_transform_iterator(first, thrust::negate<int>()),
              thrust::make_transform_iterator(last, thrust::negate<int>()),
              output.begin());

  thrust::device_vector<int> ref{-1, -2, -3, -4, -5};
  ASSERT_EQUAL(output, ref);
}
DECLARE_UNITTEST(TestDeviceDereferenceTransformedCountingIterator);

THRUST_DIAG_POP
