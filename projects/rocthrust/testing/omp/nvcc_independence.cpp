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
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <thrust/transform.h>

#include <unittest/unittest.h>

void TestNvccIndependenceTransform(void)
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>());
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>());

  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_UNITTEST(TestNvccIndependenceTransform);

void TestNvccIndependenceReduce(void)
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;

  T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
  T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

  ASSERT_ALMOST_EQUAL(h_result, d_result);
}
DECLARE_UNITTEST(TestNvccIndependenceReduce);

void TestNvccIndependenceExclusiveScan(void)
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
  ASSERT_EQUAL(d_output, h_output);
}
DECLARE_UNITTEST(TestNvccIndependenceExclusiveScan);

void TestNvccIndependenceSort(void)
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end(), thrust::less<T>());
  thrust::sort(d_data.begin(), d_data.end(), thrust::less<T>());

  ASSERT_EQUAL(h_data, d_data);
}
DECLARE_UNITTEST(TestNvccIndependenceSort);
