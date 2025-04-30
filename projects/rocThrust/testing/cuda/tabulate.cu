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

#include <unittest/unittest.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


#ifdef THRUST_TEST_DEVICE_SIDE
template<typename ExecutionPolicy, typename Iterator, typename Function>
__global__
void tabulate_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f)
{
  thrust::tabulate(exec, first, last, f);
}


template<typename ExecutionPolicy>
void TestTabulateDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using namespace thrust::placeholders;
  using T = typename Vector::value_type;

  Vector v(5);

  tabulate_kernel<<<1,1>>>(exec, v.begin(), v.end(), thrust::identity<T>());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  tabulate_kernel<<<1,1>>>(exec, v.begin(), v.end(), -_1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(v[0],  0);
  ASSERT_EQUAL(v[1], -1);
  ASSERT_EQUAL(v[2], -2);
  ASSERT_EQUAL(v[3], -3);
  ASSERT_EQUAL(v[4], -4);
  
  tabulate_kernel<<<1,1>>>(exec, v.begin(), v.end(), _1 * _1 * _1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 8);
  ASSERT_EQUAL(v[3], 27);
  ASSERT_EQUAL(v[4], 64);
}

void TestTabulateDeviceSeq()
{
  TestTabulateDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTabulateDeviceSeq);

void TestTabulateDeviceDevice()
{
  TestTabulateDevice(thrust::device);
}
DECLARE_UNITTEST(TestTabulateDeviceDevice);
#endif

void TestTabulateCudaStreams()
{
  using namespace thrust::placeholders;
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector v(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), thrust::identity<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), -_1);
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0],  0);
  ASSERT_EQUAL(v[1], -1);
  ASSERT_EQUAL(v[2], -2);
  ASSERT_EQUAL(v[3], -3);
  ASSERT_EQUAL(v[4], -4);
  
  thrust::tabulate(thrust::cuda::par.on(s), v.begin(), v.end(), _1 * _1 * _1);
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 8);
  ASSERT_EQUAL(v[3], 27);
  ASSERT_EQUAL(v[4], 64);

  cudaStreamSynchronize(s);
}
DECLARE_UNITTEST(TestTabulateCudaStreams);

