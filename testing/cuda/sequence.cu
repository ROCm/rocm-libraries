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
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>


#ifdef THRUST_TEST_DEVICE_SIDE
template<typename ExecutionPolicy, typename Iterator>
__global__
void sequence_kernel(ExecutionPolicy exec, Iterator first, Iterator last)
{
  thrust::sequence(exec, first, last);
}


template<typename ExecutionPolicy, typename Iterator, typename T>
__global__
void sequence_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T init)
{
  thrust::sequence(exec, first, last, init);
}


template<typename ExecutionPolicy, typename Iterator, typename T>
__global__
void sequence_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T init, T step)
{
  thrust::sequence(exec, first, last, init, step);
}


template<typename ExecutionPolicy>
void TestSequenceDevice(ExecutionPolicy exec)
{
  thrust::device_vector<int> v(5);
  
  sequence_kernel<<<1,1>>>(exec, v.begin(), v.end());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
 
  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);
  
  sequence_kernel<<<1,1>>>(exec, v.begin(), v.end(), 10);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 11);
  ASSERT_EQUAL(v[2], 12);
  ASSERT_EQUAL(v[3], 13);
  ASSERT_EQUAL(v[4], 14);
  
  sequence_kernel<<<1,1>>>(exec, v.begin(), v.end(), 10, 2);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  
  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 12);
  ASSERT_EQUAL(v[2], 14);
  ASSERT_EQUAL(v[3], 16);
  ASSERT_EQUAL(v[4], 18);
}

void TestSequenceDeviceSeq()
{
  TestSequenceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSequenceDeviceSeq);

void TestSequenceDeviceDevice()
{
  TestSequenceDevice(thrust::device);
}
DECLARE_UNITTEST(TestSequenceDeviceDevice);
#endif

void TestSequenceCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector v(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::sequence(thrust::cuda::par.on(s), v.begin(), v.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 0);
  ASSERT_EQUAL(v[1], 1);
  ASSERT_EQUAL(v[2], 2);
  ASSERT_EQUAL(v[3], 3);
  ASSERT_EQUAL(v[4], 4);

  thrust::sequence(thrust::cuda::par.on(s), v.begin(), v.end(), 10);
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 11);
  ASSERT_EQUAL(v[2], 12);
  ASSERT_EQUAL(v[3], 13);
  ASSERT_EQUAL(v[4], 14);
  
  thrust::sequence(thrust::cuda::par.on(s), v.begin(), v.end(), 10, 2);
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[1], 12);
  ASSERT_EQUAL(v[2], 14);
  ASSERT_EQUAL(v[3], 16);
  ASSERT_EQUAL(v[4], 18);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSequenceCudaStreams);

