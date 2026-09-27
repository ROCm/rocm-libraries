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

#include <thrust/execution_policy.h>
#include <thrust/swap.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void swap_ranges_kernel(ExecutionPolicy exec, Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  thrust::swap_ranges(exec, first1, last1, first2);
}

template <typename ExecutionPolicy>
void TestSwapRangesDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};
  Vector v2{5, 6, 7, 8, 9};
  Vector v1_ref(v2);
  Vector v2_ref(v1);

  swap_ranges_kernel<<<1, 1>>>(exec, v1.begin(), v1.end(), v2.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(v1, v1_ref);
  ASSERT_EQUAL(v2, v2_ref);
}

void TestSwapRangesDeviceSeq()
{
  TestSwapRangesDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSwapRangesDeviceSeq);

void TestSwapRangesDeviceDevice()
{
  TestSwapRangesDevice(thrust::device);
}
DECLARE_UNITTEST(TestSwapRangesDeviceDevice);
#endif

void TestSwapRangesCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector v1{0, 1, 2, 3, 4};
  Vector v2{5, 6, 7, 8, 9};
  Vector v1_ref(v2);
  Vector v2_ref(v1);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::swap_ranges(thrust::cuda::par.on(s), v1.begin(), v1.end(), v2.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(v1, v1_ref);
  ASSERT_EQUAL(v2, v2_ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestSwapRangesCudaStreams);
