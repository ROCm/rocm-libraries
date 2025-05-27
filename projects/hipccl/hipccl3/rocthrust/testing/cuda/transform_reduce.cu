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
#include <thrust/transform_reduce.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Function1, typename T, typename Function2, typename Iterator2>
__global__ void transform_reduce_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Function1 f1, T init, Function2 f2, Iterator2 result)
{
  *result = thrust::transform_reduce(exec, first, last, f1, init, f2);
}

template <typename ExecutionPolicy>
void TestTransformReduceDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  Vector data(3);
  data[0] = 1;
  data[1] = -2;
  data[2] = 3;

  T init = 10;

  thrust::device_vector<T> result(1);

  transform_reduce_kernel<<<1, 1>>>(
    exec, data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>(), result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(8, (T) result[0]);
}

void TestTransformReduceDeviceSeq()
{
  TestTransformReduceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformReduceDeviceSeq);

void TestTransformReduceDeviceDevice()
{
  TestTransformReduceDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformReduceDeviceDevice);
#endif

void TestTransformReduceCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data(3);
  data[0] = 1;
  data[1] = -2;
  data[2] = 3;

  T init = 10;

  cudaStream_t s;
  cudaStreamCreate(&s);

  T result = thrust::transform_reduce(
    thrust::cuda::par.on(s), data.begin(), data.end(), thrust::negate<T>(), init, thrust::plus<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(8, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformReduceCudaStreams);
