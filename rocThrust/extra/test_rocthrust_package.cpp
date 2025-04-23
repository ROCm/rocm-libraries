/*
 *  Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/version.h>
#include <thrust/rocthrust_version.hpp>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

template<class T>
struct unary_transform
{
  __device__ __host__ inline
  constexpr T operator()(const T& a) const
  {
    return a + 5;
  }
};

int main(void)
{
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;

  std::cout << "Thrust v" << major << "." << minor << std::endl;

  major = ROCTHRUST_VERSION_MAJOR;
  minor = ROCTHRUST_VERSION_MINOR;

  std::cout << "rocThrust v" << major << "." << minor << std::endl;

  using T = int;
  using U = long;

  size_t  size = 1<<16;
  thrust::host_vector<T> h_input(size);
  for(size_t i = 0; i < size; i++)
  {
    h_input[i] = i;
  }

  // Calculate expected results on host
  thrust::host_vector<U> expected(size);
  thrust::transform(h_input.begin(), h_input.end(), expected.begin(), unary_transform<U>());

  thrust::device_vector<T> d_input(h_input);
  thrust::device_vector<U> d_output(size);
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), unary_transform<U>());

  thrust::host_vector<U> h_output = d_output;
  for(size_t i = 0; i < size; i++)
  {
    if(h_output[i] != expected[i])
    {
      std::cout
          << "Failure: output (" << h_output[i]
          << ") != expected (" << expected[i] << ")"
          << "  at index:" << i
          << std::endl;
      return 1;
    }
  }

  return 0;
}
