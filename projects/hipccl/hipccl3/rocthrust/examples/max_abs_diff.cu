// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

#include <cmath>
#include <iostream>

#include "include/host_device.h"

// this example computes the maximum absolute difference
// between the elements of two vectors

template <typename T>
struct abs_diff
{
  __host__ __device__ T operator()(const T& a, const T& b)
  {
    return fabsf(b - a);
  }
};

int main(void)
{
  thrust::device_vector<float> d_a(4);
  thrust::device_vector<float> d_b(4);

  // clang-format off
  d_a[0] = 1.0;  d_b[0] = 2.0;
  d_a[1] = 2.0;  d_b[1] = 4.0;
  d_a[2] = 3.0;  d_b[2] = 3.0;
  d_a[3] = 4.0;  d_b[3] = 0.0;
  // clang-format on

  // initial value of the reduction
  float init = 0;

  // binary operations
  thrust::maximum<float> binary_op1;
  abs_diff<float> binary_op2;

  float max_abs_diff = thrust::inner_product(d_a.begin(), d_a.end(), d_b.begin(), init, binary_op1, binary_op2);

  std::cout << "maximum absolute difference: " << max_abs_diff << std::endl;
  return 0;
}
