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
#include <thrust/gather.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <iostream>

#include "include/host_device.h"
struct Functor
{
  template <class Tuple>
  __host__ __device__ float operator()(const Tuple& tuple) const
  {
    const float x = thrust::get<0>(tuple);
    const float y = thrust::get<1>(tuple);
    return x * y * 2.0f / 3.0f;
  }
};

int main(void)
{
  float u[4] = {4, 3, 2, 1};
  float v[4] = {-1, 1, 1, -1};
  int idx[3] = {3, 0, 1};
  float w[3] = {0, 0, 0};

  thrust::device_vector<float> U(u, u + 4);
  thrust::device_vector<float> V(v, v + 4);
  thrust::device_vector<int> IDX(idx, idx + 3);
  thrust::device_vector<float> W(w, w + 3);

  // gather multiple elements and apply a function before writing result in memory
  thrust::gather(IDX.begin(),
                 IDX.end(),
                 thrust::make_zip_iterator(thrust::make_tuple(U.begin(), V.begin())),
                 thrust::make_transform_output_iterator(W.begin(), Functor()));

  std::cout << "result= [ ";
  for (size_t i = 0; i < 3; i++)
  {
    std::cout << W[i] << " ";
  }
  std::cout << "] \n";

  return 0;
}
