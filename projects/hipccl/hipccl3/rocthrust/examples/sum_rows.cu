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
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

#include <iostream>

#include "include/host_device.h"

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index
{
  T C; // number of columns

  __host__ __device__ linear_index_to_row_index(T C)
      : C(C)
  {}

  __host__ __device__ T operator()(T i)
  {
    return i / C;
  }
};

int main(void)
{
  int R = 5; // number of rows
  int C = 8; // number of columns
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(10, 99);

  // initialize data
  thrust::device_vector<int> array(R * C);
  for (size_t i = 0; i < array.size(); i++)
  {
    array[i] = dist(rng);
  }

  // allocate storage for row sums and indices
  thrust::device_vector<int> row_sums(R);
  thrust::device_vector<int> row_indices(R);

  // compute row sums by summing values with equal row indices
  thrust::reduce_by_key(
    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R * C),
    array.begin(),
    row_indices.begin(),
    row_sums.begin(),
    thrust::equal_to<int>(),
    thrust::plus<int>());

  // print data
  for (int i = 0; i < R; i++)
  {
    std::cout << "[ ";
    for (int j = 0; j < C; j++)
    {
      std::cout << array[i * C + j] << " ";
    }
    std::cout << "] = " << row_sums[i] << "\n";
  }

  return 0;
}
