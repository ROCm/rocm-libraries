<<<<<<< ours
// Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>

#include <iomanip>
#include <iostream>

#include "include/host_device.h"

int main(void)
{
  const int rows = 32;
  const int cols = 16;

  // Create a 2D row-major multidimensional array of ints.
  thrust::universal_vector<int> data(rows * cols, 42);
  int* M         = thrust::raw_pointer_cast(data.data());
  const int size = rows * cols;

  // Create an iterator to the flat linear index space for the multidimensional array.
  auto flat_idx = thrust::counting_iterator<int>(0);

  // Fill the array with pseudorandom inputs in parallel on the device.
  thrust::tabulate(thrust::device, M, M + size, [] __host__ __device__(int flat) {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 3);
    rng.discard(flat); // Advance to the current element's position.
    return dist(rng);
  });

  // Create a range to the row index of each element.
  auto row_idx_begin = thrust::make_transform_iterator(flat_idx, [=] __host__ __device__(int flat) {
    return flat / cols;
  });
  auto row_idx_end   = row_idx_begin + size;

  // Sum each row, storing the result in a new vector.
  thrust::universal_vector<int> sums(rows);
  thrust::reduce_by_key(thrust::device, row_idx_begin, row_idx_end, M, thrust::make_discard_iterator(), sums.begin());

  // Output the result.
  thrust::for_each_n(thrust::seq, flat_idx, rows, [&](int i) {
    std::cout << "[ ";
    thrust::for_each_n(thrust::seq, flat_idx, cols, [&](int j) {
      std::cout << std::setw(2) << M[i * cols + j] << " ";
    });
    std::cout << "] = " << sums[i] << "\n";
  });

  return 0;
}
