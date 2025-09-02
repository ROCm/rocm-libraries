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
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <iostream>
#include <iterator>

// This example compute the mode [1] of a set of numbers.  If there
// are multiple modes, one with the smallest value it returned.
//
// [1] http://en.wikipedia.org/wiki/Mode_(statistics)

int main(void)
{
  const size_t N = 30;
  const size_t M = 10;
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, M - 1);

  // generate random data on the host
  thrust::host_vector<int> h_data(N);
  for (size_t i = 0; i < N; i++)
  {
    h_data[i] = dist(rng);
  }

  // transfer data to device
  thrust::device_vector<int> d_data(h_data);

  // print the initial data
  std::cout << "initial data" << std::endl;
  thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // sort data to bring equal elements together
  thrust::sort(d_data.begin(), d_data.end());

  // print the sorted data
  std::cout << "sorted data" << std::endl;
  thrust::copy(d_data.begin(), d_data.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // count number of unique keys
  size_t num_unique = thrust::unique_count(d_data.begin(), d_data.end());

  // count multiplicity of each key
  thrust::device_vector<int> d_output_keys(num_unique);
  thrust::device_vector<int> d_output_counts(num_unique);
  thrust::reduce_by_key(
    d_data.begin(), d_data.end(), thrust::constant_iterator<int>(1), d_output_keys.begin(), d_output_counts.begin());

  // print the counts
  std::cout << "values" << std::endl;
  thrust::copy(d_output_keys.begin(), d_output_keys.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // print the counts
  std::cout << "counts" << std::endl;
  thrust::copy(d_output_counts.begin(), d_output_counts.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // find the index of the maximum count
  thrust::device_vector<int>::iterator mode_iter;
  mode_iter = thrust::max_element(d_output_counts.begin(), d_output_counts.end());

  int mode       = d_output_keys[mode_iter - d_output_counts.begin()];
  int occurances = *mode_iter;

  std::cout << "Modal value " << mode << " occurs " << occurances << " times " << std::endl;

  return 0;
}
