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
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <iostream>

// This example shows how to perform a lexicographical sort on multiple keys.
//
// http://en.wikipedia.org/wiki/Lexicographical_order

template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
  // temporary storage for keys
  KeyVector temp(keys.size());

  // permute the keys with the current reordering
  thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

  // stable_sort the permuted keys and update the permutation
  thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}

template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys, PermutationVector& permutation)
{
  // copy keys to temporary vector
  KeyVector temp(keys.begin(), keys.end());

  // permute the keys
  thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

thrust::host_vector<int> random_vector(size_t N)
{
  thrust::host_vector<int> vec(N);
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9);

  for (size_t i = 0; i < N; i++)
  {
    vec[i] = dist(rng);
  }

  return vec;
}

int main(void)
{
  size_t N = 20;

  // generate three arrays of random values
  thrust::device_vector<int> upper  = random_vector(N);
  thrust::device_vector<int> middle = random_vector(N);
  thrust::device_vector<int> lower  = random_vector(N);

  std::cout << "Unsorted Keys" << std::endl;
  for (size_t i = 0; i < N; i++)
  {
    std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
  }

  // initialize permutation to [0, 1, 2, ... ,N-1]
  thrust::device_vector<int> permutation(N);
  thrust::sequence(permutation.begin(), permutation.end());

  // sort from least significant key to most significant keys
  update_permutation(lower, permutation);
  update_permutation(middle, permutation);
  update_permutation(upper, permutation);

  // Note: keys have not been modified
  // Note: permutation now maps unsorted keys to sorted order

  // permute the key arrays by the final permuation
  apply_permutation(lower, permutation);
  apply_permutation(middle, permutation);
  apply_permutation(upper, permutation);

  std::cout << "Sorted Keys" << std::endl;
  for (size_t i = 0; i < N; i++)
  {
    std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
  }

  return 0;
}
