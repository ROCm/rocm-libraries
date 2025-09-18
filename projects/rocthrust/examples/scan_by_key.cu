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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <iostream>

#include "include/host_device.h"

// BinaryPredicate for the head flag segment representation
// equivalent to thrust::not_fn(thrust::project2nd<int,int>()));
template <typename HeadFlagType>
struct head_flag_predicate
{
  __host__ __device__ bool operator()(HeadFlagType, HeadFlagType right) const
  {
    return !right;
  }
};

template <typename Vector>
void print(const Vector& v)
{
  for (size_t i = 0; i < v.size(); i++)
  {
    std::cout << v[i] << " ";
  }
  std::cout << "\n";
}

int main(void)
{
  int keys[]   = {0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5}; // segments represented with keys
  int flags[]  = {1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0}; // segments represented with head flags
  int values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; // values corresponding to each key

  int N = sizeof(keys) / sizeof(int); // number of elements

  // copy input data to device
  thrust::device_vector<int> d_keys(keys, keys + N);
  thrust::device_vector<int> d_flags(flags, flags + N);
  thrust::device_vector<int> d_values(values, values + N);

  // allocate storage for output
  thrust::device_vector<int> d_output(N);

  // inclusive scan using keys
  thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), d_output.begin());

  std::cout << "Inclusive Segmented Scan w/ Key Sequence\n";
  std::cout << " keys          : ";
  print(d_keys);
  std::cout << " input values  : ";
  print(d_values);
  std::cout << " output values : ";
  print(d_output);

  // inclusive scan using head flags
  thrust::inclusive_scan_by_key(
    d_flags.begin(), d_flags.end(), d_values.begin(), d_output.begin(), head_flag_predicate<int>());

  std::cout << "\nInclusive Segmented Scan w/ Head Flag Sequence\n";
  std::cout << " head flags    : ";
  print(d_flags);
  std::cout << " input values  : ";
  print(d_values);
  std::cout << " output values : ";
  print(d_output);

  // exclusive scan using keys
  thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), d_output.begin());

  std::cout << "\nExclusive Segmented Scan w/ Key Sequence\n";
  std::cout << " keys          : ";
  print(d_keys);
  std::cout << " input values  : ";
  print(d_values);
  std::cout << " output values : ";
  print(d_output);

  // exclusive scan using head flags
  thrust::exclusive_scan_by_key(
    d_flags.begin(), d_flags.end(), d_values.begin(), d_output.begin(), 0, head_flag_predicate<int>());

  std::cout << "\nExclusive Segmented Scan w/ Head Flag Sequence\n";
  std::cout << " head flags    : ";
  print(d_flags);
  std::cout << " input values  : ";
  print(d_values);
  std::cout << " output values : ";
  print(d_output);

  return 0;
}
