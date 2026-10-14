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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>

#include <iostream>

// this example fuses a gather operation with a reduction for
// greater efficiency than separate gather() and reduce() calls

int main(void)
{
  // gather locations
  thrust::device_vector<int> map(4);
  map[0] = 3;
  map[1] = 1;
  map[2] = 0;
  map[3] = 5;

  // array to gather from
  thrust::device_vector<int> source(6);
  source[0] = 10;
  source[1] = 20;
  source[2] = 30;
  source[3] = 40;
  source[4] = 50;
  source[5] = 60;

  // fuse gather with reduction:
  //   sum = source[map[0]] + source[map[1]] + ...
  int sum = thrust::reduce(thrust::make_permutation_iterator(source.begin(), map.begin()),
                           thrust::make_permutation_iterator(source.begin(), map.end()));

  // print sum
  std::cout << "sum is " << sum << std::endl;

  return 0;
}
