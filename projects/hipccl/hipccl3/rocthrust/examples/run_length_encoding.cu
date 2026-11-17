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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

#include <iostream>
#include <iterator>

// This example computes a run-length code [1] for an array of characters.
//
// [1] http://en.wikipedia.org/wiki/Run-length_encoding

int main(void)
{
  // input data on the host
  const char data[] = "aaabbbbbcddeeeeeeeeeff";

  const size_t N = (sizeof(data) / sizeof(char)) - 1;

  // copy input data to the device
  thrust::device_vector<char> input(data, data + N);

  // allocate storage for output data and run lengths
  thrust::device_vector<char> output(N);
  thrust::device_vector<int> lengths(N);

  // print the initial data
  std::cout << "input data:" << std::endl;
  thrust::copy(input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
  std::cout << std::endl << std::endl;

  // compute run lengths
  size_t num_runs =
    thrust::reduce_by_key(
      input.begin(),
      input.end(), // input key sequence
      thrust::constant_iterator<int>(1), // input value sequence
      output.begin(), // output key sequence
      lengths.begin() // output value sequence
      )
      .first
    - output.begin(); // compute the output size

  // print the output
  std::cout << "run-length encoded output:" << std::endl;
  for (size_t i = 0; i < num_runs; i++)
  {
    std::cout << "(" << output[i] << "," << lengths[i] << ")";
  }
  std::cout << std::endl;

  return 0;
}
