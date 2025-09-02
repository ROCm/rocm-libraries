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
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <iostream>

#include "include/host_device.h"

//   This example computes the norm [1] of a vector.  The norm is
// computed by squaring all numbers in the vector, summing the
// squares, and taking the square root of the sum of squares.  In
// Thrust this operation is efficiently implemented with the
// transform_reduce() algorith.  Specifically, we first transform
// x -> x^2 and the compute a standard plus reduction.  Since there
// is no built-in functor for squaring numbers, we define our own
// square functor.
//
// [1] http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
  __host__ __device__ T operator()(const T& x) const
  {
    return x * x;
  }
};

int main(void)
{
  // initialize host array
  float x[4] = {1.0, 2.0, 3.0, 4.0};

  // transfer to device
  thrust::device_vector<float> d_x(x, x + 4);

  // setup arguments
  square<float> unary_op;
  thrust::plus<float> binary_op;
  float init = 0;

  // compute norm
  float norm = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op));

  std::cout << "norm is " << norm << std::endl;

  return 0;
}
