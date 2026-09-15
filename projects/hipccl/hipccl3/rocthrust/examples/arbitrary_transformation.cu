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

#include <thrust/detail/config.h>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <iostream>

#if !defined(THRUST_LEGACY_GCC)
#  include <thrust/zip_function.h>
#endif // >= C++11

#include "include/host_device.h"

// This example shows how to implement an arbitrary transformation of
// the form output[i] = F(first[i], second[i], third[i], ... ).
// In this example, we use a function with 3 inputs and 1 output.
//
// Iterators for all four vectors (3 inputs + 1 output) are "zipped"
// into a single sequence of tuples with the zip_iterator.
//
// The arbitrary_functor receives a tuple that contains four elements,
// which are references to values in each of the four sequences. When we
// access the tuple 't' with the get() function,
//      get<0>(t) returns a reference to A[i],
//      get<1>(t) returns a reference to B[i],
//      get<2>(t) returns a reference to C[i],
//      get<3>(t) returns a reference to D[i].
//
// In this example, we can implement the transformation,
//      D[i] = A[i] + B[i] * C[i];
// by invoking arbitrary_functor() on each of the tuples using for_each.
//
// If we are using a functor that is not designed for zip iterators by taking a
// tuple instead of individual arguments we can adapt this function using the
// zip_function adaptor (C++11 only).
//
// Note that we could extend this example to implement functions with an
// arbitrary number of input arguments by zipping more sequence together.
// With the same approach we can have multiple *output* sequences, if we
// wanted to implement something like
//      D[i] = A[i] + B[i] * C[i];
//      E[i] = A[i] + B[i] + C[i];
//
// The possibilities are endless! :)

struct arbitrary_functor1
{
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    // D[i] = A[i] + B[i] * C[i];
    thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
  }
};

#if !defined(THRUST_LEGACY_GCC)
struct arbitrary_functor2
{
  __host__ __device__ void operator()(const float& a, const float& b, const float& c, float& d)
  {
    // D[i] = A[i] + B[i] * C[i];
    d = a + b * c;
  }
};
#endif // >= C++11

int main(void)
{
  // allocate storage
  thrust::device_vector<float> A(5);
  thrust::device_vector<float> B(5);
  thrust::device_vector<float> C(5);
  thrust::device_vector<float> D1(5);

  // initialize input vectors
  A[0] = 3;
  B[0] = 6;
  C[0] = 2;
  A[1] = 4;
  B[1] = 7;
  C[1] = 5;
  A[2] = 0;
  B[2] = 2;
  C[2] = 7;
  A[3] = 8;
  B[3] = 1;
  C[3] = 4;
  A[4] = 2;
  B[4] = 8;
  C[4] = 3;

  // apply the transformation
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D1.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end(), D1.end())),
                   arbitrary_functor1());

  // print the output
  std::cout << "Tuple functor" << std::endl;
  for (int i = 0; i < 5; i++)
  {
    std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D1[i] << std::endl;
  }

  // apply the transformation using zip_function
#if !defined(THRUST_LEGACY_GCC)
  thrust::device_vector<float> D2(5);
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D2.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end(), D2.end())),
                   thrust::make_zip_function(arbitrary_functor2()));

  // print the output
  std::cout << "N-ary functor" << std::endl;
  for (int i = 0; i < 5; i++)
  {
    std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D2[i] << std::endl;
  }
#endif // >= C++11
}
