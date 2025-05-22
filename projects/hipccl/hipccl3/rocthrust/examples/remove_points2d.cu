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

#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "include/host_device.h"

// This example generates random points in the
// unit square [0,1)x[0,1) and then removes all
// points where x^2 + y^2 > 1
//
// The x and y coordinates are stored in separate arrays
// and a zip_iterator is used to combine them together

template <typename T>
struct is_outside_circle
{
  template <typename Tuple>
  inline __host__ __device__ bool operator()(const Tuple& tuple) const
  {
    // unpack the tuple into x and y coordinates
    const T x = thrust::get<0>(tuple);
    const T y = thrust::get<1>(tuple);

    if (x * x + y * y > 1)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

int main(void)
{
  const size_t N = 20;

  // generate random points in the unit square on the host
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
  thrust::host_vector<float> x(N);
  thrust::host_vector<float> y(N);
  for (size_t i = 0; i < N; i++)
  {
    x[i] = u01(rng);
    y[i] = u01(rng);
  }

  // print the initial points
  std::cout << std::fixed;
  std::cout << "Generated " << N << " points" << std::endl;
  for (size_t i = 0; i < N; i++)
  {
    std::cout << "(" << x[i] << "," << y[i] << ")" << std::endl;
  }
  std::cout << std::endl;

  // remove points where x^2 + y^2 > 1 and determine new array sizes
  size_t new_size =
    thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end())),
                      is_outside_circle<float>())
    - thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));

  // resize the vectors (note: this does not free any memory)
  x.resize(new_size);
  y.resize(new_size);

  // print the filtered points
  std::cout << "After stream compaction, " << new_size << " points remain" << std::endl;
  for (size_t i = 0; i < new_size; i++)
  {
    std::cout << "(" << x[i] << "," << y[i] << ")" << std::endl;
  }

  return 0;
}
