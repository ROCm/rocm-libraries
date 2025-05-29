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
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iterator>

int main(void)
{
  // this example computes indices for all the nonzero values in a sequence

  // sequence of zero and nonzero values
  thrust::device_vector<int> stencil(8);
  stencil[0] = 0;
  stencil[1] = 1;
  stencil[2] = 1;
  stencil[3] = 0;
  stencil[4] = 0;
  stencil[5] = 1;
  stencil[6] = 0;
  stencil[7] = 1;

  // storage for the nonzero indices
  thrust::device_vector<int> indices(8);

  // counting iterators define a sequence [0, 8)
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + 8;

  // compute indices of nonzero elements
  using IndexIterator = thrust::device_vector<int>::iterator;

  IndexIterator indices_end = thrust::copy_if(first, last, stencil.begin(), indices.begin(), thrust::identity<int>());
  // indices now contains [1,2,5,7]

  // print result
  std::cout << "found " << (indices_end - indices.begin()) << " nonzero values at indices:\n";
  thrust::copy(indices.begin(), indices_end, std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}
