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
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <iostream>

#include "include/host_device.h"

// this example illustrates how to make repeated access to a range of values
// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
//   ...

template <typename Iterator>
class repeated_range
{
public:
  using difference_type = typename thrust::iterator_difference<Iterator>::type;

  struct repeat_functor
  {
    difference_type repeats;

    repeat_functor(difference_type repeats)
        : repeats(repeats)
    {}

    __host__ __device__ difference_type operator()(const difference_type& i) const
    {
      return i / repeats;
    }
  };

  using CountingIterator    = typename thrust::counting_iterator<difference_type>;
  using TransformIterator   = typename thrust::transform_iterator<repeat_functor, CountingIterator>;
  using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

  // type of the repeated_range iterator
  using iterator = PermutationIterator;

  // construct repeated_range for the range [first,last)
  repeated_range(Iterator first, Iterator last, difference_type repeats)
      : first(first)
      , last(last)
      , repeats(repeats)
  {}

  iterator begin(void) const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
  }

  iterator end(void) const
  {
    return begin() + repeats * (last - first);
  }

protected:
  Iterator first;
  Iterator last;
  difference_type repeats;
};

int main(void)
{
  thrust::device_vector<int> data(4);
  data[0] = 10;
  data[1] = 20;
  data[2] = 30;
  data[3] = 40;

  // print the initial data
  std::cout << "range        ";
  thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  using Iterator = thrust::device_vector<int>::iterator;

  // create repeated_range with elements repeated twice
  repeated_range<Iterator> twice(data.begin(), data.end(), 2);
  std::cout << "repeated x2: ";
  thrust::copy(twice.begin(), twice.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // create repeated_range with elements repeated x3
  repeated_range<Iterator> thrice(data.begin(), data.end(), 3);
  std::cout << "repeated x3: ";
  thrust::copy(thrice.begin(), thrice.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
