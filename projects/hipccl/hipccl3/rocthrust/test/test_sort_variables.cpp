// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

// Thrust
#include <thrust/sort.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(SortByKeyVariableTests, AllIntegerTestsParams);


TYPED_TEST(SortByKeyVariableTests, TestSortVariableBits)
{
  using T = typename TestFixture::input_type;

  for (auto size : get_sizes())
  {
    for(size_t num_bits = 0; num_bits < 8 * sizeof(T); num_bits += 3)
    {
      SCOPED_TRACE(testing::Message() << "with size = " << size);

      thrust::host_vector<T> h_keys = get_random_data<T>(size,
                                                         std::numeric_limits<T>::min(),
                                                         std::numeric_limits<T>::max());

      size_t mask = (1 << num_bits) - 1;
      for(size_t i = 0; i < size; i++)
         h_keys[i] &= mask;

      thrust::host_vector<T>   reference = h_keys;
      thrust::device_vector<T> d_keys    = h_keys;

      std::sort(reference.begin(), reference.end());

      thrust::sort(h_keys.begin(), h_keys.end());
      thrust::sort(d_keys.begin(), d_keys.end());

      ASSERT_EQ(reference, h_keys);
      ASSERT_EQ(h_keys, d_keys);
    }
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

