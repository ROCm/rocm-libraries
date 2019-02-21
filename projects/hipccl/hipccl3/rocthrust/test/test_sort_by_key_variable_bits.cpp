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

// Google Test
#include <gtest/gtest.h>

#include <thrust/sort.h>

#include <algorithm>

#include "test_utils.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class SortByKeyVariableTests : public ::testing::Test
{
public:
  using input_type = typename Params::input_type;
};

typedef ::testing::Types<
  Params<thrust::detail::uint8_t>,
  Params<thrust::detail::uint16_t>,
  Params<thrust::detail::uint32_t>,
  Params<thrust::detail::uint64_t>
> SortByKeyVariableTestsParams;

TYPED_TEST_CASE(SortByKeyVariableTests, SortByKeyVariableTestsParams);

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

      const T mask = (1 << num_bits) - 1;
      for(size_t i = 0; i < size; i++)
         h_keys[i] &= mask;

      thrust::host_vector<T>   reference = h_keys;
      thrust::device_vector<T> d_keys    = h_keys;

      thrust::host_vector<T>   h_values = h_keys;
      thrust::device_vector<T> d_values = d_keys;

      std::sort(reference.begin(), reference.end());

      thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
      thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

      ASSERT_EQ(reference, h_keys);
      ASSERT_EQ(reference, h_values);

      ASSERT_EQ(h_keys,    d_keys);
      ASSERT_EQ(h_values,  d_values);
    }
  }
}
