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

#include <cstdlib>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/sort.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_utils.hpp"

template<
  class Key,
  class Item,
  class CompareFunction = thrust::less<Key>
>
struct Params
{
  using key_type = Key;
  using value_type = Item;
  using compare_function = CompareFunction;
};

template<class Params>
class SortTests : public ::testing::Test
{
public:
  using key_type = typename Params::key_type;
  using value_type = typename Params::value_type;
  using compare_function = typename Params::compare_function;
};

typedef ::testing::Types<
  Params<unsigned short, int, thrust::less<unsigned short> >,
  Params<unsigned short, int, thrust::greater<unsigned short> >,
  Params<unsigned short, int, custom_compare_less<unsigned short> >,
  Params<unsigned short, double>,
  Params<int, long long>
> SortTestsParams;

TYPED_TEST_CASE(SortTests, SortTestsParams);



TYPED_TEST(SortTests, Sort)
{
  using key_type = typename TestFixture::key_type;
  using compare_function = typename TestFixture::compare_function;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<key_type> h_keys;
    if (std::is_floating_point<key_type>::value)
    {
      h_keys = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
      h_keys = get_random_data<key_type>(
        size,
        std::numeric_limits<key_type>::min(),
        std::numeric_limits<key_type>::max()
      );
    }

    // Calculate expected results on host
    thrust::host_vector<key_type> expected(h_keys);
    thrust::sort(expected.begin(), expected.end(), compare_function());

    thrust::device_vector<key_type> d_keys(h_keys);
    thrust::sort(d_keys.begin(), d_keys.end(), compare_function());

    h_keys = d_keys;
    for (size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_keys[i], expected[i]) << "where index = " << i;
    }
  }
}

TYPED_TEST(SortTests, SortByKey)
{
  using key_type = typename TestFixture::key_type;
  using value_type = typename TestFixture::value_type;
  using compare_function = typename TestFixture::compare_function;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    // Check if non-stable sort can be used (no equal keys with different values)
    if (size > static_cast<size_t>(std::numeric_limits<key_type>::max())) continue;

    thrust::host_vector<key_type> h_keys(size);
    std::iota(h_keys.begin(), h_keys.end(), 0);
    std::shuffle(
      h_keys.begin(),
      h_keys.end(),
      std::default_random_engine(std::random_device{}())
    );

    thrust::host_vector<value_type> h_values(size);
    std::iota(h_values.begin(), h_values.end(), 0);

    // Calculate expected results on host
    thrust::host_vector<key_type> expected_keys(h_keys);
    thrust::host_vector<value_type> expected_values(h_values);
    thrust::sort_by_key(
      expected_keys.begin(), expected_keys.end(), expected_values.begin(),
      compare_function());

    thrust::device_vector<key_type> d_keys(h_keys);
    thrust::device_vector<value_type> d_values(h_values);
    thrust::sort_by_key(
      d_keys.begin(), d_keys.end(), d_values.begin(),
      compare_function());

    h_keys = d_keys;
    h_values = d_values;
    for (size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_keys[i], expected_keys[i]) << "where index = " << i;
      ASSERT_EQ(h_values[i], expected_values[i]) << "where index = " << i;
    }
  }
}

TYPED_TEST(SortTests, StableSort)
{
  using key_type = typename TestFixture::key_type;
  using compare_function = typename TestFixture::compare_function;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<key_type> h_keys;
    if (std::is_floating_point<key_type>::value)
    {
      h_keys = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
      h_keys = get_random_data<key_type>(
        size,
        std::numeric_limits<key_type>::min(),
        std::numeric_limits<key_type>::max()
      );
    }

    // Calculate expected results on host
    thrust::host_vector<key_type> expected(h_keys);
    thrust::stable_sort(expected.begin(), expected.end(), compare_function());

    thrust::device_vector<key_type> d_keys(h_keys);
    thrust::stable_sort(d_keys.begin(), d_keys.end(), compare_function());

    h_keys = d_keys;
    for (size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_keys[i], expected[i]) << "where index = " << i;
    }
  }
}

TYPED_TEST(SortTests, StableSortByKey)
{
  using key_type = typename TestFixture::key_type;
  using value_type = typename TestFixture::value_type;
  using compare_function = typename TestFixture::compare_function;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<key_type> h_keys;
    if (std::is_floating_point<key_type>::value)
    {
      h_keys = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000);
    }
    else
    {
      h_keys = get_random_data<key_type>(
        size,
        std::numeric_limits<key_type>::min(),
        std::numeric_limits<key_type>::max()
      );
    }

    thrust::host_vector<value_type> h_values(size);
    std::iota(h_values.begin(), h_values.end(), 0);

    // Calculate expected results on host
    thrust::host_vector<key_type> expected_keys(h_keys);
    thrust::host_vector<value_type> expected_values(h_values);
    thrust::stable_sort_by_key(
      expected_keys.begin(), expected_keys.end(), expected_values.begin(),
      compare_function());

    thrust::device_vector<key_type> d_keys(h_keys);
    thrust::device_vector<value_type> d_values(h_values);
    thrust::stable_sort_by_key(
      d_keys.begin(), d_keys.end(), d_values.begin(),
      compare_function());

    h_keys = d_keys;
    h_values = d_values;
    for (size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_keys[i], expected_keys[i]) << "where index = " << i;
      ASSERT_EQ(h_values[i], expected_values[i]) << "where index = " << i;
    }
  }
}
