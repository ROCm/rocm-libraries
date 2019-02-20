// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class UniqueByKeyTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<thrust::host_vector<short>>,
    Params<thrust::host_vector<int>>,
    Params<thrust::host_vector<long long>>,
    Params<thrust::host_vector<unsigned short>>,
    Params<thrust::host_vector<unsigned int>>,
    Params<thrust::host_vector<unsigned long long>>,
    Params<thrust::host_vector<float>>,
    Params<thrust::host_vector<double>>,
    Params<thrust::device_vector<short>>,
    Params<thrust::device_vector<int>>,
    Params<thrust::device_vector<long long>>,
    Params<thrust::device_vector<unsigned short>>,
    Params<thrust::device_vector<unsigned int>>,
    Params<thrust::device_vector<unsigned long long>>,
    Params<thrust::device_vector<float>>,
    Params<thrust::device_vector<double>>
> UniqueByKeyTestsParams;

TYPED_TEST_CASE(UniqueByKeyTests, UniqueByKeyTestsParams);


template <typename ForwardIterator1,
          typename ForwardIterator2>
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(my_system &system,
              ForwardIterator1 keys_first,
              ForwardIterator1,
              ForwardIterator2 values_first)
{
    system.validate_dispatch();
    return thrust::make_pair(keys_first,values_first);
}

TEST(UniqueByKeyTests, TestUniqueByKeyDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator1,
          typename ForwardIterator2>
thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(my_tag,
              ForwardIterator1 keys_first,
              ForwardIterator1,
              ForwardIterator2 values_first)
{
    *keys_first = 13;
    return thrust::make_pair(keys_first,values_first);
}

TEST(UniqueByKeyTests, TestUniqueByKeyDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::unique_by_key(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(my_system &system,
                   InputIterator1,
                   InputIterator1,
                   InputIterator2,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output)
{
    system.validate_dispatch();
    return thrust::make_pair(keys_output, values_output);
}

TEST(UniqueByKeyTests, TestUniqueByKeyCopyDispatchExplicit)
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique_by_key_copy(sys,
                             vec.begin(),
                             vec.begin(),
                             vec.begin(),
                             vec.begin(),
                             vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(my_tag,
                   InputIterator1,
                   InputIterator1,
                   InputIterator2,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output)
{
    *keys_output = 13;
    return thrust::make_pair(keys_output, values_output);
}

TEST(UniqueByKeyTests, TestUniqueByKeyCopyDispatchImplicit)
{
  thrust::device_vector<int> vec(1);

  thrust::unique_by_key_copy(thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template<typename T>
struct is_equal_div_10_unique
{
    __host__ __device__
    bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};

template <typename Vector>
void initialize_keys(Vector& keys)
{
    keys.resize(9);
    keys[0] = 11;
    keys[1] = 11;
    keys[2] = 21;
    keys[3] = 20;
    keys[4] = 21;
    keys[5] = 21;
    keys[6] = 21;
    keys[7] = 37;
    keys[8] = 37;
}

template <typename Vector>
void initialize_values(Vector& values)
{
    values.resize(9);
    values[0] = 0;
    values[1] = 1;
    values[2] = 2;
    values[3] = 3;
    values[4] = 4;
    values[5] = 5;
    values[6] = 6;
    values[7] = 7;
    values[8] = 8;
}


TYPED_TEST(UniqueByKeyTests, TestUniqueByKeySimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector keys;
  Vector values;

  typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);  initialize_values(values);

  new_last = thrust::unique_by_key(keys.begin(), keys.end(), values.begin());

  ASSERT_EQ(new_last.first  - keys.begin(),   5);
  ASSERT_EQ(new_last.second - values.begin(), 5);
  ASSERT_EQ(keys[0], 11);
  ASSERT_EQ(keys[1], 21);
  ASSERT_EQ(keys[2], 20);
  ASSERT_EQ(keys[3], 21);
  ASSERT_EQ(keys[4], 37);

  ASSERT_EQ(values[0], 0);
  ASSERT_EQ(values[1], 2);
  ASSERT_EQ(values[2], 3);
  ASSERT_EQ(values[3], 4);
  ASSERT_EQ(values[4], 7);

  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);

  new_last = thrust::unique_by_key(keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>());

  ASSERT_EQ(new_last.first  - keys.begin(),   3);
  ASSERT_EQ(new_last.second - values.begin(), 3);
  ASSERT_EQ(keys[0], 11);
  ASSERT_EQ(keys[1], 21);
  ASSERT_EQ(keys[2], 37);

  ASSERT_EQ(values[0], 0);
  ASSERT_EQ(values[1], 2);
  ASSERT_EQ(values[2], 7);
}

TYPED_TEST(UniqueByKeyTests, TestUniqueCopyByKeySimple)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;

  Vector keys;
  Vector values;

  typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  new_last = thrust::unique_by_key_copy(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

  ASSERT_EQ(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQ(new_last.second - output_values.begin(), 5);
  ASSERT_EQ(output_keys[0], 11);
  ASSERT_EQ(output_keys[1], 21);
  ASSERT_EQ(output_keys[2], 20);
  ASSERT_EQ(output_keys[3], 21);
  ASSERT_EQ(output_keys[4], 37);

  ASSERT_EQ(output_values[0], 0);
  ASSERT_EQ(output_values[1], 2);
  ASSERT_EQ(output_values[2], 3);
  ASSERT_EQ(output_values[3], 4);
  ASSERT_EQ(output_values[4], 7);

  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);

  new_last = thrust::unique_by_key_copy(keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_unique<T>());

  ASSERT_EQ(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQ(new_last.second - output_values.begin(), 3);
  ASSERT_EQ(output_keys[0], 11);
  ASSERT_EQ(output_keys[1], 21);
  ASSERT_EQ(output_keys[2], 37);

  ASSERT_EQ(output_values[0], 0);
  ASSERT_EQ(output_values[1], 2);
  ASSERT_EQ(output_values[2], 7);
}
