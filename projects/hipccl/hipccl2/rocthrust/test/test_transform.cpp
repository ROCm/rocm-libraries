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

#include <iostream>
#include <type_traits>
#include <cstdlib>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/transform.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

template<
  class Input,
  class Output = Input
>
struct Params
{
  using input_type = Input;
  using output_type = Output;
};

template<class Params>
class TransformTests : public ::testing::Test
{
public:
  using input_type = typename Params::input_type;
  using output_type = typename Params::output_type;
};

typedef ::testing::Types<
  Params<int>,
  Params<unsigned short>,
  Params<int, long long>
> TransformTestsParams;

TYPED_TEST_CASE(TransformTests, TransformTestsParams);

std::vector<size_t> get_sizes()
{
  std::vector<size_t> sizes = {
    0, 1, 2, 12, 63, 64, 211, 256, 344,
    1024, 2048, 5096, 34567, (1 << 17) - 1220
  };
  return sizes;
}

template<class T>
struct unary_transform
{
  __device__ __host__ inline
  constexpr T operator()(const T& a) const
  {
    return a + 5;
  }
};

template<class T>
struct binary_transform
{
  __device__ __host__ inline
  constexpr T operator()(const T& a, const T& b) const
  {
    return a * 2 + b * 5;
  }
};

TYPED_TEST(TransformTests, UnaryTransform)
{
  using T = typename TestFixture::input_type;
  using U = typename TestFixture::output_type;

  for(auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h_input(size);
    for(size_t i = 0; i < size; i++)
    {
      h_input[i] = i;
    }

    // Calculate expected results on host
    thrust::host_vector<U> expected(size);
    thrust::transform(h_input.begin(), h_input.end(), expected.begin(), unary_transform<U>());

    thrust::device_vector<T> d_input(h_input);
    thrust::device_vector<U> d_output(size);
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), unary_transform<U>());

    thrust::host_vector<U> h_output = d_output;
    for(size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_output[i], expected[i]) << "where index = " << i;
    }
  }
}

TYPED_TEST(TransformTests, BinaryTransform)
{
  using T = typename TestFixture::input_type;
  using U = typename TestFixture::output_type;

  for(auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h_input1(size);
    thrust::host_vector<T> h_input2(size);
    for(size_t i = 0; i < size; i++)
    {
      h_input1[i] = i * 3;
      h_input2[i] = i;
    }

    // Calculate expected results on host
    thrust::host_vector<U> expected(size);
    thrust::transform(
      h_input1.begin(), h_input1.end(), h_input2.begin(), expected.begin(),
      binary_transform<U>()
    );

    thrust::device_vector<T> d_input1(h_input1);
    thrust::device_vector<T> d_input2(h_input2);
    thrust::device_vector<U> d_output(size);
    thrust::transform(
      d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(),
      binary_transform<U>()
    );

    thrust::host_vector<U> h_output = d_output;
    for(size_t i = 0; i < size; i++)
    {
      ASSERT_EQ(h_output[i], expected[i]) << "where index = " << i;
    }
  }
}
