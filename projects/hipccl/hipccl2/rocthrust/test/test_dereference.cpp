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
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(DereferenceTests, FullTestsParams);

template <typename Iterator1, typename Iterator2>
__global__
void simple_copy_on_device(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  while(first1 != last1)
    *(first2++) = *(first1++);
}

template <typename Iterator1, typename Iterator2>
void simple_copy(Iterator1 first1, Iterator1 last1, Iterator2 first2)
{
  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(simple_copy_on_device<Iterator1, Iterator2>),
    dim3(1), dim3(1), 0, 0,
    first1, last1, first2
  );
}

TEST(DereferenceTests, TestDeviceDereferenceDeviceVectorIterator)
{
  thrust::device_vector<int> input = get_random_data<int>(100,
                                                          std::numeric_limits<int>::min(),
                                                          std::numeric_limits<int>::max());
  thrust::device_vector<int> output(input.size(), 0);

  simple_copy(input.begin(), input.end(), output.begin());

  ASSERT_EQ(input, output);
}

TEST(DereferenceTests, TestDeviceDereferenceDevicePtr)
{
  thrust::device_vector<int> input = get_random_data<int>(100,
                                                          std::numeric_limits<int>::min(),
                                                          std::numeric_limits<int>::max());
  thrust::device_vector<int> output(input.size(), 0);

  thrust::device_ptr<int> _first1 = &input[0];
  thrust::device_ptr<int> _last1  = _first1 + input.size();
  thrust::device_ptr<int> _first2 = &output[0];

  simple_copy(_first1, _last1, _first2);

  ASSERT_EQ(input, output);
}

TEST(DereferenceTests, TestDeviceDereferenceTransformIterator)
{
  thrust::device_vector<int> input = get_random_data<int>(100,
                                                          std::numeric_limits<int>::min(),
                                                          std::numeric_limits<int>::max());
  thrust::device_vector<int> output(input.size(), 0);

  simple_copy(thrust::make_transform_iterator(input.begin(), thrust::identity<int>()),
              thrust::make_transform_iterator(input.end (),  thrust::identity<int>()),
              output.begin());

  ASSERT_EQ(input, output);
}

TEST(DereferenceTests, TestDeviceDereferenceCountingIterator)
{
  thrust::counting_iterator<int> first(1);
  thrust::counting_iterator<int> last(6);

  thrust::device_vector<int> output(5);

  simple_copy(first, last, output.begin());

  ASSERT_EQ(output[0], 1);
  ASSERT_EQ(output[1], 2);
  ASSERT_EQ(output[2], 3);
  ASSERT_EQ(output[3], 4);
  ASSERT_EQ(output[4], 5);
}

TEST(DereferenceTests, TestDeviceDereferenceTransformedCountingIterator)
{
  thrust::counting_iterator<int> first(1);
  thrust::counting_iterator<int> last(6);

  thrust::device_vector<int> output(5);

  simple_copy(thrust::make_transform_iterator(first, thrust::negate<int>()),
              thrust::make_transform_iterator(last,  thrust::negate<int>()),
              output.begin());

  ASSERT_EQ(output[0], -1);
  ASSERT_EQ(output[1], -2);
  ASSERT_EQ(output[2], -3);
  ASSERT_EQ(output[3], -4);
  ASSERT_EQ(output[4], -5);
}
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

