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
#include <thrust/distance.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TESTS_DEFINE(DistanceTests, FullTestsParams);

TYPED_TEST(DistanceTests, TestDistance)
{
  using Vector = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  Vector v(100);

  Iterator i = v.begin();

  ASSERT_EQ(thrust::distance(i, v.end()), 100);

  i++;

  ASSERT_EQ(thrust::distance(i, v.end()), 99);

  i += 49;

  ASSERT_EQ(thrust::distance(i, v.end()), 50);

  ASSERT_EQ(thrust::distance(i, i), 0);
}

TYPED_TEST(DistanceTests, TestDistanceLarge)
{
  using Vector = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  Vector v(1000);

  Iterator i = v.begin();

  ASSERT_EQ(thrust::distance(i, v.end()), 1000);

  i++;

  ASSERT_EQ(thrust::distance(i, v.end()), 999);

  i += 49;

  ASSERT_EQ(thrust::distance(i, v.end()), 950);

  i += 950;

  ASSERT_EQ(thrust::distance(i, v.end()), 0);

  ASSERT_EQ(thrust::distance(i, i), 0);
}


#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
