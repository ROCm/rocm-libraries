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
#include <vector>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/memory.h>
#include <thrust/transform.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class DevicePtrTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<int>
> DevicePtrTestsParams;

TYPED_TEST_CASE(DevicePtrTests, DevicePtrTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(DevicePtrTests, MakeDevicePointer)
{
  using T = typename TestFixture::input_type;

  T *raw_ptr = 0;
  thrust::device_ptr<T> p0 = thrust::device_pointer_cast(raw_ptr);

  ASSERT_EQ(thrust::raw_pointer_cast(p0), raw_ptr);
  thrust::device_ptr<T> p1 = thrust::device_pointer_cast(p0);
  ASSERT_EQ(p0, p1);
}

TEST(DevicePtrTests,TestDevicePointerManipulation)
{
    thrust::device_vector<int> data(5);

    thrust::device_ptr<int> begin(&data[0]);
    thrust::device_ptr<int> end(&data[0] + 5);

    ASSERT_EQ(end - begin, 5);

    begin++;
    begin--;

    ASSERT_EQ(end - begin, 5);

    begin += 1;
    begin -= 1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (int) 1;
    begin = begin - (int) 1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (unsigned int) 1;
    begin = begin - (unsigned int) 1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (size_t) 1;
    begin = begin - (size_t) 1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (ptrdiff_t) 1;
    begin = begin - (ptrdiff_t) 1;

    ASSERT_EQ(end - begin, 5);

    begin = begin + (thrust::device_ptr<int>::difference_type) 1;
    begin = begin - (thrust::device_ptr<int>::difference_type) 1;

    ASSERT_EQ(end - begin, 5);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
