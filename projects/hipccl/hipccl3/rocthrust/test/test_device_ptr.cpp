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

#include "test_utils.hpp"

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
    Params<short>,
    Params<int>,
    Params<long long>,
    Params<unsigned short>,
    Params<unsigned int>,
    Params<unsigned long long>,
    Params<float>,
    Params<double>
> DevicePtrTestsParams;

TYPED_TEST_CASE(DevicePtrTests, DevicePtrTestsParams);

template <typename T>
struct mark_processed_functor
{
    thrust::device_ptr<T> ptr;
    __host__ __device__ void operator()(size_t x){ ptr[x] = 1; }
};

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

TYPED_TEST(DevicePtrTests,TestRawPointerCast)
{
    using T = typename TestFixture::input_type;
    thrust::device_vector<T> vec(3);

    T * first;
    T * last;

    first = thrust::raw_pointer_cast(&vec[0]);
    last  = thrust::raw_pointer_cast(&vec[3]);
    ASSERT_EQ(last - first, 3);

    first = thrust::raw_pointer_cast(&vec.front());
    last  = thrust::raw_pointer_cast(&vec.back());
    ASSERT_EQ(last - first, 2);
}

TYPED_TEST(DevicePtrTests, TestDevicePointerValue)
{
  using T = typename TestFixture::input_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::device_vector<T> d_data(size);

    thrust::device_ptr<T> begin(&d_data[0]);

    auto raw_ptr_begin = thrust::raw_pointer_cast(begin);
    if(size > 0) ASSERT_NE(raw_ptr_begin, nullptr);

    // Zero input memory
    if(size > 0) HIP_CHECK(hipMemset(raw_ptr_begin, 0, sizeof(T) * size));

    // Create unary function
    mark_processed_functor<T> func;
    func.ptr = begin;

    // Run for_each in [0; end] range
    auto end = size < 2 ? size : size/2;
    auto result = thrust::for_each(
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(end),
      func
    );
    ASSERT_EQ(result, thrust::make_counting_iterator<size_t>(end));

    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < size; i++)
    {
      if(i < end)
      {
        ASSERT_EQ(h_data[i], T(1)) << "where index = " << i;
      }
      else
      {
        ASSERT_EQ(h_data[i], T(0)) << "where index = " << i;
      }
    }
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
