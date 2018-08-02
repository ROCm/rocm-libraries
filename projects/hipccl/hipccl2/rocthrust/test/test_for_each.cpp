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
#include <algorithm>
#include <vector>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/memory.h>
#include <thrust/for_each.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

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
class ForEachTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<int>,
    Params<unsigned short>
> ForEachTestsParams;

TYPED_TEST_CASE(ForEachTests, ForEachTestsParams);

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        0, 1, 2, 12, 63, 64, 211, 256, 344,
        1024, 2048, 5096, 34567, (1 << 17) - 1220
    };
    return sizes;
}

template <typename T>
struct mark_processed_functor
{
    T * ptr;
    __host__ __device__ void operator()(T x){ ptr[static_cast<int>(x)] = 1; }
};

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
TYPED_TEST(ForEachTests, HostPathSimpleTest)
{
  thrust::device_system_tag tag;
  using T = typename TestFixture::input_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    auto ptr = thrust::malloc<T>(tag, sizeof(T) * size);
    auto raw_ptr = thrust::raw_pointer_cast(ptr);
    if(size > 0) ASSERT_NE(raw_ptr, nullptr);

    // Zero input memory
    if(size > 0) HIP_CHECK(hipMemset(raw_ptr, 0, sizeof(T) * size));

    // Create unary function
    mark_processed_functor<T> func;
    func.ptr = raw_ptr;

    // Run for_each in [0; end] range
    auto end = size < 2 ? size : size/2;
    auto result = thrust::for_each(
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(end),
      func
    );
    ASSERT_EQ(result, thrust::make_counting_iterator<size_t>(end));

    std::vector<T> output(size);
    HIP_CHECK(
      hipMemcpy(
        output.data(), raw_ptr,
        size * sizeof(T),
        hipMemcpyDeviceToHost
      )
    );

    for(size_t i = 0; i < size; i++)
    {
      if(i < end)
      {
        ASSERT_EQ(output[i], T(1)) << "where index = " << i;
      }
      else
      {
        ASSERT_EQ(output[i], T(0)) << "where index = " << i;
      }
    }

    // Free
    thrust::free(tag, ptr);
  }
}

template<class F>
__global__
void simple_test_kernel(F func, int size)
{
  // (void) func; (void) size;
  thrust::for_each(
    thrust::seq,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(size),
    func
  );
}

TYPED_TEST(ForEachTests, DevicePathSimpleTest)
{
  thrust::device_system_tag tag;
  using T = typename TestFixture::input_type;
  const size_t size = 1024;

  auto ptr = thrust::malloc<T>(tag, sizeof(T) * size);
  auto raw_ptr = thrust::raw_pointer_cast(ptr);
  ASSERT_NE(raw_ptr, nullptr);

  // Zero input memory
  HIP_CHECK(hipMemset(raw_ptr, 0, sizeof(T) * size));

  // Create unary function
  mark_processed_functor<T> func;
  func.ptr = raw_ptr;

  // Run for_each in [0; end] range
  size_t end = 375;
  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(simple_test_kernel<mark_processed_functor<T>>),
    dim3(1), dim3(1), 0, 0,
    func, static_cast<int>(end)
  );

  std::vector<T> output(size);
  HIP_CHECK(
    hipMemcpy(
      output.data(), raw_ptr,
      size * sizeof(T),
      hipMemcpyDeviceToHost
    )
  );

  for(size_t i = 0; i < size; i++)
  {
    if(i < end)
    {
      ASSERT_EQ(output[i], T(1)) << "where index = " << i;
    }
    else
    {
      ASSERT_EQ(output[i], T(0)) << "where index = " << i;
    }
  }

  // Free
  thrust::free(tag, ptr);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
