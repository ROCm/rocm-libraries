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
#include <thrust/transform.h>
// STREAMHPC TODO replace <thrust/detail/seq.h> with <thrust/execution_policy.h>
#include <thrust/detail/seq.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/iterator/counting_iterator.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<
    class InputType,
    class OutputType = InputType
>
struct Params
{
    using input_type = InputType;
    using output_type = OutputType;
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
    Params<unsigned short>
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
struct transform
{
    __device__ __host__ inline
    constexpr T operator()(const T& a) const
    {
        return a + 5;
    }
};

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(TransformTests, UnaryTransform)
{
  using T = typename TestFixture::input_type;
  using U = typename TestFixture::output_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    std::vector<T> input(size, 0);
    std::vector<U> output(size, 0);

    thrust::device_ptr<T> input_ptr = thrust::device_malloc<T>(size);
    thrust::device_ptr<U> output_ptr = thrust::device_malloc<U>(size);

    for(size_t i = 0; i < size; i++)
    {
        input[i] = i;
    }

    hipMemcpy(
        thrust::raw_pointer_cast(input_ptr), input.data(),
        input.size() * sizeof(T),
        hipMemcpyHostToDevice
    );

    // Calculate expected results on host
    std::vector<U> expected(input.size());
    std::transform(input.begin(), input.end(), expected.begin(), transform<U>());

    thrust::transform(input_ptr, input_ptr + size, output_ptr, transform<U>());

    hipMemcpy(
        output.data(), thrust::raw_pointer_cast(output_ptr),
        size * sizeof(U),
        hipMemcpyDeviceToHost
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    // Free
    thrust::device_free(input_ptr);
    thrust::device_free(output_ptr);
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
