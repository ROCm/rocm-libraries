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
#include <thrust/complex.h>


#include <complex>
#include <iostream>
#include <sstream>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"
#include "test_assertations.hpp"

template<
    class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class ComplexTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
    Params<float>,
    Params<double>
> ComplexTestsParams;

TYPED_TEST_CASE(ComplexTests, ComplexTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(ComplexTests, TestComplexConstructors)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data = get_random_data<T>(2,
                                                   std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());


  thrust::complex<T> a(data[0],data[1]);
  thrust::complex<T> b(a);
  a = thrust::complex<T>(data[0],data[1]);
  ASSERT_NEAR_COMPLEX(a, b);

  a = thrust::complex<T>(data[0]);
  ASSERT_EQ(data[0], a.real());
  ASSERT_EQ(T(0), a.imag());

  a = thrust::complex<T>();
  ASSERT_NEAR_COMPLEX(a,std::complex<T>(0));

  a = thrust::complex<T>(thrust::complex<float>(data[0],data[1]));
  ASSERT_NEAR_COMPLEX(a, b);

  a = thrust::complex<T>(thrust::complex<double>(data[0],data[1]));
  ASSERT_NEAR_COMPLEX(a, b);

  a = thrust::complex<T>(std::complex<float>(data[0],data[1]));
  ASSERT_NEAR_COMPLEX(a, b);

  a = thrust::complex<T>(std::complex<double>(data[0],data[1]));
  ASSERT_NEAR_COMPLEX(a, b);
  ASSERT_NEAR_COMPLEX(a, b);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
