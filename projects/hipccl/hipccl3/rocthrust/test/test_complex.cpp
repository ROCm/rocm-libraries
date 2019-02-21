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
#include "test_assertions.hpp"

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

TYPED_TEST(ComplexTests, TestComplexGetters)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data = get_random_data<T>(2,
                                                   std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());

  thrust::complex<T> z(data[0], data[1]);

  ASSERT_EQ(data[0], z.real());
  ASSERT_EQ(data[1], z.imag());

  z.real(data[1]);
  z.imag(data[0]);
  ASSERT_EQ(data[1], z.real());
  ASSERT_EQ(data[0], z.imag());

  volatile thrust::complex<T> v(data[0], data[1]);

  ASSERT_EQ(data[0], v.real());
  ASSERT_EQ(data[1], v.imag());

  v.real(data[1]);
  v.imag(data[0]);
  ASSERT_EQ(data[1], v.real());
  ASSERT_EQ(data[0], v.imag());
}

TYPED_TEST(ComplexTests, TestComplexMemberOperators)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::host_vector<T> data_b = get_random_data<T>(2,
                                                    std::numeric_limits<T>::min(),
                                                    std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);
  thrust::complex<T> b(data_b[0], data_b[1]);

  std::complex<T> c(a);
  std::complex<T> d(b);

  a += b;
  c += d;
  ASSERT_NEAR_COMPLEX(a,c);

  a -= b;
  c -= d;
  ASSERT_NEAR_COMPLEX(a,c);

  a *= b;
  c *= d;
  ASSERT_NEAR_COMPLEX(a,c);

  a /= b;
  c /= d;
  ASSERT_NEAR_COMPLEX(a,c);

  // casting operator
  c = (std::complex<T>)a;
}

TYPED_TEST(ComplexTests, TestComplexBasicArithmetic)
{
  using T = typename TestFixture::input_type;


  thrust::host_vector<T> data = get_random_data<T>(2,
                                                   std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());

  thrust::complex<T> a(data[0], data[1]);
  std::complex<T> b(a);

  // Test the basic arithmetic functions against std

  ASSERT_NEAR(abs(a),abs(b),std::numeric_limits<T>::epsilon());

  ASSERT_NEAR(arg(a),arg(b),std::numeric_limits<T>::epsilon());

  ASSERT_NEAR(norm(a),norm(b),std::numeric_limits<T>::epsilon());

  ASSERT_EQ(conj(a),conj(b));

  // TODO: Find a good assert for this.
  //ASSERT_NEAR(thrust::polar(data[0],data[1]),std::polar(data[0],data[1]));

  // random_samples does not seem to produce infinities so proj(z) == z
  ASSERT_EQ(proj(a),a);
}

TYPED_TEST(ComplexTests, TestComplexBinaryArithmetic)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::host_vector<T> data_b = get_random_data<T>(2,
                                                    std::numeric_limits<T>::min(),
                                                    std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);
  thrust::complex<T> b(data_b[0], data_b[1]);

  ASSERT_NEAR_COMPLEX(a * b,std::complex<T>(a) * std::complex<T>(b));
  ASSERT_NEAR_COMPLEX(a * data_b[0],std::complex<T>(a) * data_b[0]);
  ASSERT_NEAR_COMPLEX(data_a[0]*b,data_b[0] * std::complex<T>(b));

  ASSERT_NEAR_COMPLEX(a / b, std::complex<T>(a) / std::complex<T>(b));
  ASSERT_NEAR_COMPLEX(a / data_b[0], std::complex<T>(a) / data_b[0]);
  ASSERT_NEAR_COMPLEX(data_a[0] / b, data_b[0] / std::complex<T>(b));

  ASSERT_EQ(a + b, std::complex<T>(a) + std::complex<T>(b));
  ASSERT_EQ(a + data_b[0], std::complex<T>(a) + data_b[0]);
  ASSERT_EQ(data_a[0] + b, data_b[0] + std::complex<T>(b));

  ASSERT_EQ(a - b, std::complex<T>(a) - std::complex<T>(b));
  ASSERT_EQ(a - data_b[0], std::complex<T>(a) - data_b[0]);
  ASSERT_EQ(data_a[0] - b, data_b[0] - std::complex<T>(b));
}

TYPED_TEST(ComplexTests, TestComplexUnaryArithmetic)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);

  ASSERT_EQ(+a,+std::complex<T>(a));
  ASSERT_EQ(-a,-std::complex<T>(a));
}

TYPED_TEST(ComplexTests, TestComplexExponentialFunctions)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);
  std::complex<T> b(a);

  ASSERT_NEAR_COMPLEX(exp(a),exp(b));
  ASSERT_NEAR_COMPLEX(log(a),log(b));
  ASSERT_NEAR_COMPLEX(log10(a),log10(b));
}

TYPED_TEST(ComplexTests, TestComplexPowerFunctions)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::host_vector<T> data_b = get_random_data<T>(2,
                                                    std::numeric_limits<T>::min(),
                                                    std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);
  thrust::complex<T> b(data_b[0], data_b[1]);
  std::complex<T> c(a);
  std::complex<T> d(b);

  ASSERT_NEAR_COMPLEX(pow(a,b),pow(c,d));
  ASSERT_NEAR_COMPLEX(pow(a,b.real()),pow(c,d.real()));
  ASSERT_NEAR_COMPLEX(pow(a.real(),b),pow(c.real(),d));

  ASSERT_NEAR_COMPLEX(sqrt(a),sqrt(c));
}

TYPED_TEST(ComplexTests, TestComplexTrigonometricFunctions)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);
  std::complex<T> c(a);

  ASSERT_NEAR_COMPLEX(cos(a),cos(c));
  ASSERT_NEAR_COMPLEX(sin(a),sin(c));
  ASSERT_NEAR_COMPLEX(tan(a),tan(c));
  ASSERT_NEAR_COMPLEX(cosh(a),cosh(c));
  ASSERT_NEAR_COMPLEX(sinh(a),sinh(c));
  ASSERT_NEAR_COMPLEX(tanh(a),tanh(c));

  ASSERT_NEAR_COMPLEX(acos(a),acos(c));
  ASSERT_NEAR_COMPLEX(asin(a),asin(c));
  ASSERT_NEAR_COMPLEX(atan(a),atan(c));
  ASSERT_NEAR_COMPLEX(acosh(a),acosh(c));
  ASSERT_NEAR_COMPLEX(asinh(a),asinh(c));
  ASSERT_NEAR_COMPLEX(atanh(a),atanh(c));
}

TYPED_TEST(ComplexTests, TestComplexStreamOperators)
{
  using T = typename TestFixture::input_type;

  thrust::host_vector<T> data_a = get_random_data<T>(2,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());

  thrust::complex<T> a(data_a[0], data_a[1]);
  std::stringstream out;
  out << a;
  thrust::complex<T> b;
  out >> b;
  ASSERT_NEAR_COMPLEX(a,b);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
