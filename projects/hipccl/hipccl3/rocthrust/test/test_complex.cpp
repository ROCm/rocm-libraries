/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/complex.h>

#include "test_header.hpp"

TESTS_DEFINE(ComplexTests, FloatTestsParams);

TYPED_TEST(ComplexTests, TestComplexConstructors)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data = get_random_data<T>(2, T(-1000), T(1000));

    thrust::complex<T>     a(data[0], data[1]);
    thrust::complex<T>     b(a);
    thrust::complex<float> float_b(a);
    a = thrust::complex<T>(data[0], data[1]);
    ASSERT_NEAR_COMPLEX(a, b);

    a = thrust::complex<T>(data[0]);
    ASSERT_EQ(data[0], a.real());
    ASSERT_EQ(T(0), a.imag());

    a = thrust::complex<T>();
    ASSERT_NEAR_COMPLEX(a, std::complex<T>(0));

    a = thrust::complex<T>(
        thrust::complex<float>(static_cast<float>(data[0]), static_cast<float>(data[1])));
    ASSERT_NEAR_COMPLEX(a, float_b);

    a = thrust::complex<T>(
        thrust::complex<double>(static_cast<double>(data[0]), static_cast<double>(data[1])));
    ASSERT_NEAR_COMPLEX(a, b);

    a = thrust::complex<T>(
        std::complex<float>(static_cast<float>(data[0]), static_cast<float>(data[1])));
    ASSERT_NEAR_COMPLEX(a, float_b);

    a = thrust::complex<T>(
        std::complex<double>(static_cast<double>(data[0]), static_cast<double>(data[1])));
    ASSERT_NEAR_COMPLEX(a, b);
}

TYPED_TEST(ComplexTests, TestComplexGetters)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data
        = get_random_data<T>(2, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

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

    thrust::host_vector<T> data_a = get_random_data<T>(2, 10000, 10000);

    thrust::host_vector<T> data_b = get_random_data<T>(2, 10000, 10000);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    std::complex<T> c(a);
    std::complex<T> d(b);

    a += b;
    c += d;
    ASSERT_NEAR_COMPLEX(a, c);

    a -= b;
    c -= d;
    ASSERT_NEAR_COMPLEX(a, c);

    a *= b;
    c *= d;
    ASSERT_NEAR_COMPLEX(a, c);

    a /= b;
    c /= d;
    ASSERT_NEAR_COMPLEX(a, c);

    // casting operator
    c = (std::complex<T>)a;
}

TYPED_TEST(ComplexTests, TestComplexBasicArithmetic)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data = get_random_data<T>(2, T(-100), T(100));

    thrust::complex<T> a(data[0], data[1]);
    std::complex<T>    b(a);

    // Test the basic arithmetic functions against std

    ASSERT_NEAR(abs(a), abs(b), T(0.01));

    ASSERT_NEAR(arg(a), arg(b), T(0.01));

    ASSERT_NEAR(norm(a), norm(b), T(0.01));

    ASSERT_EQ(conj(a), conj(b));

    ASSERT_NEAR_COMPLEX(thrust::polar(data[0], data[1]), std::polar(data[0], data[1]));

    // random_samples does not seem to produce infinities so proj(z) == z
    ASSERT_EQ(proj(a), a);
}

TYPED_TEST(ComplexTests, TestComplexBinaryArithmetic)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data_a = get_random_data<T>(2, -10000, 10000);

    thrust::host_vector<T> data_b = get_random_data<T>(2, -10000, 10000);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    ASSERT_NEAR_COMPLEX(a * b, std::complex<T>(a) * std::complex<T>(b));
    ASSERT_NEAR_COMPLEX(a * data_b[0], std::complex<T>(a) * data_b[0]);
    ASSERT_NEAR_COMPLEX(data_a[0] * b, data_a[0] * std::complex<T>(b));

    ASSERT_NEAR_COMPLEX(a / b, std::complex<T>(a) / std::complex<T>(b));
    ASSERT_NEAR_COMPLEX(a / data_b[0], std::complex<T>(a) / data_b[0]);
    ASSERT_NEAR_COMPLEX(data_a[0] / b, data_a[0] / std::complex<T>(b));

    ASSERT_EQ(a + b, std::complex<T>(a) + std::complex<T>(b));
    ASSERT_EQ(a + data_b[0], std::complex<T>(a) + data_b[0]);
    ASSERT_EQ(data_a[0] + b, data_a[0] + std::complex<T>(b));

    ASSERT_EQ(a - b, std::complex<T>(a) - std::complex<T>(b));
    ASSERT_EQ(a - data_b[0], std::complex<T>(a) - data_b[0]);
    ASSERT_EQ(data_a[0] - b, data_a[0] - std::complex<T>(b));
}

TYPED_TEST(ComplexTests, TestComplexUnaryArithmetic)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data_a
        = get_random_data<T>(2, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    thrust::complex<T> a(data_a[0], data_a[1]);

    ASSERT_EQ(+a, +std::complex<T>(a));
    ASSERT_EQ(-a, -std::complex<T>(a));
}

TYPED_TEST(ComplexTests, TestComplexExponentialFunctions)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data_a = get_random_data<T>(2, -100, 100);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T>    b(a);

    ASSERT_NEAR_COMPLEX(exp(a), exp(b));
    ASSERT_NEAR_COMPLEX(log(a), log(b));
    ASSERT_NEAR_COMPLEX(log10(a), log10(b));
}

TYPED_TEST(ComplexTests, TestComplexPowerFunctions)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data_a = get_random_data<T>(2, -100, 100);

    thrust::host_vector<T> data_b = get_random_data<T>(2, -100, 100);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);
    std::complex<T>    c(a);
    std::complex<T>    d(b);

    ASSERT_NEAR_COMPLEX(pow(a, b), pow(c, d));
    ASSERT_NEAR_COMPLEX(pow(a, b.real()), pow(c, d.real()));
    ASSERT_NEAR_COMPLEX(pow(a.real(), b), pow(c.real(), d));

    ASSERT_NEAR_COMPLEX(sqrt(a), sqrt(c));
}

TYPED_TEST(ComplexTests, TestComplexTrigonometricFunctions)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data_a = get_random_data<T>(2, T(-1), T(1));

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T>    c(a);

    ASSERT_NEAR_COMPLEX(cos(a), cos(c));
    ASSERT_NEAR_COMPLEX(sin(a), sin(c));
    ASSERT_NEAR_COMPLEX(tan(a), tan(c));
    ASSERT_NEAR_COMPLEX(cosh(a), cosh(c));
    ASSERT_NEAR_COMPLEX(sinh(a), sinh(c));
    ASSERT_NEAR_COMPLEX(tanh(a), tanh(c));

    ASSERT_NEAR_COMPLEX(acos(a), acos(c));
    ASSERT_NEAR_COMPLEX(asin(a), asin(c));
    ASSERT_NEAR_COMPLEX(atan(a), atan(c));
    ASSERT_NEAR_COMPLEX(acosh(a), acosh(c));
    ASSERT_NEAR_COMPLEX(asinh(a), asinh(c));
    ASSERT_NEAR_COMPLEX(atanh(a), atanh(c));
}

TYPED_TEST(ComplexTests, TestComplexStreamOperators)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data_a = get_random_data<T>(2, T(-1000), T(1000));

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::stringstream  out;
    out << a;
    thrust::complex<T> b;
    out >> b;
    ASSERT_NEAR_COMPLEX(a, b);
}
