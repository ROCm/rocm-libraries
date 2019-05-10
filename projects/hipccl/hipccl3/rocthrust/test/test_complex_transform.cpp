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
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include "test_header.hpp"

TESTS_DEFINE(ComplexTransformTests, FloatTestsParams);

struct basic_arithmetic_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x,
                                                      const thrust::complex<T>& y)
    {
        // exercise unary and binary arithmetic operators
        // Should return approximately 1
        return (+x + +y) + (x * y) / (y * x) + (-y + -x);
    } // end operator()()
}; // end make_pair_functor

struct complex_plane_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        // Should return a proximately 1
        return thrust::proj((thrust::polar(abs(x), arg(x)) * conj(x)) / norm(x));
    } // end operator()()
}; // end make_pair_functor

struct pow_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x,
                                                      const thrust::complex<T>& y)
    {
        // exercise power functions
        return pow(x, y);
    } // end operator()()
}; // end make_pair_functor

struct sqrt_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        // exercise power functions
        return sqrt(x);
    } // end operator()()
}; // end make_pair_functor

struct log_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return log(x);
    } // end operator()()
}; // end make_pair_functor

struct exp_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return exp(x);
    } // end operator()()
}; // end make_pair_functor

struct log10_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return log10(x);
    } // end operator()()
}; // end make_pair_functor

struct cos_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return cos(x);
    }
};

struct sin_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return sin(x);
    }
};

struct tan_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return tan(x);
    }
};

struct cosh_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return cosh(x);
    }
};

struct sinh_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return sinh(x);
    }
};

struct tanh_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return tanh(x);
    }
};

struct acos_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return acos(x);
    }
};

struct asin_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return asin(x);
    }
};

struct atan_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return atan(x);
    }
};

struct acosh_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return acosh(x);
    }
};

struct asinh_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return asinh(x);
    }
};

struct atanh_functor
{
    template <typename T>
    __host__ __device__ thrust::complex<T> operator()(const thrust::complex<T>& x)
    {
        return atanh(x);
    }
};

template <typename T>
thrust::complex<T> epsilonComplex()
{
    return thrust::complex<T>(std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::epsilon());
};

template <typename T>
thrust::host_vector<thrust::complex<T>> random_complex_samples(size_t size)
{
    thrust::host_vector<T> real = get_random_data<T>(
        2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    thrust::host_vector<thrust::complex<T>> h_p1(size);
    for(size_t i = 0; i < size; i++)
    {
        h_p1[i].real(real[i]);
        h_p1[i].imag(real[2 * i]);
    }
    return h_p1;
};

TYPED_TEST(ComplexTransformTests, TestComplexArithmeticTransform)
{
    using T    = typename TestFixture::input_type;
    using type = thrust::complex<T>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_p2 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_result(size);

        thrust::device_vector<type> d_p1 = h_p1;
        thrust::device_vector<type> d_p2 = h_p2;
        thrust::device_vector<type> d_result(size);

        thrust::transform(
            h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), basic_arithmetic_functor());
        thrust::transform(
            d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), basic_arithmetic_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);
    }
}

TYPED_TEST(ComplexTransformTests, TestComplexPlaneTransform)
{
    using T    = typename TestFixture::input_type;
    using type = thrust::complex<T>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_result(size);

        thrust::device_vector<type> d_p1 = h_p1;
        thrust::device_vector<type> d_result(size);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), complex_plane_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), complex_plane_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);
    }
}

TYPED_TEST(ComplexTransformTests, TestComplexPowerTransform)
{
    using T    = typename TestFixture::input_type;
    using type = thrust::complex<T>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_p2 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_result(size);

        thrust::device_vector<type> d_p1 = h_p1;
        thrust::device_vector<type> d_p2 = h_p2;
        thrust::device_vector<type> d_result(size);

        thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), pow_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), pow_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sqrt_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sqrt_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);
    }
}

TYPED_TEST(ComplexTransformTests, TestComplexExponentialTransform)
{
    using T    = typename TestFixture::input_type;
    using type = thrust::complex<T>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_result(size);

        thrust::device_vector<type> d_p1 = h_p1;
        thrust::device_vector<type> d_result(size);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), exp_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), exp_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), log_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), log_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), log10_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), log10_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);
    }
}

TYPED_TEST(ComplexTransformTests, TestComplexTrigonometricTransform)
{
    using T    = typename TestFixture::input_type;
    using type = thrust::complex<T>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
        thrust::host_vector<type> h_result(size);

        thrust::device_vector<type> d_p1 = h_p1;
        thrust::device_vector<type> d_result(size);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sin_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sin_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), cos_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), cos_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), tan_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), tan_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sinh_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sinh_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), cosh_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), cosh_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), tanh_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), tanh_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), asin_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), asin_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), acos_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), acos_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), atan_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), atan_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), asinh_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), asinh_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), acosh_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), acosh_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);

        thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), atanh_functor());
        thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), atanh_functor());
        ASSERT_NEAR_COMPLEX_VECTOR(h_result, d_result);
    }
}
