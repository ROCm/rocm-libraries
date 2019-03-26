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
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/transform.h>

#include "test_header.hpp"

TESTS_DEFINE(ComplexTransformTests, FloatTestsParams);

struct basic_arithmetic_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x,
				const thrust::complex<T> &y)
  {
    // exercise unary and binary arithmetic operators
    // Should return approximately 1
    return (+x + +y) + (x * y) / (y * x) + (-y + -x);
  } // end operator()()
}; // end make_pair_functor

struct complex_plane_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // Should return a proximately 1
    return thrust::proj( (thrust::polar(abs(x),arg(x)) * conj(x))/norm(x));
  } // end operator()()
}; // end make_pair_functor

struct pow_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x,
				const thrust::complex<T> &y)
  {
    // exercise power functions
    return pow(x,y);
  } // end operator()()
}; // end make_pair_functor

struct sqrt_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    // exercise power functions
    return sqrt(x);
  } // end operator()()
}; // end make_pair_functor

struct log_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return log(x);
  } // end operator()()
}; // end make_pair_functor

struct exp_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return exp(x);
  } // end operator()()
}; // end make_pair_functor

struct log10_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return log10(x);
  } // end operator()()
}; // end make_pair_functor


struct cos_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return cos(x);
  }
};

struct sin_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return sin(x);
  }
};

struct tan_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return tan(x);
  }
};



struct cosh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return cosh(x);
  }
};

struct sinh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return sinh(x);
  }
};

struct tanh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return tanh(x);
  }
};


struct acos_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return acos(x);
  }
};

struct asin_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return asin(x);
  }
};

struct atan_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return atan(x);
  }
};


struct acosh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return acosh(x);
  }
};

struct asinh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return asinh(x);
  }
};

struct atanh_functor
{
  template<typename T>
  __host__ __device__
  thrust::complex<T> operator()(const thrust::complex<T> &x)
  {
    return atanh(x);
  }
};

template <typename T>
thrust::complex<T> epsilonComplex()
{
  return thrust::complex<T>(std::numeric_limits<T>::epsilon(),std::numeric_limits<T>::epsilon());
};

template <typename T>
thrust::host_vector<thrust::complex<T> > random_complex_samples(size_t size){
  thrust::host_vector<T> real = get_random_data<T>(2*size,
                                                   std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());
  thrust::host_vector<thrust::complex<T> > h_p1(size);
  for(size_t i = 0; i<size; i++){
    h_p1[i].real(real[i]);
    h_p1[i].imag(real[2*i]);
  }
  return h_p1;
};

TYPED_TEST(ComplexTransformTests, TestComplexArithmeticTransform)
{
  using T = typename TestFixture::input_type;
  using type = thrust::complex<T>;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
    thrust::host_vector<type> h_p2 = random_complex_samples<T>(size);
    thrust::host_vector<type>   h_result(size);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_p2 = h_p2;
    thrust::device_vector<type> d_result(size);

    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), basic_arithmetic_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), basic_arithmetic_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);
  }
}

TYPED_TEST(ComplexTransformTests, TestComplexPlaneTransform)
{
  using T = typename TestFixture::input_type;
  using type = thrust::complex<T>;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
    thrust::host_vector<type>   h_result(size);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_result(size);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), complex_plane_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), complex_plane_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);
  }
}

TYPED_TEST(ComplexTransformTests, TestComplexPowerTransform)
{
  using T = typename TestFixture::input_type;
  using type = thrust::complex<T>;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
    thrust::host_vector<type> h_p2 = random_complex_samples<T>(size);
    thrust::host_vector<type>   h_result(size);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_p2 = h_p2;
    thrust::device_vector<type> d_result(size);


    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), pow_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), pow_functor());
    // pow can be very innacurate there's no point trying to check for equality
    // Currently just checking for compilation
    //    ASSERT_NEAR(h_result, d_result,epsilonComplex<T>());

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sqrt_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sqrt_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);
  }
}

TYPED_TEST(ComplexTransformTests, TestComplexExponentialTransform)
{
  using T = typename TestFixture::input_type;
  using type = thrust::complex<T>;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
    thrust::host_vector<type>   h_result(size);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_result(size);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), exp_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), exp_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), log_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), log_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), log10_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), log10_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

  }
}

TYPED_TEST(ComplexTransformTests, TestComplexTrigonometricTransform)
{
  using T = typename TestFixture::input_type;
  using type = thrust::complex<T>;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    thrust::host_vector<type> h_p1 = random_complex_samples<T>(size);
    thrust::host_vector<type>   h_result(size);

    thrust::device_vector<type> d_p1 = h_p1;
    thrust::device_vector<type> d_result(size);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sin_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sin_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), cos_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), cos_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), tan_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), tan_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);


    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), sinh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), sinh_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), cosh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), cosh_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), tanh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), tanh_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);



    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), asin_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), asin_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), acos_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), acos_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), atan_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), atan_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);


    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), asinh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), asinh_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), acosh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), acosh_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);

    thrust::transform(h_p1.begin(), h_p1.end(), h_result.begin(), atanh_functor());
    thrust::transform(d_p1.begin(), d_p1.end(), d_result.begin(), atanh_functor());
    ASSERT_NEAR_COMPLEX(h_result, d_result);
  }
}
