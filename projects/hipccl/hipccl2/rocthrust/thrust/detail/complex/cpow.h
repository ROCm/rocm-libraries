/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *  Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> pow(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return exp(log(complex<T>(x)) * complex<T>(y));
}

// Specialized version for complex<float> on AMD GPUs to use FMA-based multiplication
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
namespace detail
{
// FMA-aware complex multiplication for float precision on AMD GPUs.
// This prevents SLP vectorizer from breaking FMA formation, which causes
// numerical precision loss in complex arithmetic.
// The issue occurs when vectorizer packs scalar multiplies before backend
// can form FMA instructions, resulting in double rounding instead of single.
THRUST_HOST_DEVICE inline thrust::complex<float> complex_mul_fma(thrust::complex<float> a, thrust::complex<float> b)
{
  // Complex multiplication: (a.r + a.i*i) * (b.r + b.i*i)
  // = (a.r*b.r - a.i*b.i) + (a.r*b.i + a.i*b.r)*i
  // Using __builtin_fmaf ensures FMA at source level:
  // real: a.r*b.r + (-(a.i*b.i)) = FMA(a.r, b.r, -(a.i*b.i))
  // imag: a.i*b.r + a.r*b.i = FMA(a.r, b.i, a.i*b.r)
  float real_part = __builtin_fmaf(a.real(), b.real(), -(a.imag() * b.imag()));
  float imag_part = __builtin_fmaf(a.real(), b.imag(), a.imag() * b.real());
  return thrust::complex<float>(real_part, imag_part);
}
} // namespace detail

template <>
THRUST_HOST_DEVICE inline complex<float> pow(const complex<float>& x, const complex<float>& y)
{
  auto log_x   = thrust::log(static_cast<thrust::complex<float>>(x));
  auto y_log_x = detail::complex_mul_fma(static_cast<thrust::complex<float>>(y), log_x);
  return static_cast<complex<float>>(thrust::exp(y_log_x));
}
#endif

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> pow(const complex<T0>& x, const T1& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return exp(log(complex<T>(x)) * T(y));
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> pow(const T0& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  // Find `log` by ADL.
  using std::log;
  return exp(log(T(x)) * complex<T>(y));
}

THRUST_NAMESPACE_END
