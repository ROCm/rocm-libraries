/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
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
#include <thrust/detail/complex/math_private.h>

#include _THRUST_STD_INCLUDE(cmath)
#include _THRUST_STD_INCLUDE(limits)

THRUST_NAMESPACE_BEGIN
namespace detail::complex
{
THRUST_HOST_DEVICE inline complex<float> cprojf(const complex<float>& z)
{
  if (!_THRUST_STD::isinf(z.real()) && !_THRUST_STD::isinf(z.imag()))
  {
    return z;
  }
  else
  {
    return complex<float>(_THRUST_STD::numeric_limits<float>::infinity(), _THRUST_STD::copysignf(0.0, z.imag()));
  }
}

THRUST_HOST_DEVICE inline complex<double> cproj(const complex<double>& z)
{
  if (!_THRUST_STD::isinf(z.real()) && !_THRUST_STD::isinf(z.imag()))
  {
    return z;
  }
  else
  {
    return complex<double>(_THRUST_STD::numeric_limits<double>::infinity(), _THRUST_STD::copysign(0.0, z.imag()));
  }
}
} // namespace detail::complex

template <typename T>
THRUST_HOST_DEVICE inline thrust::complex<T> proj(const thrust::complex<T>& z)
{
  return detail::complex::cproj(z);
}

template <>
THRUST_HOST_DEVICE inline thrust::complex<double> proj(const thrust::complex<double>& z)
{
  return detail::complex::cproj(z);
}

template <>
THRUST_HOST_DEVICE inline thrust::complex<float> proj(const thrust::complex<float>& z)
{
  return detail::complex::cprojf(z);
}

THRUST_NAMESPACE_END
