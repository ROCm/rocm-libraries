// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation
// SPDX-FileCopyrightText: Copyright (c) 2013, Filipe RNC Maia
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>

#include _THRUST_STD_INCLUDE(__cmath/copysign.h)
#include _THRUST_STD_INCLUDE(__cmath/isinf.h)
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
