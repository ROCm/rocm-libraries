/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include _THRUST_STD_INCLUDE(cfloat)
#include _THRUST_STD_INCLUDE(cmath)

THRUST_NAMESPACE_BEGIN

/* --- Binary Arithmetic Operators --- */

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator+(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() + y.real(), x.imag() + y.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator+(const complex<T0>& x, const T1& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() + y, x.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator+(const T0& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x + y.real(), y.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator-(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() - y.real(), x.imag() - y.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator-(const complex<T0>& x, const T1& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() - y, x.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator-(const T0& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x - y.real(), -y.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator*(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() * y.real() - x.imag() * y.imag(), x.real() * y.imag() + x.imag() * y.real());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator*(const complex<T0>& x, const T1& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() * y, x.imag() * y);
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator*(const T0& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x * y.real(), x * y.imag());
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator/(const complex<T0>& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  T s     = _THRUST_STD::abs(y.real()) + _THRUST_STD::abs(y.imag());

  T oos = T(1.0) / s;

  T ars = x.real() * oos;
  T ais = x.imag() * oos;
  T brs = y.real() * oos;
  T bis = y.imag() * oos;

  s = (brs * brs) + (bis * bis);

  oos = T(1.0) / s;

  complex<T> quot(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
  return quot;
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator/(const complex<T0>& x, const T1& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x.real() / y, x.imag() / y);
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> operator/(const T0& x, const complex<T1>& y)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(x) / y;
}

/* --- Unary Arithmetic Operators --- */

template <typename T>
THRUST_HOST_DEVICE complex<T> operator+(const complex<T>& y)
{
  return y;
}

template <typename T>
THRUST_HOST_DEVICE complex<T> operator-(const complex<T>& y)
{
  return y * -T(1);
}

/* --- Other Basic Arithmetic Functions --- */

// As std::hypot is only C++11 we have to use the C interface
template <typename T>
THRUST_HOST_DEVICE T abs(const complex<T>& z)
{
  return _THRUST_STD::hypot(z.real(), z.imag());
}

template <typename T>
THRUST_HOST_DEVICE T arg(const complex<T>& z)
{
  return _THRUST_STD::atan2(z.imag(), z.real());
}

template <typename T>
THRUST_HOST_DEVICE complex<T> conj(const complex<T>& z)
{
  return complex<T>(z.real(), -z.imag());
}

template <typename T>
THRUST_HOST_DEVICE T norm(const complex<T>& z)
{
  return z.real() * z.real() + z.imag() * z.imag();
}

// XXX Why specialize these, we could just rely on ADL.
template <>
THRUST_HOST_DEVICE inline float norm(const complex<float>& z)
{
  if (_THRUST_STD::abs(z.real()) < _THRUST_STD::sqrt(FLT_MIN)
      && _THRUST_STD::abs(z.imag()) < _THRUST_STD::sqrt(FLT_MIN))
  {
    float a = z.real() * 4.0f;
    float b = z.imag() * 4.0f;
    return (a * a + b * b) / 16.0f;
  }

  return z.real() * z.real() + z.imag() * z.imag();
}

template <>
THRUST_HOST_DEVICE inline double norm(const complex<double>& z)
{
  if (_THRUST_STD::abs(z.real()) < _THRUST_STD::sqrt(DBL_MIN)
      && _THRUST_STD::abs(z.imag()) < _THRUST_STD::sqrt(DBL_MIN))
  {
    double a = z.real() * 4.0;
    double b = z.imag() * 4.0;
    return (a * a + b * b) / 16.0;
  }

  return z.real() * z.real() + z.imag() * z.imag();
}

template <typename T0, typename T1>
THRUST_HOST_DEVICE complex<::internal::common_type_t<T0, T1>> polar(const T0& m, const T1& theta)
{
  using T = ::internal::common_type_t<T0, T1>;
  return complex<T>(m * _THRUST_STD::cos(theta), m * _THRUST_STD::sin(theta));
}

THRUST_NAMESPACE_END
