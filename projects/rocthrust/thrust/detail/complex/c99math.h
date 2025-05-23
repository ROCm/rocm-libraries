/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/complex/math_private.h>

#include <cmath>

#include <math.h> // IWYU pragma: export

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace complex
{

// Define basic arithmetic functions so we can use them without explicit scope
// keeping the code as close as possible to FreeBSDs for ease of maintenance.
// It also provides an easy way to support compilers with missing C99 functions.
// When possible, just use the names in the global scope.
// Some platforms define these as macros, others as free functions.
// Avoid using the std:: form of these as nvcc may treat std::foo() as __host__ functions.

using ::acos;
using ::asin;
using ::atan;
using ::cos;
using ::cosh;
using ::exp;
using ::log;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;

template <typename T>
inline THRUST_HOST_DEVICE T infinity();

template <>
inline THRUST_HOST_DEVICE float infinity<float>()
{
  float res;
  set_float_word(res, 0x7f800000);
  return res;
}

template <>
inline THRUST_HOST_DEVICE double infinity<double>()
{
  double res;
  insert_words(res, 0x7ff00000, 0);
  return res;
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  ifdef __HIP_DEVICE_COMPILE__
using ::atan2;
using ::cos;
using ::exp;
using ::log;
using ::sin;
using ::sqrt;
#  else
using std::atan2;
using std::cos;
using std::exp;
using std::log;
using std::sin;
using std::sqrt;
#  endif
#endif // HIP compiler

#if defined _MSC_VER
THRUST_HOST_DEVICE inline int isinf(float x)
{
  return std::abs(x) == infinity<float>();
}

THRUST_HOST_DEVICE inline int isinf(double x)
{
  return std::abs(x) == infinity<double>();
}

THRUST_HOST_DEVICE inline int isnan(float x)
{
  return x != x;
}

THRUST_HOST_DEVICE inline int isnan(double x)
{
  return x != x;
}

THRUST_HOST_DEVICE inline int signbit(float x)
{
  return ((*((uint32_t*) &x)) & 0x80000000) != 0 ? 1 : 0;
}

THRUST_HOST_DEVICE inline int signbit(double x)
{
  return ((*((uint64_t*) &x)) & 0x8000000000000000) != 0ull ? 1 : 0;
}

THRUST_HOST_DEVICE inline int isfinite(float x)
{
  return !isnan(x) && !isinf(x);
}

THRUST_HOST_DEVICE inline int isfinite(double x)
{
  return !isnan(x) && !isinf(x);
}

#else

#  if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__)) && !defined(_NVHPC_CUDA)
// NVCC implements at least some signature of these as functions not macros.
using ::isfinite;
using ::isinf;
using ::isnan;
using ::signbit;
#  else

#    ifdef __HIP_DEVICE_COMPILE__

// hip_runtime.h provides these functions in the global scope
using ::isfinite;
using ::isinf;
using ::isnan;
using ::signbit;

#    else

// Some compilers do not provide these in the global scope, because they are
// supposed to be macros. The versions in `std` are supposed to be functions.
// Since we're not compiling with nvcc, it's safe to use the functions in std::
using std::isfinite;
using std::isinf;
using std::isnan;
using std::signbit;
#    endif // __HIP_COMPILER__

#  endif // __CUDACC__
#endif // _MSC_VER

using ::atanh;

#if defined _MSC_VER

THRUST_HOST_DEVICE inline double copysign(double x, double y)
{
  uint32_t hx, hy;
  get_high_word(hx, x);
  get_high_word(hy, y);
  set_high_word(x, (hx & 0x7fffffff) | (hy & 0x80000000));
  return x;
}

THRUST_HOST_DEVICE inline float copysignf(float x, float y)
{
  uint32_t ix, iy;
  get_float_word(ix, x);
  get_float_word(iy, y);
  set_float_word(x, (ix & 0x7fffffff) | (iy & 0x80000000));
  return x;
}

#  if !defined(__CUDACC__) && !defined(_NVHPC_CUDA)

// Simple approximation to log1p as Visual Studio is lacking one
THRUST_HOST_DEVICE inline double log1p(double x)
{
  double u = 1.0 + x;
  if (u == 1.0)
  {
    return x;
  }
  else
  {
    if (u > 2.0)
    {
      // Use normal log for large arguments
      return log(u);
    }
    else
    {
      return log(u) * (x / (u - 1.0));
    }
  }
}

THRUST_HOST_DEVICE inline float log1pf(float x)
{
  float u = 1.0f + x;
  if (u == 1.0f)
  {
    return x;
  }
  else
  {
    if (u > 2.0f)
    {
      // Use normal log for large arguments
      return logf(u);
    }
    else
    {
      return logf(u) * (x / (u - 1.0f));
    }
  }
}

#  endif // __HIP__

#  if _MSC_VER <= 1500 && !defined(__clang__)
#    include <complex>

inline float hypotf(float x, float y)
{
  return abs(std::complex<float>(x, y));
}

inline double hypot(double x, double y)
{
  return _hypot(x, y);
}

#  endif // _MSC_VER <= 1500

#endif // __CUDACC__

} // namespace complex

} // namespace detail

THRUST_NAMESPACE_END
