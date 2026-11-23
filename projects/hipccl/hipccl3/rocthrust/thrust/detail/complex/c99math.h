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

#include <thrust/detail/complex/math_private.h>

#include _THRUST_STD_INCLUDE(cmath)

#include <math.h>

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

using _THRUST_STD::acos;
using _THRUST_STD::asin;
using _THRUST_STD::atan;
using _THRUST_STD::atanh;
using _THRUST_STD::copysign;
using _THRUST_STD::cos;
using _THRUST_STD::cosh;
using _THRUST_STD::exp;
using _THRUST_STD::hypot;
using _THRUST_STD::isfinite;
using _THRUST_STD::isinf;
using _THRUST_STD::isnan;
using _THRUST_STD::log;
using _THRUST_STD::log1p;
using _THRUST_STD::signbit;
using _THRUST_STD::sin;
using _THRUST_STD::sinh;
using _THRUST_STD::sqrt;
using _THRUST_STD::tan;

} // namespace complex

} // namespace detail

THRUST_NAMESPACE_END
