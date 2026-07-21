// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file Double.hpp
 * @brief Forwarding functions for double type to enable uniform unqualified calls.
 *
 * This header provides forwarding functions that delegate to std:: implementations
 * for double type. This enables code using `using namespace hipdnn_data_sdk::types`
 * or `using hipdnn_data_sdk::types::fabs` to work uniformly with float, double,
 * and custom types like bfloat16 and half.
 *
 * All functions simply forward to the corresponding std:: function.
 */

#include <algorithm>
#include <cmath>

namespace hipdnn_data_sdk::types
{

// ============================================================================
// Forwarding functions for double
// ============================================================================
// These allow unqualified calls (via ADL or using-directives) to work
// uniformly for double and custom types.
// ============================================================================

// Basic math functions
inline double abs(double x)
{
    return std::abs(x);
}
inline double fabs(double x)
{
    return std::fabs(x);
}
inline bool isnan(double x)
{
    return std::isnan(x);
}
inline bool isinf(double x)
{
    return std::isinf(x);
}
inline bool signbit(double x)
{
    return std::signbit(x);
}
inline bool isfinite(double x)
{
    return std::isfinite(x);
}
inline double copysign(double x, double y)
{
    return std::copysign(x, y);
}

// Min/max functions
inline double max(double a, double b)
{
    return std::max(a, b);
}
inline double min(double a, double b)
{
    return std::min(a, b);
}
inline double fmax(double a, double b)
{
    return std::fmax(a, b);
}
inline double fmin(double a, double b)
{
    return std::fmin(a, b);
}

// Rounding functions
inline double floor(double x)
{
    return std::floor(x);
}
inline double ceil(double x)
{
    return std::ceil(x);
}
inline double round(double x)
{
    return std::round(x);
}
inline double trunc(double x)
{
    return std::trunc(x);
}

// Exponential and logarithmic functions
inline double exp(double x)
{
    return std::exp(x);
}
inline double exp2(double x)
{
    return std::exp2(x);
}
inline double log(double x)
{
    return std::log(x);
}
inline double log2(double x)
{
    return std::log2(x);
}
inline double log10(double x)
{
    return std::log10(x);
}

// Power functions
inline double sqrt(double x)
{
    return std::sqrt(x);
}
inline double rsqrt(double x)
{
    return 1.0 / std::sqrt(x);
}
inline double pow(double x, double y)
{
    return std::pow(x, y);
}

// Trigonometric functions
inline double sin(double x)
{
    return std::sin(x);
}
inline double cos(double x)
{
    return std::cos(x);
}
inline double tan(double x)
{
    return std::tan(x);
}
inline double asin(double x)
{
    return std::asin(x);
}
inline double acos(double x)
{
    return std::acos(x);
}
inline double atan(double x)
{
    return std::atan(x);
}

// Hyperbolic functions
inline double sinh(double x)
{
    return std::sinh(x);
}
inline double cosh(double x)
{
    return std::cosh(x);
}
inline double tanh(double x)
{
    return std::tanh(x);
}

// Error function
inline double erf(double x)
{
    return std::erf(x);
}

// Floating-point manipulation
inline double fmod(double x, double y)
{
    return std::fmod(x, y);
}

// Fused multiply-add
inline double fma(double x, double y, double z)
{
    return std::fma(x, y, z);
}

} // namespace hipdnn_data_sdk::types
