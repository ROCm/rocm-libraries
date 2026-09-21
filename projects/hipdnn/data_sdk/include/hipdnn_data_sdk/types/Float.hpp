// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file Float.hpp
 * @brief Forwarding functions for float type to enable uniform unqualified calls.
 *
 * This header provides forwarding functions that delegate to std:: implementations
 * for float type. This enables code using `using namespace hipdnn_data_sdk::types`
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
// Forwarding functions for float
// ============================================================================
// These allow unqualified calls (via ADL or using-directives) to work
// uniformly for float and custom types.
// ============================================================================

// Basic math functions
inline float abs(float x)
{
    return std::abs(x);
}
inline float fabs(float x)
{
    return std::fabs(x);
}
inline bool isnan(float x)
{
    return std::isnan(x);
}
inline bool isinf(float x)
{
    return std::isinf(x);
}
inline bool signbit(float x)
{
    return std::signbit(x);
}
inline bool isfinite(float x)
{
    return std::isfinite(x);
}
inline float copysign(float x, float y)
{
    return std::copysign(x, y);
}

// Min/max functions
inline float max(float a, float b)
{
    return std::max(a, b);
}
inline float min(float a, float b)
{
    return std::min(a, b);
}
inline float fmax(float a, float b)
{
    return std::fmax(a, b);
}
inline float fmin(float a, float b)
{
    return std::fmin(a, b);
}

// Rounding functions
inline float floor(float x)
{
    return std::floor(x);
}
inline float ceil(float x)
{
    return std::ceil(x);
}
inline float round(float x)
{
    return std::round(x);
}
inline float trunc(float x)
{
    return std::trunc(x);
}

// Exponential and logarithmic functions
inline float exp(float x)
{
    return std::exp(x);
}
inline float exp2(float x)
{
    return std::exp2(x);
}
inline float log(float x)
{
    return std::log(x);
}
inline float log2(float x)
{
    return std::log2(x);
}
inline float log10(float x)
{
    return std::log10(x);
}

// Power functions
inline float sqrt(float x)
{
    return std::sqrt(x);
}
inline float rsqrt(float x)
{
    return 1.0f / std::sqrt(x);
}
inline float pow(float x, float y)
{
    return std::pow(x, y);
}

// Trigonometric functions
inline float sin(float x)
{
    return std::sin(x);
}
inline float cos(float x)
{
    return std::cos(x);
}
inline float tan(float x)
{
    return std::tan(x);
}
inline float asin(float x)
{
    return std::asin(x);
}
inline float acos(float x)
{
    return std::acos(x);
}
inline float atan(float x)
{
    return std::atan(x);
}

// Hyperbolic functions
inline float sinh(float x)
{
    return std::sinh(x);
}
inline float cosh(float x)
{
    return std::cosh(x);
}
inline float tanh(float x)
{
    return std::tanh(x);
}

// Error function
inline float erf(float x)
{
    return std::erf(x);
}

// Floating-point manipulation
inline float fmod(float x, float y)
{
    return std::fmod(x, y);
}

// Fused multiply-add
inline float fma(float x, float y, float z)
{
    return std::fma(x, y, z);
}

} // namespace hipdnn_data_sdk::types
