// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file Int32.hpp
 * @brief Forwarding functions for int32_t type to enable uniform unqualified calls.
 *
 * This header provides forwarding functions that delegate to std:: implementations
 * for int32_t type. This enables code using `using namespace hipdnn_data_sdk::types`
 * to work uniformly with int32_t and other types.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hipdnn_data_sdk::types
{

// ============================================================================
// Forwarding functions for int32_t
// ============================================================================

/// Returns the absolute value of x.
/// @note abs(INT32_MIN) returns INT32_MIN (the input unchanged), since its magnitude
/// (2^31) cannot be represented in int32_t. The negation is computed in unsigned
/// arithmetic, so no signed overflow occurs.
inline int32_t abs(int32_t x)
{
    if(x >= 0)
    {
        return x;
    }
    return static_cast<int32_t>(0u - static_cast<uint32_t>(x));
}

inline int32_t max(int32_t a, int32_t b)
{
    return std::max(a, b);
}

inline int32_t min(int32_t a, int32_t b)
{
    return std::min(a, b);
}

/// Returns the integer square root of x, truncated toward zero.
/// @note A negative x has no real square root and int32_t has no NaN, so sqrt(x < 0)
/// returns 0.
inline int32_t sqrt(int32_t x)
{
    if(x < 0)
    {
        return 0;
    }
    return static_cast<int32_t>(std::sqrt(x));
}

} // namespace hipdnn_data_sdk::types
