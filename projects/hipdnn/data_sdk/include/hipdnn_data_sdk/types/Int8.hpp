// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file Int8.hpp
 * @brief Forwarding functions for int8_t type to enable uniform unqualified calls.
 *
 * This header provides forwarding functions that delegate to std:: implementations
 * for int8_t type. This enables code using `using namespace hipdnn_data_sdk::types`
 * to work uniformly with int8_t and other types.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hipdnn_data_sdk::types
{

// ============================================================================
// Forwarding functions for int8_t
// ============================================================================

/// Returns the absolute value of x.
/// @note abs(INT8_MIN) (-128) exhibits undefined behavior per C++ standard,
/// as the result (128) cannot be represented in int8_t. On two's complement
/// systems, this typically returns -128 (the input unchanged).
inline int8_t abs(int8_t x)
{
    return static_cast<int8_t>(std::abs(x));
}

inline int8_t max(int8_t a, int8_t b)
{
    return std::max(a, b);
}

inline int8_t min(int8_t a, int8_t b)
{
    return std::min(a, b);
}

} // namespace hipdnn_data_sdk::types
