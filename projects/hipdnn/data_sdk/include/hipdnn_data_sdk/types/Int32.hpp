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

inline int32_t abs(int32_t x)
{
    return std::abs(x);
}

inline int32_t max(int32_t a, int32_t b)
{
    return std::max(a, b);
}

inline int32_t min(int32_t a, int32_t b)
{
    return std::min(a, b);
}

} // namespace hipdnn_data_sdk::types
