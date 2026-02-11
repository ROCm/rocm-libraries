// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file Uint8.hpp
 * @brief Forwarding functions for uint8_t type to enable uniform unqualified calls.
 *
 * This header provides forwarding functions that delegate to std:: implementations
 * for uint8_t type. This enables code using `using namespace hipdnn_data_sdk::types`
 * to work uniformly with uint8_t and other types.
 *
 * Note: abs is not provided for unsigned types as they are always non-negative.
 */

#include <algorithm>
#include <cstdint>

namespace hipdnn_data_sdk::types
{

// ============================================================================
// Forwarding functions for uint8_t
// ============================================================================

inline uint8_t max(uint8_t a, uint8_t b)
{
    return std::max(a, b);
}

inline uint8_t min(uint8_t a, uint8_t b)
{
    return std::min(a, b);
}

} // namespace hipdnn_data_sdk::types
