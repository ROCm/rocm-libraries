// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <string>
#include <string_view>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Converts a heuristic policy name string to a deterministic int64_t ID
 *
 * Uses the FNV-1a hash. The same input always produces the same output, so the
 * hash is stable across processes and platforms. Used to identify selection
 * heuristic policies (e.g., "SelectionHeuristic::StaticOrdering") on the C ABI
 * without shipping null-separated string blobs.
 *
 * @param policyName The policy name to convert to an ID
 * @return int64_t The policy ID
 */
inline int64_t policyNameToId(const char* policyName) noexcept
{
    return static_cast<int64_t>(fnv1aHash(policyName));
}

/**
 * @brief Overload for std::string
 */
inline int64_t policyNameToId(const std::string& policyName)
{
    return static_cast<int64_t>(fnv1aHash(policyName));
}

/**
 * @brief Overload for std::string_view
 */
inline int64_t policyNameToId(std::string_view policyName)
{
    return static_cast<int64_t>(fnv1aHash(policyName));
}

} // namespace hipdnn_data_sdk::utilities
