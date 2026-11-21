// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <string>
#include <string_view>

namespace hipdnn_data_sdk::utilities
{

/// Built-in policy that applies the file-driven engine override rules and the exact-match
/// autotune cache; declines when neither decides. Added in hipdnn_data_sdk 0.3.0.
inline constexpr const char* CONFIG_POLICY_NAME = "SelectionHeuristic::Config";
/// Built-in policy that applies the deterministic engine ordering (MIOPEN first, deterministic
/// variants last). Added in hipdnn_data_sdk 0.3.0.
inline constexpr const char* STATIC_ORDERING_POLICY_NAME = "SelectionHeuristic::StaticOrdering";
/// Quick policy: ranks by the engine-level (L1) prediction in the request's ranking metric
/// only; never asks for a configuration-level prediction (RFC 0019 §11.2).
inline constexpr const char* MODE_A_POLICY_NAME = "SelectionHeuristic::ModeA";
/// Thorough policy: per engine, the configuration-level (L2) prediction in the request's
/// ranking metric, falling back to the engine-level (L1) one (RFC 0019 §11.2).
inline constexpr const char* MODE_B_POLICY_NAME = "SelectionHeuristic::ModeB";

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
