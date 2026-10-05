// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

/// @file FallbackEngineOrder.hpp
/// @brief The operator's HIPDNN_HEUR_FALLBACK_ENGINE_ORDER, shared by every built-in policy
///        that orders engines by the static rules.
namespace hipdnn_backend::heuristics
{

inline constexpr const char* FALLBACK_ORDERING_ENV = "HIPDNN_HEUR_FALLBACK_ENGINE_ORDER";

/// Parse HIPDNN_HEUR_FALLBACK_ENGINE_ORDER (comma-separated engine names or IDs) into a
/// list of engine IDs in the order the user wrote them. Empty / unset env →
/// empty vector (caller falls back to the legacy sortEngineIds ordering).
/// Blank tokens are skipped.
///
/// engineNameOrIdToId accepts every spelling hipdnn_list_engines prints: a declared
/// name, which hashes to the engine's ID, or the hex ID an engine that declares no
/// name displays under. An unrecognized token still hashes to a deterministic ID and
/// the caller filters against the candidate list, so a typo'd name simply won't match
/// anything.
inline std::vector<int64_t> parseFallbackOrderingEnv()
{
    const std::string raw = hipdnn_data_sdk::utilities::getEnv(FALLBACK_ORDERING_ENV, "");
    if(hipdnn_data_sdk::utilities::trim(raw).empty())
    {
        return {};
    }

    std::vector<int64_t> ids;
    std::stringstream stream(raw);
    std::string token;
    while(std::getline(stream, token, ','))
    {
        const std::string name = hipdnn_data_sdk::utilities::trim(token);
        if(name.empty())
        {
            continue;
        }
        ids.push_back(hipdnn_data_sdk::utilities::engineNameOrIdToId(name));
    }
    return ids;
}

/// Restrict @p candidates to engines named in @p envOrder, preserving the env
/// order. Engines not listed in the env are dropped — when the operator sets
/// HIPDNN_HEUR_FALLBACK_ENGINE_ORDER they are explicitly opting out of every
/// other engine. Names in the env that are not in @p candidates are silently
/// skipped (the policy loop only sees engines the rest of the stack already
/// filtered down to).
inline std::vector<int64_t> applyFallbackOrdering(const std::vector<int64_t>& candidates,
                                                  const std::vector<int64_t>& envOrder)
{
    const std::unordered_set<int64_t> candidateSet(candidates.begin(), candidates.end());
    std::vector<int64_t> out;
    out.reserve(envOrder.size());
    for(const int64_t id : envOrder)
    {
        if(candidateSet.count(id) != 0U)
        {
            out.push_back(id);
        }
    }
    return out;
}

} // namespace hipdnn_backend::heuristics
