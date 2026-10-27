// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "plugin/HeuristicPluginManager.hpp"

#include <cstdint>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <set>

namespace hipdnn_backend::test_utilities
{

/// Policy IDs the backend built-ins serve; a manager registers them in its constructor.
inline std::set<int64_t> builtInPolicyIds()
{
    using hipdnn_data_sdk::utilities::policyNameToId;
    return {policyNameToId("SelectionHeuristic::Config"),
            policyNameToId("SelectionHeuristic::StaticOrdering"),
            policyNameToId(hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME),
            policyNameToId(hipdnn_data_sdk::utilities::MODE_B_POLICY_NAME)};
}

inline std::set<int64_t> registeredPolicyIds(const plugin::HeuristicPluginManager& manager)
{
    std::set<int64_t> ids;
    for(const auto& plugin : manager.getPlugins())
    {
        const auto pluginIds = plugin->getAllPolicyIds();
        ids.insert(pluginIds.begin(), pluginIds.end());
    }
    return ids;
}

} // namespace hipdnn_backend::test_utilities
