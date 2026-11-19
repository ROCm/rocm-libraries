// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_integration_tests
{

/// Allocates the workspace that the built plan of `graph` needs. On failure,
/// returns std::nullopt and sets `error`.
inline std::optional<hipdnn_data_sdk::utilities::Workspace<>>
    allocatePlanWorkspace(hipdnn_frontend::graph::Graph& graph, std::string& error)
{
    int64_t workspaceSize = 0;
    if(auto status = graph.get_workspace_size(workspaceSize); !status.is_good())
    {
        error = status.get_message();
        return std::nullopt;
    }
    if(workspaceSize < 0)
    {
        error = "engine reported a negative workspace size";
        return std::nullopt;
    }
    return hipdnn_data_sdk::utilities::Workspace<>(static_cast<size_t>(workspaceSize));
}

} // namespace hipdnn_integration_tests
