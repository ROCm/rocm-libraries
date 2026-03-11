// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>
#include <hipdnn_frontend/node/Node.hpp>

#include <optional>

inline hipdnn_frontend::detail::EngineOverrideDesc
    hipdnn_frontend::detail::EngineOverrideAccess::getDesc(
        const hipdnn_frontend::graph::INode& node)
{
    return node.getEngineOverrideDesc();
}

namespace hipdnn_frontend::engine_override
{

/// Walk the graph using the node visitor to find the first operation
/// that participates in engine override and return the preferred engine
/// ID from the lazily-loaded engine override config (pointed to by
/// HIPDNN_ENGINE_OVERRIDE_FILE).
///
/// Returns nullopt when:
/// - no participating node is present in the graph,
/// - no rule in the config matches the operation's tensors, or
/// - JSON support is compiled out (HIPDNN_FRONTEND_SKIP_JSON_LIB defined).
inline std::optional<int64_t> getPreferredIdFromOverrideConfig(const graph::INode& root)
{
    std::optional<int64_t> result;

    root.visit([&result](const graph::INode& node) {
        if(result.has_value())
        {
            return;
        }
        auto desc = hipdnn_frontend::detail::EngineOverrideAccess::getDesc(node);
        if(desc.enabled)
        {
            result = checkEngineOverride(desc.name, desc.tensors);
        }
    });

    return result;
}

} // namespace hipdnn_frontend::engine_override
