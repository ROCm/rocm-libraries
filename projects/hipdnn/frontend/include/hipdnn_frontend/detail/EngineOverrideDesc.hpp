// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <string_view>
#include <vector>

namespace hipdnn_frontend::graph
{
class INode;
} // namespace hipdnn_frontend::graph

namespace hipdnn_frontend::detail
{

/// Engine override descriptor returned by nodes that participate in engine
/// override selection. Contains the operation name and ordered input tensors
/// used for rule matching.
struct EngineOverrideDesc
{
    bool enabled = false;
    std::string_view name;
    std::vector<std::shared_ptr<graph::TensorAttributes>> tensors;
};

/// Provides access to INode::getEngineOverrideDesc() for internal use
/// by EngineOverrideUtils without exposing it in the public API.
struct EngineOverrideAccess
{
    static EngineOverrideDesc getDesc(const graph::INode& node);
};

} // namespace hipdnn_frontend::detail
