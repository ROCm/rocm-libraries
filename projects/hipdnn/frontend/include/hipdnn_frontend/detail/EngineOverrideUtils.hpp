// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>

#include <optional>

namespace hipdnn_frontend::engine_override
{

/// Walk the graph using the node visitor to find the first convolution operation
/// and return the preferred engine ID from the lazily-loaded engine override config
/// (pointed to by HIPDNN_ENGINE_OVERRIDE_FILE).
///
/// Returns nullopt when:
/// - no convolution node is present in the graph,
/// - no rule in the config matches the convolution's tensors, or
/// - JSON support is compiled out (HIPDNN_FRONTEND_SKIP_JSON_LIB defined).
inline std::optional<int64_t> getPreferredIdFromOverrideConfig(const graph::INode& root)
{
    std::optional<int64_t> result;

    root.visit([&result](const graph::INode& node) {
        if(result.has_value())
        {
            return;
        }
        if(const auto* fprop = dynamic_cast<const graph::ConvolutionFpropNode*>(&node))
        {
            result = checkEngineOverride("conv_fprop",
                                         {fprop->attributes.get_x(), fprop->attributes.get_w()});
        }
        else if(const auto* dgrad = dynamic_cast<const graph::ConvolutionDgradNode*>(&node))
        {
            result = checkEngineOverride("conv_dgrad",
                                         {dgrad->attributes.get_dy(), dgrad->attributes.get_w()});
        }
        else if(const auto* wgrad = dynamic_cast<const graph::ConvolutionWgradNode*>(&node))
        {
            result = checkEngineOverride("conv_wgrad",
                                         {wgrad->attributes.get_x(), wgrad->attributes.get_dy()});
        }
    });

    return result;
}

} // namespace hipdnn_frontend::engine_override
