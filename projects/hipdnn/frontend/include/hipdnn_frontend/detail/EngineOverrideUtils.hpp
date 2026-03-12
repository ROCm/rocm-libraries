// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>

#include <optional>

namespace hipdnn_frontend::engine_override
{

/// Walk the graph using the node visitor to find the first convolution
/// operation and return the preferred engine ID from the lazily-loaded
/// engine override config (pointed to by HIPDNN_ENGINE_OVERRIDE_FILE).
///
/// Returns nullopt when:
/// - no convolution node is present in the graph,
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

        switch(node.getNodeType())
        {
        case graph::NodeType::CONVOLUTION_FPROP:
        {
            const auto& conv = static_cast<const graph::ConvolutionFpropNode&>(node);
            result = checkEngineOverride("conv_fprop",
                                         {conv.attributes.get_x(), conv.attributes.get_w()});
            break;
        }
        case graph::NodeType::CONVOLUTION_DGRAD:
        {
            const auto& conv = static_cast<const graph::ConvolutionDgradNode&>(node);
            result = checkEngineOverride("conv_dgrad",
                                         {conv.attributes.get_dy(), conv.attributes.get_w()});
            break;
        }
        case graph::NodeType::CONVOLUTION_WGRAD:
        {
            const auto& conv = static_cast<const graph::ConvolutionWgradNode&>(node);
            result = checkEngineOverride("conv_wgrad",
                                         {conv.attributes.get_x(), conv.attributes.get_dy()});
            break;
        }
        default:
            break;
        }
    });

    return result;
}

} // namespace hipdnn_frontend::engine_override
