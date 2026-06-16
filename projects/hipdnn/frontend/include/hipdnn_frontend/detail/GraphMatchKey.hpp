// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/NodeType.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hipdnn_frontend::detail
{

/// Determine the core operation name written to the autotune config file for a
/// graph (passed as its root INode). Priority: convolution/GEMM/SDPA (highest)
/// > normalization > pointwise (lowest); the graph name is the fallback when no
/// recognized operation is found. Reconstructed through the public node
/// traversal API (INode::visit + getNodeType + graph_attributes) so no friend
/// or new accessor is required.
inline std::string getCoreOperationName(const graph::INode& root)
{
    // Priority levels: 0 = default/unknown, 1 = pointwise, 2 = normalization,
    // 3 = conv/matmul/SDPA
    int bestPriority = 0;
    std::string bestName;

    root.visit([&](const graph::INode& node) {
        if(bestPriority == 3)
        {
            // Short-circuit: can't do better than priority 3.
            return;
        }

        int priority = 0;
        std::string name;
        switch(node.getNodeType())
        {
        case graph::NodeType::CONVOLUTION_FPROP:
            priority = 3;
            name = "conv_fprop";
            break;
        case graph::NodeType::CONVOLUTION_DGRAD:
            priority = 3;
            name = "conv_dgrad";
            break;
        case graph::NodeType::CONVOLUTION_WGRAD:
            priority = 3;
            name = "conv_wgrad";
            break;
        default:
            break;
        }

        if(priority > bestPriority)
        {
            bestPriority = priority;
            bestName = std::move(name);
        }
    });

    return bestPriority > 0 ? bestName : std::string(root.graph_attributes.get_name());
}

/// Append a tensor to @p out when it participates in the match key: non-null,
/// has an assigned UID, and is not virtual (intermediate). Mirrors the
/// non-virtual / has_uid filter the backend index applies to physical tensors.
inline void appendMatchKeyTensor(std::vector<std::shared_ptr<graph::TensorAttributes>>& out,
                                 const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    if(tensor && tensor->has_uid() && !tensor->get_is_virtual())
    {
        out.push_back(tensor);
    }
}

/// Collect the ordered match-key tensors for a graph (passed as its root INode)
/// written to the autotune config file.
///
/// CANONICAL-TENSOR-ORDER: for every operation the match-key tensor order is
/// the op's flatbuffer `*_attributes.fbs` INPUT-field declaration order with the
/// output field(s) dropped; the match is POSITIONAL (index-by-index). This rule
/// MUST stay in lockstep with the backend reader that consumes these tensors:
/// `hipdnn_backend::heuristics::config::matchOverrideConfig` (the conv dispatch)
/// in `backend/src/heuristics/config/ConfigBuiltIn.cpp`. Any change to the
/// per-op set/order here MUST be mirrored there and vice versa.
///
/// Convolution (output excluded, exactly 2 inputs each):
///   - conv_fprop → (x, w)
///   - conv_dgrad → (dy, w)
///   - conv_wgrad → (x, dy)
///
/// Each operation supported for config round-trip has an explicit op-aware
/// branch above; the branch defines its canonical match-key tensor set/order. An
/// operation with NO op-aware branch (e.g. a reduction, or a ternary pointwise)
/// is intentionally unsupported for config round-trip: it produces an EMPTY match
/// key and therefore never matches a backend config rule.
inline std::vector<std::shared_ptr<graph::TensorAttributes>>
    getMatchKeyTensors(const graph::INode& root)
{
    std::vector<std::shared_ptr<graph::TensorAttributes>> result;

    // Op-aware selection: the first node matching an op-aware branch wins and
    // emits its tensors in the canonical order above; `handled` then stops the
    // visit. A graph with no recognized op leaves `handled` false and `result`
    // empty.
    bool handled = false;
    root.visit([&](const graph::INode& node) {
        if(handled)
        {
            return;
        }
        switch(node.getNodeType())
        {
        case graph::NodeType::CONVOLUTION_FPROP:
        {
            const auto& conv = static_cast<const graph::ConvolutionFpropNode&>(node);
            appendMatchKeyTensor(result, conv.attributes.get_x());
            appendMatchKeyTensor(result, conv.attributes.get_w());
            handled = true;
            break;
        }
        case graph::NodeType::CONVOLUTION_DGRAD:
        {
            const auto& conv = static_cast<const graph::ConvolutionDgradNode&>(node);
            appendMatchKeyTensor(result, conv.attributes.get_dy());
            appendMatchKeyTensor(result, conv.attributes.get_w());
            handled = true;
            break;
        }
        case graph::NodeType::CONVOLUTION_WGRAD:
        {
            const auto& conv = static_cast<const graph::ConvolutionWgradNode&>(node);
            appendMatchKeyTensor(result, conv.attributes.get_x());
            appendMatchKeyTensor(result, conv.attributes.get_dy());
            handled = true;
            break;
        }
        default:
            break;
        }
    });

    // An op with no op-aware branch above leaves `handled` false and `result`
    // empty: an unsupported op produces an EMPTY match key (it never matches a
    // backend config rule).
    return result;
}

} // namespace hipdnn_frontend::detail
