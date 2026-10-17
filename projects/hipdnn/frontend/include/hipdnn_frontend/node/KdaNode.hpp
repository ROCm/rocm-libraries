// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/KdaAttributes.hpp>
#include <hipdnn_frontend/detail/GatedDeltaRuleNodeHelpers.hpp>
#include <hipdnn_frontend/detail/KdaPacker.hpp>
#include <hipdnn_frontend/detail/KdaUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::graph
{
class KdaNode : public BaseNode<KdaNode, NodeType::KDA>
{
public:
    KdaAttributes attributes;

    KdaNode(KdaAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        KdaAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackKdaOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(attributes.get_q(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "KdaNode missing q (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_k(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "KdaNode missing k (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_v(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "KdaNode missing v (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_g(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "KdaNode missing g (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_beta(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "KdaNode missing beta (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_cu_seqlens(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "KdaNode missing cu_seqlens (input) for pre-validation");

        // Validate required tensor dimensions
        HIPDNN_RETURN_IF_TRUE(attributes.get_q()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "KdaNode missing q dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_k()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "KdaNode missing k dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_v()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "KdaNode missing v dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_g()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "KdaNode missing g dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_beta()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "KdaNode missing beta dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_cu_seqlens()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "KdaNode missing cu_seqlens dimensions for pre-validation");

        // Layout, head-count and gate-shape rules shared with GdnNode.
        detail::GatedDeltaRuleShape shape;
        HIPDNN_CHECK_ERROR(
            detail::validateGatedDeltaRule(attributes, /*perChannelGate=*/true, "KdaNode", shape));
        HIPDNN_RETURN_IF_TRUE(attributes.get_gate_lower_bound().has_value()
                                  && !attributes.get_safe_gate(),
                              ErrorCode::INVALID_VALUE,
                              "KdaNode: gate_lower_bound requires safe_gate");
        return {};
    }

    Error infer_properties_node() override
    {
        return detail::inferGatedDeltaRuleProperties(
            attributes, /*perChannelGate=*/true, graph_attributes, "KdaNode");
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createKdaOperation(attributes, tensorDescs, operations);
    }
};
}
