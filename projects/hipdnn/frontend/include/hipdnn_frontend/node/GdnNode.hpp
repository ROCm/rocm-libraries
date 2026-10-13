// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/GatedDeltaRuleNodeHelpers.hpp>
#include <hipdnn_frontend/detail/GdnPacker.hpp>
#include <hipdnn_frontend/detail/GdnUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::graph
{
class GdnNode : public BaseNode<GdnNode, NodeType::GDN>
{
public:
    GdnAttributes attributes;

    GdnNode(GdnAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        GdnAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackGdnOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(attributes.get_q(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GdnNode missing q (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_k(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GdnNode missing k (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_v(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GdnNode missing v (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_g(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GdnNode missing g (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_beta(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GdnNode missing beta (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_cu_seqlens(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GdnNode missing cu_seqlens (input) for pre-validation");

        // Validate required tensor dimensions
        HIPDNN_RETURN_IF_TRUE(attributes.get_q()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "GdnNode missing q dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_k()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "GdnNode missing k dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_v()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "GdnNode missing v dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_g()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "GdnNode missing g dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_beta()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "GdnNode missing beta dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_cu_seqlens()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "GdnNode missing cu_seqlens dimensions for pre-validation");

        // Layout, head-count and gate-shape rules shared with KdaNode.
        detail::GatedDeltaRuleShape shape;
        return detail::validateGatedDeltaRule(
            attributes, /*perChannelGate=*/false, "GdnNode", shape);
    }

    Error infer_properties_node() override
    {
        return detail::inferGatedDeltaRuleProperties(
            attributes, /*perChannelGate=*/false, graph_attributes, "GdnNode");
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createGdnOperation(attributes, tensorDescs, operations);
    }
};
}
