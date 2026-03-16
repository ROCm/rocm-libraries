// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>

namespace hipdnn_frontend::graph
{
class ReductionNode : public BaseNode<ReductionNode>
{
public:
    ReductionAttributes attributes;

    ReductionNode(ReductionAttributes&& reductionAttributes, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(reductionAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ReductionNode missing input X for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ReductionNode missing output Y for pre-validation"};
        }
        if(!attributes.get_mode().has_value() || attributes.get_mode() == ReductionMode::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "ReductionNode missing mode for pre-validation"};
        }
        return {};
    }

    Error infer_properties_node() override
    {
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));
        return {};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::ReductionAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
} // namespace hipdnn_frontend::graph
