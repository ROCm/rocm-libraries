// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace hipdnn_frontend::graph
{
class BatchnormInferenceNode : public BaseNode<BatchnormInferenceNode>
{
public:
    BatchnormInferenceAttributes attributes;

    BatchnormInferenceNode(BatchnormInferenceAttributes&& batchnormAttrs,
                           const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing x for pre-validation"};
        }
        if(!attributes.get_scale())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing scale for pre-validation"};
        }
        if(!attributes.get_bias())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing bias for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing y for pre-validation"};
        }
        if(!attributes.get_mean())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing mean for pre-validation"};
        }
        if(!attributes.get_inv_variance())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing inv_variance for pre-validation"};
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing y for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
}
