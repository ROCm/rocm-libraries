// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

namespace hipdnn_frontend::graph
{

class BatchnormBackwardNode : public BaseNode<BatchnormBackwardNode>
{
public:
    BatchnormBackwardAttributes attributes;

    BatchnormBackwardNode(BatchnormBackwardAttributes&& batchnormAttrs,
                          const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_dy())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dy for pre-validation"};
        }
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing x for pre-validation"};
        }
        if(!attributes.get_scale())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing scale for pre-validation"};
        }
        if(!attributes.get_dx())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dx for pre-validation"};
        }
        if(!attributes.get_dscale())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dscale for pre-validation"};
        }
        if(!attributes.get_dbias())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dbias for pre-validation"};
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto dx = attributes.get_dx();
        auto dscale = attributes.get_dscale();
        auto dbias = attributes.get_dbias();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing x for setting properties"};
        }
        if(!dx)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dx for setting properties"};
        }
        if(!dscale)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dscale for setting properties"};
        }
        if(!dbias)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dbias for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(dx->get_dim().empty())
        {
            dx->set_dim(x->get_dim());
        }

        if(dx->get_stride().empty())
        {
            dx->set_stride(x->get_stride());
        }

        auto inferCTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer->get_dim().empty())
            {
                std::vector<int64_t> tensorDims(x->get_dim().size(), 1);
                tensorDims[1] = x->get_dim()[1];
                tensorToInfer->set_dim(tensorDims);
            }

            if(tensorToInfer->get_stride().empty())
            {
                auto strideOrder
                    = hipdnn_sdk::utilities::strideOrderNhwc(tensorToInfer->get_dim().size());
                tensorToInfer->set_stride(
                    hipdnn_sdk::utilities::generateStrides(tensorToInfer->get_dim(), strideOrder));
            }
        };

        inferCTensor(dscale);
        inferCTensor(dbias);

        return {};
    }

    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        BaseNode<BatchnormBackwardNode>::gather_hipdnn_tensors(allTensors);

        for(auto& tensor : attributes.peer_stats)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef BatchnormBackwardNode DBNNode;
}
