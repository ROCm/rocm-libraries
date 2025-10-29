// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

namespace hipdnn_frontend::graph
{
class BatchnormNode : public BaseNode<BatchnormNode>
{
public:
    BatchnormAttributes attributes;

    BatchnormNode(BatchnormAttributes&& batchnormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing x for pre-validation"};
        }
        if(!attributes.get_scale())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing scale for pre-validation"};
        }
        if(!attributes.get_bias())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing bias for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing y for pre-validation"};
        }
        if(!attributes.get_epsilon())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormNode missing epsilon for pre-validation"};
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing y for setting properties"};
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

        auto mean = attributes.get_mean();
        auto invVar = attributes.get_inv_variance();

        if(mean)
        {
            inferCTensor(mean);
        }

        if(invVar)
        {
            inferCTensor(invVar);
        }

        auto prevRunningMean = attributes.get_prev_running_mean();
        auto prevRunningVar = attributes.get_prev_running_variance();

        auto nextRunningMean = attributes.get_next_running_mean();
        auto nextRunningVar = attributes.get_next_running_variance();

        if(prevRunningMean && prevRunningVar && nextRunningMean && nextRunningVar)
        {
            inferCTensor(nextRunningMean);
            inferCTensor(nextRunningVar);
        }

        return {};
    }

    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        BaseNode<BatchnormNode>::gather_hipdnn_tensors(allTensors);

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
            hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef BatchnormNode BatchNormNode;
}
