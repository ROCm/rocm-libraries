// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/detail/PointwisePacker.hpp>
#include <hipdnn_frontend/detail/PointwiseUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class PointwiseNode : public BaseNode<PointwiseNode, NodeType::POINTWISE>
{

public:
    PointwiseAttributes attributes;

    PointwiseNode(PointwiseAttributes&& pointwiseAttributes, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(pointwiseAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_output_0())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "PointwiseNode missing OUT_0 for pre-validation"};
        }
        if(attributes.get_mode() == PointwiseMode::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing operation for pre-validation"};
        }
        if(attributes.outputs.size() != 1)
        {
            return {ErrorCode::INVALID_VALUE, "PointwiseNode must have exactly one output"};
        }

        auto checkInputsMatchOp = checkInputsMatchOperationType();
        if(checkInputsMatchOp.is_bad())
        {
            return checkInputsMatchOp;
        }

        return checkInputsAndOutputsAreBroadcastCompatible();
    }

    Error infer_properties_node() override
    {
        auto out = attributes.get_output_0();
        if(!out)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing output for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(out->get_dim().empty())
        {
            std::vector<std::vector<int64_t>> inputShapes;

            for(const auto& [_, tensor] : attributes.inputs)
            {
                if(tensor)
                {
                    inputShapes.push_back(tensor->get_dim());
                }
                else
                {
                    return {ErrorCode::INVALID_VALUE, "PointwiseNode has null input tensor"};
                }
            }

            auto outputDims = out->get_dim();
            HIPDNN_CHECK_ERROR(detail::findCommonShape(inputShapes, outputDims));
            out->set_dim(outputDims);
        }

        if(out->get_stride().empty())
        {
            for(const auto& [_, tensor] : attributes.inputs)
            {
                if(!tensor)
                {
                    return {ErrorCode::INVALID_VALUE, "PointwiseNode has null input tensor"};
                }

                if(tensor->get_dim() == out->get_dim())
                {
                    HIPDNN_FE_LOG_INFO("PointwiseNode " << attributes.get_name()
                                                        << " inferring stride from input tensor "
                                                        << tensor->get_name() << " for output "
                                                        << out->get_name());
                    out->set_stride(tensor->get_stride());
                    break;
                }
            }

            if(out->get_stride().empty())
            {
                return {ErrorCode::ATTRIBUTE_NOT_SET, "PointwiseNode output missing stride"};
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createPointwiseOperation(attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        PointwiseAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackPointwiseOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

private:
    Error checkInputsMatchOperationType() const
    {
        switch(attributes.inputs.size())
        {
        case 0:
            return {ErrorCode::INVALID_VALUE, "PointwiseNode must have at least one input"};
        case 1:
            if(!isUnaryPointwiseMode(attributes.get_mode()))
            {
                return {ErrorCode::INVALID_VALUE,
                        "PointwiseNode with one input must have a unary operation"};
            }
            break;
        case 2:
            if(!isBinaryPointwiseMode(attributes.get_mode()))
            {
                return {ErrorCode::INVALID_VALUE,
                        "PointwiseNode with two inputs must have a binary operation"};
            }
            break;
        case 3:
            if(!isTernaryPointwiseMode(attributes.get_mode()))
            {
                return {ErrorCode::INVALID_VALUE,
                        "PointwiseNode with three inputs must have a ternary operation"};
            }
            break;
        default:
            return {ErrorCode::INVALID_VALUE,
                    "PointwiseNode can only have one, two, or three inputs"};
        }

        return {};
    }

    Error checkInputsAndOutputsAreBroadcastCompatible() const
    {
        auto output = attributes.get_output_0();
        if(output && !output->get_dim().empty())
        {
            const auto& outputDims = output->get_dim();

            for(const auto& [_, inputTensor] : attributes.inputs)
            {
                if(inputTensor)
                {
                    const auto& inputDims = inputTensor->get_dim();

                    if(!hipdnn_data_sdk::utilities::areDimensionsBroadcastCompatible(inputDims,
                                                                                     outputDims))
                    {
                        return {ErrorCode::INVALID_VALUE,
                                "PointwiseNode input '" + inputTensor->get_name()
                                    + "' has dimensions incompatible with output. "
                                    + "All inputs must be broadcastable to output dimensions."};
                    }
                }
                else
                {
                    return {ErrorCode::INVALID_VALUE, "PointwiseNode has null input tensor"};
                }
            }
        }

        return {};
    }
};
}
