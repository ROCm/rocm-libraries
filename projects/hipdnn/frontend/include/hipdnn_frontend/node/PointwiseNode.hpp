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
        if(!toBackendPointwiseMode(attributes.get_mode()).has_value())
        {
            return {ErrorCode::INVALID_VALUE,
                    std::string("PointwiseNode mode ") + to_string(attributes.get_mode())
                        + " is not supported by hipDNN lowering"};
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
        const auto out = attributes.get_output_0();
        if(!out)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "PointwiseNode missing output for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // Every set input, including any the mode does not use, takes part in shape inference.
        const size_t usedInputCount = std::max<size_t>(modeInputCount(attributes.get_mode()), 1);
        std::vector<std::shared_ptr<TensorAttributes>> inputs;
        for(size_t slot = 0; slot < K_MAX_INPUT_COUNT; ++slot)
        {
            const auto it
                = attributes.inputs.find(static_cast<PointwiseAttributes::InputNames>(slot));
            if(it == attributes.inputs.end())
            {
                continue;
            }
            if(it->second)
            {
                inputs.push_back(it->second);
            }
            else if(slot < usedInputCount)
            {
                return nullInputError(slot);
            }
        }

        if(out->get_dim().empty())
        {
            std::vector<std::vector<int64_t>> inputShapes;
            inputShapes.reserve(inputs.size());
            for(const auto& tensor : inputs)
            {
                inputShapes.push_back(tensor->get_dim());
            }

            std::vector<int64_t> outputDims;
            HIPDNN_CHECK_ERROR(detail::findCommonShape(inputShapes, outputDims));
            out->set_dim(outputDims);
        }

        if(out->get_stride().empty())
        {
            for(const auto& tensor : inputs)
            {
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

            if(out->get_stride().empty() && out->get_is_virtual())
            {
                inferVirtualOutputStride(*out);
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
        const size_t usedInputCount = modeInputCount(attributes.get_mode());
        if(attributes.inputs.size() <= usedInputCount)
        {
            return detail::createPointwiseOperation(attributes, tensorDescs, operations);
        }

        // Inputs beyond the mode's operand count are not part of the operation.
        auto usedAttributes = attributes;
        if(usedInputCount < 2)
        {
            usedAttributes.inputs.erase(PointwiseAttributes::InputNames::IN_1);
        }
        if(usedInputCount < 3)
        {
            usedAttributes.inputs.erase(PointwiseAttributes::InputNames::IN_2);
        }
        return detail::createPointwiseOperation(usedAttributes, tensorDescs, operations);
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

    // Number of inputs the mode reads: IN_0 for unary, IN_0 and IN_1 for binary, all three for
    // ternary. Returns 0 for a mode in none of these classes.
    static size_t modeInputCount(PointwiseMode mode)
    {
        if(isTernaryPointwiseMode(mode))
        {
            return 3;
        }
        if(isBinaryPointwiseMode(mode))
        {
            return 2;
        }
        if(isUnaryPointwiseMode(mode))
        {
            return 1;
        }
        return 0;
    }

private:
    static constexpr size_t K_MAX_INPUT_COUNT = 3;

    Error nullInputError(size_t slot) const
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET,
                "PointwiseNode " + attributes.get_name() + " has null input tensor IN_"
                    + std::to_string(slot)};
    }

    // Inputs beyond the mode's operand count are accepted and left out of the operation.
    Error checkInputsMatchOperationType() const
    {
        if(attributes.inputs.empty())
        {
            return {ErrorCode::INVALID_VALUE, "PointwiseNode must have at least one input"};
        }
        if(attributes.inputs.size() > K_MAX_INPUT_COUNT)
        {
            return {ErrorCode::INVALID_VALUE,
                    "PointwiseNode can only have one, two, or three inputs"};
        }

        const size_t usedInputCount = modeInputCount(attributes.get_mode());
        if(usedInputCount == 0)
        {
            return {ErrorCode::INVALID_VALUE,
                    "PointwiseNode " + attributes.get_name() + " mode "
                        + std::to_string(static_cast<int>(attributes.get_mode()))
                        + " has no known operand count"};
        }

        for(size_t slot = 0; slot < usedInputCount; ++slot)
        {
            const auto it
                = attributes.inputs.find(static_cast<PointwiseAttributes::InputNames>(slot));
            if(it == attributes.inputs.end())
            {
                return {ErrorCode::INVALID_VALUE,
                        "PointwiseNode " + attributes.get_name() + " mode requires "
                            + std::to_string(usedInputCount) + " inputs but IN_"
                            + std::to_string(slot) + " is not set"};
            }
            if(!it->second)
            {
                return nullInputError(slot);
            }
        }

        return {};
    }

    // Generates strides for a virtual output that no input matches, following IN_0's stride
    // order. Axes the output has beyond IN_0's rank are leading and outermost.
    void inferVirtualOutputStride(TensorAttributes& out) const
    {
        const auto in0 = attributes.get_input_0();
        const auto& outDims = out.get_dim();
        if(!in0 || in0->get_stride().empty() || in0->get_stride().size() > outDims.size())
        {
            return;
        }

        const auto in0Order = hipdnn_data_sdk::utilities::extractStrideOrder(in0->get_stride());
        const size_t leadingAxes = outDims.size() - in0Order.size();
        std::vector<int64_t> strideOrder(outDims.size());
        for(size_t i = 0; i < outDims.size(); ++i)
        {
            strideOrder[i] = i < leadingAxes ? static_cast<int64_t>(outDims.size() - 1 - i)
                                             : in0Order[i - leadingAxes];
        }

        HIPDNN_FE_LOG_INFO("PointwiseNode " << attributes.get_name()
                                            << " generating stride from the stride order of "
                                            << in0->get_name() << " for virtual output "
                                            << out.get_name());
        out.set_stride(hipdnn_data_sdk::utilities::generateStrides(outDims, strideOrder));
    }

    Error checkInputsAndOutputsAreBroadcastCompatible() const
    {
        const auto output = attributes.get_output_0();
        if(output && !output->get_dim().empty())
        {
            const auto& outputDims = output->get_dim();

            for(const auto& [_, inputTensor] : attributes.inputs)
            {
                if(!inputTensor)
                {
                    continue;
                }

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
        }

        return {};
    }
};
}
