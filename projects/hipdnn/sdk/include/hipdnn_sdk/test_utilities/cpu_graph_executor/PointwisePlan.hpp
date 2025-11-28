// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>

namespace hipdnn_sdk::test_utilities
{

struct PointwiseParams
{
    PointwiseParams() = default;
    PointwiseParams(const hipdnn_sdk::data_objects::PointwiseMode pointwiseMode,
                    const hipdnn_sdk::data_objects::TensorAttributes& in0Attributes,
                    const hipdnn_sdk::data_objects::TensorAttributes* optionalIn1Attributes,
                    const hipdnn_sdk::data_objects::TensorAttributes& out0Attributes,
                    std::optional<float> reluLowerClipLocal,
                    std::optional<float> reluUpperClipLocal,
                    std::optional<float> reluLowerClipSlopeLocal,
                    std::optional<float> swishBetaLocal,
                    std::optional<float> eluAlphaLocal,
                    std::optional<float> softplusBetaLocal)
        : in0Tensor(unpackTensorAttributes(in0Attributes))
        , out0Tensor(unpackTensorAttributes(out0Attributes))
        , mode(pointwiseMode)
        , reluLowerClip(reluLowerClipLocal)
        , reluUpperClip(reluUpperClipLocal)
        , reluLowerClipSlope(reluLowerClipSlopeLocal)
        , swishBeta(swishBetaLocal)
        , eluAlpha(eluAlphaLocal)
        , softplusBeta(softplusBetaLocal)
    {
        if(optionalIn1Attributes != nullptr)
        {
            in1Tensor = unpackTensorAttributes(*optionalIn1Attributes);
        }
    }

    hipdnn_sdk::data_objects::TensorAttributesT in0Tensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> in1Tensor;
    hipdnn_sdk::data_objects::TensorAttributesT out0Tensor;
    hipdnn_sdk::data_objects::PointwiseMode mode;

    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    std::optional<float> reluLowerClipSlope;
    std::optional<float> swishBeta;
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;
};

template <typename Input0Type, typename Input1Type, typename OutputType>
class PointwisePlan : public IGraphNodePlanExecutor
{
public:
    PointwisePlan(PointwiseParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        if(_params.reluLowerClip.has_value() || _params.reluUpperClip.has_value()
           || _params.reluLowerClipSlope.has_value())
        {
            executeParameterized(variantPack);
        }
        else
        {
            executeNonParameterized(variantPack);
        }
    }

private:
    void executeNonParameterized(const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto shallowIn0Tensor = createShallowTensor<Input0Type>(
            _params.in0Tensor, variantPack.at(_params.in0Tensor.uid));

        auto shallowOut0Tensor = createShallowTensor<OutputType>(
            _params.out0Tensor, variantPack.at(_params.out0Tensor.uid));

        if(utilities::isUnaryPointwiseMode(_params.mode))
        {
            CpuReferencePointwiseImpl<OutputType, Input0Type>::pointwiseCompute(
                _params.mode, *shallowOut0Tensor, *shallowIn0Tensor);
        }
        else if(utilities::isBinaryPointwiseMode(_params.mode))
        {
            if(!_params.in1Tensor.has_value())
            {
                throw std::runtime_error("Binary pointwise operation requires in1 tensor");
            }

            auto shallowIn1Tensor = createShallowTensor<Input1Type>(
                _params.in1Tensor.value(), variantPack.at(_params.in1Tensor.value().uid));

            CpuReferencePointwiseImpl<OutputType, Input0Type, Input1Type>::pointwiseCompute(
                _params.mode, *shallowOut0Tensor, *shallowIn0Tensor, *shallowIn1Tensor);
        }
        else
        {
            throw std::runtime_error("Unsupported pointwise operation mode");
        }
    }

    void executeParameterized(const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto shallowIn0Tensor = createShallowTensor<Input0Type>(
            _params.in0Tensor, variantPack.at(_params.in0Tensor.uid));

        auto shallowOut0Tensor = createShallowTensor<OutputType>(
            _params.out0Tensor, variantPack.at(_params.out0Tensor.uid));

        if(utilities::isUnaryPointwiseMode(_params.mode))
        {
            CpuReferencePointwiseImpl<OutputType, Input0Type>::pointwiseCompute(
                _params.mode,
                *shallowOut0Tensor,
                *shallowIn0Tensor,
                static_cast<OutputType>(
                    _params.reluLowerClip.has_value() ? _params.reluLowerClip.value() : 0.0f),
                static_cast<OutputType>(_params.reluUpperClip.has_value()
                                            ? _params.reluUpperClip.value()
                                            : std::numeric_limits<float>::max()),
                static_cast<OutputType>(_params.reluLowerClipSlope.has_value()
                                            ? _params.reluLowerClipSlope.value()
                                            : 0.0f));
        }
        else if(utilities::isBinaryPointwiseMode(_params.mode))
        {
            if(!_params.in1Tensor.has_value())
            {
                throw std::runtime_error("Binary pointwise operation requires in1 tensor");
            }

            auto shallowIn1Tensor = createShallowTensor<Input1Type>(
                _params.in1Tensor.value(), variantPack.at(_params.in1Tensor.value().uid));

            CpuReferencePointwiseImpl<OutputType, Input0Type, Input1Type>::pointwiseCompute(
                _params.mode,
                *shallowOut0Tensor,
                *shallowIn0Tensor,
                *shallowIn1Tensor,
                static_cast<OutputType>(
                    _params.reluLowerClip.has_value() ? _params.reluLowerClip.value() : 0.0f),
                static_cast<OutputType>(_params.reluUpperClip.has_value()
                                            ? _params.reluUpperClip.value()
                                            : std::numeric_limits<float>::max()),
                static_cast<OutputType>(_params.reluLowerClipSlope.has_value()
                                            ? _params.reluLowerClipSlope.value()
                                            : 0.0f));
        }
        else
        {
            throw std::runtime_error("Unsupported pointwise operation mode");
        }
    }

    PointwiseParams _params;
};

template <hipdnn_sdk::data_objects::DataType Input0DataTypeEnum,
          hipdnn_sdk::data_objects::DataType Input1DataTypeEnum,
          hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
          hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
class PointwisePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using Input0Type = DataTypeToNative<Input0DataTypeEnum>;
    using Input1Type = DataTypeToNative<Input1DataTypeEnum>;
    using ComputeType = DataTypeToNative<ComputeDataTypeEnum>;
    using OutputType = DataTypeToNative<OutputDataTypeEnum>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        // Check that the operation is implemented
        auto mode = nodeAttributes->operation();
        bool isImplemented = false;
        if(utilities::isUnaryPointwiseMode(mode))
        {
            isImplemented = utilities::isImplementedUnaryPointwiseMode(mode);
        }
        else if(utilities::isBinaryPointwiseMode(mode))
        {
            isImplemented = utilities::isImplementedBinaryPointwiseMode(mode);
        }

        if(!isImplemented)
        {
            return false;
        }

        // Check required tensors exist
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_0_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_0_tensor_uid());

        // Check required tensor types
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_0_tensor_uid(), Input0DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_0_tensor_uid(), OutputDataTypeEnum);

        // Check optional tensors based on operation mode
        if(utilities::isBinaryPointwiseMode(mode))
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->in_1_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->in_1_tensor_uid(), Input1DataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PointwiseAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        // Get required tensors
        const auto* in0Tensor = tensorMap.at(nodeAttributes->in_0_tensor_uid());
        const auto* out0Tensor = tensorMap.at(nodeAttributes->out_0_tensor_uid());

        // Get optional tensors
        const auto* in1Tensor = (nodeAttributes->in_1_tensor_uid().has_value()
                                     ? tensorMap.at(*nodeAttributes->in_1_tensor_uid())
                                     : nullptr);

        PointwiseParams params(nodeAttributes->operation(),
                               *in0Tensor,
                               in1Tensor,
                               *out0Tensor,
                               nodeAttributes->relu_lower_clip(),
                               nodeAttributes->relu_upper_clip(),
                               nodeAttributes->relu_lower_clip_slope(),
                               nodeAttributes->swish_beta(),
                               nodeAttributes->elu_alpha(),
                               nodeAttributes->softplus_beta());

        // Throw if these values get set so its clear to any future users they are not supported.
        // Throwing here also makes it clear that we need to update PointwisePlan::execute
        // to use the params once there are implementations that can use them.
        // Cant throw in isApplicable and I want better error messaging than just returning false
        if(params.swishBeta.has_value() || params.eluAlpha.has_value()
           || params.softplusBeta.has_value())
        {
            throw std::runtime_error("Swish, ELU, and Softplus parameters are not supported "
                                     "in PointwisePlanBuilder for the Cpu Graph Executor yet");
        }

        return std::make_unique<PointwisePlan<Input0Type, Input1Type, OutputType>>(
            std::move(params));
    }
};

}
