// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuReferencePointwise.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <limits>
#include <optional>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuPointwiseUnaryParams
{
    GpuPointwiseUnaryParams() = default;
    GpuPointwiseUnaryParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& inputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode,
        const std::optional<float> reluFwdLowerClip,
        const std::optional<float> reluFwdUpperClip,
        const std::optional<float> reluFwdLowerSlope,
        const std::optional<float> swishFwdBeta,
        const std::optional<float> eluFwdAlpha,
        const std::optional<float> softplusFwdBeta)
        : inputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(inputAttributes))
        , outputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(outputAttributes))
        , pointwiseMode(pointwiseMode)
        , reluFwdLowerClip(reluFwdLowerClip)
        , reluFwdUpperClip(reluFwdUpperClip)
        , reluFwdLowerSlope(reluFwdLowerSlope)
        , swishFwdBeta(swishFwdBeta)
        , eluFwdAlpha(eluFwdAlpha)
        , softplusFwdBeta(softplusFwdBeta)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT inputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode;
    std::optional<float> reluFwdLowerClip;
    std::optional<float> reluFwdUpperClip;
    std::optional<float> reluFwdLowerSlope;
    std::optional<float> swishFwdBeta;
    std::optional<float> eluFwdAlpha;
    std::optional<float> softplusFwdBeta;
};

struct GpuPointwiseBinaryParams
{
    GpuPointwiseBinaryParams() = default;
    GpuPointwiseBinaryParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& input0Attributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& input1Attributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode,
        const std::optional<float> reluBwdLowerClip,
        const std::optional<float> reluBwdUpperClip,
        const std::optional<float> reluBwdLowerSlope)
        : input0Tensor(hipdnn_test_sdk::detail::unpackTensorAttributes(input0Attributes))
        , input1Tensor(hipdnn_test_sdk::detail::unpackTensorAttributes(input1Attributes))
        , outputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(outputAttributes))
        , pointwiseMode(pointwiseMode)
        , reluBwdLowerClip(reluBwdLowerClip)
        , reluBwdUpperClip(reluBwdUpperClip)
        , reluBwdLowerSlope(reluBwdLowerSlope)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT input0Tensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT input1Tensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode;
    std::optional<float> reluBwdLowerClip;
    std::optional<float> reluBwdUpperClip;
    std::optional<float> reluBwdLowerSlope;
};

// A node deeper than the kernels address is declined in isApplicable(), so the harness falls
// through to the CPU reference instead of the launch throwing.
inline bool exceedsMaxRank(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& output)
{
    return output.dims() != nullptr
           && output.dims()->size() > hipdnn_gpu_ref::GpuReferencePointwise::MAX_DIMS;
}

struct GpuPointwiseTernaryParams
{
    GpuPointwiseTernaryParams() = default;
    GpuPointwiseTernaryParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& input0Attributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& input1Attributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& maskAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode)
        : input0Tensor(hipdnn_test_sdk::detail::unpackTensorAttributes(input0Attributes))
        , input1Tensor(hipdnn_test_sdk::detail::unpackTensorAttributes(input1Attributes))
        , maskTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(maskAttributes))
        , outputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(outputAttributes))
        , pointwiseMode(pointwiseMode)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT input0Tensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT input1Tensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT maskTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode pointwiseMode;
};

template <typename Input0DataType,
          typename Input1DataType,
          typename MaskDataType,
          typename OutputDataType>
class GpuPointwiseTernaryPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuPointwiseTernaryPlan(GpuPointwiseTernaryParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<Input0DataType> input0Tensor(
            variantPack.at(_params.input0Tensor.uid),
            _params.input0Tensor.dims,
            _params.input0Tensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<Input1DataType> input1Tensor(
            variantPack.at(_params.input1Tensor.uid),
            _params.input1Tensor.dims,
            _params.input1Tensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<MaskDataType> maskTensor(
            variantPack.at(_params.maskTensor.uid),
            _params.maskTensor.dims,
            _params.maskTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> outputTensor(
            variantPack.at(_params.outputTensor.uid),
            _params.outputTensor.dims,
            _params.outputTensor.strides);

        hipdnn_gpu_ref::GpuReferencePointwise::
            pointwiseCompute<OutputDataType, Input0DataType, Input1DataType, MaskDataType>(
                _params.pointwiseMode, outputTensor, input0Tensor, input1Tensor, maskTensor);
    }

private:
    GpuPointwiseTernaryParams _params;
};

template <typename Input0DataType,
          typename Input1DataType,
          typename OutputDataType,
          typename ComputeDataType>
class GpuPointwiseBinaryPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuPointwiseBinaryPlan(GpuPointwiseBinaryParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<Input0DataType> input0Tensor(
            variantPack.at(_params.input0Tensor.uid),
            _params.input0Tensor.dims,
            _params.input0Tensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<Input1DataType> input1Tensor(
            variantPack.at(_params.input1Tensor.uid),
            _params.input1Tensor.dims,
            _params.input1Tensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> outputTensor(
            variantPack.at(_params.outputTensor.uid),
            _params.outputTensor.dims,
            _params.outputTensor.strides);

        const float lowerClip = _params.reluBwdLowerClip.value_or(0.0f);
        const float upperClip
            = _params.reluBwdUpperClip.value_or(std::numeric_limits<float>::max());
        const float lowerSlope = _params.reluBwdLowerSlope.value_or(0.0f);

        hipdnn_gpu_ref::GpuReferencePointwise::
            pointwiseCompute<OutputDataType, Input0DataType, Input1DataType, ComputeDataType>(
                _params.pointwiseMode,
                outputTensor,
                input0Tensor,
                input1Tensor,
                lowerClip,
                upperClip,
                lowerSlope);
    }

private:
    GpuPointwiseBinaryParams _params;
};

template <typename InputDataType, typename OutputDataType, typename ComputeDataType>
class GpuPointwiseUnaryPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuPointwiseUnaryPlan(GpuPointwiseUnaryParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<InputDataType> inputTensor(
            variantPack.at(_params.inputTensor.uid),
            _params.inputTensor.dims,
            _params.inputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> outputTensor(
            variantPack.at(_params.outputTensor.uid),
            _params.outputTensor.dims,
            _params.outputTensor.strides);

        const float lowerClip = _params.reluFwdLowerClip.value_or(0.0f);
        const float upperClip
            = _params.reluFwdUpperClip.value_or(std::numeric_limits<float>::max());
        const float lowerSlope = _params.reluFwdLowerSlope.value_or(0.0f);
        const float swishBeta = _params.swishFwdBeta.value_or(1.0f);
        const float eluAlpha = _params.eluFwdAlpha.value_or(1.0f);
        const float softplusBeta = _params.softplusFwdBeta.value_or(1.0f);

        hipdnn_gpu_ref::GpuReferencePointwise::
            pointwiseCompute<OutputDataType, InputDataType, ComputeDataType>(_params.pointwiseMode,
                                                                             outputTensor,
                                                                             inputTensor,
                                                                             lowerClip,
                                                                             upperClip,
                                                                             lowerSlope,
                                                                             swishBeta,
                                                                             eluAlpha,
                                                                             softplusBeta);
    }

private:
    GpuPointwiseUnaryParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType Input0DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType Input1DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuPointwiseBinaryPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using Input0DataType = hipdnn_test_sdk::utilities::DataTypeToNative<Input0DataTypeEnum>;
    using Input1DataType = hipdnn_test_sdk::utilities::DataTypeToNative<Input1DataTypeEnum>;
    using OutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        if(!nodeAttributes->in_1_tensor_uid().has_value())
        {
            return false;
        }

        // CMP_GT is the only mode with a BOOLEAN output, and BOOLEAN output is only valid for it.
        // NOLINTNEXTLINE(misc-const-correctness)
        auto op = nodeAttributes->operation();
        if(op == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_GT)
        {
            if(OutputDataTypeEnum != hipdnn_flatbuffers_sdk::data_objects::DataType::BOOLEAN)
            {
                return false;
            }
        }
        else if(OutputDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::BOOLEAN)
        {
            return false;
        }
        else
        {
            switch(op)
            {
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MAX_OP:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MIN_OP:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD:
            case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD:
                break;
            default:
                return false;
            }
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_0_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_1_tensor_uid().value());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_0_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_0_tensor_uid(), Input0DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_1_tensor_uid().value(), Input1DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_0_tensor_uid(), OutputDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        if(exceedsMaxRank(*tensorMap.at(nodeAttributes->out_0_tensor_uid())))
        {
            return false;
        }

        // Reject if any operand is runtime pass-by-value; the plan cannot resolve a PBV host scalar.
        return !anyOperandIsRuntimePassByValue(tensorMap,
                                               {nodeAttributes->in_0_tensor_uid(),
                                                nodeAttributes->out_0_tensor_uid(),
                                                nodeAttributes->in_1_tensor_uid().value()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PointwiseAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuPointwiseBinaryParams params(*tensorMap.at(nodeAttributes->in_0_tensor_uid()),
                                        *tensorMap.at(nodeAttributes->in_1_tensor_uid().value()),
                                        *tensorMap.at(nodeAttributes->out_0_tensor_uid()),
                                        nodeAttributes->operation(),
                                        nodeAttributes->relu_lower_clip(),
                                        nodeAttributes->relu_upper_clip(),
                                        nodeAttributes->relu_lower_clip_slope());
        return std::make_unique<GpuPointwiseBinaryPlan<Input0DataType,
                                                       Input1DataType,
                                                       OutputDataType,
                                                       ComputeDataType>>(std::move(params));
    }
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuPointwiseUnaryPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using InputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<InputDataTypeEnum>;
    using OutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        // NOLINTNEXTLINE(misc-const-correctness)
        auto op = nodeAttributes->operation();

        switch(op)
        {
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::NEG:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::EXP:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOG:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SQRT:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RSQRT:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RECIPROCAL:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIN:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TAN:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ERF:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CEIL:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::FLOOR:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_FWD:
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD:
            break;
        default:
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_0_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_0_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_0_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_0_tensor_uid(), OutputDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        if(exceedsMaxRank(*tensorMap.at(nodeAttributes->out_0_tensor_uid())))
        {
            return false;
        }

        // Reject if any operand is runtime pass-by-value; the plan cannot resolve a PBV host scalar.
        return !anyOperandIsRuntimePassByValue(
            tensorMap, {nodeAttributes->in_0_tensor_uid(), nodeAttributes->out_0_tensor_uid()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PointwiseAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuPointwiseUnaryParams params(*tensorMap.at(nodeAttributes->in_0_tensor_uid()),
                                       *tensorMap.at(nodeAttributes->out_0_tensor_uid()),
                                       nodeAttributes->operation(),
                                       nodeAttributes->relu_lower_clip(),
                                       nodeAttributes->relu_upper_clip(),
                                       nodeAttributes->relu_lower_clip_slope(),
                                       nodeAttributes->swish_beta(),
                                       nodeAttributes->elu_alpha(),
                                       nodeAttributes->softplus_beta());
        return std::make_unique<
            GpuPointwiseUnaryPlan<InputDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

// BINARY_SELECT: output = mask ? input0 : input1. The compute type does not affect the result.
template <hipdnn_flatbuffers_sdk::data_objects::DataType Input0DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType Input1DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuPointwiseTernaryPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using Input0DataType = hipdnn_test_sdk::utilities::DataTypeToNative<Input0DataTypeEnum>;
    using Input1DataType = hipdnn_test_sdk::utilities::DataTypeToNative<Input1DataTypeEnum>;
    using MaskDataType = hipdnn_test_sdk::utilities::DataTypeToNative<MaskDataTypeEnum>;
    using OutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<OutputDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        if(nodeAttributes->operation()
           != hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::BINARY_SELECT)
        {
            return false;
        }

        if(!nodeAttributes->in_1_tensor_uid().has_value()
           || !nodeAttributes->in_2_tensor_uid().has_value())
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_0_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_1_tensor_uid().value());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_2_tensor_uid().value());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_0_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_0_tensor_uid(), Input0DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_1_tensor_uid().value(), Input1DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_2_tensor_uid().value(), MaskDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_0_tensor_uid(), OutputDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        if(exceedsMaxRank(*tensorMap.at(nodeAttributes->out_0_tensor_uid())))
        {
            return false;
        }

        // Reject if any operand is runtime pass-by-value; the plan cannot resolve a PBV host scalar.
        return !anyOperandIsRuntimePassByValue(tensorMap,
                                               {nodeAttributes->in_0_tensor_uid(),
                                                nodeAttributes->in_1_tensor_uid().value(),
                                                nodeAttributes->in_2_tensor_uid().value(),
                                                nodeAttributes->out_0_tensor_uid()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PointwiseAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuPointwiseTernaryParams params(*tensorMap.at(nodeAttributes->in_0_tensor_uid()),
                                         *tensorMap.at(nodeAttributes->in_1_tensor_uid().value()),
                                         *tensorMap.at(nodeAttributes->in_2_tensor_uid().value()),
                                         *tensorMap.at(nodeAttributes->out_0_tensor_uid()),
                                         nodeAttributes->operation());
        return std::make_unique<
            GpuPointwiseTernaryPlan<Input0DataType, Input1DataType, MaskDataType, OutputDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
