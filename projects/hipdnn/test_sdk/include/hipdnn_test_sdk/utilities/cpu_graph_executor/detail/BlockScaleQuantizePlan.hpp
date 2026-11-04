// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleQuantize.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct BlockScaleQuantizeParams
{
    BlockScaleQuantizeParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        int32_t blockSize,
        std::optional<int64_t> axis = std::nullopt)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , blockSize(blockSize)
        , axis(axis)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    int32_t blockSize;
    std::optional<int64_t> axis;
};

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleDataType,
          typename ComputeDataType>
class BlockScaleQuantizePlan : public IGraphNodePlanExecutor
{
public:
    BlockScaleQuantizePlan(BlockScaleQuantizeParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.yTensor.uid, _params.scaleTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor = createShallowTensor<InputDataType>(
            _params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        utilities::CpuFpReferenceBlockScaleQuantize::
            quantize<InputDataType, OutputDataType, ScaleDataType, ComputeDataType>(
                *shallowXTensor,
                *shallowYTensor,
                *shallowScaleTensor,
                _params.blockSize,
                _params.axis);
    }

private:
    BlockScaleQuantizeParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class BlockScaleQuantizePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using InputDataType = utilities::DataTypeToNative<InputDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ScaleDataType = utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BlockScaleQuantizeAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type BlockScaleQuantizeAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const auto blockSize = nodeAttributes->block_size();
        const auto axis = nodeAttributes->axis()
                              ? std::make_optional(nodeAttributes->axis().value())
                              : std::nullopt;

        BlockScaleQuantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                        *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                        *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                        blockSize,
                                        axis);

        return std::make_unique<
            BlockScaleQuantizePlan<InputDataType, OutputDataType, ScaleDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
