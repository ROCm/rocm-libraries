// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct BlockScaleDequantizeParams
{
    BlockScaleDequantizeParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        std::vector<int32_t> blockSize,
        bool isNegativeScale)
        : xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , blockSize(std::move(blockSize))
        , isNegativeScale(isNegativeScale)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    std::vector<int32_t> blockSize;
    bool isNegativeScale;
};

template <typename XDataType,
          typename ScaleDataType,
          typename OutputDataType,
          typename ComputeDataType>
class BlockScaleDequantizePlan : public IGraphNodePlanExecutor
{
public:
    BlockScaleDequantizePlan(BlockScaleDequantizeParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.yTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        utilities::CpuFpReferenceBlockScaleDequantize::
            dequantize<XDataType, ScaleDataType, OutputDataType, ComputeDataType>(
                *shallowXTensor,
                *shallowScaleTensor,
                *shallowYTensor,
                _params.blockSize,
                _params.isNegativeScale);
    }

private:
    BlockScaleDequantizeParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class BlockScaleDequantizePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleDataType = utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type BlockScaleDequantizeAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        std::vector<int32_t> blockSize;
        if(nodeAttributes->block_size() != nullptr)
        {
            const auto* bs = nodeAttributes->block_size();
            blockSize.assign(bs->begin(), bs->end());
        }

        BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                          *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                          *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                          std::move(blockSize),
                                          nodeAttributes->is_negative_scale());

        return std::make_unique<
            BlockScaleDequantizePlan<XDataType, ScaleDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};
} // namespace hipdnn_test_sdk::detail
