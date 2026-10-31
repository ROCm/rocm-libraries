// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceBlockScale.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBlockScaleQuantizeParams
{
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT x;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT y;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scale;
    int32_t blockSize;
    std::optional<int64_t> axis;
};

struct GpuBlockScaleDequantizeParams
{
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT x;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scale;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT y;
    std::vector<int32_t> blockSize;
    bool isNegativeScale;
};

template <typename XType, typename YType, typename ScaleType, typename ComputeType>
class GpuBlockScaleQuantizePlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuBlockScaleQuantizePlan(GpuBlockScaleQuantizeParams params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XType> x(
            variantPack.at(_params.x.uid), _params.x.dims, _params.x.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<YType> y(
            variantPack.at(_params.y.uid), _params.y.dims, _params.y.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleType> scale(
            variantPack.at(_params.scale.uid), _params.scale.dims, _params.scale.strides);
        hipdnn_gpu_ref::GpuFpReferenceBlockScale::quantize<XType, YType, ScaleType, ComputeType>(
            x, y, scale, _params.blockSize, _params.axis);
    }

private:
    GpuBlockScaleQuantizeParams _params;
};

template <typename XType, typename ScaleType, typename YType, typename ComputeType>
class GpuBlockScaleDequantizePlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuBlockScaleDequantizePlan(GpuBlockScaleDequantizeParams params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XType> x(
            variantPack.at(_params.x.uid), _params.x.dims, _params.x.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleType> scale(
            variantPack.at(_params.scale.uid), _params.scale.dims, _params.scale.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<YType> y(
            variantPack.at(_params.y.uid), _params.y.dims, _params.y.strides);
        hipdnn_gpu_ref::GpuFpReferenceBlockScale::dequantize<XType, ScaleType, YType, ComputeType>(
            x, scale, y, _params.blockSize, _params.isNegativeScale);
    }

private:
    GpuBlockScaleDequantizeParams _params;
};

inline bool
    validTensorMetadata(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    return tensor.dims() != nullptr && tensor.strides() != nullptr && !tensor.dims()->empty()
           && tensor.dims()->size() == tensor.strides()->size() && tensor.dims()->size() <= 5;
}

template <hipdnn_flatbuffers_sdk::data_objects::DataType XEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeEnum>
class GpuBlockScaleQuantizePlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* attributes = node.attributes_as_BlockScaleQuantizeAttributes();
        if(attributes == nullptr || node.compute_data_type() != ComputeEnum)
        {
            return false;
        }
        CHECK_TENSOR_EXISTS(tensorMap, attributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, attributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, attributes->scale_tensor_uid());
        CHECK_TENSOR_TYPE(tensorMap, attributes->x_tensor_uid(), XEnum);
        CHECK_TENSOR_TYPE(tensorMap, attributes->y_tensor_uid(), YEnum);
        CHECK_TENSOR_TYPE(tensorMap, attributes->scale_tensor_uid(), ScaleEnum);
        CHECK_NO_RAGGED_TENSORS(tensorMap);

        const auto& x = *tensorMap.at(attributes->x_tensor_uid());
        const auto& y = *tensorMap.at(attributes->y_tensor_uid());
        const auto& scale = *tensorMap.at(attributes->scale_tensor_uid());
        if(!validTensorMetadata(x) || !validTensorMetadata(y) || !validTensorMetadata(scale)
           || attributes->block_size() <= 0)
        {
            return false;
        }
        const std::vector<int64_t> xDims(x.dims()->begin(), x.dims()->end());
        const std::vector<int64_t> yDims(y.dims()->begin(), y.dims()->end());
        const std::vector<int64_t> scaleDims(scale.dims()->begin(), scale.dims()->end());
        const auto axis = attributes->axis().value_or(static_cast<int64_t>(xDims.size()) - 1);
        if(xDims != yDims || scaleDims.size() != xDims.size() || axis < 0
           || axis >= static_cast<int64_t>(xDims.size()))
        {
            return false;
        }
        auto expectedScaleDims = xDims;
        expectedScaleDims[static_cast<size_t>(axis)]
            = (xDims[static_cast<size_t>(axis)] + attributes->block_size() - 1)
              / attributes->block_size();
        return scaleDims == expectedScaleDims;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* attributes = node.attributes_as_BlockScaleQuantizeAttributes();
        if(attributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not BlockScaleQuantizeAttributes");
        }
        const auto& tensors = graph.getTensorMap();
        GpuBlockScaleQuantizeParams params{
            hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensors.at(attributes->x_tensor_uid())),
            hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensors.at(attributes->y_tensor_uid())),
            hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensors.at(attributes->scale_tensor_uid())),
            attributes->block_size(),
            attributes->axis() ? std::make_optional(attributes->axis().value()) : std::nullopt};
        using XType = hipdnn_test_sdk::utilities::DataTypeToNative<XEnum>;
        using YType = hipdnn_test_sdk::utilities::DataTypeToNative<YEnum>;
        using ScaleType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleEnum>;
        using ComputeType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeEnum>;
        return std::make_unique<GpuBlockScaleQuantizePlan<XType, YType, ScaleType, ComputeType>>(
            std::move(params));
    }
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeEnum>
class GpuBlockScaleDequantizePlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* attributes = node.attributes_as_BlockScaleDequantizeAttributes();
        if(attributes == nullptr || node.compute_data_type() != ComputeEnum
           || attributes->block_size() == nullptr)
        {
            return false;
        }
        CHECK_TENSOR_EXISTS(tensorMap, attributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, attributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, attributes->scale_tensor_uid());
        CHECK_TENSOR_TYPE(tensorMap, attributes->x_tensor_uid(), XEnum);
        CHECK_TENSOR_TYPE(tensorMap, attributes->y_tensor_uid(), YEnum);
        CHECK_TENSOR_TYPE(tensorMap, attributes->scale_tensor_uid(), ScaleEnum);
        CHECK_NO_RAGGED_TENSORS(tensorMap);

        const auto& x = *tensorMap.at(attributes->x_tensor_uid());
        const auto& y = *tensorMap.at(attributes->y_tensor_uid());
        const auto& scale = *tensorMap.at(attributes->scale_tensor_uid());
        if(!validTensorMetadata(x) || !validTensorMetadata(y) || !validTensorMetadata(scale))
        {
            return false;
        }
        const std::vector<int64_t> xDims(x.dims()->begin(), x.dims()->end());
        const std::vector<int64_t> yDims(y.dims()->begin(), y.dims()->end());
        const std::vector<int64_t> scaleDims(scale.dims()->begin(), scale.dims()->end());
        if(xDims != yDims || scaleDims.size() != xDims.size()
           || attributes->block_size()->size() > xDims.size())
        {
            return false;
        }
        std::vector<int64_t> effectiveBlockSize(xDims.size(), 1);
        const auto offset = xDims.size() - attributes->block_size()->size();
        size_t blockDimension = 0;
        for(const auto value : *attributes->block_size())
        {
            if(value <= 0)
            {
                return false;
            }
            effectiveBlockSize[offset + blockDimension] = value;
            ++blockDimension;
        }
        for(size_t dim = 0; dim < xDims.size(); ++dim)
        {
            if(scaleDims[dim]
               != (xDims[dim] + effectiveBlockSize[dim] - 1) / effectiveBlockSize[dim])
            {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* attributes = node.attributes_as_BlockScaleDequantizeAttributes();
        if(attributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not BlockScaleDequantizeAttributes");
        }
        const auto& tensors = graph.getTensorMap();
        std::vector<int32_t> blockSize(attributes->block_size()->begin(),
                                       attributes->block_size()->end());
        GpuBlockScaleDequantizeParams params{hipdnn_test_sdk::detail::unpackTensorAttributes(
                                                 *tensors.at(attributes->x_tensor_uid())),
                                             hipdnn_test_sdk::detail::unpackTensorAttributes(
                                                 *tensors.at(attributes->scale_tensor_uid())),
                                             hipdnn_test_sdk::detail::unpackTensorAttributes(
                                                 *tensors.at(attributes->y_tensor_uid())),
                                             std::move(blockSize),
                                             attributes->is_negative_scale()};
        using XType = hipdnn_test_sdk::utilities::DataTypeToNative<XEnum>;
        using ScaleType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleEnum>;
        using YType = hipdnn_test_sdk::utilities::DataTypeToNative<YEnum>;
        using ComputeType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeEnum>;
        return std::make_unique<GpuBlockScaleDequantizePlan<XType, ScaleType, YType, ComputeType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
