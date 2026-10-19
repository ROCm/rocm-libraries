// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "GpuResampleUtils.hpp"
#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuResampleBwdParams
{
    GpuResampleBwdParams() = default;
    GpuResampleBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& resampleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* indexAttributes = nullptr)
        : dyTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dyAttributes))
        , dxTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dxAttributes))
        , indexTensor(indexAttributes != nullptr
                          ? std::make_optional(
                                hipdnn_test_sdk::detail::unpackTensorAttributes(*indexAttributes))
                          : std::nullopt)
        , prePadding(toSpatialVector(resampleAttributes.pre_padding()))
        , postPadding(toSpatialVector(resampleAttributes.post_padding()))
        , stride(toSpatialVector(resampleAttributes.stride()))
        , window(toSpatialVector(resampleAttributes.window()))
        , resampleMode(resampleAttributes.resample_mode())
        , paddingMode(resampleAttributes.padding_mode())
    {
        if(postPadding.empty())
        {
            postPadding.assign(prePadding.size(), 0);
        }
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> indexTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode
        = hipdnn_flatbuffers_sdk::data_objects::ResampleMode::NOT_SET;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode
        = hipdnn_flatbuffers_sdk::data_objects::PaddingMode::NOT_SET;
};

template <typename DyDataType,
          typename DxDataType,
          typename ComputeDataType,
          typename IndexDataType = int32_t>
class GpuResampleBwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuResampleBwdPlan(GpuResampleBwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<DyDataType> dyTensor(
            variantPack.at(_params.dyTensor.uid), _params.dyTensor.dims, _params.dyTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<DxDataType> dxTensor(
            variantPack.at(_params.dxTensor.uid), _params.dxTensor.dims, _params.dxTensor.strides);

        std::unique_ptr<hipdnn_gpu_ref::ShallowGpuTensor<IndexDataType>> indexTensor;
        if(_params.indexTensor.has_value())
        {
            indexTensor = std::make_unique<hipdnn_gpu_ref::ShallowGpuTensor<IndexDataType>>(
                variantPack.at(_params.indexTensor->uid),
                _params.indexTensor->dims,
                _params.indexTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceResample::
            backward<DyDataType, DxDataType, ComputeDataType, IndexDataType>(dyTensor,
                                                                             dxTensor,
                                                                             _params.prePadding,
                                                                             _params.postPadding,
                                                                             _params.stride,
                                                                             _params.window,
                                                                             _params.resampleMode,
                                                                             _params.paddingMode,
                                                                             indexTensor.get());
    }

private:
    GpuResampleBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
class GpuResampleBwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using DyDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DyDataTypeEnum>;
    using DxDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DxDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;
    using IndexDataType = typename GpuResampleIndexNative<IndexDataTypeEnum>::type;

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

        const auto* nodeAttributes = node.attributes_as_ResampleBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        if(!isSupportedResampleMode(nodeAttributes->resample_mode())
           || !isSupportedPaddingMode(nodeAttributes->padding_mode()))
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);

        std::vector<int64_t> operandUids{nodeAttributes->dy_tensor_uid(),
                                         nodeAttributes->dx_tensor_uid()};

        // The backward reference requires an index tensor for MAXPOOL and rejects one otherwise.
        const bool isMaxPool = nodeAttributes->resample_mode()
                               == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL;
        if constexpr(IndexDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        {
            if(isMaxPool || nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
        }
        else
        {
            if(!isMaxPool || !nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->index_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->index_tensor_uid().value(), IndexDataTypeEnum);
            operandUids.push_back(nodeAttributes->index_tensor_uid().value());
        }

        const auto* dxAttributes = tensorMap.at(nodeAttributes->dx_tensor_uid());
        if(!hasSupportedResampleGeometry(dxAttributes->dims(),
                                         nodeAttributes->pre_padding(),
                                         nodeAttributes->post_padding(),
                                         nodeAttributes->stride(),
                                         nodeAttributes->window(),
                                         /*requirePaddingBelowWindow=*/false))
        {
            return false;
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ResampleBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::invalid_argument("Node attributes are not of type ResampleBwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const auto* indexAttributes = nodeAttributes->index_tensor_uid().has_value()
                                          ? tensorMap.at(*nodeAttributes->index_tensor_uid())
                                          : nullptr;

        GpuResampleBwdParams params(*nodeAttributes,
                                    *tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                    indexAttributes);
        return std::make_unique<
            GpuResampleBwdPlan<DyDataType, DxDataType, ComputeDataType, IndexDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
