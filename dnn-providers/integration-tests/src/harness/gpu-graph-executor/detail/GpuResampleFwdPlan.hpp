// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
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

struct GpuResampleFwdParams
{
    GpuResampleFwdParams() = default;
    GpuResampleFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* indexAttributes = nullptr)
        : xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , yTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(yAttributes))
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
        // post_padding is optional in the schema; default it to zeros so the GPU reference,
        // which always takes both paddings, matches the CPU reference behavior.
        if(postPadding.empty())
        {
            postPadding.assign(prePadding.size(), 0);
        }
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
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

template <typename XDataType,
          typename YDataType,
          typename ComputeDataType,
          typename IndexDataType = int32_t>
class GpuResampleFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuResampleFwdPlan(GpuResampleFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XDataType> xTensor(
            variantPack.at(_params.xTensor.uid), _params.xTensor.dims, _params.xTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<YDataType> yTensor(
            variantPack.at(_params.yTensor.uid), _params.yTensor.dims, _params.yTensor.strides);

        std::unique_ptr<hipdnn_gpu_ref::ShallowGpuTensor<IndexDataType>> indexTensor;
        if(_params.indexTensor.has_value())
        {
            indexTensor = std::make_unique<hipdnn_gpu_ref::ShallowGpuTensor<IndexDataType>>(
                variantPack.at(_params.indexTensor->uid),
                _params.indexTensor->dims,
                _params.indexTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceResample::
            forward<XDataType, YDataType, ComputeDataType, IndexDataType>(xTensor,
                                                                          yTensor,
                                                                          _params.prePadding,
                                                                          _params.postPadding,
                                                                          _params.stride,
                                                                          _params.window,
                                                                          _params.resampleMode,
                                                                          _params.paddingMode,
                                                                          indexTensor.get());
    }

private:
    GpuResampleFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
class GpuResampleFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using XDataType = hipdnn_test_sdk::utilities::DataTypeToNative<XDataTypeEnum>;
    using YDataType = hipdnn_test_sdk::utilities::DataTypeToNative<YDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        if(!isSupportedResampleMode(nodeAttributes->resample_mode())
           || !isSupportedPaddingMode(nodeAttributes->padding_mode()))
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), YDataTypeEnum);

        std::vector<int64_t> operandUids{nodeAttributes->x_tensor_uid(),
                                         nodeAttributes->y_tensor_uid()};

        // An index output is produced only for MAXPOOL, and only an INT32 index is registered.
        if constexpr(IndexDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        {
            if(nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
        }
        else
        {
            if(!nodeAttributes->index_tensor_uid().has_value()
               || nodeAttributes->resample_mode()
                      != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->index_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->index_tensor_uid().value(), IndexDataTypeEnum);
            operandUids.push_back(nodeAttributes->index_tensor_uid().value());
        }

        // The GPU reference only supports 4D/5D tensors with spatial parameters of matching rank
        // and a padding smaller than the window in every spatial dimension.
        const auto* xAttributes = tensorMap.at(nodeAttributes->x_tensor_uid());
        if(!hasSupportedResampleGeometry(xAttributes->dims(),
                                         nodeAttributes->pre_padding(),
                                         nodeAttributes->post_padding(),
                                         nodeAttributes->stride(),
                                         nodeAttributes->window(),
                                         /*requirePaddingBelowWindow=*/true))
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
        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::invalid_argument("Node attributes are not of type ResampleFwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const auto* indexAttributes = nodeAttributes->index_tensor_uid().has_value()
                                          ? tensorMap.at(*nodeAttributes->index_tensor_uid())
                                          : nullptr;

        GpuResampleFwdParams params(*nodeAttributes,
                                    *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                    indexAttributes);
        return std::make_unique<
            GpuResampleFwdPlan<XDataType, YDataType, ComputeDataType, IndexDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
