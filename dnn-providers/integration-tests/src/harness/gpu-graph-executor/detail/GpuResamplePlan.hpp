// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <optional>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// ===========================================================================
// GPU resample forward and backward plans
// ===========================================================================

struct GpuResampleFwdParams
{
    GpuResampleFwdParams() = default;
    GpuResampleFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const std::vector<int64_t>& prePadding,
        const std::vector<int64_t>& postPadding,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& window,
        const hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
        const hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* indexAttributes = nullptr)
        : xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , yTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(yAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , window(window)
        , resampleMode(resampleMode)
        , paddingMode(paddingMode)
        , indexTensor(indexAttributes != nullptr
                          ? std::make_optional(
                                hipdnn_test_sdk::detail::unpackTensorAttributes(*indexAttributes))
                          : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode
        = hipdnn_flatbuffers_sdk::data_objects::ResampleMode::NOT_SET;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode
        = hipdnn_flatbuffers_sdk::data_objects::PaddingMode::PADDING_NOT_SET;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> indexTensor;
};

struct GpuResampleBwdParams
{
    GpuResampleBwdParams() = default;
    GpuResampleBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const std::vector<int64_t>& prePadding,
        const std::vector<int64_t>& postPadding,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& window,
        const hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
        const hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* indexAttributes = nullptr)
        : dyTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dyAttributes))
        , dxTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dxAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , window(window)
        , resampleMode(resampleMode)
        , paddingMode(paddingMode)
        , indexTensor(indexAttributes != nullptr
                          ? std::make_optional(
                                hipdnn_test_sdk::detail::unpackTensorAttributes(*indexAttributes))
                          : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode
        = hipdnn_flatbuffers_sdk::data_objects::ResampleMode::NOT_SET;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode
        = hipdnn_flatbuffers_sdk::data_objects::PaddingMode::PADDING_NOT_SET;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> indexTensor;
};

// ===========================================================================
// GPU resample forward and backward plans
// ===========================================================================

template <typename XDataType, typename YDataType, typename ComputeDataType, typename IndexDataType>
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

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<IndexDataType>> indexTensor;
        if(_params.indexTensor.has_value())
        {
            indexTensor.emplace(variantPack.at(_params.indexTensor->uid),
                                _params.indexTensor->dims,
                                _params.indexTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceResample::
            forward<XDataType, YDataType, ComputeDataType, IndexDataType>(
                xTensor,
                yTensor,
                _params.prePadding,
                _params.postPadding,
                _params.stride,
                _params.window,
                _params.resampleMode,
                _params.paddingMode,
                indexTensor.has_value() ? &indexTensor.value() : nullptr);
    }

private:
    GpuResampleFwdParams _params;
};

template <typename DyDataType,
          typename DxDataType,
          typename ComputeDataType,
          typename IndexDataType>
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

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<IndexDataType>> indexTensor;
        if(_params.indexTensor.has_value())
        {
            indexTensor.emplace(variantPack.at(_params.indexTensor->uid),
                                _params.indexTensor->dims,
                                _params.indexTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceResample::
            backward<DyDataType, DxDataType, ComputeDataType, IndexDataType>(
                dyTensor,
                dxTensor,
                _params.prePadding,
                _params.postPadding,
                _params.stride,
                _params.window,
                _params.resampleMode,
                _params.paddingMode,
                indexTensor.has_value() ? &indexTensor.value() : nullptr);
    }

private:
    GpuResampleBwdParams _params;
};

// ==========================================================================
// GPU resample forward and backward plan builders
// ==========================================================================

template <hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
struct ResampleIndexNative
{
    using type = hipdnn_test_sdk::utilities::DataTypeToNative<IndexDataTypeEnum>;
};

template <>
struct ResampleIndexNative<hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
{
    using type = int32_t;
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
    using IndexDataType = typename ResampleIndexNative<IndexDataTypeEnum>::type;

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

        const auto resampleMode = nodeAttributes->resample_mode();
        if(resampleMode != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL
           && resampleMode
                  != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
           && resampleMode
                  != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
        {
            return false;
        }

        const auto paddingMode = nodeAttributes->padding_mode();
        if(paddingMode != hipdnn_flatbuffers_sdk::data_objects::PaddingMode::NEG_INF_PAD
           && paddingMode != hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), YDataTypeEnum);

        if constexpr(IndexDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        {
            if(nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
        }
        else
        {
            if(!nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->index_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->index_tensor_uid().value(), IndexDataTypeEnum);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        const std::vector<int64_t> operandUids = [&]() {
            std::vector<int64_t> uids
                = {nodeAttributes->x_tensor_uid(), nodeAttributes->y_tensor_uid()};

            if constexpr(IndexDataTypeEnum != hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
            {
                if(nodeAttributes->index_tensor_uid().has_value())
                {
                    uids.push_back(nodeAttributes->index_tensor_uid().value());
                }
            }
            return uids;
        }();

        // Reject if any operand is runtime pass-by-value. This plan has no scalar
        // operands, but check anyway so the guard holds if the schema ever adds one.
        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ResampleFwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuResampleFwdParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->pre_padding()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->post_padding()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->stride()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->window()),
            nodeAttributes->resample_mode(),
            nodeAttributes->padding_mode(),
            nodeAttributes->index_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->index_tensor_uid().value())
                : nullptr);

        return std::make_unique<
            GpuResampleFwdPlan<XDataType, YDataType, ComputeDataType, IndexDataType>>(
            std::move(params));
    }
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
    using IndexDataType = typename ResampleIndexNative<IndexDataTypeEnum>::type;

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

        const auto resampleMode = nodeAttributes->resample_mode();
        if(resampleMode != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL
           && resampleMode
                  != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
           && resampleMode
                  != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
        {
            return false;
        }

        const auto paddingMode = nodeAttributes->padding_mode();
        if(paddingMode != hipdnn_flatbuffers_sdk::data_objects::PaddingMode::NEG_INF_PAD
           && paddingMode != hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);

        if constexpr(IndexDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        {
            if(nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }

            if(resampleMode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
            {
                return false;
            }
        }
        else
        {
            if(!nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->index_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->index_tensor_uid().value(), IndexDataTypeEnum);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        const std::vector<int64_t> operandUids = [&]() {
            std::vector<int64_t> uids
                = {nodeAttributes->dy_tensor_uid(), nodeAttributes->dx_tensor_uid()};

            if constexpr(IndexDataTypeEnum != hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
            {
                if(nodeAttributes->index_tensor_uid().has_value())
                {
                    uids.push_back(nodeAttributes->index_tensor_uid().value());
                }
            }
            return uids;
        }();

        // Reject if any operand is runtime pass-by-value. This plan has no scalar
        // operands, but check anyway so the guard holds if the schema ever adds one.
        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ResampleBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ResampleBwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuResampleBwdParams params(
            *tensorMap.at(nodeAttributes->dy_tensor_uid()),
            *tensorMap.at(nodeAttributes->dx_tensor_uid()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->pre_padding()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->post_padding()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->stride()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->window()),
            nodeAttributes->resample_mode(),
            nodeAttributes->padding_mode(),
            nodeAttributes->index_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->index_tensor_uid().value())
                : nullptr);

        return std::make_unique<
            GpuResampleBwdPlan<DyDataType, DxDataType, ComputeDataType, IndexDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
