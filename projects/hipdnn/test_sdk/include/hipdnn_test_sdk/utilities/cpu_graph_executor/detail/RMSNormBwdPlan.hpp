// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRMSNorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct RMSNormBwdParams
{
    RMSNormBwdParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
                     const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
                     const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
                     const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& invRmsAttributes,
                     const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
                     const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dscaleAttributes,
                     const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dbiasAttributes
                     = nullptr)
        : dyTensor(unpackTensorAttributes(dyAttributes))
        , xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , invRmsTensor(unpackTensorAttributes(invRmsAttributes))
        , dxTensor(unpackTensorAttributes(dxAttributes))
        , dscaleTensor(unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(dbiasAttributes != nullptr
                          ? std::make_optional(unpackTensorAttributes(*dbiasAttributes))
                          : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT invRmsTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dscaleTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dbiasTensor;
};

template <typename DyDataType,
          typename XDataType,
          typename ScaleDataType,
          typename DxDataType,
          typename ComputeDataType>
class RMSNormBwdPlan : public IGraphNodePlanExecutor
{
public:
    RMSNormBwdPlan(RMSNormBwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids = {_params.dxTensor.uid, _params.dscaleTensor.uid};
        if(_params.dbiasTensor.has_value())
        {
            ids.push_back(_params.dbiasTensor->uid);
        }
        return ids;
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowDyTensor = createShallowTensor<DyDataType>(
            _params.dyTensor, variantPack.at(_params.dyTensor.uid));

        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        auto shallowInvRmsTensor = createShallowTensor<ComputeDataType>(
            _params.invRmsTensor, variantPack.at(_params.invRmsTensor.uid));

        auto shallowDxTensor = createShallowTensor<DxDataType>(
            _params.dxTensor, variantPack.at(_params.dxTensor.uid));

        auto shallowDScaleTensor = createShallowTensor<ScaleDataType>(
            _params.dscaleTensor, variantPack.at(_params.dscaleTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>> shallowDBiasTensor;
        if(_params.dbiasTensor.has_value())
        {
            shallowDBiasTensor = createShallowTensor<ScaleDataType>(
                *_params.dbiasTensor, variantPack.at(_params.dbiasTensor->uid));
        }

        utilities::CpuFpReferenceRMSNorm::
            backward<DyDataType, XDataType, ScaleDataType, DxDataType, ComputeDataType>(
                *shallowDyTensor,
                *shallowXTensor,
                *shallowScaleTensor,
                *shallowInvRmsTensor,
                *shallowDxTensor,
                *shallowDScaleTensor,
                shallowDBiasTensor.get());
    }

private:
    RMSNormBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class RMSNormBwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DyDataType = utilities::DataTypeToNative<DyDataTypeEnum>;
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleDataType = utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using DxDataType = utilities::DataTypeToNative<DxDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_RMSNormBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_rms_tensor_uid());

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dscale_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->inv_rms_tensor_uid(), ComputeDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleDataTypeEnum);

        if(nodeAttributes->dbias_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dbias_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->dbias_tensor_uid().value(), ScaleDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type RMSNormBwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const auto* dbiasPtr = nodeAttributes->dbias_tensor_uid().has_value()
                                   ? tensorMap.at(nodeAttributes->dbias_tensor_uid().value())
                                   : nullptr;

        return std::make_unique<
            RMSNormBwdPlan<DyDataType, XDataType, ScaleDataType, DxDataType, ComputeDataType>>(
            RMSNormBwdParams(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                             *tensorMap.at(nodeAttributes->x_tensor_uid()),
                             *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                             *tensorMap.at(nodeAttributes->inv_rms_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                             dbiasPtr));
    }
};

} // namespace hipdnn_test_sdk::detail
