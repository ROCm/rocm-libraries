// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h"
#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct LayernormBpropParams
{
    LayernormBpropParams() = default;
    LayernormBpropParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dscaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dbiasAttributes,
        const int64_t normalizedDimCount,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* epsilonAttributes = nullptr)
        : dyTensor(unpackTensorAttributes(dyAttributes))
        , xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , dxTensor(unpackTensorAttributes(dxAttributes))
        , dscaleTensor(unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(unpackTensorAttributes(dbiasAttributes))
        , normalizedDimCount(normalizedDimCount)
        , meanTensor(meanAttributes != nullptr
                         ? std::make_optional(unpackTensorAttributes(*meanAttributes))
                         : std::nullopt)
        , invVarianceTensor(invVarianceAttributes != nullptr
                                ? std::make_optional(unpackTensorAttributes(*invVarianceAttributes))
                                : std::nullopt)
        , epsilonTensor(epsilonAttributes != nullptr
                            ? std::make_optional(unpackTensorAttributes(*epsilonAttributes))
                            : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dscaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dbiasTensor;
    int64_t normalizedDimCount;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invVarianceTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> epsilonTensor;
};

template <typename DyDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType,
          typename DxDataType,
          typename ComputeDataType>
class LayernormBpropPlan : public IGraphNodePlanExecutor
{
public:
    LayernormBpropPlan(LayernormBpropParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.dxTensor.uid, _params.dscaleTensor.uid, _params.dbiasTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowDyTensor = createShallowTensor<DyDataType>(
            _params.dyTensor, variantPack.at(_params.dyTensor.uid));

        auto shallowXTensor
            = createShallowTensor<DxDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<MeanInvVarianceDataType>>
            shallowMeanTensor;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<MeanInvVarianceDataType>>
            shallowInvVarianceTensor;

        if(_params.meanTensor.has_value() && _params.invVarianceTensor.has_value())
        {
            shallowMeanTensor = createShallowTensor<MeanInvVarianceDataType>(
                _params.meanTensor.value(), variantPack.at(_params.meanTensor.value().uid));

            shallowInvVarianceTensor = createShallowTensor<MeanInvVarianceDataType>(
                _params.invVarianceTensor.value(),
                variantPack.at(_params.invVarianceTensor.value().uid));
        }

        auto shallowDxTensor = createShallowTensor<DxDataType>(
            _params.dxTensor, variantPack.at(_params.dxTensor.uid));

        auto shallowDscaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.dscaleTensor, variantPack.at(_params.dscaleTensor.uid));

        auto shallowDbiasTensor = createShallowTensor<ScaleBiasDataType>(
            _params.dbiasTensor, variantPack.at(_params.dbiasTensor.uid));

        // Extract epsilon from pass-by-value tensor (cast to double)
        double epsilon = 1e-5;
        if(_params.epsilonTensor.has_value())
        {
            epsilon = hipdnn_flatbuffers_sdk::utilities::resolveDoubleScalarFromVariantPack(
                _params.epsilonTensor.value(), variantPack, "Epsilon");
        }

        utilities::CpuFpReferenceLayernorm::bprop<DyDataType,
                                                  ScaleBiasDataType,
                                                  DxDataType,
                                                  MeanInvVarianceDataType,
                                                  ComputeDataType>(*shallowDyTensor,
                                                                   *shallowXTensor,
                                                                   *shallowScaleTensor,
                                                                   *shallowDxTensor,
                                                                   *shallowDscaleTensor,
                                                                   *shallowDbiasTensor,
                                                                   epsilon,
                                                                   shallowMeanTensor.get(),
                                                                   shallowInvVarianceTensor.get(),
                                                                   _params.normalizedDimCount);
    }

private:
    LayernormBpropParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class LayernormBpropPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DyDataType = utilities::DataTypeToNative<DyDataTypeEnum>;
    using ScaleBiasDataType = utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanInvVarianceDataType = utilities::DataTypeToNative<MeanInvVarianceDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_LayernormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        // Check required tensors
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dscale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dbias_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), DxDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dbias_tensor_uid(), ScaleBiasDataTypeEnum);

        const bool hasMean = nodeAttributes->mean_tensor_uid().has_value();
        const bool hasInvVariance = nodeAttributes->inv_variance_tensor_uid().has_value();
        if(hasMean != hasInvVariance)
        {
            return false;
        }

        if(hasMean)
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid(), MeanInvVarianceDataTypeEnum);
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanInvVarianceDataTypeEnum);
        }

        // Optional epsilon tensor
        if(nodeAttributes->epsilon_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_LayernormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type LayernormBackwardAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttr = nullptr;
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarAttr = nullptr;

        const bool hasMean = nodeAttributes->mean_tensor_uid().has_value();
        const bool hasInvVariance = nodeAttributes->inv_variance_tensor_uid().has_value();
        if(hasMean && hasInvVariance)
        {
            meanAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            invVarAttr = tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value());
        }

        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* epsilon = nullptr;

        if(nodeAttributes->epsilon_tensor_uid().has_value())
        {
            epsilon = tensorMap.at(nodeAttributes->epsilon_tensor_uid().value());
        }

        LayernormBpropParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->dbias_tensor_uid()),
                                    nodeAttributes->normalized_dim_count(),
                                    meanAttr,
                                    invVarAttr,
                                    epsilon);

        return std::make_unique<LayernormBpropPlan<DyDataType,
                                                   ScaleBiasDataType,
                                                   MeanInvVarianceDataType,
                                                   DxDataType,
                                                   ComputeDataType>>(std::move(params));
    }
};
} // namespace hipdnn_test_sdk::detail
