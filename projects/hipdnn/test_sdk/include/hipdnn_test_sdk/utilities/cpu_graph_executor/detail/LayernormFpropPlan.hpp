// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct LayernormFpropParams
{
    LayernormFpropParams() = default;

    LayernormFpropParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& epsilonAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& biasAttributes,
        const int64_t normalizedDimCount,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes = nullptr)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , epsilonTensor(unpackTensorAttributes(epsilonAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , biasTensor(unpackTensorAttributes(biasAttributes))
        , normalizedDimCount(normalizedDimCount)
        , meanTensor(meanAttributes != nullptr
                         ? std::make_optional(unpackTensorAttributes(*meanAttributes))
                         : std::nullopt)
        , invVarianceTensor(invVarianceAttributes != nullptr
                                ? std::make_optional(unpackTensorAttributes(*invVarianceAttributes))
                                : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT biasTensor;
    int64_t normalizedDimCount;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invVarianceTensor;
};

template <typename XDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType,
          typename OutputDataType,
          typename ComputeDataType>
class LayernormFpropPlan : public IGraphNodePlanExecutor
{
public:
    explicit LayernormFpropPlan(LayernormFpropParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids = {_params.yTensor.uid};
        if(_params.meanTensor.has_value())
        {
            ids.push_back(_params.meanTensor->uid);
        }
        if(_params.invVarianceTensor.has_value())
        {
            ids.push_back(_params.invVarianceTensor->uid);
        }
        return ids;
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        const double epsilon = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
            _params.epsilonTensor, "Epsilon");

        auto scaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        auto biasTensor = createShallowTensor<ScaleBiasDataType>(
            _params.biasTensor, variantPack.at(_params.biasTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<MeanInvVarianceDataType>> meanTensor;
        hipdnn_data_sdk::utilities::TensorBase<MeanInvVarianceDataType>* meanPtr = nullptr;

        if(_params.meanTensor.has_value())
        {
            meanTensor = createShallowTensor<MeanInvVarianceDataType>(
                *_params.meanTensor, variantPack.at(_params.meanTensor->uid));
            meanPtr = meanTensor.get();
        }

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<MeanInvVarianceDataType>> invVarianceTensor;
        hipdnn_data_sdk::utilities::TensorBase<MeanInvVarianceDataType>* invVariancePtr = nullptr;

        if(_params.invVarianceTensor.has_value())
        {
            invVarianceTensor = createShallowTensor<MeanInvVarianceDataType>(
                *_params.invVarianceTensor, variantPack.at(_params.invVarianceTensor->uid));
            invVariancePtr = invVarianceTensor.get();
        }

        utilities::CpuReferenceLayernorm::fprop(*shallowXTensor,
                                               scaleTensor.get(),
                                               biasTensor.get(),
                                               *shallowYTensor,
                                               epsilon,
                                               _params.normalizedDimCount,
                                               meanPtr,
                                               invVariancePtr);
    }

private:
    LayernormFpropParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class LayernormFpropPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleBiasDataType = utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanInvVarianceDataType = utilities::DataTypeToNative<MeanInvVarianceDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>& tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_LayernormAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);

        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid(), MeanInvVarianceDataTypeEnum);
        }

        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanInvVarianceDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
    buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                  const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_LayernormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type LayernormAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const auto* mean = nodeAttributes->mean_tensor_uid().has_value()
                               ? tensorMap.at(nodeAttributes->mean_tensor_uid().value())
                               : nullptr;

        const auto* invVariance = nodeAttributes->inv_variance_tensor_uid().has_value()
                                     ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value())
                                     : nullptr;

        return std::make_unique<LayernormFpropPlan<XDataType,
                                                  ScaleBiasDataType,
                                                  MeanInvVarianceDataType,
                                                  OutputDataType,
                                                  ComputeDataType>>(
            LayernormFpropParams(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                 *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                 *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
                                 *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                 *tensorMap.at(nodeAttributes->bias_tensor_uid()),
                                 nodeAttributes->normalized_dim_count(),
                                 mean,
                                 invVariance));
    }
};

} // namespace hipdnn_test_sdk::detail