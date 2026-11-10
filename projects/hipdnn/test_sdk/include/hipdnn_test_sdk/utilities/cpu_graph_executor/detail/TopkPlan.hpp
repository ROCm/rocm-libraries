// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceTopk.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct TopkParams
{
    TopkParams() = default;
    TopkParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
               const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& indicesAttributes,
               const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* valuesAttributes,
               int64_t topK,
               bool selectLargest)
        : xTensor(unpackTensorAttributes(xAttributes))
        , indicesTensor(unpackTensorAttributes(indicesAttributes))
        , k(topK)
        , largest(selectLargest)
    {
        if(valuesAttributes != nullptr)
        {
            valuesTensor = unpackTensorAttributes(*valuesAttributes);
        }
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT indicesTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> valuesTensor;
    int64_t k = 0;
    bool largest = true;
};

template <typename DataType, typename IndexType>
class TopkPlan : public IGraphNodePlanExecutor
{
public:
    explicit TopkPlan(TopkParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        if(_params.valuesTensor.has_value())
        {
            return {_params.indicesTensor.uid, _params.valuesTensor->uid};
        }
        return {_params.indicesTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto x
            = createShallowTensor<DataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));
        auto indices = createShallowTensor<IndexType>(_params.indicesTensor,
                                                      variantPack.at(_params.indicesTensor.uid));

        if(_params.valuesTensor.has_value())
        {
            auto values = createShallowTensor<DataType>(*_params.valuesTensor,
                                                        variantPack.at(_params.valuesTensor->uid));
            utilities::CpuFpReferenceTopk::topk<DataType, IndexType>(
                *x, *indices, values.get(), _params.k, _params.largest);
        }
        else
        {
            utilities::CpuFpReferenceTopk::topk<DataType, IndexType>(
                *x, *indices, nullptr, _params.k, _params.largest);
        }
    }

private:
    TopkParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
class TopkPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DataType = utilities::DataTypeToNative<DataTypeEnum>;
    using IndexType = utilities::DataTypeToNative<IndexDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_TopkAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->indices_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->indices_tensor_uid(), IndexDataTypeEnum);

        if(nodeAttributes->values_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->values_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->values_tensor_uid().value(), DataTypeEnum);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_TopkAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type TopkAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* valuesAttributes = nullptr;
        if(nodeAttributes->values_tensor_uid().has_value())
        {
            valuesAttributes = tensorMap.at(nodeAttributes->values_tensor_uid().value());
        }

        return std::make_unique<TopkPlan<DataType, IndexType>>(
            TopkParams(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                       *tensorMap.at(nodeAttributes->indices_tensor_uid()),
                       valuesAttributes,
                       nodeAttributes->k(),
                       nodeAttributes->largest()));
    }
};

} // namespace hipdnn_test_sdk::detail
