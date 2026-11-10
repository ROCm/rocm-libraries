// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGather.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <cstdint>
#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct GatherParams
{
    GatherParams() = default;
    GatherParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& indicesAttributes,
                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
                 int64_t gatherAxis)
        : xTensor(unpackTensorAttributes(xAttributes))
        , indicesTensor(unpackTensorAttributes(indicesAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , axis(gatherAxis)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT indicesTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    int64_t axis = 0;
};

template <typename DataType, typename IndexType>
class GatherPlan : public IGraphNodePlanExecutor
{
public:
    explicit GatherPlan(GatherParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.yTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto x
            = createShallowTensor<DataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));
        auto indices = createShallowTensor<IndexType>(_params.indicesTensor,
                                                      variantPack.at(_params.indicesTensor.uid));
        auto y
            = createShallowTensor<DataType>(_params.yTensor, variantPack.at(_params.yTensor.uid));

        utilities::CpuFpReferenceGather::gather<DataType, IndexType>(
            *x, *indices, *y, _params.axis);
    }

private:
    GatherParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
class GatherPlanBuilder : public IGraphNodePlanBuilder
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
        const auto* nodeAttributes = node.attributes_as_GatherAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->indices_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->indices_tensor_uid(), IndexDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), DataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_GatherAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type GatherAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        return std::make_unique<GatherPlan<DataType, IndexType>>(
            GatherParams(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                         *tensorMap.at(nodeAttributes->indices_tensor_uid()),
                         *tensorMap.at(nodeAttributes->y_tensor_uid()),
                         nodeAttributes->axis()));
    }
};

} // namespace hipdnn_test_sdk::detail
