// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceReduction.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct ReductionParams
{
    ReductionParams() = default;
    ReductionParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
                    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
                    hipdnn_flatbuffers_sdk::data_objects::ReductionMode reductionMode)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , mode(reductionMode)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode
        = hipdnn_flatbuffers_sdk::data_objects::ReductionMode::NOT_SET;
};

template <typename XDataType, typename YDataType, typename ComputeDataType>
class ReductionPlan : public IGraphNodePlanExecutor
{
public:
    explicit ReductionPlan(ReductionParams&& params)
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
        auto shallowYTensor
            = createShallowTensor<YDataType>(_params.yTensor, variantPack.at(_params.yTensor.uid));

        utilities::CpuFpReferenceReduction::reduce<XDataType, YDataType, ComputeDataType>(
            *shallowXTensor, *shallowYTensor, _params.mode);
    }

private:
    ReductionParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class ReductionPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using YDataType = utilities::DataTypeToNative<YDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_ReductionAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_tensor_uid(), YDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ReductionAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ReductionAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        return std::make_unique<ReductionPlan<XDataType, YDataType, ComputeDataType>>(
            ReductionParams(*tensorMap.at(nodeAttributes->in_tensor_uid()),
                            *tensorMap.at(nodeAttributes->out_tensor_uid()),
                            nodeAttributes->mode()));
    }
};

} // namespace hipdnn_test_sdk::detail
