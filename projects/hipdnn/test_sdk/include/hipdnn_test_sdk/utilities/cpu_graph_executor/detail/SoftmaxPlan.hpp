// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSoftmax.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <memory>
#include <optional>
#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct SoftmaxParams
{
    SoftmaxParams() = default;
    SoftmaxParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& pAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& sAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statsAttributes,
                  bool useLog2)
        : pTensor(unpackTensorAttributes(pAttributes))
        , sTensor(unpackTensorAttributes(sAttributes))
        , statsUseLog2(useLog2)
    {
        if(statsAttributes != nullptr)
        {
            statsTensor = unpackTensorAttributes(*statsAttributes);
        }
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT pTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT sTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> statsTensor;
    bool statsUseLog2 = false;
};

/// Executes one softmax node. The optional stats output is always FLOAT.
template <typename PDataType, typename SDataType, typename ComputeDataType>
class SoftmaxPlan : public IGraphNodePlanExecutor
{
public:
    explicit SoftmaxPlan(SoftmaxParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        if(_params.statsTensor.has_value())
        {
            return {_params.sTensor.uid, _params.statsTensor->uid};
        }
        return {_params.sTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto p
            = createShallowTensor<PDataType>(_params.pTensor, variantPack.at(_params.pTensor.uid));
        auto s
            = createShallowTensor<SDataType>(_params.sTensor, variantPack.at(_params.sTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<float>> stats;
        if(_params.statsTensor.has_value())
        {
            stats = createShallowTensor<float>(*_params.statsTensor,
                                               variantPack.at(_params.statsTensor->uid));
        }

        utilities::CpuFpReferenceSoftmax::forward<PDataType, SDataType, float, ComputeDataType>(
            *p, *s, stats.get(), _params.statsUseLog2);
    }

private:
    SoftmaxParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType PDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType SDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class SoftmaxPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using PDataType = utilities::DataTypeToNative<PDataTypeEnum>;
    using SDataType = utilities::DataTypeToNative<SDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_SoftmaxAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->p_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->s_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->p_tensor_uid(), PDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->s_tensor_uid(), SDataTypeEnum);

        const auto statsUid = nodeAttributes->stats_tensor_uid();
        if(statsUid.has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, statsUid);
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, statsUid, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SoftmaxAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SoftmaxAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const auto statsUid = nodeAttributes->stats_tensor_uid();
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statsAttributes
            = statsUid.has_value() ? tensorMap.at(*statsUid) : nullptr;

        return std::make_unique<SoftmaxPlan<PDataType, SDataType, ComputeDataType>>(
            SoftmaxParams(*tensorMap.at(nodeAttributes->p_tensor_uid()),
                          *tensorMap.at(nodeAttributes->s_tensor_uid()),
                          statsAttributes,
                          nodeAttributes->stats_use_log2()));
    }
};

} // namespace hipdnn_test_sdk::detail
