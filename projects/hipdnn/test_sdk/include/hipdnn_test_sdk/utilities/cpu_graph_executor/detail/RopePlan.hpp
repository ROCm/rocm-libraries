// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRope.hpp>
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

struct RopeParams
{
    RopeParams() = default;
    RopeParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
               const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& freqsAttributes,
               const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
               const hipdnn_flatbuffers_sdk::data_objects::RopeAttributes& attributes)
        : xTensor(unpackTensorAttributes(xAttributes))
        , freqsTensor(unpackTensorAttributes(freqsAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , outputScale(attributes.output_scale())
        , ropeDim(attributes.rope_dim())
    {
        if(attributes.rope_offset_ext().has_value())
        {
            ropeOffset = *attributes.rope_offset_ext();
        }
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT freqsTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    float outputScale = 1.0f;
    int64_t ropeDim = 0;
    std::optional<int64_t> ropeOffset;
};

template <typename DataType>
class RopePlan : public IGraphNodePlanExecutor
{
public:
    explicit RopePlan(RopeParams&& params)
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
        auto freqs = createShallowTensor<float>(_params.freqsTensor,
                                                variantPack.at(_params.freqsTensor.uid));
        auto y
            = createShallowTensor<DataType>(_params.yTensor, variantPack.at(_params.yTensor.uid));

        utilities::CpuFpReferenceRope::forward<DataType>(
            *x, *freqs, *y, _params.outputScale, _params.ropeDim, _params.ropeOffset);
    }

private:
    RopeParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DataTypeEnum>
class RopePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DataType = utilities::DataTypeToNative<DataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_RopeAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->freqs_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), DataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap,
                          nodeAttributes->freqs_tensor_uid(),
                          hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), DataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_RopeAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type RopeAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        return std::make_unique<RopePlan<DataType>>(
            RopeParams(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                       *tensorMap.at(nodeAttributes->freqs_tensor_uid()),
                       *tensorMap.at(nodeAttributes->y_tensor_uid()),
                       *nodeAttributes));
    }
};

} // namespace hipdnn_test_sdk::detail
