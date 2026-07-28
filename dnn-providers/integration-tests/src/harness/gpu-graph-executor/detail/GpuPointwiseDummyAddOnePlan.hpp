// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <stdexcept>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Dummy GPU plan that computes output[i] = input[i] + 1.0f for each element.
// This is a proof-of-concept plan to exercise the GPU executor infrastructure.
// Real GPU plans will use HipRTC-compiled kernels operating on device memory.
class GpuDummyAddOnePlanExecutor : public IGpuGraphNodePlanExecutor
{
public:
    GpuDummyAddOnePlanExecutor(int64_t inputUid, int64_t outputUid, size_t elementCount)
        : _inputUid(inputUid)
        , _outputUid(outputUid)
        , _elementCount(elementCount)
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        const auto* input = static_cast<const float*>(variantPack.at(_inputUid));
        auto* output = static_cast<float*>(variantPack.at(_outputUid));

        for(size_t i = 0; i < _elementCount; ++i)
        {
            output[i] = input[i] + 1.0f;
        }
    }

private:
    int64_t _inputUid;
    int64_t _outputUid;
    size_t _elementCount;
};

class GpuDummyAddOnePlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.attributes_type()
           != hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes)
        {
            return false;
        }

        const auto* attrs = node.attributes_as_PointwiseAttributes();

        // Reject if any operand this op consumes is runtime pass-by-value; the
        // dummy add-one plan cannot resolve a PBV host scalar. Enumerate every
        // operand (required in_0/out_0 and optional in_1/in_2/axis) so the check
        // is exhaustive.
        std::vector<int64_t> operandUids{attrs->in_0_tensor_uid(), attrs->out_0_tensor_uid()};
        for(const auto optional :
            {attrs->in_1_tensor_uid(), attrs->in_2_tensor_uid(), attrs->axis_tensor_uid()})
        {
            if(optional.has_value())
            {
                operandUids.push_back(*optional);
            }
        }
        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* attrs = node.attributes_as_PointwiseAttributes();
        if(attrs == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PointwiseAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        auto inputUid = attrs->in_0_tensor_uid();
        auto outputUid = attrs->out_0_tensor_uid();

        const auto* outTensor = tensorMap.at(outputUid);
        size_t elementCount = 1;
        for(unsigned int i = 0; i < outTensor->dims()->size(); ++i)
        {
            elementCount *= static_cast<size_t>(outTensor->dims()->Get(i));
        }

        return std::make_unique<GpuDummyAddOnePlanExecutor>(inputUid, outputUid, elementCount);
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
