// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Returns true if any of the given operand uids resolves to a runtime
// pass-by-value tensor. A builder whose plan cannot read PBV host scalars uses
// this to report itself not-applicable, so the harness falls back to the CPU
// reference. Plans that read host scalars (e.g. ragged SDPA, via
// GpuScalarOperand.hpp) don't need it.
// Callers MUST pass every operand uid their node consumes (required and
// optional) so the check is exhaustive per node rather than graph-wide.
inline bool anyOperandIsRuntimePassByValue(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    const std::vector<int64_t>& operandUids)
{
    for(const int64_t uid : operandUids)
    {
        const auto it = tensorMap.find(uid);
        if(it != tensorMap.end()
           && hipdnn_flatbuffers_sdk::utilities::isTensorRuntimePassByValue(it->second))
        {
            return true;
        }
    }
    return false;
}

class IGpuGraphNodePlanBuilder
{
public:
    virtual ~IGpuGraphNodePlanBuilder() = default;

    virtual bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const
        = 0;

    virtual std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const
        = 0;
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
