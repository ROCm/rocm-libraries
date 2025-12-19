// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>

namespace hipdnn_test_sdk::utilities
{

class IGraphNodePlanBuilder
{
public:
    virtual ~IGraphNodePlanBuilder() = default;
    virtual bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const
        = 0;

    virtual std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin_sdk::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const
        = 0;
};

}
