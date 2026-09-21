// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <set>

#include "EngineInterface.hpp"
#include "plans/PlanBuilderInterface.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipblaslt_plugin
{

class HipblasltEngine : public IEngine
{
public:
    HipblasltEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(
        HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    void getDetails(HipdnnEnginePluginHandle& handle,
                    hipdnnPluginConstData_t& detailsOut) const override;
    size_t getWorkspaceSize(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void initializeExecutionContext(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        HipdnnEnginePluginExecutionContext& executionContext) const override;

    void addPlanBuilder(std::unique_ptr<IPlanBuilder> planBuilder);

private:
    int64_t _id;
    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

} // namespace hipblaslt_plugin
