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

/// An engine that serves a graph through the first of its plan builders that applies. Backend
/// neutral: the hipBLASLt and rocBLAS engines are each one of these with their own builders.
/// It reports HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE only when constructed with
/// @p supportsGraphCapture: every plan its builders make must be capture-safe (no host
/// synchronization, allocation or benchmarking on the execute path).
class PlanBuilderEngine : public IEngine
{
public:
    explicit PlanBuilderEngine(int64_t id, bool supportsGraphCapture = false);

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
    bool _supportsGraphCapture;
    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

} // namespace hipblaslt_plugin
