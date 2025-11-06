// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

namespace miopen_legacy_plugin
{

class MiopenBatchnormFwdTrainingActivPlanBuilder : public IPlanBuilder
{
public:
    MiopenBatchnormFwdTrainingActivPlanBuilder() = default;
    ~MiopenBatchnormFwdTrainingActivPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenBatchnormFwdTrainingActivPlanBuilder(const MiopenBatchnormFwdTrainingActivPlanBuilder&)
        = delete;
    MiopenBatchnormFwdTrainingActivPlanBuilder&
        operator=(const MiopenBatchnormFwdTrainingActivPlanBuilder&)
        = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_plugin::IGraph& opGraph) const override;
    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_plugin::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;
};

}
