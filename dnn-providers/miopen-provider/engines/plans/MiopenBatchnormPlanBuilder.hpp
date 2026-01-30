// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MiopenPlanBuilderBase.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_plugin
{

class MiopenBatchnormPlanBuilder : public MiopenPlanBuilderBase
{
public:
    MiopenBatchnormPlanBuilder() = default;
    ~MiopenBatchnormPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenBatchnormPlanBuilder(const MiopenBatchnormPlanBuilder&) = delete;
    MiopenBatchnormPlanBuilder& operator=(const MiopenBatchnormPlanBuilder&) = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_plugin_sdk::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_plugin_sdk::IGraph& opGraph,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;
};

}
