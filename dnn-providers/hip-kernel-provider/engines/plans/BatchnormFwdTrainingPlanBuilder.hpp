// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnHipKernelContext.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "HipdnnHipKernelSettings.hpp"
#include "hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"

namespace hip_kernel_plugin
{

class BatchnormFwdTrainingPlanBuilder
    : public hipdnn_plugin_sdk::
          IPlanBuilder<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>
{
public:
    BatchnormFwdTrainingPlanBuilder() = default;
    ~BatchnormFwdTrainingPlanBuilder() override = default;

    // Disallow copy and assignment
    BatchnormFwdTrainingPlanBuilder(const BatchnormFwdTrainingPlanBuilder&) = delete;
    BatchnormFwdTrainingPlanBuilder& operator=(const BatchnormFwdTrainingPlanBuilder&) = delete;

    bool isApplicable(const HipdnnHipKernelHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipdnnHipKernelHandle& handle,
                               const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const HipdnnHipKernelSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const HipdnnHipKernelHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnHipKernelSettings& executionSettings) const override;

    void buildPlan(const HipdnnHipKernelHandle& handle,
                   const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipdnnHipKernelContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnHipKernelHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

}
