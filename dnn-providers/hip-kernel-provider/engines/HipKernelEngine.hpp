// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "HipdnnHipKernelContext.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "HipdnnHipKernelSettings.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

namespace hip_kernel_plugin
{

class HipKernelEngine : public hipdnn_plugin_sdk::IEngine<HipdnnHipKernelHandle,
                                                          HipdnnHipKernelSettings,
                                                          HipdnnHipKernelContext>
{
public:
    HipKernelEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(HipdnnHipKernelHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    void getDetails(HipdnnHipKernelHandle& handle,
                    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;
    size_t getMaxWorkspaceSize(
        const HipdnnHipKernelHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const override;

    void initializeExecutionContext(
        const HipdnnHipKernelHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnHipKernelContext& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnHipKernelHandle,
                                                        HipdnnHipKernelSettings,
                                                        HipdnnHipKernelContext>> planBuilder);

private:
    int64_t _id;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<HipdnnHipKernelHandle,
                                                                HipdnnHipKernelSettings,
                                                                HipdnnHipKernelContext>>>
        _planBuilders;
};

}
