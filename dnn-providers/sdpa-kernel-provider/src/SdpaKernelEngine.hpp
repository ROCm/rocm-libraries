// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "SdpaKernelContext.hpp"
#include "SdpaKernelHandle.hpp"
#include "SdpaKernelPlanBuilder.hpp"
#include "SdpaKernelSettings.hpp"

namespace sdpa_kernel_provider
{

using IEngine = hipdnn_plugin_sdk::IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>;
using IPlanBuilder
    = hipdnn_plugin_sdk::IPlanBuilder<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>;

class SdpaKernelEngine
    : public hipdnn_plugin_sdk::IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>
{
public:
    SdpaKernelEngine(std::vector<std::unique_ptr<IPlanBuilder>>&& planBuilders)
        : _planBuilders(std::move(planBuilders))
    {
    }
    SdpaKernelEngine() = default;

    void addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder);

    static int64_t staticId();

    static constexpr const char* engineName()
    {
        return "SdpaKernelPlugin";
    }

    int64_t id() const override;

    bool isApplicable(SdpaKernelHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(SdpaKernelHandle& handle,
                    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getMaxWorkspaceSize(const SdpaKernelHandle& handle,
                            const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig&
                                engineConfig) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void initializeExecutionContext(
        const SdpaKernelHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        SdpaKernelContext& executionContext) const override;

protected:
    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

}
