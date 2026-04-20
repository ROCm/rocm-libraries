// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace asm_sdpa_engine
{

using IEngine = hipdnn_plugin_sdk::IEngine<HipKernelHandle, HipKernelSettings, HipKernelContext>;
using IPlanBuilder
    = hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>;

class AsmSdpaEngine
    : public hipdnn_plugin_sdk::IEngine<HipKernelHandle, HipKernelSettings, HipKernelContext>
{
public:
    AsmSdpaEngine();

    void addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder);

    static int64_t staticId();

    static const char* engineName()
    {
        return hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_NAME;
    }

    int64_t id() const override;

    bool isApplicable(
        HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(HipKernelHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getMaxWorkspaceSize(const HipKernelHandle& handle,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                engineConfig) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void initializeExecutionContext(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipKernelContext& executionContext) const override;

private:
    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

} // namespace asm_sdpa_engine
