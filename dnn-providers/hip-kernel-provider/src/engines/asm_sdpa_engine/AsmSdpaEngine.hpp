// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace asm_sdpa_engine
{

using IEngine = hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>;
using IPlanBuilder = hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>;

class AsmSdpaEngine : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
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
        Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(Handle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getMaxWorkspaceSize(const Handle& handle,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                engineConfig) const override;

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void initializeExecutionContext(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Context& executionContext) const override;

private:
    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

} // namespace asm_sdpa_engine
