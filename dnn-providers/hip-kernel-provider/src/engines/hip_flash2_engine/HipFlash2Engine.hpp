// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2Engine: hipDNN IEngine plugin wrapping our V7 Flash-Attention 2 kernel
// (rocWMMA MFMA + causal tile skip) for FP16 SDPA on gfx942/gfx950.
//
// Registered via HIPDNN_REGISTER_ENGINE(HIP_FLASH2_ENGINE) in EngineNames.hpp.
// Enabled at build time with -DENABLE_HIP_FLASH2_ENGINE=OFF (default: off).

#pragma once

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <memory>
#include <vector>

namespace hip_flash2_engine
{

using IEngine = hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>;
using IPlanBuilder = hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>;

class HipFlash2Engine : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
{
public:
    explicit HipFlash2Engine(int64_t engineId);

    void addPlanBuilder(std::unique_ptr<IPlanBuilder>&& planBuilder);

    static int64_t staticId()
    {
        return hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_ID;
    }

    static const char* engineName()
    {
        return hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_NAME;
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
    int64_t _id;
    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

} // namespace hip_flash2_engine
