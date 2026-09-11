// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2Engine: hipDNN IEngine plugin wrapping our V7 Flash-Attention 2 kernel
// (rocWMMA MFMA + causal tile skip) for FP16 SDPA on gfx942.
//
// Registered via HIPDNN_REGISTER_ENGINE(HIP_FLASH2_ENGINE) in EngineNames.hpp.
// Enabled at build time with -DENABLE_HIP_FLASH2_ENGINE=OFF (default: off).

#pragma once

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <memory>
#include <vector>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

namespace hip_flash2_engine
{

using IEngine = hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>;
using IPlanBuilder = hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>;

// final: this engine is not designed to be subclassed
class HipFlash2Engine final : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
{
public:
    HipFlash2Engine() = default;

    // Takes ownership of the plan builder (pass by value communicates this clearly)
    void addPlanBuilder(std::unique_ptr<IPlanBuilder> planBuilder);

    static int64_t staticId()
    {
        return hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_ID;
    }

    static const char* engineName()
    {
        return hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_NAME;
    }

    // id() returns the fixed constant -- no per-instance ID needed
    int64_t id() const override
    {
        return hipdnn_data_sdk::utilities::HIP_FLASH2_ENGINE_ID;
    }

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

} // namespace hip_flash2_engine
