// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaModuleCache.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <optional>
#include <string>

// Generated forward kernel table row (asm_fmha_v3_fwd_configs.hpp).
struct fmha_v3_fwdConfig; // NOLINT(readability-identifier-naming)

namespace asm_sdpa_engine
{

class SdpaFwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>
{
public:
    static SdpaModuleCache& moduleCache()
    {
        static SdpaModuleCache s_cache;
        return s_cache;
    }

    bool isApplicable(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const Settings& executionSettings) const override;

    void initializeExecutionSettings(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Settings& executionSettings) const override;

    void buildPlan(const Handle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   Context& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const Handle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

protected:
    // Largest block, in threads, the resolved kernel can be launched with. The
    // generated table records it from each code object, so no kernel is loaded
    // to answer it. Virtual so tests can drive the launch-limit decline: every
    // vendored kernel accepts the block it is launched with.
    virtual unsigned int maxThreadsPerBlock(const fmha_v3_fwdConfig& config) const;

private:
    // Describes why the resolved kernel cannot be launched with the block the
    // plan uses, or std::nullopt when it can.
    std::optional<std::string> launchLimitViolation(const fmha_v3_fwdConfig& config) const;

    // launchLimitViolation as an isApplicable verdict: logs the reason and
    // returns false when the kernel cannot be launched with the plan's block.
    bool acceptsLaunchBlock(const fmha_v3_fwdConfig& config) const;
};

} // namespace asm_sdpa_engine
