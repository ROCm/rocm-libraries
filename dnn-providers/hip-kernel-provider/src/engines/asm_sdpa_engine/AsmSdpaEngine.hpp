// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace asm_sdpa_engine
{

// True only when the forward plan builder declines every O and stats tensor whose strides map two
// elements to one address or that the kernel does not honour, and the forward kernels have a
// passing bitwise run-to-run repeat test on every code-object target they ship. Backward is never
// claimed: its dQ accumulation uses fp32 atomics.
inline constexpr bool ASM_SDPA_FORWARD_CLAIMS_DETERMINISM = false;

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

    // Numerical notes this engine reports for the operation in opGraph. An engine ID covers both
    // forward and backward, so the note depends on the operation: an explicit empty list (a
    // determinism claim) only for a graph with an SDPA forward node, no SDPA backward node and
    // forwardClaimed set; std::nullopt (field absent, i.e. NONDETERMINISTIC) for everything else.
    static std::optional<std::vector<int32_t>>
        numericalNotesFor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                          bool forwardClaimed);

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
    // The first registered builder that accepts opGraph; every entry point uses it,
    // so knobs, workspace sizing and plan building agree on one builder.
    const IPlanBuilder* findApplicablePlanBuilder(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const;

    std::vector<std::unique_ptr<IPlanBuilder>> _planBuilders;
};

} // namespace asm_sdpa_engine
