// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <string_view>
#include <utility>

#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
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

    /// @brief The L1 throughput models this engine binds, by architecture.
    ///
    /// RFC 0019 Open Question 7 (RESOLVED): an engine that ships no UED binds its
    /// `predict_engine_tflops` model by naming that UHD's UUID in its provider's own
    /// engine definition -- this table. The binding identity therefore comes from
    /// compiled-in provider code, never from the document: a UHD keeps §4.1's shape and
    /// carries no `engine`, `role` or `arch` member, and the loader resolves the id out
    /// of the descriptor catalog it already parses, validating provenance exactly as it
    /// does for a UED role reference (§3.1, §8.1).
    ///
    /// One entry per architecture this engine ships its own kernels for: gfx942 and
    /// gfx950 run different code objects at different throughput, so one model cannot
    /// answer for both. An id nothing deploys resolves to nothing and the engine reports
    /// UNAVAILABLE -- the behaviour it had before any model existed (§11.2's "no declared
    /// model" row).
    ///
    /// To install a model, a deployer drops a `.uhd.json` carrying one of these ids, plus
    /// its artifact, into any descriptor root this provider reads
    /// (kernel_ingestor_engine::descriptorSearchDirectories()).
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 2> L1_MODEL_IDS{{
        {"gfx942", "5f2a7c14-9d3b-4e86-b0a1-6c4f21d8e370"},
        {"gfx950", "8b61d0c9-24af-4d17-9e52-3a7c06b8f145"},
    }};

    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        getPrediction(Handle& handle,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
                      hipdnnEnginePredictionKind_t kind,
                      bool evaluate) const override;

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
    /// Resolved once at construction from @ref L1_MODEL_IDS; empty when no descriptor
    /// tree is installed, which the engine reports as UNAVAILABLE rather than an error.
    hipdnn_plugin_sdk::uhd::EngineModelBinding _l1Models;
};

} // namespace asm_sdpa_engine
