// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#endif

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

    /// @brief The selector revision every L1 model for this engine must record; a model
    /// whose `trained_against.selector_revision` differs is refused at load.
    static const char* selectorRevision();

    int64_t id() const override;

    bool isApplicable(
        Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(Handle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    /// @brief The L1 models this engine binds: architecture to UHD UUID (RFC 0019 OQ 7).
    ///
    /// The binding identity comes from this compiled-in table, never from the document. An
    /// architecture may have one row per ranking metric (§4.4); each model answers in its
    /// own `score.metric`, and a second model for one (architecture, metric) disables it.
    /// An id nothing deploys leaves the engine UNAVAILABLE.
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 2> L1_MODEL_IDS{{
        {"gfx942", "5f2a7c14-9d3b-4e86-b0a1-6c4f21d8e370"}, // tflops
        {"gfx950", "8b61d0c9-24af-4d17-9e52-3a7c06b8f145"}, // tflops
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
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    /// Resolved once at construction from @ref L1_MODEL_IDS; empty (UNAVAILABLE, not an
    /// error) when no descriptor tree is installed.
    hipdnn_plugin_sdk::uhd::EngineModelBinding _l1Models;
#endif
};

} // namespace asm_sdpa_engine
