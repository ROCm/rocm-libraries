// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "HipMlopsKernelCompiler.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/heuristics/uhd/EnginePredictor.hpp>
#endif

namespace hip_kernel_provider
{

class HipMlopsEngine : public hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>
{
public:
    explicit HipMlopsEngine(int64_t id);

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
    size_t getMaxWorkspaceSize(const Handle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    /// @brief The engine's L1 estimate from the model HipMlopsL1Models.hpp declares for the
    /// device's architecture and the requested metric; UNAVAILABLE without one. The engine
    /// exposes no knobs, so it has no exact configuration to predict.
    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT
        getPrediction(Handle& handle,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
                      hipdnnEnginePredictionKind_t kind,
                      bool evaluate) const override;

    void initializeExecutionContext(
        const Handle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        Context& executionContext) const override;

    void addPlanBuilder(
        std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>> planBuilder);

    const compilation::IKernelCompiler& getKernelCompiler() const
    {
        return _kernelCompiler;
    }

private:
    int64_t _id;
    std::vector<std::unique_ptr<hipdnn_plugin_sdk::IPlanBuilder<Handle, Settings, Context>>>
        _planBuilders;
    HipMlopsKernelCompiler _kernelCompiler;
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    /// Resolved once at construction from HIP_MLOPS_L1_MODEL_IDS; empty (UNAVAILABLE, not
    /// an error) when no descriptor tree is installed.
    hipdnn_plugin_sdk::uhd::EngineModelBinding _l1Models;
#endif
};

}
