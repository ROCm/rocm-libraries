// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <memory>

#include "../../CkDslContext.hpp"
#include "../../CkDslHandle.hpp"
#include "../../CkDslSettings.hpp"
#include "ConvImplicitGemmPlanBuilder.hpp"

namespace ck_dsl_provider {

class CompileServiceBridge;
class JitCache;

/// IEngine implementation for CK DSL implicit-GEMM forward
/// convolution.
///
/// One engine per CK DSL op kind; this is the first. The engine owns
/// exactly one plan builder. ``getDetails`` publishes an EngineDetails
/// FlatBuffer (engine id + empty knob vector for now) via the handle's
/// detached-buffer map so the SDK's plugin-API surface keeps working.
class CkDslConvImplicitGemmEngine
    : public hipdnn_plugin_sdk::IEngine<::CkDslHandle, CkDslSettings, CkDslContext> {
   public:
    CkDslConvImplicitGemmEngine(std::int64_t id, CompileServiceBridge& bridge, JitCache& cache);

    std::int64_t id() const override;

    bool isApplicable(
        ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(::CkDslHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    std::size_t getMaxWorkspaceSize(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig)
        const override;

    void initializeExecutionContext(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkDslContext& executionContext) const override;

    /// Test-only accessor: lets the plan-builder test exercise the
    /// same cache the engine uses for cache-hit verification.
    ConvImplicitGemmPlanBuilder& planBuilderForTesting() const {
        return *_planBuilder;
    }

   private:
    std::int64_t _id;
    std::unique_ptr<ConvImplicitGemmPlanBuilder> _planBuilder;
};

}  // namespace ck_dsl_provider
