// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstddef>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <vector>

#include "../../CkDslContext.hpp"
#include "../../CkDslHandle.hpp"
#include "../../CkDslSettings.hpp"
#include "../../runtime/JitCache.hpp"

namespace ck_dsl_provider {

class CompileServiceBridge;

/// IPlanBuilder for the implicit-GEMM convolution kernel.
///
/// Owns the per-op ``JitCache`` (each op gets its own cache so a key
/// collision across ops is impossible by construction). The bridge
/// reference comes from the container; the builder does not own it.
///
/// **isApplicable**: returns true iff the graph has exactly one node,
/// that node is a ``ConvolutionFwdAttributes`` node, and the adapter
/// would successfully build a spec for it (matching dtype, 2-D, etc.).
/// Throws are caught + downgraded to ``false`` so the SDK can fall
/// through to other engines without surfacing an exception.
///
/// **buildPlan**: derives the cache key via ``GraphSignature``, calls
/// ``JitCache::getOrLoad`` with a loader that runs the adapter +
/// ``CompileServiceBridge::compile``, wraps the resulting
/// ``HipModule`` in a ``ConvImplicitGemmPlan``, and stores it on the
/// execution context via ``setPlan``. The plan's ``execute()`` is a
/// stub for I-7 (wired in I-8).
class ConvImplicitGemmPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<::CkDslHandle, CkDslSettings, CkDslContext> {
   public:
    explicit ConvImplicitGemmPlanBuilder(CompileServiceBridge& bridge);

    bool isApplicable(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    std::size_t getMaxWorkspaceSize(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const CkDslSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkDslSettings& executionSettings) const override;

    void buildPlan(const ::CkDslHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   CkDslContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    /// Test-only access to the cache so the I-7 plan-builder test can
    /// assert miss-then-hit behaviour across two buildPlan calls.
    JitCache& cacheForTesting() const {
        return _cache;
    }

    /// Stable op-kind string used as the cache-key partition and as
    /// the dispatch tag the Python compile_service sees. Defined here
    /// so the test can cross-check the signature derivation.
    static constexpr const char* opKind() {
        return "conv_implicit_gemm";
    }

   private:
    CompileServiceBridge& _bridge;

    // ``mutable`` because IPlanBuilder methods are const, but the
    // cache fundamentally mutates on a miss. The mutation is
    // thread-safe via the JitCache's internal mutex.
    mutable JitCache _cache;
};

}  // namespace ck_dsl_provider
