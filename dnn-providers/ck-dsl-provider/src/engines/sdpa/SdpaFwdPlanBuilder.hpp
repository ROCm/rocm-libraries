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

/// IPlanBuilder for the FMHA-forward kernel.
///
/// Holds non-owning references to the container's
/// ``CompileServiceBridge`` and the process-wide ``JitCache``. A single
/// cache is shared by every engine in the provider: the cache key
/// already includes ``op_kind`` as its first folded field, so
/// cross-engine collisions are impossible by construction.
///
/// **isApplicable**: returns true iff the graph has exactly one node,
/// that node is an ``SdpaAttributes`` node, the adapter would build a
/// spec for it (matching dtype, shape constraints, supported mask), and
/// the DSL accepts the spec on the detected device arch. Throws are
/// caught + downgraded to ``false`` so the SDK can fall through to other
/// engines without surfacing an exception.
///
/// **buildPlan**: derives the cache key via ``GraphSignature``, calls
/// ``JitCache::getOrLoad`` with a loader that runs the adapter +
/// ``CompileServiceBridge::compile``, wraps the resulting ``HipModule``
/// in an ``SdpaFwdPlan``, and stores it on the execution context via
/// ``setPlan``.
class SdpaFwdPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<::CkDslHandle, CkDslSettings, CkDslContext> {
   public:
    SdpaFwdPlanBuilder(CompileServiceBridge& bridge, JitCache& cache);

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

    /// Test-only access to the shared cache so unit tests can assert
    /// miss-then-hit behaviour across two buildPlan calls.
    JitCache& cacheForTesting() const {
        return _cache;
    }

    /// Stable op-kind string used as the cache-key partition and as the
    /// dispatch tag the Python compile_service sees. Defined here so the
    /// test can cross-check the signature derivation.
    static constexpr const char* opKind() {
        return "sdpa_fmha_fwd";
    }

   private:
    CompileServiceBridge& _bridge;
    JitCache& _cache;
};

}  // namespace ck_dsl_provider
