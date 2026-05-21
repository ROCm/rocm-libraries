// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "../../CkDslContext.hpp"
#include "../../CkDslHandle.hpp"
#include "../../CkDslSettings.hpp"

namespace ck_dsl_provider {

/// IEngine implementation for CK DSL implicit-GEMM forward
/// convolution.
///
/// The M1 plan registers one engine per CK DSL op kind; this is the
/// first. For the I-1 milestone the engine is a load-bearing stub:
///
///  - isApplicable() returns false unconditionally (no graphs match
///    until the adapter and plan builder land in I-6/I-7).
///  - getDetails() returns an empty FlatBuffer payload via the
///    handle's detached-buffer map, matching the SDK contract.
///  - getMaxWorkspaceSize() returns 0.
///  - initializeExecutionContext() throws because isApplicable()
///    promised "not applicable" and the SDK should never reach it.
class CkDslConvImplicitGemmEngine
    : public hipdnn_plugin_sdk::IEngine<::CkDslHandle, CkDslSettings, CkDslContext> {
   public:
    explicit CkDslConvImplicitGemmEngine(int64_t id);

    int64_t id() const override;

    bool isApplicable(
        ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void getDetails(::CkDslHandle& handle,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                    hipdnnPluginConstData_t& detailsOut) const override;

    size_t getMaxWorkspaceSize(const ::CkDslHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
                                   engineConfig) const override;

    void initializeExecutionContext(
        const ::CkDslHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        CkDslContext& executionContext) const override;

   private:
    int64_t _id;
};

}  // namespace ck_dsl_provider
