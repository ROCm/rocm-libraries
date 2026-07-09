// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: Second PlanBuilder example demonstrating the same pattern as ReluPlanBuilder
// but for a convolution operation. Compare with ReluPlanBuilder to see how different operations
// handle graph matching (isApplicable), parameter extraction (buildPlan), and knob definitions
// (getCustomKnobs).

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "ExampleProviderContext.hpp"
#include "ExampleProviderHandle.hpp"
#include "ExampleProviderSettings.hpp"
#include "hip/IKernelCompiler.hpp"

namespace example_provider
{

/// PlanBuilder for GPU-based naive convolution forward.
///
/// Handles single-node ConvolutionFwdAttributes graphs with CROSS_CORRELATION
/// mode and FLOAT data type. Provides a custom BLOCK_SIZE knob with
/// choices [64, 128, 256].
///
/// The built plan computes a real convolution and then busy-waits for a fixed
/// wall-clock duration so the autotune framework, which ranks candidates by
/// measured GPU time, sees a deterministic per-engine duration. Each instance
/// is configured with its own spin duration. The exhaustive variant registers
/// and reads the "global.benchmarking" knob and, once primed, switches to a
/// shorter spin.
class ConvFwdPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder<ExampleProviderHandle,
                                                                  ExampleProviderSettings,
                                                                  ExampleProviderContext>
{
public:
    ConvFwdPlanBuilder(const IKernelCompiler& compiler,
                       uint64_t normalSpinNs,
                       uint64_t primedSpinNs,
                       bool isExhaustive,
                       size_t workspaceBytes = 0);
    ~ConvFwdPlanBuilder() override = default;

    ConvFwdPlanBuilder(const ConvFwdPlanBuilder&) = delete;
    ConvFwdPlanBuilder& operator=(const ConvFwdPlanBuilder&) = delete;

    bool isApplicable(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const ExampleProviderHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const ExampleProviderSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        ExampleProviderSettings& executionSettings) const override;

    void buildPlan(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
            engineConfig,
        ExampleProviderContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const ExampleProviderHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

private:
    const IKernelCompiler& _compiler;
    uint64_t _normalSpinNs;
    uint64_t _primedSpinNs;
    bool _isExhaustive;
    size_t _workspaceBytes;

    // The primed flag must outlive the plan/context rebuild between the exhaustive priming pass and
    // the timing pass. This builder is owned by its engine for the engine's lifetime, so an
    // instance member is a stable home.
    mutable std::atomic<bool> _primed{false};
};

} // namespace example_provider
