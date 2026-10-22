// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file FlydslSdpaObjectTestSupport.hpp
 * @brief The SDPA device harnesses, reached with a named object.
 *
 * The RDNA and CDNA dispatch suites each own a launch-and-compare harness that picks the
 * object to run itself. The per-object suite (TestFlydslSdpaObjects.cpp) names the object
 * instead, so these entry points run a given staged object through the same harness: the
 * graph @p spec describes is matched, the object's own kernel matcher must admit it, and
 * every output element (and the LSE when @p spec asks for one) is checked against the
 * harness's double-precision reference. Declared here, defined beside each harness.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The RDNA engine a forced object belongs to.
enum class RdnaObjectEngine
{
    PREFILL,
    DECODE,
    AIW,
};

/// Runs @p kernel, an object of @p engine, on the graph @p spec describes and checks it
/// against the RDNA harness's reference at the dispatch suite's tolerance for the dtype.
/// Defined in TestFlydslSdpaRdnaEngine.cpp.
void expectRdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                      RdnaObjectEngine engine,
                                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Runs @p kernel, an object of any of the three CDNA engines, on the graph @p spec
/// describes and checks it against the CDNA harness's reference. Defined in
/// TestFlydslSdpaCdnaEngine.cpp.
void expectCdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
