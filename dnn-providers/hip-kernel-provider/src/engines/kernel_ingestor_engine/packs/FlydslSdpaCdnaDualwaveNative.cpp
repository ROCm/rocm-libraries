// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaCommon.hpp"

/**
 * @file FlydslSdpaCdnaDualwaveNative.cpp
 * @brief The flyDSL CDNA SDPA engine of gfx950's dual-wave kernel: its kernel matcher, score and dispatch,
 *        and the one function that registers them.
 *
 * gfx950's dual-wave software-pipelined kernel, compiled ahead of time from
 * `flydsl/kernels_src/kernels/attention_cdna/flash_attn_gfx950.py`: its own argument list
 * and launch, so an engine of its own. The shared matcher, checks and dispatch are in
 * FlydslSdpaCdnaCommon.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_sdpa_cdna;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna_dualwave.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.flydsl_sdpa_cdna_dualwave.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_cdna_dualwave.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_cdna_dualwave.dispatch";

bool flydslSdpaCdnaDualwaveKernelMatches(const MatchContext& context,
                                         const BoundTokens& bound,
                                         const KernelDefinition& kernel)
{
    return familyKernelMatches(context, bound, kernel, CdnaFamily::DUALWAVE);
}

/// Ranks candidates that survived kernelMatches by their descriptor `priority`.
double flydslSdpaCdnaDualwaveScore(const MatchContext& /*context*/,
                                   const BoundTokens& /*bound*/,
                                   const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

} // namespace

compilation::KpackModuleCache& flydslSdpaCdnaDualwaveKpackModuleCache()
{
    // Process-lifetime and this pack's own; see flydslSdpaRdnaKpackModuleCache.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaCdnaDualwaveModuleCache()
{
    flydslSdpaCdnaDualwaveKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it.
const FlydslSdpaCdnaDispatchHandler& flydslSdpaCdnaDualwaveDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        flydslSdpaCdnaDualwaveKpackModuleCache());
    static const FlydslSdpaCdnaDispatchHandler s_dispatchHandler(
        s_kernelCompiler, s_kpackLoader, CdnaFamily::DUALWAVE);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaCdnaDualwaveSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydsl_sdpa_cdna::flydslSdpaCdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaCdnaDualwaveKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaCdnaDualwaveScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaCdnaDualwaveDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
