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
 * @file FlydslSdpaCdnaDecodeNative.cpp
 * @brief The flyDSL CDNA SDPA engine of the generic kernel's split-KV decode objects: its kernel matcher, score and dispatch,
 *        and the one function that registers them.
 *
 * The generic kernel in its split-KV mode (vendored-kernel modifications 14, 15): the keys
 * split across workgroups, a GQA group packed into each row tile, and a merge kernel in the
 * same object -- a different algorithm from prefill's, with its own launch and workspace,
 * so an engine of its own. The shared matcher, checks and dispatch are in
 * FlydslSdpaCdnaCommon.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_sdpa_cdna;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna_decode.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna_decode.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_cdna_decode.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_cdna_decode.dispatch";

bool flydslSdpaCdnaDecodeKernelMatches(const MatchContext& context,
                                       const BoundTokens& bound,
                                       const KernelDefinition& kernel)
{
    return familyKernelMatches(context, bound, kernel, CdnaFamily::DECODE);
}

/// Ranks candidates that survived kernelMatches by their descriptor `priority`.
double flydslSdpaCdnaDecodeScore(const MatchContext& /*context*/,
                                 const BoundTokens& /*bound*/,
                                 const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

} // namespace

compilation::KpackModuleCache& flydslSdpaCdnaDecodeKpackModuleCache()
{
    // Process-lifetime and this pack's own; see flydslSdpaRdnaKpackModuleCache.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaCdnaDecodeModuleCache()
{
    flydslSdpaCdnaDecodeKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it.
const FlydslSdpaCdnaDispatchHandler& flydslSdpaCdnaDecodeDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        flydslSdpaCdnaDecodeKpackModuleCache());
    static const FlydslSdpaCdnaDispatchHandler s_dispatchHandler(
        s_kernelCompiler, s_kpackLoader, CdnaFamily::DECODE);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaCdnaDecodeSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydsl_sdpa_cdna::flydslSdpaCdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaCdnaDecodeKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaCdnaDecodeScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaCdnaDecodeDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
