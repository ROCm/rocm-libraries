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
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaHeadPad.hpp"

/**
 * @file FlydslSdpaCdnaNative.cpp
 * @brief The flyDSL CDNA SDPA engine of the generic MFMA kernel's prefill objects: its kernel matcher, score and dispatch,
 *        and the one function that registers them.
 *
 * Prefill objects of the generic MFMA kernel (gfx942, gfx950), compiled ahead of time from
 * `flydsl/kernels_src/kernels/attention_cdna/flash_attn_generic.py`. The shared matcher,
 * checks and dispatch are in FlydslSdpaCdnaCommon.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

using namespace flydsl_sdpa_cdna;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_cdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_cdna.dispatch";

bool flydslSdpaCdnaKernelMatches(const MatchContext& context,
                                 const BoundTokens& bound,
                                 const KernelDefinition& kernel)
{
    return familyKernelMatches(context, bound, kernel, CdnaFamily::GENERIC);
}

/**
 * @brief The default ranking: an object's priority, plus PREFERRED_BONUS where its
 *        class's measured rule prefers it for this graph's size: the generic kernel's
 *        tiles ranked against each other, the engine's interim native ranker until a
 *        trained UHD ranker replaces it.
 */
double flydslSdpaCdnaScore(const MatchContext& context,
                           const BoundTokens& /*bound*/,
                           const KernelDefinition& kernel)
{
    auto score = static_cast<double>(kernel.priority);
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return score;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    if(q != nullptr && k != nullptr && isPreferredFor(problemFor(*q, *k, nullptr), kernel))
    {
        score += PREFERRED_BONUS;
    }
    return score;
}

} // namespace

compilation::KpackModuleCache& flydslSdpaCdnaKpackModuleCache()
{
    // Process-lifetime and this pack's own; see flydslSdpaRdnaKpackModuleCache.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaCdnaModuleCache()
{
    flydslSdpaCdnaKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it.
/// A graph whose head dim is not a multiple of 8 runs on padded copies (FlydslSdpaHeadPad.hpp);
/// every other graph reaches the engine's own handler unchanged.
const flydsl_sdpa::HeadPadDispatchHandler& flydslSdpaCdnaDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaCdnaKpackModuleCache());
    static const FlydslSdpaCdnaDispatchHandler s_engineHandler(
        s_kernelCompiler, s_kpackLoader, CdnaFamily::GENERIC);
    static const flydsl_sdpa::HeadPadDispatchHandler s_dispatchHandler(s_engineHandler,
                                                                       s_kpackLoader);
    return s_dispatchHandler;
}

/// The engine's matchers and score, padding a head dim off 8 (FlydslSdpaHeadPad.hpp).
using CdnaHeadPad = flydsl_sdpa::HeadPadMatchers<&flydsl_sdpa_cdna::flydslSdpaCdnaGraphMatches,
                                                 &flydslSdpaCdnaKernelMatches,
                                                 &flydslSdpaCdnaScore>;

} // namespace

void registerFlydslSdpaCdnaSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &CdnaHeadPad::graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &CdnaHeadPad::kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &CdnaHeadPad::score);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaCdnaDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
