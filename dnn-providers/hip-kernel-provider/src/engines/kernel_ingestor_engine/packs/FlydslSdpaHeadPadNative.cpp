// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file FlydslSdpaHeadPadNative.cpp
 * @brief `hipkernel:flydsl_sdpa_head_pad`: the engine the head-dim padding copy ships under,
 *        which serves no graph.
 *
 * The copy (`sdpa_head_pad`, one object per target) is launched by the SDPA engines' natives
 * around an attention launch when a graph's Q/K or V head dim is not a multiple of 8
 * (FlydslSdpaHeadPad.hpp). Its objects ship as an op of their own so the packer packs them into
 * each arch shard and the attention objects stay untouched; an op is an engine to the
 * descriptors, so this engine exists, and declines every graph: no graph asks for a copy alone.
 * Its kernel matcher declines every kernel too, so nothing reaches its dispatch handler, which
 * refuses rather than launching anything.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_head_pad.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_head_pad.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_head_pad.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_head_pad.dispatch";

constexpr std::string_view SERVES_NO_GRAPH
    = "flydsl_sdpa_head_pad serves no graph: its kernel is the SDPA engines' copy around an "
      "attention launch at a head dim off 8";

std::optional<BoundTokens> flydslSdpaHeadPadGraphMatches(const MatchContext& /*context*/)
{
    return std::nullopt;
}

bool flydslSdpaHeadPadKernelMatches(const MatchContext& /*context*/,
                                    const BoundTokens& /*bound*/,
                                    const KernelDefinition& /*kernel*/)
{
    return false;
}

double flydslSdpaHeadPadScore(const MatchContext& /*context*/,
                              const BoundTokens& /*bound*/,
                              const KernelDefinition& /*kernel*/)
{
    return 0.0;
}

class FlydslSdpaHeadPadDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::string(SERVES_NO_GRAPH));
    }

    void launch(const Handle& /*handle*/,
                const PreparedDispatch& /*prepared*/,
                const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                uint32_t /*numDeviceBuffers*/,
                void* /*workspace*/) const override
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::string(SERVES_NO_GRAPH));
    }
};

/// Process-lifetime: the registry holds a non-owning pointer to it.
const FlydslSdpaHeadPadDispatchHandler& flydslSdpaHeadPadDispatchHandler()
{
    static const FlydslSdpaHeadPadDispatchHandler s_dispatchHandler;
    return s_dispatchHandler;
}

} // namespace

void resetFlydslSdpaHeadPadModuleCache()
{
    // Nothing of its own: the copy's modules are loaded, and cached, by the SDPA engines'
    // natives, whose own resets drop them.
}

void registerFlydslSdpaHeadPadSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSdpaHeadPadGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaHeadPadKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaHeadPadScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaHeadPadDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
