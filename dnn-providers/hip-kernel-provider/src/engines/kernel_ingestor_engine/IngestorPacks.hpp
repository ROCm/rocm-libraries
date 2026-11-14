// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"

/// @file IngestorPacks.hpp
/// The provider's pack inventory: the one file adding an engine edits. A pack contributes
/// native symbols only; its descriptors are installed JSON read by discoverDescriptorSets().
/// Registration is a table entry rather than a self-registering static, because a linker
/// drops an unreferenced archive member (as in the unit-test static-archive build) even
/// though the same TU survives in the plugin .so -- this table is that reference.
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// One pack's contribution to the provider.
struct IngestorPack
{
    std::string_view label;
    /// May throw; caller rolls back this pack alone.
    void (*registerSymbols)(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
    /// Drops this pack's cached kpack modules, or null for a pack that loads no kpack
    /// archive. Tests only -- see resetIngestorModuleCachesForTesting().
    void (*resetModuleCache)();
};

/// Every pack this provider ships. **Adding an engine edits this table.**
const std::vector<IngestorPack>& ingestorPacks();

/// One function per pack, one per file: each pack's matchers, scorer, and dispatch handler
/// are internal to its native file, reachable only through the registry, so there's no
/// per-pack header.

/// @see packs/PointwiseNative.cpp
void registerPointwiseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetPointwiseModuleCache();

/// @see packs/ConvNative.cpp
void registerConvFwdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetConvFwdModuleCache();

/// The engine name gfx950_attention_dense.ued.json declares. The pack table, the pack's
/// own log lines and the descriptor census (which looks the loaded set up by this name)
/// all read it here, so a rename that misses the descriptor fails the census.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_ENGINE_NAME
    = "hipkernel:Gfx950AttentionDense";

/// @see packs/FlydslRmsNormNative.cpp
void registerFlydslRmsNormSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslRmsNormModuleCache();

/// @see packs/FlydslRmsNormBwdNative.cpp
void registerFlydslRmsNormBwdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslRmsNormBwdModuleCache();

/// @see packs/FlydslSdpaRdnaNative.cpp
void registerFlydslSdpaRdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaRdnaModuleCache();

/// @see packs/FlydslSdpaRdnaDecodeNative.cpp
void registerFlydslSdpaRdnaDecodeSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaRdnaDecodeModuleCache();

/// @see packs/FlydslSdpaAiwNative.cpp
void registerFlydslSdpaAiwSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaAiwModuleCache();

/// @see packs/FlydslSdpaCdnaNative.cpp
void registerFlydslSdpaCdnaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaCdnaModuleCache();

/// @see packs/FlydslSdpaCdnaDualwaveNative.cpp
void registerFlydslSdpaCdnaDualwaveSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaCdnaDualwaveModuleCache();

/// @see packs/FlydslSdpaCdnaDecodeNative.cpp
void registerFlydslSdpaCdnaDecodeSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaCdnaDecodeModuleCache();

/// @see packs/FlydslSdpaDropoutMaskNative.cpp
void registerFlydslSdpaDropoutMaskSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaDropoutMaskModuleCache();

/// @see packs/FlydslSdpaHeadPadNative.cpp
void registerFlydslSdpaHeadPadSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaHeadPadModuleCache();

/// @see packs/Gfx950AttentionDenseNative.cpp
void registerGfx950AttentionDenseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx950AttentionDenseModuleCache();

/// The engine name gfx942_attention_unified3d.ued.json declares. Read here by the pack
/// table, the pack's own log lines and the census, like GFX950_ATTENTION_DENSE_ENGINE_NAME.
inline constexpr std::string_view GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME
    = "hipkernel:Gfx942AttentionUnified3d";

/// @see packs/Gfx942AttentionUnified3dNative.cpp
void registerGfx942AttentionUnified3dSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx942AttentionUnified3dModuleCache();
/// The engine name gfx950_attention_unified_2d.ued.json declares, shared the same way.
inline constexpr std::string_view GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME
    = "hipkernel:Gfx950AttentionUnified2D";

/// @see packs/Gfx950AttentionUnified2DNative.cpp
void registerGfx950AttentionUnified2DSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx950AttentionUnified2DModuleCache();

/// The engine name gfx950_attention_unified_3d.ued.json declares, shared the same way.
inline constexpr std::string_view GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME
    = "hipkernel:Gfx950AttentionUnified3D";

/// @see packs/Gfx950AttentionUnified3DNative.cpp
void registerGfx950AttentionUnified3DSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx950AttentionUnified3DModuleCache();
/// The engine name gfx942_attention_dense.ued.json declares; see
/// GFX950_ATTENTION_DENSE_ENGINE_NAME.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_ENGINE_NAME
    = "hipkernel:Gfx942AttentionDense";

/// @see packs/Gfx942AttentionDenseNative.cpp
void registerGfx942AttentionDenseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx942AttentionDenseModuleCache();

/// The engine name gfx942_attention_unified_2d.ued.json declares, shared the same way.
inline constexpr std::string_view GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME
    = "hipkernel:Gfx942AttentionUnified2D";

/// @see packs/Gfx942AttentionUnified2DNative.cpp
void registerGfx942AttentionUnified2DSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx942AttentionUnified2DModuleCache();

/// A preference between two engines that accept the same graph: when both are in a call's
/// applicable list, @p preferred is listed ahead of @p other.
struct EngineOrderRule
{
    std::string_view preferred;
    std::string_view other;
};

/// The preferences the provider states, applied by the ApplicabilityManager to every list it
/// returns. hipDNN ranks an engine that has no rule of its own (sortEngineIds) in the order a
/// provider lists it, which for this provider is its engine map's hash order, so the order of
/// two engines serving the same graph would otherwise depend on the hash.
///
/// Each pair is a dense-attention engine and the paged unified-attention 2D engine of the same
/// arch that accept some of the same graphs; the preferred engine comes first. On gfx942 both
/// accept a dense BSHD SDPA graph of head size 64 or 128 whenever seqlen_q equals seqlen_kv (a
/// top-left and a bottom-right causal mask coincide there), and the dense engine was measured
/// faster on every shared shape of the ground truth. A pair is listed only once its order is
/// measured.
inline constexpr std::array<EngineOrderRule, 1> ENGINE_ORDER_RULES{{
    {GFX942_ATTENTION_DENSE_ENGINE_NAME, GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME},
}};

/// @see packs/Gfx1151WmmaAttentionNative.cpp, which serves both engines
void registerGfx1151WmmaAttentionSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx1151WmmaAttentionModuleCache();
void registerGfx1151WmmaAttentionDecodeSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx1151WmmaAttentionDecodeModuleCache();

/// @see packs/ConvDgradAotNative.cpp, which serves both engines
/// (hipkernel:Gfx950ConvDgradAot, hipkernel:Gfx1250ConvDgradAot)
void registerGfx950ConvDgradAotSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx950ConvDgradAotModuleCache();
void registerGfx1250ConvDgradAotSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx1250ConvDgradAotModuleCache();

/// Sets the per-image byte limit conv_dgrad_aot::batchChunks() slices a batch at (both
/// dgrad engines share it), in the provider's own loaded copy of that test seam -- see
/// ConvDgradAotGeometry.hpp::batchChunkByteLimitForTesting(). A test in another TU
/// (an integration test that loads this engine as a separate plugin .so) mutating its
/// own copy of that inline function's local static never reaches the one the plugin's
/// plan build actually reads; this function runs inside the plugin. FOR TESTS ONLY.
void setConvDgradAotBatchChunkByteLimitForTesting(int64_t limit);

/// The implicit-GEMM kernel launches hipkernel:Gfx950ConvDgradAot has made in this
/// process, one per batch slice: how a test sees that a plan ran sliced. FOR TESTS ONLY.
int64_t convDgradAotImplicitLaunchCountForTesting();

/// Drops every pack's cached kpack modules and closes every kpack archive
/// SharedKpackArchives retains, so the next dispatch re-reads its archive from disk. Leases
/// stay held: the archive that next dispatch opens is retained again.
///
/// FOR TESTS ONLY. Nothing in the product calls this: module residency is a deliberate
/// process-lifetime guarantee -- one hipModule_t per (archive, toc_key, arch) -- not a
/// cache to be invalidated. It exists because a test that deliberately corrupts a
/// staged archive cannot otherwise observe the failure it asserts on: a resident module
/// serves the plan and nothing reads the damaged bytes.
///
/// Clearing releases each cache's own reference only. A module still held by a live
/// plan stays loaded until that plan drops it, so this cannot unload a module out from
/// under a running dispatch.
void resetIngestorModuleCachesForTesting();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
