// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
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

/// @see packs/FlydslSdpaBwdMergedNative.cpp
void registerFlydslSdpaBwdMergedSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaBwdMergedModuleCache();

/// What one launch of the merged SDPA backward engine hands the device, resolved without
/// touching it. FOR TESTS ONLY: the product launches through the dispatch handler, and this
/// is the same argument computation, so a test can read the kernarg images, grids and
/// workspace offsets a launch would use.
struct FlydslSdpaBwdMergedLaunchDump
{
    /// The merged kernel's arguments, each at its natural offset, and its launch.
    std::vector<unsigned char> kernarg;
    std::array<unsigned int, 3> grid{};
    unsigned int block = 0;
    /// Set when the graph's dBias is broadcast: the reduce kernel's symbol, arguments and launch.
    bool reduces = false;
    std::string reduceSymbol;
    std::vector<unsigned char> reduceKernarg;
    std::array<unsigned int, 3> reduceGrid{};
    unsigned int reduceBlock = 0;
    /// Set when the graph needs the padded-row tail-zero launch after the kernel (and the
    /// reduce): its arguments and launch.
    bool tailzero = false;
    std::vector<unsigned char> tailzeroKernarg;
    std::array<unsigned int, 3> tailzeroGrid{};
    unsigned int tailzeroBlock = 0;
    size_t workspaceBytes = 0;
    /// Set when a pass-by-value seed and offset are written into the workspace ahead of the
    /// launch, as the four words in seedWords (seed low, seed high, offset low, offset high).
    bool seedUploaded = false;
    std::vector<uint32_t> seedWords;
};

/// The launch of @p kernel on the graph of @p context, over @p buffers (by tensor uid) and a
/// workspace that starts at @p workspace; the address is only offset into, never dereferenced.
/// @throws HipdnnPluginException with the status prepare() or launch() would throw.
FlydslSdpaBwdMergedLaunchDump describeFlydslSdpaBwdMergedLaunchForTesting(
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
    const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
    void* workspace);

/// @see packs/FlydslSdpaBwdSplitNative.cpp
void registerFlydslSdpaBwdSplitSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaBwdSplitModuleCache();

/// The slots of the split SDPA backward engine, in launch order.
enum class FlydslSdpaBwdSplitSlot
{
    PREP,
    DKDV,
    DQ
};

/// What one launch of a split slot hands the device, resolved without touching it. FOR TESTS
/// ONLY, as FlydslSdpaBwdMergedLaunchDump is for the merged engine: the product launches
/// through the slot's dispatch handler, and this is the same argument computation.
struct FlydslSdpaBwdSplitLaunchDump
{
    /// The slot kernel's arguments, each at its natural offset, and its launch.
    std::vector<unsigned char> kernarg;
    std::array<unsigned int, 3> grid{};
    unsigned int block = 0;
    /// Set on the dkdv slot when the row serves V heads that differ from K's: the launch above is
    /// the dK-only object over the K heads and is followed by the dV-only object, with the same
    /// arguments, over the V heads: its launch.
    bool dvonly = false;
    std::array<unsigned int, 3> dvonlyGrid{};
    unsigned int dvonlyBlock = 0;
    /// Set when the launch is followed by the dBias reduce kernel: its symbol, arguments and
    /// launch.
    bool reduces = false;
    std::string reduceSymbol;
    std::vector<unsigned char> reduceKernarg;
    std::array<unsigned int, 3> reduceGrid{};
    unsigned int reduceBlock = 0;
    /// Set on the dq slot when the graph needs the padded-row tail-zero launch after the
    /// kernel (and the reduce): its arguments and launch.
    bool tailzero = false;
    std::vector<unsigned char> tailzeroKernarg;
    std::array<unsigned int, 3> tailzeroGrid{};
    unsigned int tailzeroBlock = 0;
    /// The slot handler's workspaceBytes(): the scratch this slot needs, shared region excluded.
    size_t scratchBytes = 0;
    /// Set when the launch writes the seed and offset into the shared region ahead of the
    /// kernel (the prep slot only); see FlydslSdpaBwdMergedLaunchDump.
    bool seedUploaded = false;
    std::vector<uint32_t> seedWords;
};

/// The launch of @p slot's @p kernel on the graph of @p context, with the tokens @p bound that
/// graph_match published, over @p buffers (by tensor uid) and a workspace that starts at
/// @p workspace; the address is only offset into, never dereferenced.
/// @throws HipdnnPluginException with the status prepare() or launch() would throw.
FlydslSdpaBwdSplitLaunchDump describeFlydslSdpaBwdSplitLaunchForTesting(
    FlydslSdpaBwdSplitSlot slot,
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
    const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
    const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
    void* workspace);

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
