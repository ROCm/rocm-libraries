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
    size_t workspaceBytes = 0;
    /// Set when the seed and offset are written into the workspace ahead of the launch: either
    /// as the four words in seedWords (seed low, seed high, offset low, offset high), or, with
    /// seedCopiedFromDevice, by copying the 4 bytes at seedSource and offsetSource (the device
    /// addresses of the 32-bit seed and offset tensors).
    bool seedUploaded = false;
    bool seedCopiedFromDevice = false;
    std::vector<uint32_t> seedWords;
    const void* seedSource = nullptr;
    const void* offsetSource = nullptr;
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
    /// Set when the launch is followed by the dBias reduce kernel: its symbol, arguments and
    /// launch.
    bool reduces = false;
    std::string reduceSymbol;
    std::vector<unsigned char> reduceKernarg;
    std::array<unsigned int, 3> reduceGrid{};
    unsigned int reduceBlock = 0;
    /// The slot handler's workspaceBytes(): the scratch this slot needs, shared region excluded.
    size_t scratchBytes = 0;
    /// Set when the launch writes the seed and offset into the shared region ahead of the
    /// kernel (the prep slot only); see FlydslSdpaBwdMergedLaunchDump for the two forms.
    bool seedUploaded = false;
    bool seedCopiedFromDevice = false;
    std::vector<uint32_t> seedWords;
    const void* seedSource = nullptr;
    const void* offsetSource = nullptr;
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

/// @see packs/FlydslSdpaDropoutMaskNative.cpp
void registerFlydslSdpaDropoutMaskSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetFlydslSdpaDropoutMaskModuleCache();

/// @see packs/Gfx950AttentionDenseNative.cpp
void registerGfx950AttentionDenseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx950AttentionDenseModuleCache();

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

/// @see packs/Gfx1151WmmaAttentionNative.cpp, which serves both engines
void registerGfx1151WmmaAttentionSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx1151WmmaAttentionModuleCache();
void registerGfx1151WmmaAttentionDecodeSymbols(
    hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
void resetGfx1151WmmaAttentionDecodeModuleCache();

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
