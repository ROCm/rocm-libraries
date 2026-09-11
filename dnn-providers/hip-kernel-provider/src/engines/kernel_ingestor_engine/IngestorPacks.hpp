// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

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

/// Drops every pack's cached kpack modules, so the next dispatch re-reads its archive
/// from disk.
///
/// FOR TESTS ONLY. Nothing in the product calls this: module residency is a deliberate
/// process-lifetime guarantee -- one hipModule_t per (archive, toc_key, arch) -- not a
/// cache to be invalidated. It exists because a test that deliberately corrupts a
/// staged archive cannot otherwise observe the failure it asserts on: a resident module
/// serves the plan and the damaged bytes are read by nothing.
///
/// Clearing releases each cache's own reference only. A module still held by a live
/// plan stays loaded until that plan drops it, so this cannot unload a module out from
/// under a running dispatch.
void resetIngestorModuleCachesForTesting();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
