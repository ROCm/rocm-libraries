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
};

/// Every pack this provider ships. **Adding an engine edits this table.**
const std::vector<IngestorPack>& ingestorPacks();

/// One function per pack, one per file: each pack's matchers, scorer, and dispatch handler
/// are internal to its native file, reachable only through the registry, so there's no
/// per-pack header.

/// @see packs/PointwiseNative.cpp
void registerPointwiseSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);

/// @see packs/ConvNative.cpp
void registerConvFwdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
