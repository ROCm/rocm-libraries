// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"

/**
 * @file IngestorPacks.hpp
 * @brief The provider's pack inventory: the one file adding an engine edits.
 *
 * A pack contributes native symbols, which no descriptor file can supply and which
 * outlive ALMIOPEN-2401, and a descriptor set, which that ticket turns into a parsed
 * file.
 *
 * Registration is a table entry rather than a self-registering static because unit
 * tests link the provider as a static archive, and a linker takes an archive member
 * only to resolve a reference. A self-registering TU nothing names is dropped from the
 * test binary while surviving in the plugin .so, which links objects directly. This
 * table is that reference.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// One pack's contribution to the provider.
struct IngestorPack
{
    /// Names the pack in a log line.
    std::string_view label;
    /// May throw; the caller rolls back this pack alone and carries on.
    void (*registerSymbols)(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
    /// Must not touch any registry: called for enumeration alone, before symbols are
    /// registered.
    hipdnn_plugin_sdk::ingestor::DescriptorSet (*buildDescriptorSet)();
};

/**
 * @brief Every pack this provider ships. **Adding an engine edits this table.**
 *
 * Registration and enumeration follow this order, so a name collision resolves the
 * same way on every run.
 */
const std::vector<IngestorPack>& ingestorPacks();

/**
 * @name Pack entry points
 *
 * Two functions per pack, one per file, and nothing else. A pack's matchers, scorer,
 * and dispatch handler are internal to its native file, reachable only through the
 * registry its descriptors name, so there is no per-pack header.
 *
 * Declared here rather than in the .cpp because a definition with no visible
 * declaration reads as one that should have been internal.
 *
 * ALMIOPEN-2401 replaces the descriptor column with installed files and takes those
 * declarations with it; the native ones are unaffected.
 * @{
 */

/// @see packs/PointwiseAddNative.cpp
void registerPointwiseAddSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
/// @see packs/PointwiseAddDescriptors.cpp
hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseAddDescriptorSet();

/// @see packs/PointwiseSubNative.cpp
void registerPointwiseSubSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
/// @see packs/PointwiseSubDescriptors.cpp
hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseSubDescriptorSet();

/** @} */

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
