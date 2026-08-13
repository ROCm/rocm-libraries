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
 * @brief The provider's inventory of ingestor packs: the one file adding an engine
 *        edits.
 *
 * A pack contributes two things that arrive by different routes and have opposite
 * lifetimes:
 *
 *  - its **native symbols** (matchers, scorer, dispatch handler), which no descriptor
 *    file can ever supply, so this stays after ALMIOPEN-2401;
 *  - its **descriptor set**, C++ today and a parsed file after ALMIOPEN-2401, which is
 *    why discoverDescriptorSets() is deliberately the only place that builds one.
 *
 * Registration is a named function called from a table rather than a self-registering
 * static, and that is load-bearing rather than stylistic: unit tests link the provider
 * as a static archive (hip_kernel_provider_private), and a linker takes an archive
 * member only to resolve a symbol something references. A self-registering translation
 * unit nothing names is silently dropped from the test binary while surviving in the
 * plugin .so, which links the objects directly. The table below is that reference.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// One pack's contribution to the provider.
struct IngestorPack
{
    /// Diagnostic label, so a failing pack can be named in a log line.
    std::string_view label;
    /// Registers this pack's native symbols into @p scope. May throw; the caller
    /// rolls back this pack alone and carries on with the rest.
    void (*registerSymbols)(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
    /// Builds this pack's descriptor set. Must not touch any registry: it is called
    /// for enumeration alone, before symbols are registered.
    hipdnn_plugin_sdk::ingestor::DescriptorSet (*buildDescriptorSet)();
};

/**
 * @brief Every pack this provider ships. **Adding an engine edits this table.**
 *
 * Ordered: the sweep registers and enumerates in this order, so a name collision
 * resolves the same way on every run.
 */
const std::vector<IngestorPack>& ingestorPacks();

/**
 * @name Pack entry points
 *
 * Each pack contributes exactly these two functions, one per file, and nothing else:
 * the native seam from `<Op>Native.cpp` and the descriptor set from
 * `<Op>Descriptors.cpp`. A pack's matchers, scorer, and dispatch handler are internal
 * to its native file -- the registry the descriptors name is the only way to them --
 * so there is deliberately no per-pack header.
 *
 * Declared here rather than locally in IngestorPacks.cpp because a definition with no
 * visible declaration reads as one that should have been internal, and the table is
 * the only caller either has.
 *
 * Post-ALMIOPEN-2401 an installed descriptor file replaces the second column, and its
 * declaration goes with it; the native one is unaffected.
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
