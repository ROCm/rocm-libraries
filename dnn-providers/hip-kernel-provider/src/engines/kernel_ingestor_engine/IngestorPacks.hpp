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
/// The provider's pack inventory: the one file adding an engine edits.
///
/// Registration is a table entry, not a self-registering static: a linker drops an
/// archive member nothing references (as in the unit-test static-archive build), even
/// though the same TU survives in the plugin .so. This table is that reference.
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// One pack's contribution to the provider.
struct IngestorPack
{
    std::string_view label;
    /// May throw; caller rolls back this pack alone.
    void (*registerSymbols)(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
    /// Must not touch any registry: called for enumeration before symbols register.
    hipdnn_plugin_sdk::ingestor::DescriptorSet (*buildDescriptorSet)();
};

/// Every pack this provider ships. **Adding an engine edits this table.**
const std::vector<IngestorPack>& ingestorPacks();

/// Two functions per pack, one per file; no per-pack header.

void registerPointwiseAddSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseAddDescriptorSet();

void registerPointwiseSubSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope);
hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseSubDescriptorSet();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
